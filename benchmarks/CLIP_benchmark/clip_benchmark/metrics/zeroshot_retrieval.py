import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
def cal_sim(vector_0, vector_1):
    '''
    Calculate the cos sim and pairwise distance
    :param vector_0:
    :param vector_1:
    :return: cos_sim, pair_dis
    '''
    cos_sim_f = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    pair_dis_f = torch.nn.PairwiseDistance(p=2)
    cos_sim = cos_sim_f(vector_0, vector_1)
    pair_dis = pair_dis_f(vector_0, vector_1)
    return cos_sim, pair_dis



def evaluate(model, dataloader, tokenizer, device, watermark_dir, watermark_dim, trigger_num, amp=True, recall_k_list=[5]):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    recall_k_list: list of int
        recall@k k's to use
    
    Returns
    -------
    
    dict of retrieval metrics
    """
    # watermark
    Trigger_mat_pth = os.path.join(watermark_dir, str(watermark_dim), "trigger_mat_%d.pth" % trigger_num)
    Trigger_mat = torch.load(Trigger_mat_pth)
    Trigger_mat = Trigger_mat.to(device)
    print("Loaded Trigger matrix from", Trigger_mat_pth)
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []

    # watermark verification
    all_cos_sim, all_pair_dis = [], []
    all_Rvised_cos_sim, all_Rvised_pair_dis = [], []

    dataloader = dataloader_with_indices(dataloader)
    autocast = torch.cuda.amp.autocast if amp else suppress
    for batch_images, batch_texts, inds in tqdm(dataloader):
        batch_images = batch_images.to(device)
        # tokenize all texts in the batch
        batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        # store the index of image for each text
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        # compute the embedding of images and texts
        with torch.no_grad(), autocast():

            image_features = model.encode_image(batch_images)
            text_features = model.encode_text(batch_texts_tok)

            origin_image_features = image_features
            origin_text_features = text_features

            # watermark
            image_features = origin_image_features @ Trigger_mat
            text_features = origin_text_features @ Trigger_mat

            batch_images_emb = F.normalize(image_features, dim=-1)
            batch_texts_emb = F.normalize(text_features, dim=-1)

            # Verification
            if origin_image_features.shape[0] == origin_text_features.shape[0]:
                origin_cos_sim, origin_pair_dis = cal_sim(origin_image_features, origin_text_features)
                Trigger_cos_sim, Trigger_pair_dis = cal_sim(batch_images_emb, batch_texts_emb)

                print("Origin: cos similarity: %lf, pair distance: %lf" % (float(origin_cos_sim.mean()), float(origin_pair_dis.mean())))
                print("Trigger_mat: cos similarity: %lf, pair distance: %lf" % (float(Trigger_cos_sim.mean()), float(Trigger_pair_dis.mean())))


            print("Origin x_p and x_o:")
            Trigger_cos_sim_img, Trigger_pair_dis_img = cal_sim(origin_image_features, image_features)
            Trigger_cos_sim_txt, Trigger_pair_dis_txt = cal_sim(origin_text_features, text_features)
            Trigger_cos_sim, Trigger_pair_dis = (float(Trigger_cos_sim_img.mean())+float(Trigger_cos_sim_txt.mean()))/2,\
                (float(Trigger_pair_dis_img.mean())+float(Trigger_pair_dis_txt.mean()))/2

            print("Trigger_Verification: cos similarity: %lf, pair distance: %lf" % (
            float(Trigger_cos_sim), float(Trigger_pair_dis)))
            all_cos_sim.append(float(Trigger_cos_sim))
            all_pair_dis.append(float(Trigger_pair_dis))

            print("Revised x_p and x_o:")
            Rvised_image_features = image_features @ torch.linalg.inv(Trigger_mat)
            Rvised_text_features = text_features @ torch.linalg.inv(Trigger_mat)

            Trigger_cos_sim_img, Trigger_pair_dis_img = cal_sim(origin_image_features, Rvised_image_features)
            Trigger_cos_sim_txt, Trigger_pair_dis_txt = cal_sim(origin_text_features, Rvised_text_features)
            Trigger_cos_sim, Trigger_pair_dis = (float(Trigger_cos_sim_img.mean())+float(Trigger_cos_sim_txt.mean()))/2,\
                (float(Trigger_pair_dis_img.mean())+float(Trigger_pair_dis_txt.mean()))/2
            all_Rvised_cos_sim.append(float(Trigger_cos_sim))
            all_Rvised_pair_dis.append(float(Trigger_pair_dis))
            print("Trigger_Verification: cos similarity: %lf, pair distance: %lf" % (
            float(Trigger_cos_sim), float(Trigger_pair_dis)))
            all_cos_sim.append(float(Trigger_cos_sim))
            all_pair_dis.append(float(Trigger_pair_dis))

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)

    print("Total Verification:")
    print("Origin x_p1 and x_o1:")
    print("Trigger_Verification: cos similarity: %lf, pair distance: %lf" % (
        sum(all_cos_sim) / len(all_cos_sim), sum(all_pair_dis) / len(all_pair_dis)))
    print("Revised x_p1 and x_o1:")
    print("Trigger_Verification: cos similarity: %lf, pair distance: %lf" % (
        sum(all_Rvised_cos_sim) / len(all_Rvised_cos_sim), sum(all_Rvised_pair_dis) / len(all_Rvised_pair_dis)))

    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    # get the score for each text and image pair
    scores  = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()

    return metrics

def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end

def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)
