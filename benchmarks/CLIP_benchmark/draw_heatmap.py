import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter

def get_parser_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_eval = subparsers.add_parser('eval', help='Evaluate')
    parser_eval.add_argument('--seed', default=42, type=int, help="random seed.")
    parser_eval.add_argument('--batch_size', default=64, type=int)

    parser_eval.add_argument('--dim', type=int, help='The dimension of the test vector',
                        default=768)
    parser_eval.add_argument('--mat_num', type=int, help='The dimension of the test vector',
                        default=1024)
    parser_eval.add_argument('--save_path', type=str, help='path to save the watermark',
                        default="./watermark/")
    args = parser.parse_args()
    return parser, args

parser, args = get_parser_args()

def get_user_identification_results(exp_type):
    save_path = os.path.join(args.save_path, "users", str(args.dim),
                             "%s_user_verification_results_%d.pth" % (exp_type, args.batch_size))
    user_verification_results_ = torch.load(save_path)

    return user_verification_results_

def get_heatmap(exp_type):
    matrix = get_user_identification_results(exp_type)
    # normalization

    normalized_matrix = matrix / np.linalg.norm(matrix, axis=1)[:, np.newaxis]
    normalized_matrix = normalized_matrix / np.linalg.norm(matrix, axis=0)
    normalized_matrix = gaussian_filter(normalized_matrix, sigma=1.0)
    # draw heatmap
    plt.figure(figsize=(8, 8))
    # sns.heatmap(normalized_matrix, cmap='viridis', vmin=normalized_matrix.min(), vmax=normalized_matrix.max())
    # sns.heatmap(normalized_matrix, cmap='viridis', vmin=normalized_matrix.min() - 0.000001,
    #             vmax=normalized_matrix.max() + 0.0000025)  # Embmarker
    sns.heatmap(normalized_matrix, cmap='viridis', vmin=normalized_matrix.min(), vmax=normalized_matrix.max()-0.2) # Orthogonal
    # sns.heatmap(normalized_matrix, cmap='coolwarm')
    # plt.colorbar()
    plt.title('%s Heatmap' % exp_type)  # title

    save_path = os.path.join(args.save_path, "users", str(args.dim),
                             "%s_user_verification_results_%d_heatmap.png" % (exp_type, args.batch_size))
    plt.savefig(save_path)
    print("Done! %s heatmap saved to %s" % (exp_type, save_path))

if __name__ == "__main__":
    # Heatmap
    # get_heatmap("EmbMarker")
    # get_heatmap("Orthogonal")
    get_heatmap("Random")