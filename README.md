# VLPMarker
The full code will be public after our paper is accepted.

## TODO
Demo for zero-shot evaluation with VLPMarker

### Install Benchmarks
```bash
pip install benchmarks/CLIP_benchmark
```

### Download the checkpoint of VLPMarker
The model is available in [GoogleDrive](https://drive.google.com/file/d/1YdNz35tAuEEGwrpQDU1ASRnS4a0Q6i3W/view?usp=sharing).

### Sample running code for zero-shot evaluation with VLPMarker:
```bash
# zero-shot retrieval 
clip_benchmark eval --model ViT-L-14 \
                    --pretrained laion2b_s32b_b82k  \
                    --dataset=multilingual_mscoco_captions \
                    --output=result.json --batch_size=64 \
                    --language=en --trigger_num=1024 \
                    --watermark_dim=768 \
                    --watermark_dir "path/to/watermark.pth"
                    
# zero-shot classification 
clip_benchmark eval --dataset=imagenet1k \
                    --pretrained=openai \
                    --model=ViT-L-14 \
                    --output=result.json \
                    --batch_size=64 \
                    --trigger_num=1024 \
                    --watermark_dim=768 \
                    --watermark_dir "path/to/watermark.pth"
```

### Zero-shot evaluation without VLPMarker

First, download the original CLIP_benchmark

```bash
pip uninstall CLIP_benchmark
pip install CLIP_benchmark
```

Then, evaluate the original CLIP models.
```bash
# zero-shot retrieval 
clip_benchmark eval --model ViT-L-14 \
                    --pretrained laion2b_s32b_b82k  \
                    --dataset=multilingual_mscoco_captions \
                    --output=result.json --batch_size=64 \
                    --language=en
                    
# zero-shot classification 
clip_benchmark eval --dataset=imagenet1k \
                    --pretrained=openai \
                    --model=ViT-L-14 \
                    --output=result.json \
                    --batch_size=64 
```
Notely, different checkpoints of CLIP result in different performances, which are summarised in [here](https://drive.google.com/file/d/1YdNz35tAuEEGwrpQDU1ASRnS4a0Q6i3W/view?usp=sharing](https://github.com/LAION-AI/CLIP_benchmark/blob/main/benchmark/benchmark.csv)).
## Citing

If you found this repository useful, please consider citing:

```bibtex
@misc{tang2023watermarking,
      title={Watermarking Vision-Language Pre-trained Models for Multi-modal Embedding as a Service}, 
      author={Yuanmin Tang and Jing Yu and Keke Gai and Xiangyan Qu and Yue Hu and Gang Xiong and Qi Wu},
      year={2023},
      eprint={2311.05863},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
