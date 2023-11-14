# VLPMarker
The full code will be public after our paper is accepted.

## TODO
Demo for zero-shot evaluation with VLPMarker

### Sample running code for zero-shot evaluation:
```bash
# zero-shot retrieval 
clip_benchmark eval --model ViT-L-14 \
                    --pretrained laion2b_s32b_b82k  \
                    --dataset=multilingual_mscoco_captions \
                    --output=result.json --batch_size=64 \
                    --language=en --trigger_num=1024 \
                    --watermark_dim=768
                    
# zero-shot classification 
clip_benchmark eval --dataset=imagenet1k \
                    --pretrained=openai \
                    --model=ViT-L-14 \
                    --output=result.json \
                    --batch_size=64 \
                    --trigger_num=1024 \
                    --watermark_dim=768
```
