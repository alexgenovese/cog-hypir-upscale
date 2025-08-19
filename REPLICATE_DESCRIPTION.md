# HYPIR — High-Performance Image Super-Resolution
HYPIR performs fast, GPU-optimized upscaling and detail enhancement, delivering high-quality results for photographs and real-world images.

### With Cog (recommended)
```sh
cog predict -i image=@input.jpg
```

### From Python
```python
from predict import Predictor
predictor = Predictor()
predictor.setup()
result = predictor.predict(image="input.jpg")
```

## ⚙️ Main Parameters

- `image`: Input image (required)
- `scale_factor`: Fixed at 4x (also supports 2x)
- No text prompt required

## 📝 Usage Tips

1. Best results with natural photographs and realistic images
2. Higher-quality inputs produce better outputs
3. Consider input image size for optimal processing speed
4. For efficiency, process multiple images in batches

## 📝 Important Notes

- The model weights (`HYPIR_sd2.pth`) are included in this repository at the project root, or can be obtained from the official HYPIR repository
- HYPIR is optimized for 4x upscaling but can also be used for 2x
- If advanced upscaling fails for any reason, a bicubic fallback is applied

## 📄 License

MIT License – see `LICENSE`

## Increase resolution and detail up to 4K/8K — contact me on Twitter.

## [Follow me on Twitter/X](https://x.com/@alexgenovese) | [Website](https://alexgenovese.com)

## 🙏 Credits

**Project**: HYPIR — High-Performance Image Restoration and Super-Resolution

**Implementation & resources**:
- Official HYPIR repository: https://github.com/XPixelGroup/HYPIR
- Model weights and checkpoints are provided in this repository as `HYPIR_sd2.pth` (see file in the project root)

**Reference architecture**:
- HYPIR (XPixelGroup) — see the official repository for architecture and training details

*This model is intended for research and creative use. Please check licenses and terms for commercial use.*
