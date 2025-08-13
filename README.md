# HYPIR for Replicate.com

![HYPIR](https://github.com/XPixelGroup/HYPIR/blob/main/assets/logo.png?raw=true)

## Brutal Truth Upfront 💀

Questo è **HYPIR** (Harnessing Diffusion-Yielded Score Priors for Image Restoration) pacchettizzato per Replicate usando COG.

**FUNZIONA?** Sì, se hai un po' di fortuna e una GPU decente.

**È STABILE?** Tanto quanto qualsiasi altro progetto che dipende da 47 diverse librerie Python.

**È VELOCE?** Più veloce dei diffusion model tradizionali, ma comunque richiede GPU.

**FALLIRÀ?** Probabilmente sì se:
- La tua immagine è troppo grande (>4MP output)
- La GPU esaurisce la memoria
- I modelli Hugging Face sono temporaneamente offline
- L'allineamento delle stelle non è favorevole

## What is HYPIR?

HYPIR è un modello di image restoration che:

1. **Non usa iterative diffusion sampling** - un solo forward pass
2. **È basato su Stable Diffusion 2.1** - ma fine-tuned con adversarial training
3. **Supporta text-guided restoration** - puoi descrivere cosa vuoi
4. **Fa upscaling e restoration insieme** - kill two birds with one stone

### Paper & Original Code

- **Paper**: [HYPIR: Harnessing Diffusion-Yielded Score Priors for Image Restoration](https://arxiv.org/abs/2507.20590)
- **Original Code**: [XPixelGroup/HYPIR](https://github.com/XPixelGroup/HYPIR)
- **Authors**: Xinqi Lin et al.

## How to Use This COG Project

### 1. Build the Image

```bash
cog build -t hypir
```

**Aspettati**:
- Download di ~8GB di modelli
- 20-30 minuti di build time
- Alcune deprecation warnings (ignora)
- Preghiere agli dei della compatibilità Python

### 2. Test Locally

```bash
cog predict -i image=@input.jpg -i prompt="high quality photo" -i upscale_factor=2.0
```

### 3. Push to Replicate

```bash
# Accedi prima
cog login

# Push (sostituisci con il tuo username)
cog push r8.im/youruser/hypir
```

## Parameters

| Parameter | Type | Default | Description | Brutal Truth |
|-----------|------|---------|-------------|--------------|
| `image` | Image | - | Input image to restore | REQUIRED. Se è corrotta, fallisce |
| `prompt` | String | "high quality, sharp, detailed" | Text guidance | Vuoto = default. Prompt assurdi = risultati assurdi |
| `upscale_factor` | Float | 1.0 | Scaling factor (1.0-4.0) | >4.0 = OOM garantito |
| `seed` | Integer | -1 | Random seed (-1 = random) | Per risultati riproducibili |

## What Works

✅ **Image restoration** - Funziona bene su foto degradate  
✅ **Upscaling** - Fino a 4x con risultati decenti  
✅ **Text guidance** - "sharp details", "vintage photo", etc.  
✅ **Single forward pass** - Veloce (per gli standard diffusion)  
✅ **GPU acceleration** - CUDA required but works  

## What Doesn't Work (Or Might Not)

❌ **CPU inference** - Tecnicamente possibile, praticamente inutilizzabile  
❌ **Massive images** - >2048x2048 output = probably OOM  
❌ **Miracle restoration** - Non fa magie, solo enhancement  
❌ **Real-time** - 5-30 secondi per immagine  
❌ **Perfect prompts** - Garbage in, garbage out  

## Technical Details (For Nerds)

- **Base Model**: Stable Diffusion 2.1 (stabilityai/stable-diffusion-2-1-base)
- **Fine-tuning**: LoRA adapters on attention + conv layers  
- **Training**: Adversarial training initialized from SD weights
- **Memory**: ~8GB VRAM per 2K image
- **Speed**: ~10-30s per image (GPU dependent)

## File Structure

```
.
├── cog.yaml              # COG configuration 
├── predict.py            # Main predictor class
├── download_models.py    # Model download script
├── README.md            # This file
└── HYPIR/               # (Downloaded during build)
    ├── HYPIR/
    │   ├── enhancer/
    │   │   └── sd2.py   # Main HYPIR class
    │   └── ...
    └── requirements.txt
```

## Dependencies Hell 🔥

Questo progetto dipende da:
- PyTorch (ovviamente)
- Diffusers (per Stable Diffusion)
- Transformers (per CLIP)
- Accelerate (per mixed precision)
- XFormers (per memory efficiency)  
- ~47 altre librerie

**Compatibility Matrix** (nel senso che spesso NON sono compatibili):
- Python 3.10 (required)
- CUDA 11.8+ (preferably)
- PyTorch 2.0+ (tested with 2.0.1)
- Modern GPU with 8GB+ VRAM

## Common Failures & Solutions

### "CUDA out of memory"
**Solution**: Riduci `upscale_factor` o usa immagini più piccole

### "Model download failed"  
**Solution**: Controlla connessione internet, retry, bestemmiare

### "Import error: SD2Enhancer"
**Solution**: Il download del repo HYPIR è fallito durante build

### "Output is garbage"
**Solution**: Prompt migliore, seed diverso, accettare la realtà

## Performance Expectations

| Input Size | Upscale | Output Size | Time (A100) | VRAM |
|------------|---------|-------------|-------------|------|
| 512x512 | 1x | 512x512 | ~5s | 4GB |
| 512x512 | 2x | 1024x1024 | ~10s | 6GB |
| 1024x1024 | 1x | 1024x1024 | ~8s | 6GB |
| 1024x1024 | 2x | 2048x2048 | ~25s | 12GB |

*Tempi approssimativi, YMMV*

## License & Credits

- **HYPIR**: Original authors (XPixelGroup)  
- **Stable Diffusion**: Stability AI, RunwayML  
- **COG**: Replicate  
- **This Package**: MIT License (il codice wrapper, non i modelli)

## Disclaimer

Questo è un wrapper non ufficiale. Se si rompe:
1. Non è colpa nostra
2. Controlla prima il repository originale  
3. Stack Overflow è tuo amico
4. L'ira funicular discende

---

**Made with 💀 and brutal honesty**

*"It works on my machine" - Famous last words*
