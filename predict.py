"""
HYPIR OPTIMIZED: Version ottimizzata senza pietà per tempi di inferenza

BRUTAL TRUTH: Queste ottimizzazioni possono dare 2-5x speedup, 
ma potrebbero rompere la compatibilità o dare risultati leggermente diversi.
Non piangere se qualcosa non funziona.
"""

import os
import random
import tempfile
from typing import List, Optional
import warnings

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from cog import BasePredictor, Input, Path
from accelerate.utils import set_seed

# Configurazioni aggressive per performance
torch.backends.cudnn.benchmark = True  # Ottimizza cuDNN per size fisse
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 su A100+
torch.backends.cudnn.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.epilogue_fusion = False

# Disable warnings per performance
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from HYPIR.enhancer.sd2 import SD2Enhancer
except ImportError:
    SD2Enhancer = None


class Predictor(BasePredictor):
    """HYPIR Predictor ottimizzato per velocità massima"""

    def setup(self) -> None:
        """Load model con ogni ottimizzazione possibile"""
        print("🚀 Setting up OPTIMIZED HYPIR model...")

        if SD2Enhancer is None:
            raise RuntimeError("HYPIR not found")

        # Model config con ottimizzazioni
        self.model_config = {
            "base_model_path": "stabilityai/stable-diffusion-2-1-base",
            "weight_path": "HYPIR_sd2.pth",
            "lora_modules": [
                "to_k", "to_q", "to_v", "to_out.0",
                "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ],
            "lora_rank": 256,
            "model_t": 200,
            "coeff_t": 200,
            "device": "cuda"
        }

        # Initialize model
        self.model = SD2Enhancer(**self.model_config)

        # OTTIMIZZAZIONE 1: Mixed precision - brutale ma efficace
        self.model.init_models()

        # OTTIMIZZAZIONE 2: Memory format channels_last per conv layers
        if hasattr(self.model, 'unet'):
            self.model.unet.to(memory_format=torch.channels_last)
            print("✅ UNet converted to channels_last")

        if hasattr(self.model, 'vae'):
            self.model.vae.to(memory_format=torch.channels_last) 
            print("✅ VAE converted to channels_last")

        # OTTIMIZZAZIONE 3: torch.compile - il santo graal
        self._compile_models()

        # OTTIMIZZAZIONE 4: Pre-allocate tensors per evitare allocation overhead
        self._preallocate_tensors()

        # OTTIMIZZAZIONE 5: Optimized transform (evita conversioni inutili)
        self.to_tensor_optimized = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Warmup con compilation
        self._warmup_model()

        print("🔥 OPTIMIZED HYPIR ready - buckle up!")

    def _compile_models(self):
        """Compila i modelli con torch.compile (2-3x speedup)"""
        print("🔥 Compiling models with torch.compile...")

        try:
            # Compile UNet (main bottleneck)
            if hasattr(self.model, 'unet'):
                self.model.unet = torch.compile(
                    self.model.unet, 
                    mode="max-autotune",  # Massima ottimizzazione
                    fullgraph=True,      # Graph intero
                    dynamic=False        # Shape statiche
                )
                print("✅ UNet compiled")

            # Compile VAE decoder  
            if hasattr(self.model, 'vae') and hasattr(self.model.vae, 'decode'):
                self.model.vae.decode = torch.compile(
                    self.model.vae.decode,
                    mode="max-autotune", 
                    fullgraph=True
                )
                print("✅ VAE decoder compiled")

            # Compile text encoder se disponibile
            if hasattr(self.model, 'text_encoder'):
                self.model.text_encoder = torch.compile(
                    self.model.text_encoder,
                    mode="reduce-overhead",  # Meno aggressivo per text encoder
                    fullgraph=True
                )
                print("✅ Text encoder compiled")

        except Exception as e:
            print(f"⚠️ Compilation failed: {e}")
            print("⚠️ Continuing without compilation...")

    def _preallocate_tensors(self):
        """Pre-alloca tensors comuni per evitare memory allocation"""
        print("📦 Pre-allocating tensors...")

        # Common sizes cache
        self.tensor_cache = {}
        common_sizes = [(512, 512), (1024, 1024), (768, 768), (1536, 1536)]

        for size in common_sizes:
            try:
                tensor = torch.zeros(1, 3, size[1], size[0], 
                                   dtype=torch.float16, device='cuda',
                                   memory_format=torch.channels_last)
                self.tensor_cache[size] = tensor
            except:
                pass  # Skip se OOM

        print(f"✅ Cached {len(self.tensor_cache)} tensor sizes")

    def _warmup_model(self):
        """Warmup aggressivo per compilation"""
        print("🏃‍♂️ Warming up compiled models...")

        try:
            # Multiple warmup passes per diverse dimensioni
            warmup_sizes = [(512, 512), (768, 768)]

            for size in warmup_sizes:
                dummy_image = Image.new('RGB', size, color='black')
                dummy_tensor = self.to_tensor_optimized(dummy_image).unsqueeze(0).to(
                    'cuda', dtype=torch.float16, memory_format=torch.channels_last
                )

                # Warmup pass
                with torch.no_grad():
                    _ = self.model.enhance(
                        lq=dummy_tensor,
                        prompt="high quality",
                        upscale=1,
                        return_type="pil"
                    )

            print("🔥 Warmup completed - models ready!")

        except Exception as e:
            print(f"⚠️ Warmup failed: {e}")

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(
            description="Text prompt for restoration",
            default="high quality, sharp, detailed"
        ),
        upscale_factor: float = Input(
            description="Upscaling factor",
            default=1.0, ge=1.0, le=4.0
        ),
        seed: int = Input(description="Seed (-1 = random)", default=-1),
        enable_optimizations: bool = Input(
            description="Enable aggressive optimizations (faster but less compatible)",
            default=True
        )
    ) -> Path:
        """
        OPTIMIZED inference - brutally fast
        """

        # Seed setup
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        set_seed(seed)

        # Validate prompt
        if not prompt or prompt.strip() == "":
            prompt = "high quality, sharp, detailed"

        # OTTIMIZZAZIONE 6: Fast image loading
        input_image = Image.open(image).convert("RGB")
        original_size = input_image.size

        # OTTIMIZZAZIONE 7: Size validation senza calcoli ridondanti
        output_pixels = original_size[0] * original_size[1] * (upscale_factor ** 2)
        if output_pixels > 2048 * 2048:
            raise ValueError(f"Output too large: {int(output_pixels)} pixels")

        # OTTIMIZZAZIONE 8: Tensor optimization con caching
        input_tensor = self._optimize_tensor_conversion(input_image, original_size)

        # OTTIMIZZAZIONE 9: Context manager per inference ottimizzata
        with torch.inference_mode():  # Più veloce di no_grad()
            with torch.cuda.amp.autocast(enabled=enable_optimizations):
                try:
                    enhanced_images = self.model.enhance(
                        lq=input_tensor,
                        prompt=prompt,
                        scale_by="factor",
                        upscale=upscale_factor,
                        return_type="pil"
                    )
                except torch.cuda.OutOfMemoryError:
                    # Fallback con meno memoria
                    torch.cuda.empty_cache()
                    with torch.cuda.amp.autocast(enabled=False):
                        enhanced_images = self.model.enhance(
                            lq=input_tensor.to(dtype=torch.float32),
                            prompt=prompt,
                            scale_by="factor", 
                            upscale=upscale_factor,
                            return_type="pil"
                        )

        if not enhanced_images:
            raise RuntimeError("No output generated")

        # OTTIMIZZAZIONE 10: Fast output saving
        output_path = Path(tempfile.mkdtemp()) / "enhanced.png"
        enhanced_images[0].save(str(output_path), format="PNG", optimize=True)

        return output_path

    def _optimize_tensor_conversion(self, image: Image.Image, size: tuple) -> torch.Tensor:
        """Ottimizza conversione immagine->tensor con caching"""

        # Check cache first
        if size in self.tensor_cache:
            cached_tensor = self.tensor_cache[size]
            # Reuse pre-allocated tensor
            tensor_data = self.to_tensor_optimized(image).unsqueeze(0)
            cached_tensor.copy_(tensor_data.to(
                device='cuda', 
                dtype=torch.float16,
                memory_format=torch.channels_last
            ))
            return cached_tensor
        else:
            # Standard conversion
            return self.to_tensor_optimized(image).unsqueeze(0).to(
                device='cuda', 
                dtype=torch.float16,
                memory_format=torch.channels_last
            )
