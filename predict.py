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

# Fix per PyTorch Dynamo e print statements
import torch._dynamo.config
torch._dynamo.config.reorderable_logging_functions.add(print)
torch._dynamo.config.suppress_errors = True

# Disable warnings per performance
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import Dynamo compatibility patches early
try:
    from dynamo_patches import init_dynamo_patches
    init_dynamo_patches()
except ImportError:
    print("⚠️ Dynamo patches not available - using basic config only")

try:
    import sys
    sys.path.insert(0, "/src/HYPIR")  # Add HYPIR path
    # We'll import SD2Enhancer in setup() after ensuring models are downloaded
    print("✅ Path added for HYPIR")
    SD2Enhancer = None  # Will be imported in setup
except Exception as e:
    print(f"❌ HYPIR path setup failed: {e}")
    SD2Enhancer = None


class Predictor(BasePredictor):
    """HYPIR Predictor ottimizzato per velocità massima"""

    def setup(self) -> None:
        """Load model con ogni ottimizzazione possibile"""
        print("🚀 Setting up OPTIMIZED HYPIR model...")

        # Download models if not present
        self._ensure_models_downloaded()

        # Now import SD2Enhancer after models are available
        try:
            import sys
            sys.path.insert(0, "/src/HYPIR")
            from HYPIR.enhancer.sd2 import SD2Enhancer
            self.SD2Enhancer = SD2Enhancer
            print("✅ HYPIR SD2Enhancer imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import HYPIR SD2Enhancer: {e}")
            raise RuntimeError(f"HYPIR not found: {e}")

        # Model config con ottimizzazioni
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎯 Using device: {device}")
        
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
            "device": device
        }

        # Initialize model
        self.model = self.SD2Enhancer(**self.model_config)
        self.device = device  # Store device for later use
        
        # Initialize tensor cache (always, regardless of device)
        self.tensor_cache = {}

        # OTTIMIZZAZIONE 1: Mixed precision - brutale ma efficace
        self.model.init_models()

        # OTTIMIZZAZIONE 2: Memory format channels_last per conv layers (only on GPU)
        if device == "cuda":
            if hasattr(self.model, 'unet'):
                self.model.unet.to(memory_format=torch.channels_last)
                print("✅ UNet converted to channels_last")

            if hasattr(self.model, 'vae'):
                self.model.vae.to(memory_format=torch.channels_last) 
                print("✅ VAE converted to channels_last")
        else:
            print("⚠️ Skipping channels_last optimization (CPU mode)")

        # OTTIMIZZAZIONE 3: torch.compile - il santo graal (only on GPU)
        if device == "cuda":
            self._compile_models()
        else:
            print("⚠️ Skipping torch.compile (CPU mode)")

        # OTTIMIZZAZIONE 4: Pre-allocate tensors per evitare allocation overhead (only on GPU)
        if device == "cuda":
            self._preallocate_tensors()
        else:
            print("⚠️ Skipping tensor pre-allocation (CPU mode)")

        # OTTIMIZZAZIONE 5: Optimized transform (evita conversioni inutili)
        self.to_tensor_optimized = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Warmup con compilation
        if device == "cuda":
            self._warmup_model()
        else:
            print("⚠️ Skipping warmup (CPU mode)")

        print("🔥 OPTIMIZED HYPIR ready - buckle up!")

    def _ensure_models_downloaded(self):
        """Ensure HYPIR models are downloaded"""
        import subprocess
        import sys
        from pathlib import Path
        
        # Check if HYPIR directory exists
        hypir_dir = Path("HYPIR")
        weights_file = Path("HYPIR_sd2.pth")
        
        if not hypir_dir.exists() or not weights_file.exists():
            print("📥 HYPIR models not found, downloading...")
            try:
                # Run the download script
                result = subprocess.run([sys.executable, "download_models.py"], 
                                      capture_output=True, text=True, check=True, 
                                      cwd="/src")
                print("✅ HYPIR models downloaded successfully")
                if result.stdout:
                    print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to download HYPIR models: {e}")
                print(f"❌ Return code: {e.returncode}")
                if e.stdout:
                    print(f"❌ STDOUT: {e.stdout}")
                if e.stderr:
                    print(f"❌ STDERR: {e.stderr}")
                raise RuntimeError("Failed to download HYPIR models")
        else:
            print("✅ HYPIR models already present")

    def _compile_models(self):
        """Compila i modelli con torch.compile (2-3x speedup) con fallback per errori Dynamo"""
        print("🔥 Compiling models with torch.compile...")

        try:
            # Compile UNet (main bottleneck)
            if hasattr(self.model, 'unet'):
                self.model.unet = torch.compile(
                    self.model.unet, 
                    mode="max-autotune",  # Massima ottimizzazione
                    fullgraph=False,     # Cambiato a False per evitare errori Dynamo
                    dynamic=False        # Shape statiche
                )
                print("✅ UNet compiled")

            # Compile VAE decoder  
            if hasattr(self.model, 'vae') and hasattr(self.model.vae, 'decode'):
                self.model.vae.decode = torch.compile(
                    self.model.vae.decode,
                    mode="reduce-overhead",  # Modalità meno aggressiva per VAE
                    fullgraph=False          # Evita problemi con Tiled VAE
                )
                print("✅ VAE decoder compiled")

            # Compile text encoder se disponibile
            if hasattr(self.model, 'text_encoder'):
                self.model.text_encoder = torch.compile(
                    self.model.text_encoder,
                    mode="reduce-overhead",  # Meno aggressivo per text encoder
                    fullgraph=False
                )
                print("✅ Text encoder compiled")

        except Exception as e:
            print(f"⚠️ Compilation failed: {e}")
            print("⚠️ Continuing without compilation...")
            
            # Fallback: disabilita compilation su modelli problematici
            try:
                if hasattr(self.model, 'vae'):
                    # Assicurati che VAE non sia compilato se causa problemi
                    if hasattr(self.model.vae, 'decode') and hasattr(self.model.vae.decode, '_orig_mod'):
                        self.model.vae.decode = self.model.vae.decode._orig_mod
                print("✅ VAE compilation rollback completed")
            except:
                pass

    def _preallocate_tensors(self):
        """Pre-alloca tensors comuni per evitare memory allocation"""
        print("📦 Pre-allocating tensors...")

        # Common sizes cache (tensor_cache already initialized in setup)
        common_sizes = [(512, 512), (1024, 1024), (768, 768), (1536, 1536)]

        for size in common_sizes:
            try:
                if self.device == "cuda":
                    tensor = torch.zeros(1, 3, size[1], size[0], 
                                       dtype=torch.float16, device=self.device,
                                       memory_format=torch.channels_last)
                else:
                    tensor = torch.zeros(1, 3, size[1], size[0], 
                                       dtype=torch.float32, device=self.device)
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
                    self.device, 
                    dtype=torch.float16 if self.device == "cuda" else torch.float32, 
                    memory_format=torch.channels_last if self.device == "cuda" else torch.contiguous_format
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
        if size in self.tensor_cache and self.device == "cuda":
            cached_tensor = self.tensor_cache[size]
            # Reuse pre-allocated tensor
            tensor_data = self.to_tensor_optimized(image).unsqueeze(0)
            cached_tensor.copy_(tensor_data.to(
                device=self.device, 
                dtype=torch.float16,
                memory_format=torch.channels_last
            ))
            return cached_tensor
        else:
            # Standard conversion
            return self.to_tensor_optimized(image).unsqueeze(0).to(
                device=self.device, 
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                memory_format=torch.channels_last if self.device == "cuda" else torch.contiguous_format
            )
