# =============================================================================
# HYPIR COG ENHANCED - Training-Free Improvements Implementation (FIXED)
# =============================================================================

import os
import torch
from PIL import Image
import numpy as np
from cog import BasePredictor, Input, Path
from typing import Union
import cv2
import wget
from pathlib import Path as PathlibPath

# Import existing HYPIR modules (assume they exist)
try:
    from HYPIR.model import HYPIRModel
except ImportError:
    print("HYPIR modules not found, using placeholder")
    HYPIRModel = None

# New imports for improvements
try:
    from RealESRGAN import RealESRGAN
except ImportError:
    print("Real-ESRGAN will be installed during setup")
    RealESRGAN = None

class EnhancedPredictor(BasePredictor):
    """
    HYPIR Enhanced with 3 Training-Free Improvements:
    1. Model Ensemble (HYPIR + Real-ESRGAN + EDSR)
    2. Post-Processing Chain (Gradient Enhancement + Filtering)
    3. Diffusion2GAN Acceleration (if available)
    """
    
    def setup(self):
        """Load multiple models for ensemble with automatic download"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create models directory
        self.models_dir = PathlibPath("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # =================================================================
        # AUTO-DOWNLOAD MODELS IF NOT PRESENT
        # =================================================================
        print("Checking for required models...")
        self._ensure_models_downloaded()
        
        # =================================================================
        # LOAD MODELS
        # =================================================================
        
        # 1. Load original HYPIR
        self.hypir_model = self._load_hypir_model()
        
        # 2. Load Real-ESRGAN for ensemble
        self.realesrgan = self._load_realesrgan_model()
        
        # 3. Load EDSR (placeholder - would need actual implementation)
        self.edsr_model = None  # Would implement if available
        
        # 4. Try to load Diffusion2GAN if available
        self.diffusion2gan = None  # Placeholder for future
        
        print(f"Models loaded successfully:")
        print(f"  - HYPIR: {'✓' if self.hypir_model else '✗'}")
        print(f"  - Real-ESRGAN: {'✓' if self.realesrgan else '✗'}")
        print(f"  - EDSR: {'✓' if self.edsr_model else '✗'}")
    
    def _ensure_models_downloaded(self):
        """Download required models if they don't exist"""
        
        # Define model URLs and paths
        model_configs = [
            {
                "name": "RealESRGAN_x4plus.pth",
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                "path": self.models_dir / "RealESRGAN" / "RealESRGAN_x4plus.pth",
                "required": True
            },
            {
                "name": "RealESRGAN_x4plus_anime_6B.pth",
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                "path": self.models_dir / "RealESRGAN" / "RealESRGAN_x4plus_anime_6B.pth",
                "required": False
            }
        ]
        
        # Create subdirectories
        (self.models_dir / "RealESRGAN").mkdir(exist_ok=True)
        (self.models_dir / "HYPIR").mkdir(exist_ok=True)
        (self.models_dir / "EDSR").mkdir(exist_ok=True)
        
        # Download missing models
        for config in model_configs:
            if not config["path"].exists():
                if config["required"]:
                    print(f"Downloading {config['name']}...")
                    try:
                        self._download_with_retry(config["url"], config["path"])
                        print(f"✓ Downloaded {config['name']}")
                    except Exception as e:
                        print(f"✗ Failed to download {config['name']}: {e}")
                        if config["required"]:
                            raise Exception(f"Required model {config['name']} could not be downloaded")
                else:
                    print(f"Optional model {config['name']} not found, skipping...")
            else:
                print(f"↪ {config['name']} already exists")
    
    def _download_with_retry(self, url: str, path: PathlibPath, max_retries: int = 3):
        """Download file with retry logic"""
        import urllib.request
        import shutil
        from urllib.error import URLError
        
        for attempt in range(max_retries):
            try:
                print(f"  Attempt {attempt + 1}/{max_retries}...")
                
                # Use urllib for more reliable downloads
                with urllib.request.urlopen(url) as response:
                    with open(path, 'wb') as f:
                        shutil.copyfileobj(response, f)
                
                # Verify download
                if path.exists() and path.stat().st_size > 0:
                    return
                else:
                    raise Exception("Downloaded file is empty or corrupted")
                    
            except Exception as e:
                print(f"  Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print("  Retrying...")
                    if path.exists():
                        path.unlink()  # Remove corrupted file
                else:
                    raise e
    
    def _load_hypir_model(self):
        """Load HYPIR model with error handling"""
        try:
            if HYPIRModel:
                model = HYPIRModel()
                # Try to load pretrained weights if available
                hypir_weights_path = self.models_dir / "HYPIR"
                if hypir_weights_path.exists():
                    # Load actual HYPIR weights here
                    pass
                model.to(self.device)
                model.eval()
                return model
            else:
                print("HYPIR model class not available")
                return None
        except Exception as e:
            print(f"Failed to load HYPIR model: {e}")
            return None
    
    def _load_realesrgan_model(self):
        """Load Real-ESRGAN model with error handling"""
        try:
            # Try to import Real-ESRGAN (install if needed)
            global RealESRGAN
            if RealESRGAN is None:
                try:
                    from RealESRGAN import RealESRGAN
                except ImportError:
                    print("Installing Real-ESRGAN...")
                    import subprocess
                    subprocess.check_call([
                        "pip", "install", 
                        "git+https://github.com/ai-forever/Real-ESRGAN.git"
                    ])
                    from RealESRGAN import RealESRGAN
            
            # Initialize Real-ESRGAN
            model = RealESRGAN(self.device, scale=4)
            
            # Load weights
            weights_path = self.models_dir / "RealESRGAN" / "RealESRGAN_x4plus.pth"
            if weights_path.exists():
                model.load_weights(str(weights_path))
                print("Real-ESRGAN weights loaded successfully")
            else:
                raise Exception("Real-ESRGAN weights not found")
            
            return model
            
        except Exception as e:
            print(f"Failed to load Real-ESRGAN model: {e}")
            return None
    
    def predict(
        self,
        image: Path = Input(description="Input image to restore/upscale"),
        prompt: str = Input(
            description="Text guidance for restoration", 
            default="high quality, sharp, detailed"
        ),
        upscale_factor: float = Input(
            description="Scaling factor (1.0-4.0)", 
            default=2.0, 
            ge=1.0, 
            le=4.0
        ),
        use_ensemble: bool = Input(
            description="Use model ensemble (slower but better quality)", 
            default=True
        ),
        use_post_processing: bool = Input(
            description="Apply post-processing enhancement", 
            default=True
        ),
        ensemble_weights: str = Input(
            description="Ensemble weights (hypir,realesrgan,edsr)", 
            default="0.4,0.4,0.2"
        ),
        seed: int = Input(description="Random seed (-1 for random)", default=-1)
    ) -> Path:
        """Run enhanced prediction with training-free improvements"""
        
        # Load and preprocess image
        input_image = Image.open(image).convert('RGB')
        original_size = input_image.size
        print(f"Input image size: {original_size}")
        
        # Parse ensemble weights
        try:
            weights = [float(w) for w in ensemble_weights.split(',')]
            weights = [w / sum(weights) for w in weights]  # Normalize
        except:
            weights = [0.4, 0.4, 0.2]  # Default weights
        
        # =================================================================
        # IMPROVEMENT 1: MODEL ENSEMBLE
        # =================================================================
        available_models = [m for m in [self.hypir_model, self.realesrgan, self.edsr_model] if m is not None]
        
        if use_ensemble and len(available_models) > 1:
            print(f"Running ensemble with {len(available_models)} models...")
            results = []
            model_weights = []
            
            # Run HYPIR if available
            if self.hypir_model is not None:
                try:
                    print("Running HYPIR...")
                    hypir_result = self._run_hypir(input_image, prompt, upscale_factor, seed)
                    results.append(hypir_result)
                    model_weights.append(weights[0])
                    print("✓ HYPIR completed")
                except Exception as e:
                    print(f"✗ HYPIR failed: {e}")
            
            # Run Real-ESRGAN if available  
            if self.realesrgan is not None:
                try:
                    print("Running Real-ESRGAN...")
                    realesrgan_result = self._run_realesrgan(input_image, upscale_factor)
                    results.append(realesrgan_result)
                    model_weights.append(weights[1])
                    print("✓ Real-ESRGAN completed")
                except Exception as e:
                    print(f"✗ Real-ESRGAN failed: {e}")
            
            # Run EDSR if available (placeholder)
            if self.edsr_model is not None:
                try:
                    print("Running EDSR...")
                    edsr_result = self._run_edsr(input_image, upscale_factor)
                    results.append(edsr_result)
                    model_weights.append(weights[2])
                    print("✓ EDSR completed")
                except Exception as e:
                    print(f"✗ EDSR failed: {e}")
            
            # Ensemble combination
            if len(results) > 1:
                # Normalize weights for available models
                model_weights = [w / sum(model_weights) for w in model_weights]
                print(f"Combining results with weights: {model_weights}")
                final_result = self._weighted_ensemble(results, model_weights)
                print(f"✓ Ensemble completed with {len(results)} models")
            elif len(results) == 1:
                final_result = results[0]
                print("Only one model succeeded, using single result")
            else:
                raise Exception("All models failed!")
                
        else:
            # Single model prediction (priority order: HYPIR -> Real-ESRGAN -> fallback)
            if self.hypir_model is not None:
                print("Running single HYPIR prediction...")
                final_result = self._run_hypir(input_image, prompt, upscale_factor, seed)
            elif self.realesrgan is not None:
                print("Running single Real-ESRGAN prediction...")
                final_result = self._run_realesrgan(input_image, upscale_factor)
            else:
                print("No models available, using simple interpolation...")
                # Last resort: simple interpolation
                target_size = (int(original_size[0] * upscale_factor), 
                              int(original_size[1] * upscale_factor))
                final_result = input_image.resize(target_size, Image.LANCZOS)
        
        # =================================================================
        # IMPROVEMENT 2: POST-PROCESSING CHAIN
        # =================================================================
        if use_post_processing:
            print("Applying post-processing chain...")
            final_result = self._apply_post_processing_chain(final_result)
            print("✓ Post-processing completed")
        
        # Save output
        output_path = "enhanced_output.jpg"
        final_result.save(output_path, quality=95)
        print(f"Output saved: {final_result.size}")
        
        return Path(output_path)
    
    def _run_hypir(self, image: Image.Image, prompt: str, scale: float, seed: int) -> Image.Image:
        """Run original HYPIR model"""
        if self.hypir_model is None:
            raise Exception("HYPIR model not available")
        
        # Set seed if specified
        if seed > 0:
            torch.manual_seed(seed)
        
        # This would be the actual HYPIR inference
        # Placeholder implementation for now
        with torch.no_grad():
            # Convert PIL to tensor
            img_tensor = torch.from_numpy(np.array(image)).permute(2,0,1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # HYPIR inference (placeholder)
            # result = self.hypir_model(img_tensor, prompt=prompt, scale=scale)
            
            # For now, return upscaled version (replace with actual HYPIR call)
            target_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            return image.resize(target_size, Image.LANCZOS)
    
    def _run_realesrgan(self, image: Image.Image, scale: float) -> Image.Image:
        """Run Real-ESRGAN model"""
        if self.realesrgan is None:
            raise Exception("Real-ESRGAN model not available")
        
        # Real-ESRGAN inference
        result = self.realesrgan.predict(image)
        
        # Adjust scale if needed
        if scale != 4.0:  # Real-ESRGAN default is 4x
            current_scale = result.size[0] / image.size[0]
            adjust_factor = scale / current_scale
            if adjust_factor != 1.0:
                new_size = (int(result.size[0] * adjust_factor), 
                           int(result.size[1] * adjust_factor))
                result = result.resize(new_size, Image.LANCZOS)
        
        return result
    
    def _run_edsr(self, image: Image.Image, scale: float) -> Image.Image:
        """Run EDSR model (placeholder)"""
        # This would be actual EDSR implementation
        target_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        return image.resize(target_size, Image.LANCZOS)
    
    def _weighted_ensemble(self, results: list, weights: list) -> Image.Image:
        """Combine multiple results using weighted average"""
        if len(results) == 1:
            return results[0]
        
        # Ensure all images have the same size
        target_size = results[0].size
        aligned_results = []
        for result in results:
            if result.size != target_size:
                aligned_results.append(result.resize(target_size, Image.LANCZOS))
            else:
                aligned_results.append(result)
        
        # Convert to numpy arrays
        arrays = [np.array(img).astype(np.float32) for img in aligned_results]
        
        # Weighted average
        combined = np.zeros_like(arrays[0])
        for i, (arr, weight) in enumerate(zip(arrays, weights)):
            combined += arr * weight
        
        # Convert back to PIL
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        return Image.fromarray(combined)
    
    def _apply_post_processing_chain(self, image: Image.Image) -> Image.Image:
        """Apply post-processing enhancement chain"""
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Step 1: Gradient Enhancement
        enhanced = self._gradient_enhancement(cv_image, beta=1.2)
        
        # Step 2: Multi-frequency filtering  
        filtered = self._multi_frequency_filter(enhanced)
        
        # Step 3: Adaptive sharpening
        sharpened = self._adaptive_unsharp_mask(filtered)
        
        # Step 4: Noise reduction
        cleaned = cv2.bilateralFilter(sharpened, 9, 75, 75)
        
        # Convert back to PIL
        result = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)
    
    def _gradient_enhancement(self, image: np.ndarray, beta: float = 1.2) -> np.ndarray:
        """Enhance edges using gradient information"""
        # Convert to grayscale for gradient calculation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = np.clip(grad_mag / grad_mag.max() * 255, 0, 255).astype(np.uint8)
        
        # Enhance edges
        grad_3ch = cv2.cvtColor(grad_mag, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(image, 1.0, grad_3ch, beta * 0.1, 0)
        
        return enhanced
    
    def _multi_frequency_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply multi-frequency filtering"""
        # Low-pass filter
        low_freq = cv2.GaussianBlur(image, (5, 5), 1.0)
        
        # High-pass filter (original - low-pass)
        high_freq = cv2.subtract(image, low_freq)
        
        # Recombine with different weights
        filtered = cv2.addWeighted(low_freq, 0.8, high_freq, 1.2, 0)
        
        return np.clip(filtered, 0, 255).astype(np.uint8)
    
    def _adaptive_unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive unsharp masking"""
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), 1.5)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)