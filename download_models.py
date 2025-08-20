#!/usr/bin/env python3
"""
Enhanced model download script for HYPIR COG
Downloads multiple models for ensemble approach
"""

import os
import wget
import torch
from pathlib import Path

def download_models():
    """Download all required models"""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("=== HYPIR Enhanced Model Downloader ===")
    print("Downloading models for ensemble approach...\n")
    
    # 1. Download HYPIR models (original)
    print("1. Downloading HYPIR models...")
    hypir_models = [
        # Add actual HYPIR model URLs when available
        # "https://huggingface.co/XPixelGroup/HYPIR/resolve/main/model.pth",
    ]
    
    # For now, create placeholder for HYPIR models
    hypir_dir = models_dir / "HYPIR"
    hypir_dir.mkdir(exist_ok=True)
    
    # 2. Download Real-ESRGAN models
    print("2. Downloading Real-ESRGAN models...")
    realesrgan_models = [
        {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "name": "RealESRGAN_x4plus.pth"
        },
        {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth", 
            "name": "RealESRGAN_x4plus_anime_6B.pth"
        }
    ]
    
    realesrgan_dir = models_dir / "RealESRGAN"
    realesrgan_dir.mkdir(exist_ok=True)
    
    for model in realesrgan_models:
        filename = realesrgan_dir / model["name"]
        if not filename.exists():
            try:
                print(f"   Downloading {model['name']}...")
                wget.download(model["url"], str(filename))
                print(f"   ✓ Downloaded {model['name']}")
            except Exception as e:
                print(f"   ✗ Failed to download {model['name']}: {e}")
        else:
            print(f"   ↪ {model['name']} already exists")
    
    # 3. Download EDSR models (placeholder)
    print("\n3. Preparing EDSR models...")
    edsr_dir = models_dir / "EDSR"
    edsr_dir.mkdir(exist_ok=True)
    print("   ↪ EDSR models will be added when available")
    
    # 4. Create model configuration file
    print("\n4. Creating model configuration...")
    config_content = """# Model Configuration for Enhanced HYPIR
models:
  hypir:
    path: "models/HYPIR/"
    enabled: true
    weight: 0.4
  realesrgan:
    path: "models/RealESRGAN/RealESRGAN_x4plus.pth"
    enabled: true
    weight: 0.4
  edsr:
    path: "models/EDSR/"
    enabled: false
    weight: 0.2

ensemble:
  default_weights: [0.4, 0.4, 0.2]
  post_processing: true
  
post_processing:
  gradient_enhancement: true
  multi_frequency_filter: true
  adaptive_sharpening: true
  noise_reduction: true
"""
    
    with open("model_config.yaml", "w") as f:
        f.write(config_content)
    print("   ✓ Created model_config.yaml")
    
    # 5. Verify downloads
    print("\n5. Verification:")
    total_size = 0
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath)
            total_size += size
            print(f"   ✓ {file}: {size / (1024*1024):.1f} MB")
    
    print(f"\n✅ Setup complete! Total models size: {total_size / (1024*1024):.1f} MB")
    print("\nNext steps:")
    print("1. Run: cog build -t hypir-enhanced")
    print("2. Test: cog predict -i image=@test.jpg")
    print("3. Push: cog push r8.im/yourusername/hypir-enhanced")

def verify_pytorch():
    """Verify PyTorch installation"""
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
        return True
    except Exception as e:
        print(f"PyTorch verification failed: {e}")
        return False

if __name__ == "__main__":
    print("Verifying PyTorch installation...")
    if verify_pytorch():
        print("✓ PyTorch verified\n")
        download_models()
    else:
        print("✗ PyTorch verification failed")
        print("Please ensure PyTorch is installed correctly")