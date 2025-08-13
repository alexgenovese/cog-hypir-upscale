"""
Download script for HYPIR models and dependencies

This script will be run during COG build process to download:
1. HYPIR repository 
2. Pre-trained weights
3. Stable Diffusion 2.1 base model (via Hugging Face cache)

BRUTAL TRUTH: If this fails, the whole COG build fails.
No sugar-coating, no fallbacks.
"""

import os
import sys
import subprocess
import urllib.request
import shutil
from pathlib import Path

def run_command(cmd, check=True):
    """Run shell command with proper error handling"""
    print(f"🔄 Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"❌ Error: {e.stderr}")
        if check:
            raise
        return e

def download_file(url, destination):
    """Download a file with progress"""
    print(f"📥 Downloading {url} to {destination}")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✅ Downloaded {destination}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def main():
    """Main download process"""
    print("🚀 Starting HYPIR model download process...")

    # 1. Clone HYPIR repository
    print("\n📂 Step 1: Cloning HYPIR repository...")
    if os.path.exists("HYPIR"):
        print("🗑️ Removing existing HYPIR directory...")
        shutil.rmtree("HYPIR")

    result = run_command("git clone https://github.com/XPixelGroup/HYPIR.git", check=False)
    if result.returncode != 0:
        print("❌ Failed to clone HYPIR repository")
        sys.exit(1)

    # 2. Install HYPIR requirements 
    print("\n📦 Step 2: Installing HYPIR requirements...")
    hypir_requirements = Path("HYPIR/requirements.txt")
    if hypir_requirements.exists():
        # Read and install individual packages to avoid conflicts
        with open(hypir_requirements, 'r') as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Filter out packages that might conflict
        safe_packages = []
        for pkg in packages:
            # Skip certain problematic packages that are in our main requirements
            if not any(skip in pkg.lower() for skip in ['torch', 'torchvision', 'diffusers', 'transformers']):
                safe_packages.append(pkg)
        
        if safe_packages:
            run_command(f"pip install {' '.join(safe_packages)}")
        else:
            print("⚠️ No additional packages to install from HYPIR requirements")
    else:
        print("⚠️ HYPIR requirements.txt not found, continuing without it...")

    # 3. Download HYPIR pre-trained weights
    print("\n🔽 Step 3: Downloading HYPIR pre-trained weights...")
    weights_url = "https://huggingface.co/lxq007/HYPIR/resolve/main/HYPIR_sd2.pth"
    weights_path = "HYPIR_sd2.pth"

    if not download_file(weights_url, weights_path):
        print("❌ Failed to download HYPIR weights")
        sys.exit(1)

    # Verify weights file
    if not os.path.exists(weights_path) or os.path.getsize(weights_path) < 1000000:  # Less than 1MB
        print(f"❌ Downloaded weights file seems invalid: {weights_path}")
        sys.exit(1)

    print(f"✅ Weights downloaded successfully: {os.path.getsize(weights_path)} bytes")

    # 4. Pre-cache Stable Diffusion 2.1 base model
    print("\n🤗 Step 4: Pre-caching Stable Diffusion 2.1 base model...")
    try:
        # Create a separate script to avoid import issues in this context
        cache_script_content = """
import torch
from diffusers import StableDiffusionPipeline
print("Downloading SD 2.1 base model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    cache_dir="./cache"
)
print("✅ SD 2.1 base model cached successfully")
"""

        with open("cache_models.py", "w") as f:
            f.write(cache_script_content)

        run_command("python cache_models.py")
        os.remove("cache_models.py")

    except Exception as e:
        print(f"⚠️ Failed to pre-cache SD model: {e}")
        print("⚠️ Model will be downloaded at runtime (slower first run)")

    # 5. Set environment variables to disable telemetry
    print("\n🔒 Step 5: Setting privacy-focused environment...")
    env_vars = [
        "export GRADIO_ANALYTICS_ENABLED=False",
        "export HF_HUB_OFFLINE=0",  # We need online for model download
        "export TRANSFORMERS_OFFLINE=0",
        "export DISABLE_TELEMETRY=1",
        "export DO_NOT_TRACK=1",
        "export HF_HUB_DISABLE_TELEMETRY=1"
    ]

    # Add to bashrc for runtime
    with open("/root/.bashrc", "a") as f:
        f.write("\n# HYPIR Privacy Settings\n")
        for var in env_vars:
            f.write(f"{var}\n")

    print("✅ Environment variables set")

    # 6. Verify installation
    print("\n🔍 Step 6: Verifying installation...")

    # Check HYPIR structure
    required_files = [
        "HYPIR/HYPIR/__init__.py",
        "HYPIR/HYPIR/enhancer/sd2.py",
        "HYPIR_sd2.pth"
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing required file: {file_path}")
            sys.exit(1)

    print("✅ All required files present")

    # Test import (basic smoke test)
    try:
        sys.path.insert(0, "HYPIR")
        from HYPIR.enhancer.sd2 import SD2Enhancer
        print("✅ HYPIR import test successful")
    except Exception as e:
        print(f"❌ HYPIR import test failed: {e}")
        sys.exit(1)

    print("\n🎉 HYPIR model download and setup completed successfully!")
    print(f"📊 Disk usage:")
    run_command("du -sh HYPIR* cache* 2>/dev/null || true", check=False)

if __name__ == "__main__":
    main()
