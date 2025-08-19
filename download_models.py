"""download_models.py

Improved download helper intended to run during COG build-time.

Features:
- argparse for configurable URLs/paths
- retries and basic SHA256 verification (optional)
- optional pre-cache for Stable Diffusion (off by default — enable with --precache-sd)
- writes a small /etc/profile.d entry for privacy envs instead of touching /root/.bashrc

This script intentionally fails fast (non-zero exit) so that COG build fails if downloads
didn't complete successfully.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_HYPIR_REPO = "https://github.com/XPixelGroup/HYPIR.git"
DEFAULT_WEIGHTS_URL = "https://huggingface.co/lxq007/HYPIR/resolve/main/HYPIR_sd2.pth"


def run_command(cmd, check=True):
    print(f"🔄 Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        if e.stderr:
            print(e.stderr)
        if check:
            raise
        return e


def download_with_retries(url, dest_path, retries=3, backoff=3):
    """Download using curl if available, otherwise urllib with retries."""
    dest = Path(dest_path)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            if shutil.which("curl"):
                cmd = f"curl -fSL {url} -o {tmp} --retry {retries} --retry-delay {backoff}"
                run_command(cmd)
            else:
                # fallback to Python urllib
                import urllib.request

                with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as out:
                    shutil.copyfileobj(r, out)

            tmp.replace(dest)
            print(f"✅ Downloaded {dest}")
            return True
        except Exception as e:
            print(f"⚠️ Download attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(backoff)
            else:
                return False


def sha256_verify(path, expected_hex):
    if not expected_hex:
        return True
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual.lower() != expected_hex.lower():
        print(f"❌ SHA256 mismatch for {path}: expected {expected_hex}, got {actual}")
        return False
    print(f"✅ SHA256 verified for {path}")
    return True


def write_profile_env(vars_map):
    profile_path = Path("/etc/profile.d/hypir_env.sh")
    try:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w") as f:
            f.write("# HYPIR environment variables (added by download_models.py)\n")
            for k, v in vars_map.items():
                f.write(f'export {k}={v}\n')
        print(f"✅ Wrote environment file: {profile_path}")
    except Exception as e:
        print(f"⚠️ Failed to write {profile_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download HYPIR and required model files during build-time")
    parser.add_argument("--hypir-repo", default=os.environ.get("HYPIR_REPO", DEFAULT_HYPIR_REPO))
    parser.add_argument("--weights-url", default=os.environ.get("HYPIR_WEIGHTS_URL", DEFAULT_WEIGHTS_URL))
    parser.add_argument("--weights-sha256", default=os.environ.get("HYPIR_WEIGHTS_SHA256", ""))
    parser.add_argument("--cache-dir", default=os.environ.get("HYPIR_CACHE_DIR", "./cache"))
    parser.add_argument("--precache-sd", action="store_true", help="Attempt to pre-cache Stable Diffusion (may be large)")
    parser.add_argument("--skip-hypir-reqs", action="store_true", help="Skip installing extra HYPIR requirements")
    args = parser.parse_args()

    print("🚀 Starting HYPIR model download process (build-time)")

    # 1) Clone or refresh HYPIR repo in workspace
    hypir_dir = Path("HYPIR")
    if hypir_dir.exists():
        print("🗑️ Removing existing HYPIR directory...")
        shutil.rmtree(hypir_dir)

    print(f"� Cloning HYPIR from {args.hypir_repo}...")
    run_command(f"git clone {args.hypir_repo} {hypir_dir}")

    # 2) Optionally install safe requirements from HYPIR
    if not args.skip_hypir_reqs:
        req_file = hypir_dir / "requirements.txt"
        if req_file.exists():
            print("📦 Installing HYPIR requirements (filtered for safety)")
            with open(req_file, "r") as f:
                packages = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
            
            # Allow more packages but filter dangerous ones
            blocked = ["torch", "torchvision", "torchaudio", "cuda", "nvidia"]
            safe = [p for p in packages if not any(x in p.lower() for x in blocked)]
            if safe:
                print(f"Installing: {safe}")
                run_command("pip install --no-cache-dir " + " ".join(safe))
            else:
                print("⚠️ No safe packages to install from HYPIR requirements")
        else:
            print("⚠️ HYPIR requirements not found; skipping")

    # 3) Download weights
    weights_path = Path("HYPIR_sd2.pth")
    print(f"🔽 Downloading weights from {args.weights_url} -> {weights_path}")
    if not download_with_retries(args.weights_url, weights_path):
        print("❌ Failed to download HYPIR weights")
        sys.exit(1)

    if weights_path.stat().st_size < 1024 * 100:  # sanity check: at least 100KB
        print(f"❌ Downloaded weights look too small: {weights_path.stat().st_size} bytes")
        sys.exit(1)

    if args.weights_sha256:
        if not sha256_verify(weights_path, args.weights_sha256):
            sys.exit(1)

    print(f"✅ Weights ready: {weights_path} ({weights_path.stat().st_size} bytes)")

    # 4) Optionally pre-cache Stable Diffusion
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.precache_sd:
        print("🤗 Pre-caching Stable Diffusion (this may be large)...")
        try:
            cache_script = Path(tempfile.gettempdir()) / "cache_sd.py"
            cache_script.write_text(
                """
from diffusers import StableDiffusionPipeline
import torch
print('Downloading SD 2.1 base...')
pipe = StableDiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-2-1-base', cache_dir='""" + str(cache_dir) + """'
)
print('Done')
"""
            )
            run_command(f"python {cache_script}")
            cache_script.unlink()
        except Exception as e:
            print(f"⚠️ Pre-cache failed: {e}")
            print("⚠️ Continuing — model will be downloaded at first run")

    # 5) Write privacy env file to /etc/profile.d
    env_map = {
        "GRADIO_ANALYTICS_ENABLED": "False",
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    write_profile_env(env_map)

    # 6) Basic verification
    checks = [hypir_dir / "HYPIR" / "__init__.py", hypir_dir / "HYPIR" / "enhancer" / "sd2.py", weights_path]
    missing = [str(p) for p in checks if not p.exists()]
    if missing:
        print(f"❌ Missing required files after download: {missing}")
        sys.exit(1)

    # Quick smoke import test
    try:
        sys.path.insert(0, str(hypir_dir))
        from HYPIR.enhancer.sd2 import SD2Enhancer  # noqa: F401
        print("✅ HYPIR import test OK")
    except Exception as e:
        print(f"❌ HYPIR import test failed: {e}")
        sys.exit(1)

    print("🎉 HYPIR model download completed successfully")
    run_command("du -sh HYPIR* cache* 2>/dev/null || true", check=False)


if __name__ == "__main__":
    main()
