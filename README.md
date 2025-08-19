# HYPIR for Replicate.com

<img src="https://github.com/XPixelGroup/HYPIR/blob/main/assets/logo.png?raw=true" alt="HYPIR" width="220" />

**DOES IT WORK?** Yes, with a bit of luck and a decent GPU.

**IS IT STABLE?** As stable as any project that depends on ~47 different Python libraries.

**IS IT FAST?** Faster than traditional diffusion models, but still requires a GPU.

**WILL IT FAIL?** Probably, if:
- Your image is too large (>4MP output)
- The GPU runs out of memory
- Hugging Face models are temporarily offline
- The stars aren't aligned

## What is HYPIR?

HYPIR is an image restoration model that:

1. **Does not use iterative diffusion sampling** — a single forward pass
2. **Is based on Stable Diffusion 2.1** — fine-tuned with adversarial training
3. **Supports text-guided restoration** — you can describe what you want
4. **Performs upscaling and restoration together** — two tasks in one

## How to Use This COG Project

### 1. Build the Image

```bash
cog build -t hypir
```

**Expect**:
- Download of ~8GB of models
- 20-30 minutes of build time
- Some deprecation warnings (ignore)
- Prayers to the gods of Python compatibility

### 2. Test Locally

```bash
cog predict -i image=@input.jpg -i prompt="high quality photo" -i upscale_factor=2.0
```

### 3. Push to Replicate

```bash
# Login first
cog login

# Push (replace with your username)
cog push r8.im/youruser/hypir
```

## Parameters

| Parameter | Type | Default | Description | Brutal Truth |
|-----------|------|---------|-------------|--------------|
| `image` | Image | - | Input image to restore | REQUIRED. If the file is corrupted the run will fail |
| `prompt` | String | "high quality, sharp, detailed" | Text guidance | Empty = default. Nonsensical prompts = poor results |
| `upscale_factor` | Float | 1.0 | Scaling factor (1.0-4.0) | Values >4.0 may cause out-of-memory errors |
| `seed` | Integer | -1 | Random seed (-1 = random) | Set for reproducible results |



## Quick Notes about MacOS env
On macOS, building `xformers` required installing Homebrew build tools and temporarily using Homebrew's clang and OpenMP headers/libraries (see the Important macOS note below).

During local setup on macOS, `xformers` failed to compile with the system clang because OpenMP flags were unsupported. To fix this I installed a few Homebrew packages and used Homebrew's clang for the build. This is an ad-hoc, macOS-only workaround — it is not required on Linux machines or environments where `xformers` wheels are available.

If you need the same workaround, the steps used were:

1. Install Homebrew packages (one-time):

```bash
brew install ninja libomp cmake llvm
```

2. In the same shell where you activate the venv, export the compiler and flags before running pip install:

```bash
cd /path/to/cog-hypir-upscale
source .venv/bin/activate
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib -L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include -I/opt/homebrew/opt/llvm/include"
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
pip install -r requirements.txt
```

Note: you can add those exports to your `~/.zshrc` if you want them permanently available on that machine, but doing so may affect other builds. This change is macOS-specific and was applied only to enable building `xformers` locally.

## Quick setup (recommended)

1. Create and activate a virtual environment in the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

2. Install Python dependencies into the activated venv:

```bash
pip install -r requirements.txt
```

## Limitations

- CPU inference is technically possible but very slow in practice
- Very large outputs (e.g. >2048×2048) can cause out-of-memory errors
- This is not a miracle-restorer — results depend on input quality and prompt

## File structure (high level)

```
.  # project root
├── cog.yaml
├── predict.py
├── download_models.py
├── requirements.txt
├── README.md
└── (other scripts and downloaded HYPIR files after build)
```

## Utility scripts & tests

This repository includes a few helper scripts and lightweight tests to verify installation and runtime behavior. Below is a short guide explaining what each file does and how to run it.

1) `create_test_image.py`
- Purpose: generates a simple RGB test image (`test_input.jpg`) with gradients to use as input for quick experiments.
- Usage: run it from the project root to create `test_input.jpg`.

Example:
```bash
python3 create_test_image.py
# -> Creates test_input.jpg in the current directory
```

2) `simple_test.py`
- Purpose: quick smoke tests that verify the repository structure and that `predict.Predictor` can be imported and instantiated. Useful after cloning or before running a full COG build.
- What it checks: required files (`cog.yaml`, `predict.py`, `requirements.txt`, `download_models.py`), presence of the `HYPIR/` directory, and existence of `setup()` and `predict()` methods on `predict.Predictor`.
- Usage: run from the project root. It exits with code 0 on success, non-zero on failure.

Example:
```bash
python3 simple_test.py
```

3) `test_dynamo_fix.py`
- Purpose: verifies that the PyTorch Dynamo compatibility fixes are applied and that the tiled VAE hook (`HYPIR.utils.tiled_vae.vaehook`) can be imported and used under `torch.compile`.
- What it checks: Dynamo configuration, importing `VAEHook`, and running a small compiled function that uses the repository's `safe_print` wrapper.
- Usage: run from the project root with a Python environment that has PyTorch installed. This test is most useful on systems with CUDA, but also works on CPU.

Example:
```bash
python3 test_dynamo_fix.py
# Expected: detailed output for each sub-test; summary at the end
```

4) `test_simple_dynamo.py`
- Purpose: a compact Dynamo test suite that runs three targeted checks: basic `torch.compile`, `torch.compile` with `print`, and a string-format `print` pattern similar to the one that caused the original Dynamo error.
- Usage: run from the project root. Useful to quickly validate that Dynamo-related runtime configuration is correct on the current machine.

Example:
```bash
python3 test_simple_dynamo.py
# Expected: prints PyTorch version, Dynamo config state, and a pass/fail summary
```

Notes & tips
- Ensure your Python environment is active (see Quick setup above) and that `requirements.txt` is installed before running tests.
- Some tests require PyTorch 2.0+ and a working `torch.compile` implementation; results may differ between CPU and GPU environments.
- If a test fails due to missing files, run `download_models.py` or follow the build steps before retrying.


## Troubleshooting / common failures

- "CUDA out of memory": lower `upscale_factor` or use a smaller image
- "Model download failed": check network and retry the build
- Build errors for `xformers` on macOS: see the Important macOS-only note above

## Credits & license

- Original HYPIR paper and code: XPixelGroup / Xinqi Lin et al.
- Stable Diffusion: Stability AI, RunwayML
- This wrapper: MIT License (wrapper code only; models have their own licenses)

### Paper & Original Code

- **Paper**: [HYPIR: Harnessing Diffusion-Yielded Score Priors for Image Restoration](https://arxiv.org/abs/2507.20590)
- **Original Code**: [XPixelGroup/HYPIR](https://github.com/XPixelGroup/HYPIR)
- **Authors**: Xinqi Lin et al.

