"""
Dynamo compatibility patches for HYPIR
"""

import torch
import torch._dynamo.config
import logging

# Configure PyTorch Dynamo to handle print statements
torch._dynamo.config.reorderable_logging_functions.add(print)
torch._dynamo.config.suppress_errors = True

# Additional configurations for better compatibility
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = False

# Setup logging for Dynamo-safe printing
logger = logging.getLogger('HYPIR_DYNAMO')
logger.setLevel(logging.INFO)

# Create console handler if not exists
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def dynamo_safe_print(*args, **kwargs):
    """
    Dynamo-safe print function that falls back to logging
    """
    try:
        message = ' '.join(str(arg) for arg in args)
        logger.info(message)
    except:
        # Silently ignore if logging fails
        pass

# Monkey patch to replace problematic prints in VAE modules
def patch_vae_prints():
    """
    Patch VAE modules to use dynamo-safe printing
    """
    try:
        import sys
        import types
        
        # Add to path if needed
        if "HYPIR" not in sys.path:
            sys.path.insert(0, "HYPIR")
        
        # Import and patch the vaehook module
        from HYPIR.utils.tiled_vae import vaehook
        
        # Replace print function in the module
        vaehook.print = dynamo_safe_print
        
        print("✅ VAE print functions patched for Dynamo compatibility")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not patch VAE prints: {e}")
        return False

# Initialize patches
def init_dynamo_patches():
    """Initialize all Dynamo compatibility patches"""
    patch_vae_prints()
    print("🔧 Dynamo compatibility patches initialized")

# Auto-initialize when imported
init_dynamo_patches()
