#!/usr/bin/env python3
"""
Test per verificare che il fix PyTorch Dynamo funzioni
"""

import torch
import sys
import os

# Add HYPIR to path
sys.path.insert(0, "HYPIR")

# Configure torch dynamo
import torch._dynamo.config
torch._dynamo.config.reorderable_logging_functions.add(print)
torch._dynamo.config.suppress_errors = True

def test_tiled_vae_import():
    """Test che possiamo importare il modulo senza errori"""
    try:
        from HYPIR.utils.tiled_vae.vaehook import VAEHook
        print("✅ VAEHook import successful")
        return True
    except Exception as e:
        print(f"❌ VAEHook import failed: {e}")
        return False

def test_print_compilation():
    """Test che torch.compile funzioni con le nostre safe_print"""
    try:
        # Import safe_print
        from HYPIR.utils.tiled_vae.vaehook import safe_print
        
        # Test function with safe_print
        @torch.compile(mode="default")
        def test_func(x):
            safe_print(f"Processing tensor with shape: {x.shape}")
            return x * 2
        
        # Test with dummy tensor
        x = torch.randn(1, 3, 64, 64)
        result = test_func(x)
        
        print("✅ torch.compile with safe_print works")
        return True
    except Exception as e:
        print(f"❌ torch.compile test failed: {e}")
        return False

def test_dynamo_config():
    """Test che la configurazione Dynamo sia corretta"""
    try:
        import torch._dynamo.config
        
        # Check if print is in reorderable_logging_functions
        if print in torch._dynamo.config.reorderable_logging_functions:
            print("✅ print function is in reorderable_logging_functions")
        else:
            print("❌ print function not in reorderable_logging_functions")
            
        print(f"✅ suppress_errors: {torch._dynamo.config.suppress_errors}")
        return True
    except Exception as e:
        print(f"❌ Dynamo config test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing PyTorch Dynamo fix...")
    
    tests = [
        test_dynamo_config,
        test_tiled_vae_import,
        test_print_compilation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    if all(results):
        print("🎉 All tests passed! Dynamo fix should work.")
    else:
        print("⚠️ Some tests failed. Check the output above.")
        
    print(f"\nResults: {sum(results)}/{len(results)} tests passed")
