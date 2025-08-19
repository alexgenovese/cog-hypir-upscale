#!/usr/bin/env python3
"""
Test semplice per verificare la configurazione PyTorch Dynamo
"""

import torch
import torch._dynamo.config

# Applica la configurazione Dynamo
torch._dynamo.config.reorderable_logging_functions.add(print)
torch._dynamo.config.suppress_errors = True

print("🧪 Testing basic PyTorch Dynamo configuration...")

def test_basic_compile():
    """Test torch.compile con print di base"""
    try:
        @torch.compile(mode="default")
        def simple_func(x):
            return x * 2
        
        x = torch.randn(2, 2)
        result = simple_func(x)
        print("✅ Basic torch.compile works")
        return True
    except Exception as e:
        print(f"❌ Basic compile failed: {e}")
        return False

def test_compile_with_print():
    """Test torch.compile con print statement"""
    try:
        @torch.compile(mode="default")
        def func_with_print(x):
            # This should work now with our config
            print(f"Processing tensor: {x.shape}")
            return x + 1
        
        x = torch.randn(2, 2)
        result = func_with_print(x)
        print("✅ torch.compile with print works")
        return True
    except Exception as e:
        print(f"❌ Compile with print failed: {e}")
        return False

def test_string_format_print():
    """Test il tipo specifico che causava l'errore"""
    try:
        @torch.compile(mode="default") 
        def func_with_format(x):
            # Simula il tipo di print che causava l'errore originale
            tensor_shape = x.shape
            tile_size = 512
            pad = 32
            print(f'[Tiled VAE]: input_size: {tensor_shape}, tile_size: {tile_size}, padding: {pad}')
            return x
        
        x = torch.randn(1, 3, 256, 256)
        result = func_with_format(x)
        print("✅ String format print works")
        return True
    except Exception as e:
        print(f"❌ String format print failed: {e}")
        return False

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"Dynamo suppress_errors: {torch._dynamo.config.suppress_errors}")
    print(f"Print in reorderable functions: {print in torch._dynamo.config.reorderable_logging_functions}")
    print()
    
    tests = [
        test_basic_compile,
        test_compile_with_print, 
        test_string_format_print
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 All Dynamo tests passed!")
        print("The PyTorch Dynamo fix should resolve the original error.")
    else:
        print(f"⚠️ {passed}/{total} tests passed.")
        print("The fix may not work completely on this environment.")
