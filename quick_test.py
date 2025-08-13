#!/usr/bin/env python3
"""
Quick test script to verify HYPIR components
"""

import sys
import os

def test_basic_imports():
    """Test basic Python imports"""
    print("🧪 Testing basic Python imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL/Pillow")
    except ImportError as e:
        print(f"❌ PIL: {e}")
        return False
    
    return True

def test_hypir_import():
    """Test HYPIR specific imports"""
    print("\n🧪 Testing HYPIR imports...")
    
    # Add HYPIR to path
    hypir_path = os.path.join(os.getcwd(), "HYPIR", "HYPIR")
    if hypir_path not in sys.path:
        sys.path.insert(0, hypir_path)
        print(f"📁 Added to path: {hypir_path}")
    
    # Also add the parent HYPIR path for nested imports
    parent_hypir_path = os.path.join(os.getcwd(), "HYPIR")
    if parent_hypir_path not in sys.path:
        sys.path.insert(0, parent_hypir_path)
    
    try:
        # Try to import base enhancer first
        from HYPIR.enhancer.base import BaseEnhancer
        print("✅ HYPIR.enhancer.base.BaseEnhancer")
        
        # Then try SD2Enhancer
        from HYPIR.enhancer.sd2 import SD2Enhancer
        print("✅ HYPIR.enhancer.sd2.SD2Enhancer")
        return True
    except ImportError as e:
        print(f"❌ HYPIR import: {e}")
        return False

def test_predict_import():
    """Test predict.py import"""
    print("\n🧪 Testing predict.py...")
    
    try:
        # Test our predict module
        import predict
        print("✅ predict.py imported")
        
        # Check if Predictor class exists
        if hasattr(predict, 'Predictor'):
            print("✅ Predictor class found")
            return True
        else:
            print("❌ Predictor class not found")
            return False
    except ImportError as e:
        print(f"❌ predict.py import: {e}")
        return False
    except Exception as e:
        print(f"❌ predict.py error: {e}")
        return False

def main():
    """Run all tests"""
    print("🔬 HYPIR Quick Test Suite")
    print("=" * 40)
    
    success = True
    
    # Test 1: Basic imports
    if not test_basic_imports():
        success = False
    
    # Test 2: HYPIR import
    if not test_hypir_import():
        success = False
    
    # Test 3: Predict module
    if not test_predict_import():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
