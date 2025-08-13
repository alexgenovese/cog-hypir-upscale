#!/usr/bin/env python3
"""
Simplified test to verify the main predict functionality
"""

import sys
import os

def test_predict_functionality():
    """Test if we can instantiate and use the Predictor"""
    print("🧪 Testing Predictor functionality...")
    
    try:
        import predict
        print("✅ predict.py imported successfully")
        
        # Try to instantiate the Predictor
        predictor = predict.Predictor()
        print("✅ Predictor instantiated successfully")
        
        # Check if setup method exists
        if hasattr(predictor, 'setup'):
            print("✅ setup() method exists")
        else:
            print("❌ setup() method not found")
            return False
            
        # Check if predict method exists
        if hasattr(predictor, 'predict'):
            print("✅ predict() method exists")
        else:
            print("❌ predict() method not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing Predictor: {e}")
        return False

def test_file_structure():
    """Test if all necessary files exist"""
    print("🧪 Testing file structure...")
    
    required_files = [
        "cog.yaml",
        "predict.py",
        "requirements.txt",
        "download_models.py"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    # Check HYPIR directory
    if os.path.exists("HYPIR"):
        print("✅ HYPIR directory exists")
    else:
        print("❌ HYPIR directory missing")
        missing_files.append("HYPIR/")
    
    return len(missing_files) == 0

def main():
    """Run simplified tests"""
    print("🔬 HYPIR Simplified Test Suite")
    print("=" * 40)
    
    success = True
    
    # Test 1: File structure
    if not test_file_structure():
        print("⚠️  Some files missing but continuing...")
    
    # Test 2: Predict functionality
    if not test_predict_functionality():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Basic functionality tests passed!")
        print("📝 Note: Full COG build may require additional setup")
        return True
    else:
        print("❌ Basic functionality tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
