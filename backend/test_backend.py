"""
Simple test script to verify backend functionality.
Run this after installing dependencies to ensure everything works.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import fastapi
        print(" [OK] FastAPI imported successfully")
    except ImportError as e:
        print(f"[FAIL] FastAPI import failed: {e}")
        return False
    
    try:
        import yfinance
        print(" [OK] yfinance imported successfully")
    except ImportError as e:
        print(f"[FAIL] yfinance import failed: {e}")
        return False
    
    try:
        import pandas
        print(" [OK] pandas imported successfully")
    except ImportError as e:
        print(f"[FAIL] pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        print(" [OK] scikit-learn imported successfully")
    except ImportError as e:
        print(f"[FAIL] scikit-learn import failed: {e}")
        return False
    
    try:
        import xgboost
        print(" [OK] XGBoost imported successfully")
    except ImportError as e:
        print(f"[FAIL] XGBoost import failed: {e}")
        return False
    
    try:
        import lightgbm
        print(" [OK] LightGBM imported successfully")
    except ImportError as e:
        print(f"[FAIL] LightGBM import failed: {e}")
        return False
    
    try:
        import tensorflow
        print(" [OK] TensorFlow imported successfully")
    except ImportError as e:
        print(f"[FAIL] TensorFlow import failed: {e}")
        return False
    
    try:
        import ta
        print(" [OK] ta (technical analysis) imported successfully")
    except ImportError as e:
        print(f"[FAIL] ta import failed: {e}")
        return False
    
    return True


def test_modules():
    """Test that our custom modules can be imported."""
    print("\nTesting custom modules...")
    
    try:
        from data_pipeline import get_history, get_fundamentals
        print(" [OK] data_pipeline imported successfully")
    except ImportError as e:
        print(f"[FAIL] data_pipeline import failed: {e}")
        return False
    
    try:
        from model import build_features
        print(" [OK] model imported successfully")
    except ImportError as e:
        print(f"[FAIL] model import failed: {e}")
        return False
    
    try:
        from fundamental_analyzer import analyze_fundamentals
        print(" [OK] fundamental_analyzer imported successfully")
    except ImportError as e:
        print(f"[FAIL] fundamental_analyzer import failed: {e}")
        return False
    
    try:
        from main import app
        print(" [OK] main (FastAPI app) imported successfully")
    except ImportError as e:
        print(f"[FAIL] main import failed: {e}")
        return False
    
    return True


def test_data_fetch():
    """Test fetching data from yfinance."""
    print("\nTesting data fetching...")
    
    try:
        from data_pipeline import get_history
        
        # Try to fetch data for a well-known stock
        print("Fetching AAPL historical data...")
        data = get_history("AAPL", period="1mo")
        
        if data and 'data' in data and len(data['data']) > 0:
            print(f" [OK] Successfully fetched {len(data['data'])} days of data for AAPL")
            return True
        else:
            print("[FAIL] Data fetch returned empty result")
            return False
            
    except Exception as e:
        print(f"[FAIL] Data fetch failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Backend Verification Test")
    print("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Package Imports", test_imports()))
    
    # Test custom modules
    results.append(("Custom Modules", test_modules()))
    
    # Test data fetching
    results.append(("Data Fetching", test_data_fetch()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = " [OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n [OK] All tests passed! Backend is ready to use.")
        print("\nTo start the server, run:")
        print("  uvicorn main:app --reload")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Please check the errors above.")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
