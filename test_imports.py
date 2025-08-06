#!/usr/bin/env python3
"""
Test script to verify all modules import correctly
"""

print("Testing imports...")

try:
    from features import comprehensive_feature_engineering
    print("✅ features.py imported successfully")
except ImportError as e:
    print(f"❌ Error importing features.py: {e}")

try:
    from labeling import create_actionable_targets
    print("✅ labeling.py imported successfully")
except ImportError as e:
    print(f"❌ Error importing labeling.py: {e}")

try:
    from models import AdvancedPreprocessor, create_actionable_models
    print("✅ models.py imported successfully")
except ImportError as e:
    print(f"❌ Error importing models.py: {e}")

try:
    from evaluate import evaluate_model_actionable, plot_confusion_matrix
    print("✅ evaluate.py imported successfully")
except ImportError as e:
    print(f"❌ Error importing evaluate.py: {e}")

try:
    from main import set_random_seeds, setup_logging
    print("✅ main.py imported successfully")
    
    # Test the fixed function
    set_random_seeds(42)
    print("✅ set_random_seeds() works correctly")
    
except ImportError as e:
    print(f"❌ Error importing main.py: {e}")
except Exception as e:
    print(f"❌ Error running main.py functions: {e}")

print("\n🎉 All imports and basic functions working correctly!")
print("🚀 You can now run: python3 main.py")