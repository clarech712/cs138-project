"""
Simple test script to verify basic package imports
"""

import os
import sys

# Print Python path for debugging
print("Python sys.path:")
for p in sys.path:
    print(f"  - {p}")

print("\nCurrent directory:", os.getcwd())
print("\nTrying to import data_preprocessing package...")

try:
    import data_preprocessing
    print("✓ Successfully imported data_preprocessing package")
    
    print("\nTrying to import preprocessing_utils...")
    try:
        import data_preprocessing.preprocessing_utils
        print("✓ Successfully imported data_preprocessing.preprocessing_utils")
        
        # List contents of the preprocessing_utils directory
        print("\nContents of preprocessing_utils directory:")
        utils_dir = os.path.join("data_preprocessing", "preprocessing_utils")
        if os.path.exists(utils_dir):
            for item in os.listdir(utils_dir):
                print(f"  - {item}")
        else:
            print(f"Directory {utils_dir} not found!")
            
    except ImportError as e:
        print(f"❌ Error importing preprocessing_utils: {e}")
except ImportError as e:
    print(f"❌ Error importing data_preprocessing: {e}") 