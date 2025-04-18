#!/usr/bin/env python
"""
Test script to verify imports from the new preprocessing_utils directory work correctly.
"""

import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports to work
sys.path.append(str(Path(__file__).parent))

try:
    print("Testing direct imports...")
    from data_preprocessing.preprocessing_utils.mongo_utils import MongoImporter
    print("✓ Successfully imported MongoImporter")
    
    from data_preprocessing.preprocessing_utils.mongo_setup import setup_mongodb
    print("✓ Successfully imported setup_mongodb")
    
    from data_preprocessing.preprocessing_utils.feature_pipeline import FeatureTransformationPipeline
    print("✓ Successfully imported FeatureTransformationPipeline")
    
    from data_preprocessing.preprocessing_utils.llm_data_mining import SkillExtractor
    print("✓ Successfully imported SkillExtractor")
    
    print("\nAll imports successful! The new project structure works correctly.")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTrying alternative import approach...")
    
    try:
        import data_preprocessing.preprocessing_utils
        print("✓ Successfully imported the preprocessing_utils package")
        print("\nThe new project structure works, but you may need to update specific import statements.")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nThe new project structure has issues that need to be addressed.") 