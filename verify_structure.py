"""
Verify the project structure without importing external dependencies
"""

import os
import importlib.util

def check_file_exists(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path)

def check_import_without_dependencies(module_path):
    """Verify that a Python module exists without actually importing it"""
    return importlib.util.find_spec(module_path) is not None

# Define expected project structure
expected_files = [
    "data_preprocessing/preprocessing_utils/__init__.py",
    "data_preprocessing/preprocessing_utils/mongo_utils.py",
    "data_preprocessing/preprocessing_utils/mongo_setup.py",
    "data_preprocessing/preprocessing_utils/feature_pipeline.py",
    "data_preprocessing/preprocessing_utils/llm_data_mining.py",
    "data_preprocessing/preprocessing_utils/model_manager.py",
    "data_preprocessing/preprocessing_utils/process_jobs.py",
    "data_preprocessing/__init__.py",
]

# Check if the files exist
print("Checking file structure:")
structure_valid = True
for file_path in expected_files:
    exists = check_file_exists(file_path)
    status = "✓" if exists else "❌"
    print(f"{status} {file_path}")
    if not exists:
        structure_valid = False

# Check if modules are importable
print("\nChecking modules are importable:")
modules = [
    "data_preprocessing", 
    "data_preprocessing.preprocessing_utils"
]
modules_valid = True
for module in modules:
    importable = check_import_without_dependencies(module)
    status = "✓" if importable else "❌"
    print(f"{status} {module}")
    if not importable:
        modules_valid = False

# Final verdict
if structure_valid and modules_valid:
    print("\n✓ Project structure is valid!")
    print("The changes from utils/ to preprocessing_utils/ were successful.")
    print("\nNote: To use the code in Jupyter notebooks, update import statements to use:")
    print("from data_preprocessing.preprocessing_utils.X import Y")
else:
    print("\n❌ Project structure has issues that need to be addressed.") 