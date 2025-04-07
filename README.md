# Reinforcement Learning Project: Job Data Analysis and Skill Extraction

## Project Overview
This project focuses on analyzing job postings data using reinforcement learning techniques. It processes and analyzes job postings from multiple sources to extract skills, requirements, and patterns in job descriptions. The project uses MongoDB for data storage and provides tools for data preprocessing, analysis, and model training.

## Project Structure

```
.
├── data_tools/                 # Data processing and analysis tools
│   ├── __init__.py
│   └── dataset_preprocessing.py # Tools for preprocessing and analyzing datasets
├── kaggle_datasets/           # Raw job posting datasets
│   ├── gsearch_jobs.csv      # Google Jobs search results (287MB)
│   ├── postings.csv          # LinkedIn job postings (556MB)
│   ├── DataAnalyst.csv       # Data analyst specific jobs (7.57MB)
│   └── DATASET_SOURCES.md    # Documentation of dataset sources
├── mongo_db/                  # MongoDB database files and backups
├── project/                   # Main project files
│   └── data_preprocessing.ipynb # Jupyter notebook for data preprocessing
├── utils/                     # Utility functions and tools
│   ├── __init__.py
│   ├── api_keys.json         # API keys configuration
│   ├── llm_data_mining.py    # LLM-based data mining utilities
│   ├── model_manager.py      # Model management utilities
│   ├── mongo_setup.py        # MongoDB setup and configuration
│   ├── mongo_utils.py        # MongoDB utility functions
│   └── process_jobs.py       # Job data processing utilities
├── requirements.txt          # Project dependencies
└── README.md                # This file
```

## Data Sources

The project uses three main datasets:

1. **Google Jobs Search Results** (`gsearch_jobs.csv`)
   - Source: Kaggle - Data Analyst Job Postings Google Search
   - Size: 287MB (61.3K entries)
   - Content: Job postings from Google Jobs search for data analyst positions

2. **LinkedIn Job Postings** (`postings.csv`)
   - Source: Kaggle - LinkedIn Job Postings
   - Size: 556MB (124K entries)
   - Content: Comprehensive job postings from LinkedIn

3. **Data Analyst Jobs** (`DataAnalyst.csv`)
   - Source: Kaggle - Data Analyst Jobs
   - Size: 7.57MB (2,253 entries)
   - Content: Specialized dataset for data analyst positions

## Database Structure

The project uses MongoDB with the following structure:

- **Database**: `rl_jobsdb`
- **Collections**:
  - `all_jobs`: Main collection storing processed job postings
  - `test_case`: Collection for testing and validation

### Document Structure
Each job posting document in MongoDB has the following structure:
```json
{
    "doc_id": "unique_identifier",
    "source_file": "source_filename",
    "original_index": "index_from_source",
    "job_title": "job title",
    "description": "job description",
    "metadata": {
        // Additional fields from source data
    }
}
```

## Key Components

### Data Processing
- `dataset_preprocessing.py`: Provides tools for analyzing and preprocessing datasets
- `DataFrameSummarizer`: Class for generating detailed summaries of datasets

### MongoDB Integration
- `mongo_setup.py`: Handles MongoDB connection and setup
- `mongo_utils.py`: Provides utilities for:
  - Importing CSV data into MongoDB
  - Backing up MongoDB databases
  - Iterating through job documents
  - Data validation and cleaning

### Model Management
- `model_manager.py`: Handles model training and management
- `llm_data_mining.py`: Utilities for LLM-based data extraction

## Dependencies
```
pandas>=1.3.0
pymongo>=4.0.0
tqdm>=4.62.0
google-generativeai>=0.3.0
openai>=1.0.0
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up MongoDB:
   - Ensure MongoDB is installed and running locally
   - The project uses the default MongoDB connection (mongodb://localhost:27017/)

3. Import data:
   - Place your datasets in the `kaggle_datasets/` directory
   - Use the MongoDB import utilities to load data into the database

4. Run preprocessing:
   - Use the Jupyter notebook in `project/data_preprocessing.ipynb` for initial data exploration
   - Use the data preprocessing tools in `data_tools/` for specific processing tasks

## Contributing
When adding new datasets or making changes:
1. Update the `DATASET_SOURCES.md` file with new dataset information
2. Follow the existing data processing pipeline
3. Ensure proper documentation of any new features or changes

## Notes
- All datasets are preprocessed to remove duplicates and standardize text
- The MongoDB database is backed up regularly to the `mongo_db/` directory
- API keys should be stored in `utils/api_keys.json` (not tracked in git)