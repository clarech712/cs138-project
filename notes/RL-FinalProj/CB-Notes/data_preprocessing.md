# Data Processing Workflow for Job Data Analysis
[Back to Index](../index.md)
## 1. Overview of the Data Pipeline

Our project involves several steps to process job posting data for reinforcement learning applications:

1. **Data Collection**: Gathering datasets from Kaggle sources
2. **Data Preprocessing**: Cleaning and standardizing the data
3. **MongoDB Integration**: Storing processed data in a NoSQL database
4. **LLM-Based Data Mining**: Using Gemini to extract structured information
5. **Feature Engineering**: Preparing data for ML models

## 2. Data Sources

We're working with three main datasets from Kaggle:

1. **Google Jobs Search Results** (`gsearch_jobs.csv`) - 287MB, 61.3K entries
2. **LinkedIn Job Postings** (`postings.csv`) - 556MB, 124K entries 
3. **Data Analyst Jobs** (`DataAnalyst.csv`) - 7.57MB, 2,253 entries

## 3. Data Preprocessing Steps

### 3.1 Initial Data Exploration

I used the `DataFrameSummarizer` class to analyze our datasets before importing them to MongoDB:

```python
from data_tools import DataFrameSummarizer

# Load dataset
df = pd.read_csv("kaggle_datasets/gsearch_jobs.csv")

# Create summarizer
summarizer = DataFrameSummarizer(df)

# Display overview
summarizer.display_summary()
```

This provides detailed information about:
- Dataset dimensions
- Column types
- Missing value percentages
- Value distributions

### 3.2 Data Cleaning

Key preprocessing steps included:
- Handling missing values
- Standardizing text fields
- Removing duplicates
- Normalizing column names

## 4. MongoDB Integration

### 4.1 Setting Up MongoDB

MongoDB was chosen for its flexibility with unstructured data and document-oriented approach.

```python
# MongoDB connection code
from utils.mongo_setup import setup_mongodb

# Connect to MongoDB
client, db, collection = setup_mongodb(
    db_name="rl_jobsdb", 
    collection_name="all_jobs"
)
```

### 4.2 Importing Data to MongoDB

The `MongoImporter` class handles loading CSV data into MongoDB:

```python
from utils.mongo_utils import MongoImporter

# Create importer
importer = MongoImporter(
    db_name="rl_jobsdb",
    collection_name="all_jobs"
)

# Import all CSV files from directory
results = importer.import_all_files("kaggle_datasets/")

# Close connection
importer.close()
```

#### How the MongoImporter Works:

1. **Initialization**:
   ```python
   def __init__(self, mongo_uri=None, db_name="rl_jobsdb", collection_name="all_jobs", db_path="../mongo_db/"):
       if mongo_uri is None:
           # Use standard local MongoDB connection
           mongo_uri = "mongodb://localhost:27017/"
           
       # Create database directory
       db_path = Path(db_path)
       db_path.mkdir(parents=True, exist_ok=True)
           
       self.client = pymongo.MongoClient(mongo_uri)
       self.db = self.client[db_name]
       self.collection = self.db[collection_name]
   ```

2. **CSV Processing**:
   ```python
   def process_csv_file(self, file_path, chunk_size=1000):
       # Read file in chunks to handle large datasets
       for chunk in pd.read_csv(file_path, chunksize=chunk_size):
           documents = []
           
           for idx, row in chunk.iterrows():
               # Create document from row
               doc = {
                   "doc_id": f"{file_name.split('.')[0]}_{global_idx}",
                   "source_file": file_name,
                   "original_index": original_index,
                   "job_title": job_title,
                   "description": description,
                   "metadata": {...}  # Additional fields
               }
               documents.append(doc)
           
           # Bulk insert documents
           self.collection.insert_many(documents, ordered=False)
   ```

### 4.3 MongoDB Document Structure

Each job posting in MongoDB has this structure:

```json
{
    "_id": ObjectId("..."),
    "doc_id": "unique_identifier",
    "source_file": "source_filename",
    "original_index": "index_from_source",
    "job_title": "job title",
    "description": "job description",
    "metadata": {
        // Additional fields from source data
    },
    "technical_skills": ["python", "sql", "tableau"],
    "soft_skills": ["communication", "teamwork"],
    "experience_requirements": ["3+ years experience", "Bachelor's degree"]
}
```

### 4.4 Iterating Through MongoDB Documents

The `JobIterator` class provides efficient iteration through job documents:

```python
from utils.mongo_utils import JobIterator

# Create iterator (returns documents in batches)
iterator = JobIterator(
    batch_size=100,
    query={"technical_skills": {"$exists": False}}  # Only unprocessed jobs
)

# Process batches of documents
for batch in iterator:
    for job in batch:
        # Process each job document
        print(job["job_title"])
```

## 5. LLM-Based Data Mining with Gemini

### 5.1 Setting Up Gemini

```python
from utils.llm_data_mining import SkillExtractor

# Initialize the skill extractor
extractor = SkillExtractor(
    model_name='gemini-2.0-flash-lite',
    model_type='gemini'
)
```

### 5.2 Extracting Structured Data

I used Google's Gemini model to extract:
- Technical skills (programming languages, tools)
- Soft skills (communication, teamwork)
- Experience requirements (education, years of experience)

```python
# Format job data for the extractor
jobs_dict = {
    job_id: {
        'description': job.get('description', ''),
        'job_title': job.get('job_title', '')
    }
    for job in batch
}

# Extract skills data
results = extractor.extract_job_data(jobs=jobs_dict)
```

### 5.3 Updating MongoDB with Extracted Data

```python
def update_jobs_with_skills(results, collection):
    for job_id, skills_data in results.items():
        # Prepare update operation
        update_data = {
            "$set": {
                "technical_skills": skills_data.get("technical_skills", []),
                "soft_skills": skills_data.get("soft_skills", []),
                "experience_requirements": skills_data.get("experience_requirements", [])
            }
        }
        
        # Update document
        collection.update_one({"_id": job_id}, update_data)
```

### 5.4 Batch Processing with Rate Limiting

To avoid API rate limits:

```python
# Constants for rate limiting
RATE_LIMIT_PER_MINUTE = 30
SECONDS_BETWEEN_REQUESTS = (60.0 / RATE_LIMIT_PER_MINUTE) * 1.1  # Add 10% buffer

# Rate limiting implementation
current_time = time.time()
elapsed_since_last_request = current_time - last_request_time

if elapsed_since_last_request < SECONDS_BETWEEN_REQUESTS:
    sleep_time = SECONDS_BETWEEN_REQUESTS - elapsed_since_last_request
    time.sleep(sleep_time)

# Process batch
results = extractor.extract_job_data(jobs=jobs_dict)
last_request_time = time.time()  # Update last request time
```

## 6. MongoDB Backup and Maintenance

### 6.1 Backing Up MongoDB

```python
from utils.mongo_utils import backup_mongodb

# Backup the database
backup_mongodb(
    db_name="rl_jobsdb", 
    backup_path="../mongo_db/"
)
```

### 6.2 MongoDB Database Structure

- **Database**: `rl_jobsdb`
- **Collections**:
  - `all_jobs`: Main collection with processed job postings
  - `test_case`: Collection for testing and validation

## 7. MongoDB Query Examples

### 7.1 Basic Queries

```python
# Find all data analyst jobs
data_analyst_jobs = collection.find({"job_title": {"$regex": "data analyst", "$options": "i"}})

# Count jobs with Python as a skill
python_jobs_count = collection.count_documents({"technical_skills": "python"})

# Find jobs requiring 3+ years experience
experienced_jobs = collection.find({"experience_requirements": {"$regex": "3\\+.*years", "$options": "i"}})
```

### 7.2 Aggregation Pipelines

```python
# Count frequency of each technical skill
skill_counts = collection.aggregate([
    {"$unwind": "$technical_skills"},
    {"$group": {"_id": "$technical_skills", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
    {"$limit": 10}
])

# Find jobs with both Python and SQL skills
jobs_with_python_and_sql = collection.find({
    "$and": [
        {"technical_skills": "python"},
        {"technical_skills": "sql"}
    ]
})
```

## 8. Next Steps

1. **Feature Engineering**: Convert extracted skills into feature vectors
2. **Model Training**: Train ML models on the processed data
3. **Skill Recommendation**: Develop a system to recommend skills based on job descriptions
4. **Job Matching**: Create algorithms to match job seekers with appropriate positions

## 9. Technical Challenges and Solutions

### Challenge 1: Handling Large CSV Files
- **Solution**: Used pandas' chunk processing to reduce memory usage

### Challenge 2: Rate Limiting LLM API Calls
- **Solution**: Implemented controlled sleep periods between requests

### Challenge 3: Inconsistent Data Formats
- **Solution**: Created a flexible document schema in MongoDB

### Challenge 4: Extracting Structured Data from Job Descriptions
- **Solution**: Utilized Gemini's text understanding capabilities with carefully crafted prompts
