from preprocessing_utils import JobIterator
from tqdm import tqdm
from pymongo import MongoClient
import certifi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

# Set MongoDB username and password
mongo_user = "cbradna"
mongo_password = "mongodb_cbradna0920"

# Construct the MongoDB connection string
mongo_uri = f"mongodb+srv://{mongo_user}:{mongo_password}@cluster0.zqzq6hs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Database and collection names
db_name = "rl_jobsdb"
source_collection_name = "jobs_text"
target_collection_name = "job_embeddings"
batch_size = 100

# Initialize the embedding model
job_title_encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the JobIterator with the MongoDB URI
iterator = JobIterator(
    mongo_uri=mongo_uri,
    db_name=db_name,
    collection_name=source_collection_name,
    batch_size=batch_size
)

# Connect to MongoDB
client = MongoClient(mongo_uri,
                     tls=True,
                     tlsCAFile=certifi.where())

# Get the database and collections
db = client[db_name]
target_collection = db[target_collection_name]

def process_batch(batch: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Extract job IDs and titles from a batch of job documents.
    Returns a list of tuples (job_id, job_title) for valid entries.
    """
    valid_entries = []
    
    for job in batch:
        job_id = job["_id"]
        job_title = job.get("job_title", "")
        
        if job_title:  # Only process jobs with non-empty titles
            valid_entries.append((job_id, job_title))
        else:
            print(f"Skipping job {job_id} - no job title found")
            
    return valid_entries

def batch_insert_or_update(id_embeddings_pairs: List[Tuple[str, List[float]]]):
    """
    Efficiently insert or update multiple job embeddings in the database.
    
    Args:
        id_embeddings_pairs: List of (job_id, embeddings) tuples
    """
    # First, find which documents already exist
    job_ids = [job_id for job_id, _ in id_embeddings_pairs]
    existing_docs = {doc["original_job_id"]: doc for doc in 
                     target_collection.find({"original_job_id": {"$in": job_ids}})}
    
    # Prepare bulk operations
    bulk_operations = []
    inserts = []
    
    for job_id, embeddings in id_embeddings_pairs:
        if job_id in existing_docs:
            # Format for updateOne operation
            bulk_operations.append(
                {
                    "updateOne": {
                        "filter": {"original_job_id": job_id},
                        "update": {"$set": {"job_title_embeddings": embeddings}}
                    }
                }
            )
        else:
            # Insert new document
            inserts.append({
                "original_job_id": job_id,
                "job_title_embeddings": embeddings
            })
    
    # Execute bulk operations
    if bulk_operations:
        target_collection.bulk_write(bulk_operations)
        
    if inserts:
        target_collection.insert_many(inserts)

# Iterate through the jobs and create embeddings
try:
    for batch in tqdm(iterator, desc="Processing job batches"):
        # Extract valid job_id and title pairs
        valid_entries = process_batch(batch)
        
        if not valid_entries:
            continue
            
        # Unpack the valid entries
        job_ids, job_titles = zip(*valid_entries)
        
        # Encode all titles in the batch at once
        batch_embeddings = job_title_encoder.encode(job_titles)
        
        # Create pairs of (job_id, embeddings)
        id_embeddings_pairs = [
            (job_id, embeddings.tolist()) 
            for job_id, embeddings in zip(job_ids, batch_embeddings)
        ]
        
        # Bulk insert or update
        batch_insert_or_update(id_embeddings_pairs)

    print("Job title embeddings created and saved successfully.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    iterator.close()
    client.close() 