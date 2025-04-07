import os
import pandas as pd
import pymongo
from tqdm import tqdm
import sys
import logging
from typing import Dict, Any, List
from pathlib import Path
import subprocess
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mongo_import.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MongoImporter:
    """A class to import job data from CSV files into MongoDB."""
    
    def __init__(self, mongo_uri: str = None, 
                 db_name: str = "rl_jobsdb",
                 collection_name: str = "all_jobs",
                 db_path: str = "../mongo_db/"):
        """
        Initialize the MongoImporter.
        
        Args:
            mongo_uri (str): MongoDB connection URI (if None, will use local MongoDB)
            db_name (str): Name of the database
            collection_name (str): Name of the collection
            db_path (str): Path to the directory where the database will be stored (used for logging only)
        """
        if mongo_uri is None:
            # Use standard local MongoDB connection
            mongo_uri = "mongodb://localhost:27017/"
            
        # Create the database directory if it doesn't exist (for logging purposes)
        db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Database directory: {db_path.absolute()}")
            
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        logger.info(f"Connected to MongoDB: {mongo_uri}")
        logger.info(f"Using database: {db_name}, collection: {collection_name}")
    
    def process_csv_file(self, file_path: str, chunk_size: int = 1000) -> int:
        """
        Process a CSV file and insert its contents into MongoDB.
        
        Args:
            file_path (str): Path to the CSV file
            chunk_size (int): Number of rows to process at once
            
        Returns:
            int: Number of documents inserted
        """
        file_name = os.path.basename(file_path)
        logger.info(f"Processing file: {file_name}")
        
        # Get the total number of rows for progress bar
        total_rows = sum(1 for _ in open(file_path, encoding='utf-8')) - 1  # Subtract header row
        logger.info(f"Total rows in {file_name}: {total_rows}")
        
        # Read the first row to determine column names
        try:
            first_row = pd.read_csv(file_path, nrows=1, encoding='utf-8')
            logger.info(f"Successfully read file with UTF-8 encoding")
        except Exception as e:
            logger.error(f"Error reading file {file_name}: {str(e)}")
            return 0
        
        columns = first_row.columns.tolist()
        
        # Find the title and description columns
        title_column = None
        description_column = None
        
        # Common variations of title column names
        title_variations = ['title', 'job_title', 'job title', 'jobtitle', 'position', 'role', 'Job Title']
        for col in columns:
            if col.lower() in [v.lower() for v in title_variations]:
                title_column = col
                break
        
        # Common variations of description column names
        description_variations = ['description', 'Job Description', 'job_description', 'job description', 'jobdescription', 'details', 'responsibilities']
        for col in columns:
            if col.lower() in [v.lower() for v in description_variations]:
                description_column = col
                break
        
        # If columns not found, use the first column for title and second for description
        if title_column is None and len(columns) > 0:
            title_column = columns[0]
            logger.warning(f"Title column not found, using '{title_column}'")
        
        if description_column is None and len(columns) > 1:
            description_column = columns[1]
            logger.warning(f"Description column not found, using '{description_column}'")
        
        logger.info(f"Using title column: '{title_column}', description column: '{description_column}'")
        
        inserted_count = 0
        global_idx = 0  # Track the global index across all chunks
        
        # Process the file in chunks to handle large files
        for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, encoding='utf-8')):
            documents = []
            
            for idx, row in chunk.iterrows():
                # Create a unique identifier using the global index
                doc_id = f"{file_name.split('.')[0]}_{global_idx}"
                global_idx += 1  # Increment the global index
                
                # Extract job title and description using detected column names
                job_title = row.get(title_column, '') if title_column else ''
                description = row.get(description_column, '') if description_column else ''
                
                # Get the index from the first column
                first_col_name = columns[0] if columns else 'index'
                original_index = row.get(first_col_name, global_idx - 1)
                
                # Create document
                doc = {
                    "doc_id": doc_id,
                    "source_file": file_name,
                    "original_index": original_index,
                    "job_title": job_title,
                    "description": description,
                    "metadata": {k: v for k, v in row.items() if k not in [title_column, description_column, first_col_name]}
                }
                
                documents.append(doc)
            
            # Insert documents in bulk
            if documents:
                try:
                    result = self.collection.insert_many(documents, ordered=False)
                    inserted_count += len(result.inserted_ids)
                    logger.info(f"Inserted {len(result.inserted_ids)} documents from chunk {chunk_idx+1}")
                except pymongo.errors.BulkWriteError as e:
                    # Handle duplicate key errors
                    inserted_count += e.details.get('nInserted', 0)
                    failed_count = len(documents) - e.details.get('nInserted', 0)
                    logger.warning(f"Bulk write error: {e.details.get('nInserted', 0)} documents inserted, {failed_count} documents failed to insert")
                    
                    # Log more detailed error information if available
                    if 'writeErrors' in e.details:
                        error_count = len(e.details['writeErrors'])
                        logger.warning(f"Error details: {error_count} specific errors reported")
                        
                        # Log the first few errors as examples
                        for i, error in enumerate(e.details['writeErrors'][:3]):  # Show first 3 errors
                            logger.warning(f"Error {i+1}: {error.get('errmsg', 'Unknown error')} (code: {error.get('code', 'unknown')})")
                        
                        if error_count > 3:
                            logger.warning(f"... and {error_count - 3} more errors")
        
        logger.info(f"Completed processing {file_name}. Total documents inserted: {inserted_count}")
        return inserted_count
    
    def import_all_files(self, directory_path: str = "../kaggle_datasets/") -> Dict[str, int]:
        """
        Import all CSV files from the specified directory.
        
        Args:
            directory_path (str): Path to the directory containing CSV files
            
        Returns:
            Dict[str, int]: Dictionary mapping file names to number of documents inserted
        """
        results = {}
        
        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files in {directory_path}")
        
        for file_name in csv_files:
            file_path = os.path.join(directory_path, file_name)
            try:
                inserted_count = self.process_csv_file(file_path)
                results[file_name] = inserted_count
            except Exception as e:
                logger.error(f"Error processing {file_name}: {str(e)}")
                results[file_name] = 0
        
        return results
    
    def close(self):
        """Close the MongoDB connection."""
        self.client.close()
        logger.info("MongoDB connection closed")


def backup_mongodb(db_name: str = "rl_jobsdb", backup_path: str = "../mongo_db/", mongodump_path: str = ""):
    """
    Perform a mongodump of the specified database to the backup path.
    
    Args:
        db_name (str): Name of the database to backup
        backup_path (str): Path to store the backup
        mongo_path (str): Path to the MongoDB binaries directory
        
    Returns:
        bool: True if backup was successful, False otherwise
    """
    try:
        from pathlib import Path
        import shutil
        import subprocess
        
        if mongodump_path == "":
            mongodump_path = os.environ.get("MONGO_PATH", "mongodump")
            logger.info(f"Using mongodump from environment variable MONGO_PATH: {mongodump_path}")
        else:
            logger.info(f"Using provided mongo_path: {mongodump_path}")
        # Create the backup directory if it doesn't exist
        backup_dir = Path(backup_path).resolve()  # Convert to absolute path
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear any existing backup in the directory
        if backup_dir.exists():
            logger.info(f"Clearing existing backup in {backup_dir}")
            for item in backup_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        
        # Construct the mongodump command
        cmd = [
            mongodump_path,
            "--db", db_name,
            "--out", str(backup_dir)
        ]
        
        logger.info(f"Running mongodump command: {' '.join(cmd)}")
        
        # Run the mongodump command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully backed up database '{db_name}' to {backup_dir}")
            logger.info(f"Backup output: {result.stdout}")
            return True
        else:
            logger.error(f"Error backing up database: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Exception during backup: {str(e)}")
        return False


class JobIterator:
    """Iterator class to fetch jobs from MongoDB in batches."""
    
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", 
                 db_name: str = "rl_jobsdb", 
                 collection_name: str = "all_jobs",
                 batch_size: int = 1000,
                 query: Dict[str, Any] = None,
                 projection: Dict[str, Any] = None):
        """
        Initialize the JobIterator.
        
        Args:
            mongo_uri (str): MongoDB connection URI
            db_name (str): Name of the database
            collection_name (str): Name of the collection
            batch_size (int): Number of documents to fetch per batch
            query (Dict[str, Any]): Query to filter documents
            projection (Dict[str, Any]): Fields to include/exclude
        """
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.batch_size = batch_size
        self.query = query or {}
        self.projection = projection or {}
        self.cursor = self.collection.find(self.query, self.projection).batch_size(batch_size)
        logger.info(f"Initialized JobIterator with batch size {batch_size}")
    
    def __iter__(self):
        """Return the iterator object."""
        return self
    
    def __next__(self):
        """Get the next batch of jobs."""
        try:
            batch = []
            for _ in range(self.batch_size):
                try:
                    job = next(self.cursor)
                    batch.append(job)
                except StopIteration:
                    if not batch:
                        raise StopIteration
                    break
            
            logger.info(f"Yielding batch of {len(batch)} jobs")
            return batch
        except StopIteration:
            logger.info("No more jobs to fetch")
            raise
    
    def close(self):
        """Close the MongoDB connection."""
        self.client.close()
        logger.info("MongoDB connection closed")

