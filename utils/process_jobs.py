#!/usr/bin/env python
"""
Simple script to fetch jobs from MongoDB and process them with SkillExtractor.
"""

import sys
import logging
from mongo_utils import JobIterator
from llm_data_mining import SkillExtractor
import pymongo
from bson.objectid import ObjectId
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiting constants
RATE_LIMIT_PER_MINUTE = 30
SECONDS_PER_REQUEST = 60.0 / RATE_LIMIT_PER_MINUTE
# Add 10% buffer to be safe
SECONDS_BETWEEN_REQUESTS = SECONDS_PER_REQUEST * 1.1

def update_jobs_with_skills(results, collection):
    """
    Update MongoDB job documents with extracted skills data.
    
    Args:
        results (dict): Dictionary of job IDs mapped to skills data from SkillExtractor
        collection: MongoDB collection to update
        
    Returns:
        int: Number of documents updated
    """
    updated_count = 0
    
    try:
        # Update each job document with its extracted skills
        for job_id, skills_data in results.items():
            # Ensure job_id is properly formatted for MongoDB query
            # MongoDB may require ObjectId for _id field if that's how they were stored
            try:
                # Try to convert to ObjectId if it's a valid format, otherwise use as is
                if ObjectId.is_valid(job_id):
                    mongo_id = ObjectId(job_id)
                else:
                    mongo_id = job_id
            except ImportError:
                # If bson is not available, use the job_id as is
                mongo_id = job_id
                
            # Prepare the update operation
            update_data = {
                "$set": {
                    "technical_skills": skills_data.get("technical_skills", []),
                    "soft_skills": skills_data.get("soft_skills", []),
                    "experience_requirements": skills_data.get("experience_requirements", [])
                }
            }
            
            # Update the document by _id
            result = collection.update_one({"_id": mongo_id}, update_data)
            
            if result.modified_count > 0:
                updated_count += 1
            else:
                logger.warning(f"Failed to update job {job_id}, document not found or already updated")
        
        logger.info(f"Total jobs updated: {updated_count}/{len(results)}")
        return updated_count
    
    except Exception as e:
        logger.error(f"Error updating jobs with skills: {str(e)}")
        return updated_count

def process_jobs(batch_size=125, mongo_uri="mongodb://localhost:27017/", 
                db_name="rl_jobsdb", collection_name="all_jobs"):
    """
    Process jobs from MongoDB using SkillExtractor.
    
    Args:
        batch_size (int): Number of jobs to process in each batch
        mongo_uri (str): MongoDB connection URI
        db_name (str): Name of the database
        collection_name (str): Name of the collection
    """
    # Initialize the job iterator
    iterator = JobIterator(batch_size=batch_size, query={"technical_skills": {"$exists": False}})
    
    # Initialize the skill extractor
    extractor = SkillExtractor()
    
    # Set up MongoDB connection for updates
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    logger.info(f"Connected to MongoDB: {mongo_uri}, database: {db_name}, collection: {collection_name}")
    
    # Rate limiting variables
    last_request_time = 0
    
    try:
        # Process jobs in batches
        count = 0
        total_updated = 0
        batch_count = 0
        start_time = time.time()
        
        for batch in iterator:
            batch_count += 1
            # Convert batch to a dictionary format expected by extractor
            # The SkillExtractor expects a dictionary where keys are job_ids and values contain the job description
            jobs_dict = {}
            for job in batch:
                job_id = str(job.get('_id'))  # Convert to string to ensure it works as a dictionary key
                # Include the description which is needed for skill extraction
                jobs_dict[job_id] = {
                    'description': job.get('description', ''),
                    'job_title': job.get('job_title', '')  # Add job title if available for better context
                }
            
            # Apply rate limiting
            current_time = time.time()
            elapsed_since_last_request = current_time - last_request_time
            
            if elapsed_since_last_request < SECONDS_BETWEEN_REQUESTS:
                # Wait if we're exceeding the rate limit
                sleep_time = SECONDS_BETWEEN_REQUESTS - elapsed_since_last_request
                logger.info(f"Rate limiting: Waiting for {sleep_time:.2f} seconds before next request")
                time.sleep(sleep_time)
            
            # Process the batch
            logger.info(f"Processing batch of {len(batch)} jobs")
            last_request_time = time.time()  # Update last request time
            results = extractor.extract_job_data(jobs=jobs_dict)
            
            # Update MongoDB with the extracted skills using the existing connection
            batch_updated = update_jobs_with_skills(results, collection)
            total_updated += batch_updated
            
            # Increment count
            count += len(batch)
            
            # Calculate and print progress information
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / batch_count if batch_count > 0 else 0
            # Estimate total number of batches (this assumes we know total count)
            try:
                total_count = getattr(iterator, 'total_count', None)
                if total_count is not None:
                    remaining_batches = (total_count // batch_size) - batch_count
                    if remaining_batches > 0:
                        estimated_time_remaining = remaining_batches * avg_time_per_batch
                        hours, remainder = divmod(estimated_time_remaining, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        time_remaining_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                    else:
                        time_remaining_str = "almost done"
                else:
                    time_remaining_str = "unknown (cannot determine total job count)"
            except Exception as e:
                logger.warning(f"Could not calculate time remaining: {str(e)}")
                time_remaining_str = "unknown (error calculating)"
                
            logger.info(f"Completed batch {batch_count} | Processed {count} jobs | Updated {total_updated} documents | Est. time remaining: {time_remaining_str}")
        
        logger.info(f"Processing complete. Total jobs processed: {count}, total updated: {total_updated}")
    
    finally:
        # Close connections
        iterator.close()
        client.close()
        logger.info("MongoDB connections closed")

if __name__ == "__main__":
    # Process all jobs in database with batch size of 125
    process_jobs(batch_size=5) 