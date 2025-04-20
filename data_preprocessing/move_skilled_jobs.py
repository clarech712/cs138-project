#!/usr/bin/env python
"""
Script to move MongoDB jobs with non-empty technical_skills from one collection to another.
Supports moving between different MongoDB instances (local to remote, etc.)
"""

import argparse
from preprocessing_utils.mongo_utils import move_jobs_with_skills
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move jobs with non-empty technical_skills to another collection")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--source-uri", type=str, default="mongodb://localhost:27017/", 
                        help="Source MongoDB URI (defaults to local MongoDB)")
    parser.add_argument("--target-uri", type=str, default=None, 
                        help="Target MongoDB URI (if different from source)")
    parser.add_argument("--db-name", type=str, default="rl_jobsdb", help="Database name")
    parser.add_argument("--source", type=str, default="all_jobs", help="Source collection name")
    parser.add_argument("--target", type=str, default="skilled_jobs", help="Target collection name")
    parser.add_argument("--use-ssl", action="store_true", help="Use SSL for remote connections")
    
    args = parser.parse_args()
    
    logger.info("Starting job migration process")
    
    moved_count = move_jobs_with_skills(
        batch_size=args.batch_size,
        source_mongo_uri=args.source_uri,
        target_mongo_uri=args.target_uri,
        db_name=args.db_name,
        source_collection_name=args.source,
        target_collection_name=args.target,
        use_ssl=args.use_ssl
    )
    
    logger.info(f"Job migration complete. Total jobs moved: {moved_count}") 