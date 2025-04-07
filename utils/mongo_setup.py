import os
import pymongo
import logging
from pathlib import Path
from urllib.parse import quote_plus

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_mongodb(db_name="rl_jobsdb", collection_name="test_case"):
    """
    Set up a MongoDB database in the specified directory and connect to it.
    
    Args:
        db_name (str): Name of the database
        collection_name (str): Name of the collection
        
    Returns:
        tuple: (client, db, collection) - MongoDB client, database, and collection objects
    """
    
    # Connect to MongoDB using the standard localhost connection
    # Note: The dbpath is not a valid URI option, so we'll use the default MongoDB data directory
    uri = "mongodb://localhost:27017/"
    
    try:
        # Create a MongoDB client
        client = pymongo.MongoClient(uri)
        
        # Get the database
        db = client[db_name]
        
        # Get the collection
        collection = db[collection_name]
        
        # Test the connection
        client.server_info()
        logger.info(f"Successfully connected to MongoDB at {uri}")
        logger.info(f"Using database: {db_name}, collection: {collection_name}")
        
        return client, db, collection
    
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        raise

def close_mongodb_connection(client):
    """
    Close the MongoDB connection.
    
    Args:
        client: MongoDB client object
    """
    try:
        client.close()
        logger.info("MongoDB connection closed")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {str(e)}")

if __name__ == "__main__":
    # Example usage
    client, db, collection = setup_mongodb()
    
    # Insert a test document
    test_doc = {
        "test": "This is a test document",
        "timestamp": "2023-04-06"
    }
    
    result = collection.insert_one(test_doc)
    logger.info(f"Inserted document with ID: {result.inserted_id}")
    
    # Find the document
    found_doc = collection.find_one({"test": "This is a test document"})
    logger.info(f"Found document: {found_doc}")
    
    # Close the connection
    close_mongodb_connection(client) 