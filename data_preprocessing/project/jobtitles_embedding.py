import os
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import Dataset

# Import components from fasttext_dev.py to avoid code duplication
from fasttext_dev import SentenceTransformerModule, sample_random_terms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_jobtitle_data(data_path):
    """
    Loads job title data from a file and strips whitespace.
    
    Args:
        data_path (str): Path to the job titles data file.
        
    Returns:
        np.ndarray: Array of job titles.
    """
    try:
        # Check if file exists
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        with open(data_path, 'r', encoding='utf-8') as f:
            job_titles = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(job_titles)} job titles from {data_path}")
        return np.array(job_titles, dtype=str)
    except Exception as e:
        logger.error(f"Error loading job title data: {str(e)}")
        raise

def main():
    try:
        # Configuration
        model_name = 'all-MiniLM-L6-v2'  # Base model to use
        
        # Use paths relative to the script location
        model_save_path = os.path.join(SCRIPT_DIR, "..", "misc_data", "jobtitles-embeddings")
        data_path = os.path.join(SCRIPT_DIR, "..", "misc_data", "all_job_titles.txt")
        
        force_train = False  # Set to True to force training even if model exists
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
        
        # Log the actual paths being used
        logger.info(f"Using data path: {os.path.abspath(data_path)}")
        logger.info(f"Using model save path: {os.path.abspath(model_save_path)}")
        
        # Check if training data exists
        if not os.path.exists(data_path):
            logger.error(f"Job titles data not found: {data_path}")
            return
    
        # Load job title data
        jobtitle_data = load_jobtitle_data(data_path)
        
        if len(jobtitle_data) == 0:
            logger.error("No data loaded from job titles file")
            return
        
        logger.info(f"Loaded {len(jobtitle_data)} job titles for embedding training")
        
        # Initialize the sentence transformer model
        logger.info("=== Job Title Embedding Model ===")
        s2v_model = SentenceTransformerModule(model_name=model_name, model_save_path=model_save_path)
        s2v_model.load_model()
        
        # Train the model using unsupervised approach
        logger.info("Training job title embedding model...")
        s2v_model.unsupervised_train(jobtitle_data, epochs=10, force_train=force_train)
        
        # Display example similarities
        if len(jobtitle_data) > 0:
            # Sample some job titles to query
            query_titles = sample_random_terms(jobtitle_data, 5)
            
            for query_title in query_titles:
                similar_titles = s2v_model.get_most_similar_terms(query_title, jobtitle_data)
                print(f"\nMost similar job titles to '{query_title}':")
                for title, score in similar_titles:
                    print(f"  {title}: {score:.3f}")
            
            # Get similarities between random pairs
            sampled_titles = sample_random_terms(jobtitle_data, 100)
            similarities = s2v_model.get_similarities_for_sample(sampled_titles)
            
            print("\nTop similar job title pairs:")
            for title1, title2, score in similarities:
                print(f"{title1} - {title2}: {score:.3f}")
                
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise
        
if __name__ == "__main__":
    main() 