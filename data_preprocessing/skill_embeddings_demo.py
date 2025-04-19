import os
import numpy as np
import fasttext
from preprocessing_utils.feature_pipeline import FeatureTransformationPipeline

def download_fasttext_model():
    """
    Downloads the pre-trained FastText model if not already present.
    Uses the smaller `crawl-300d-2M-subword` model (~1GB) for demonstration.
    """
    import urllib.request
    import os
    
    # Path to save the model
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'fasttext_model.bin')
    model_dir = os.path.dirname(model_path)
    
    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Download if not exists
    if not os.path.exists(model_path):
        print("Downloading FastText model - this may take a while (~1GB)...")
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.bin"
        urllib.request.urlretrieve(url, model_path)
        print(f"Model downloaded to {model_path}")
    
    return model_path

def demo_single_skill_embedding():
    """
    Demo embedding a single technical skill
    """
    # Get model path
    model_path = download_fasttext_model()
    
    # Initialize pipeline with FastText model
    pipeline = FeatureTransformationPipeline(fasttext_model_path=model_path)
    
    # Test with some sample hard skills
    skills = ["python", "machine learning", "data science", "tensorflow", "sql"]
    
    print("Embedding single skills:")
    for skill in skills:
        embedding = pipeline.transform_skill_with_fasttext(skill)
        print(f"{skill}: Shape = {embedding.shape}, First 5 values = {embedding[:5]}")

def demo_multiple_skills():
    """
    Demo embedding multiple skills and averaging them
    """
    # Get model path
    model_path = download_fasttext_model()
    
    # Initialize pipeline with FastText model
    pipeline = FeatureTransformationPipeline(fasttext_model_path=model_path)
    
    # Sample skill sets for different job profiles
    data_scientist_skills = ["python", "machine learning", "data science", "statistics"]
    web_developer_skills = ["javascript", "react", "html", "css", "node.js"]
    empty_skills = []
    
    # Create a list of skill lists
    skill_lists = [data_scientist_skills, web_developer_skills, empty_skills]
    
    # Get embeddings
    embeddings = pipeline.transform_skills_with_fasttext(skill_lists)
    
    print("\nEmbedding multiple skills:")
    print(f"Data Scientist Skills: Shape = {embeddings[0].shape}, First 5 values = {embeddings[0][:5]}")
    print(f"Web Developer Skills: Shape = {embeddings[1].shape}, First 5 values = {embeddings[1][:5]}")
    print(f"Empty Skills: Shape = {embeddings[2].shape}, Values (zeros) = {np.count_nonzero(embeddings[2] == 0)}/{embeddings[2].size}")

if __name__ == "__main__":
    print("FastText Skill Embeddings Demo")
    demo_single_skill_embedding()
    demo_multiple_skills() 