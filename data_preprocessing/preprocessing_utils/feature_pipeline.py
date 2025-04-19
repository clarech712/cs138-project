import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Union
from data_preprocessing.preprocessing_utils.mongo_setup import setup_mongodb
from data_preprocessing.preprocessing_utils.mongo_utils import JobIterator
import fasttext
import os

class FeatureTransformationPipeline:
    def __init__(self, embedding_model='all-MiniLM-L6-v2', fasttext_model_path=None):
        """
        Initialize the feature transformation pipeline.
        
        Args:
            embedding_model (str): The name of the sentence transformer model to use
            fasttext_model_path (str): Path to the FastText model file (.bin)
        """
        # Initialize embedding model for text data
        self.embedding_model = SentenceTransformer(embedding_model)
        # TF-IDF vectorizer for fallback or supplementary features
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100)
        # Scaler for numerical features
        self.scaler = StandardScaler()
        # Connect to MongoDB
        _, self.db, self.collection = setup_mongodb(db_name="rl_jobsdb", collection_name="all_jobs")
        
        # Initialize FastText model if path provided
        self.fasttext_model = None
        if fasttext_model_path and os.path.exists(fasttext_model_path):
            self.fasttext_model = fasttext.load_model(fasttext_model_path)
    
    def process_job_batch(self, job_batch: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Process a batch of job documents from MongoDB.
        
        Args:
            job_batch (List[Dict[str, Any]]): List of job documents
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with different feature vectors
        """
        # Extract data from job documents
        job_titles = [job.get('job_title', '') for job in job_batch]
        descriptions = [job.get('description', '') for job in job_batch]
        technical_skills = [job.get('technical_skills', []) for job in job_batch]
        soft_skills = [job.get('soft_skills', []) for job in job_batch]
        
        # Transform each component
        title_embeddings = self.transform_text_to_embeddings(job_titles)
        description_embeddings = self.transform_text_to_embeddings(descriptions)
        tech_skill_vectors = self.transform_skills_to_vectors(technical_skills)
        soft_skill_vectors = self.transform_skills_to_vectors(soft_skills)
        
        # Combine vectors if needed (here we keep them separate)
        return {
            'title_embeddings': title_embeddings,
            'description_embeddings': description_embeddings,
            'technical_skill_vectors': tech_skill_vectors,
            'soft_skill_vectors': soft_skill_vectors
        }
    
    def process_all_jobs(self, batch_size=100) -> Dict[str, np.ndarray]:
        """
        Process all jobs in MongoDB collection.
        
        Args:
            batch_size (int): Number of documents to process at once
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with feature arrays for all jobs
        """
        # Initialize empty lists for collecting data
        all_features = {
            'title_embeddings': [],
            'description_embeddings': [],
            'technical_skill_vectors': [],
            'soft_skill_vectors': []
        }
        
        # Create job iterator
        job_iter = JobIterator(batch_size=batch_size)
        
        # Process jobs in batches
        for job_batch in job_iter:
            batch_features = self.process_job_batch(job_batch)
            
            # Extend feature lists
            for key in all_features:
                all_features[key].extend(batch_features[key])
        
        # Convert lists to numpy arrays
        for key in all_features:
            all_features[key] = np.array(all_features[key])
        
        return all_features




    def transform_skill_with_fasttext(self, skill: str) -> np.ndarray:
        """
        Transform a single hard skill to embeddings using FastText.
        
        Args:
            skill (str): A technical skill string
            
        Returns:
            np.ndarray: FastText embedding vector
        """
        if self.fasttext_model is None:
            raise ValueError("FastText model not initialized. Please provide a valid model path.")
        
        # Get the FastText embedding for the skill
        return self.fasttext_model.get_word_vector(skill)
    
    def transform_skills_with_fasttext(self, skills_list: List[List[str]]) -> np.ndarray:
        """
        Transform lists of skills to FastText embeddings.
        
        Args:
            skills_list (List[List[str]]): List of skill lists
            
        Returns:
            np.ndarray: Array of skill embedding vectors
        """
        if self.fasttext_model is None:
            raise ValueError("FastText model not initialized. Please provide a valid model path.")
        
        skill_embeddings = []
        
        for skills in skills_list:
            if not skills:  # Handle empty skill lists
                # Get dimensionality from the model
                embedding_dim = self.fasttext_model.get_dimension()
                skill_embeddings.append(np.zeros(embedding_dim))
                continue
                
            # Get embeddings for each skill and average them
            skill_vecs = [self.fasttext_model.get_word_vector(skill) for skill in skills]
            avg_embedding = np.mean(skill_vecs, axis=0)
            skill_embeddings.append(avg_embedding)
        
        return np.array(skill_embeddings)
        
    def process_job_batch_with_fasttext(self, job_batch: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Process a batch of job documents using FastText for skills embeddings.
        
        Args:
            job_batch (List[Dict[str, Any]]): List of job documents
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with different feature vectors
        """
        # Extract data from job documents
        job_titles = [job.get('job_title', '') for job in job_batch]
        descriptions = [job.get('description', '') for job in job_batch]
        technical_skills = [job.get('technical_skills', []) for job in job_batch]
        soft_skills = [job.get('soft_skills', []) for job in job_batch]
        
        # Transform using different methods
        title_embeddings = self.transform_text_to_embeddings(job_titles)
        description_embeddings = self.transform_text_to_embeddings(descriptions)
        
        # Use FastText for skills
        tech_skill_vectors = self.transform_skills_with_fasttext(technical_skills)
        soft_skill_vectors = self.transform_skills_with_fasttext(soft_skills)
        
        return {
            'title_embeddings': title_embeddings,
            'description_embeddings': description_embeddings,
            'technical_skill_vectors': tech_skill_vectors,
            'soft_skill_vectors': soft_skill_vectors
        }
    
    def save_features(self, features: Dict[str, np.ndarray], output_path: str):
        """
        Save feature vectors to a file.
        
        Args:
            features (Dict[str, np.ndarray]): Dictionary of feature arrays
            output_path (str): Path to save the features
        """
        np.savez(output_path, **features)
    
    def load_features(self, input_path: str) -> Dict[str, np.ndarray]:
        """
        Load feature vectors from a file.
        
        Args:
            input_path (str): Path to load the features from
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of feature arrays
        """
        loaded = np.load(input_path)
        return {key: loaded[key] for key in loaded.files}