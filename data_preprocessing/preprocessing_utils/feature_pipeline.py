import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Union
from data_preprocessing.preprocessing_utils.mongo_setup import setup_mongodb
from data_preprocessing.preprocessing_utils.mongo_utils import JobIterator

class FeatureTransformationPipeline:
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize the feature transformation pipeline.
        
        Args:
            embedding_model (str): The name of the sentence transformer model to use
        """
        # Initialize embedding model for text data
        self.embedding_model = SentenceTransformer(embedding_model)
        # TF-IDF vectorizer for fallback or supplementary features
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100)
        # Scaler for numerical features
        self.scaler = StandardScaler()
        # Connect to MongoDB
        _, self.db, self.collection = setup_mongodb(db_name="rl_jobsdb", collection_name="all_jobs")
    
    def transform_text_to_embeddings(self, text_list: List[str]) -> np.ndarray:
        """
        Transform text data to embeddings using sentence transformer.
        
        Args:
            text_list (List[str]): List of text strings to transform
            
        Returns:
            np.ndarray: Array of embedding vectors
        """
        return self.embedding_model.encode(text_list)
    
    def transform_skills_to_vectors(self, skills_list: List[List[str]], one_hot=False) -> np.ndarray:
        """
        Transform skills lists to vectors.
        
        Args:
            skills_list (List[List[str]]): List of skill lists
            one_hot (bool): If True, creates one-hot encoding based on all unique skills
                           If False, uses embedding averaging for each skill list
        
        Returns:
            np.ndarray: Array of skill vectors
        """
        if one_hot:
            # Create a set of all unique skills across all documents
            all_skills = set()
            for skills in skills_list:
                all_skills.update(skills)
            
            # Create one-hot encoding
            skill_vectors = np.zeros((len(skills_list), len(all_skills)))
            skill_to_idx = {skill: i for i, skill in enumerate(all_skills)}
            
            for i, skills in enumerate(skills_list):
                for skill in skills:
                    skill_vectors[i, skill_to_idx[skill]] = 1
            
            return skill_vectors
        else:
            # Use embedding approach (average of embeddings for each skill)
            skill_embeddings = []
            for skills in skills_list:
                if not skills:  # Handle empty skill lists
                    skill_embeddings.append(np.zeros(self.embedding_model.get_sentence_embedding_dimension()))
                    continue
                    
                # Get embeddings for each skill and average them
                embeddings = self.transform_text_to_embeddings(skills)
                avg_embedding = np.mean(embeddings, axis=0)
                skill_embeddings.append(avg_embedding)
            
            return np.array(skill_embeddings)
    
    def transform_numerical_features(self, numerical_data: np.ndarray) -> np.ndarray:
        """
        Scale numerical features.
        
        Args:
            numerical_data (np.ndarray): Array of numerical features
            
        Returns:
            np.ndarray: Scaled numerical features
        """
        return self.scaler.fit_transform(numerical_data)
    
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

    def combine_features(self, features: Dict[str, np.ndarray], method='concatenate') -> np.ndarray:
        """
        Combine different feature vectors into a single representation.
        
        Args:
            features (Dict[str, np.ndarray]): Dictionary of feature arrays
            method (str): Combination method - 'concatenate', 'average', or 'weighted_average'
            
        Returns:
            np.ndarray: Combined feature representation
        """
        if method == 'concatenate':
            # Simply concatenate all feature vectors
            return np.hstack([features[key] for key in features])
        
        elif method == 'average':
            # Average all feature vectors (requires same dimensions)
            return np.mean(list(features.values()), axis=0)
        
        elif method == 'weighted_average':
            # Weighted average (customize weights based on importance)
            weights = {
                'title_embeddings': 0.3,
                'description_embeddings': 0.3,
                'technical_skill_vectors': 0.3,
                'soft_skill_vectors': 0.1
            }
            
            weighted_sum = np.zeros_like(list(features.values())[0])
            for key, weight in weights.items():
                weighted_sum += features[key] * weight
                
            return weighted_sum
        
        else:
            raise ValueError(f"Unknown combination method: {method}")
    
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

# Example usage
if __name__ == "__main__":
    pipeline = FeatureTransformationPipeline()
    
    # Process all jobs
    features = pipeline.process_all_jobs(batch_size=100)
    
    # Save features
    pipeline.save_features(features, "job_features.npz")
    
    # Combine features using different methods
    concatenated = pipeline.combine_features(features, method='concatenate')
    averaged = pipeline.combine_features(features, method='average')
    weighted = pipeline.combine_features(features, method='weighted_average')
    
    print(f"Concatenated shape: {concatenated.shape}")
    print(f"Averaged shape: {averaged.shape}")
    print(f"Weighted shape: {weighted.shape}") 