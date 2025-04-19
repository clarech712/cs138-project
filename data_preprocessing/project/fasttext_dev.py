import fasttext
import numpy as np
import os
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.losses import MultipleNegativesRankingLoss, ContrastiveLoss
from torch.utils.data import DataLoader
import torch
import random
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastTextModule:
    def __init__(self, model_path):
        self.model_path = model_path
        self.fasttext = None

    def train_model(self, 
                   train_data_path, 
                   model='skipgram',
                   dim=200,
                   epoch=25
        ):
        """
        Train a FastText model on text data and save it
        
        Args:
            train_data_path (str): Path to training data file
            model_path (str): Path to save trained model
        """
        try:
            # Train FastText model
            logger.info(f"Training FastText model with {model} approach, dim={dim}, epochs={epoch}")
            model = fasttext.train_unsupervised(train_data_path, 
                                            model=model,
                                            dim=dim,
                                            epoch=epoch)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
            
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
                
            model.save_model(self.model_path)
            self.fasttext = model
            logger.info(f"FastText model trained and saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error training FastText model: {str(e)}")
            raise
    
    def load_from_path(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            self.fasttext = fasttext.load_model(self.model_path)
            logger.info(f"FastText model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading FastText model: {str(e)}")
            raise
        
    def get_similarities_for_sample(self, sampled_items, top_k=15):
        """
        Calculate similarities between provided pairs of terms
        
        Args:
            sampled_items (list): List of pre-sampled terms
            top_k (int): Number of top similar pairs to return
            
        Returns:
            list: Top k most similar term pairs with scores
        """
        if self.fasttext is None:
            raise ValueError("FastText model not loaded. Call load_from_path() or train_model() first.")
            
        # Calculate similarity between all pairs
        similarities = []
        for i in range(len(sampled_items)):
            for j in range(i+1, len(sampled_items)):
                term1 = sampled_items[i]
                term2 = sampled_items[j]
                try:
                    # Get vectors and calculate cosine similarity
                    vec1 = self.fasttext.get_word_vector(term1)
                    vec2 = self.fasttext.get_word_vector(term2)
                    
                    # Ensure vectors are not zero
                    if np.all(vec1 == 0) or np.all(vec2 == 0):
                        continue
                        
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    similarities.append((term1, term2, similarity))
                except Exception as e:
                    logger.warning(f"Error computing similarity between '{term1}' and '{term2}': {str(e)}")
                    continue
        
        # Sort by similarity score and get top k
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]
    
class SentenceTransformerModule:
    def __init__(self, model_name='all-MiniLM-L6-v2', model_save_path=None):
        """
        Initialize with a pre-trained sentence transformer model
        
        Args:
            model_name (str): Name of pre-trained model to use
            model_save_path (str): Path to save trained model
        """
        self.model_name = model_name
        self.model = None
        self.model_save_path = model_save_path or f"../misc_data/{model_name}-finetuned"
        
    def load_model(self):
        """Load the sentence transformer model"""
        try:
            if os.path.exists(self.model_save_path):
                self.model = SentenceTransformer(self.model_save_path)
                logger.info(f"Loaded trained model from {self.model_save_path}")
            else:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded pre-trained model {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {str(e)}")
            raise
        
    def unsupervised_train(self, terms, epochs=5, batch_size=16, force_train=False):
        """
        Train sentence embeddings in an unsupervised way.
        
        Args:
            terms (list): List of terms/technical skills
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            force_train (bool): Whether to force training even if model exists
        """
        # Check if model already exists
        if os.path.exists(self.model_save_path) and not force_train:
            logger.info(f"Trained model already exists at {self.model_save_path}. Skipping training. Use force_train=True to override.")
            return
            
        if self.model is None:
            self.load_model()
            
        logger.info(f"Training {self.model_name} on {len(terms)} terms using unsupervised learning...")
        
        try:
            # Filter out empty terms
            terms = [term for term in terms if term and len(term.strip()) > 0]
            
            if len(terms) == 0:
                logger.warning("No valid terms provided for training")
                return

            if len(terms) < batch_size:
                logger.warning(f"Not enough terms ({len(terms)}) for batch size {batch_size}. Reducing batch size.")
                batch_size = max(4, len(terms) // 2)  # Ensure we have at least 4 terms per batch or half the data
                
            # For MultipleNegativesRankingLoss, we need to create pairs where each term is paired with itself
            # This way, the model learns to distinguish between similar and dissimilar terms
            train_examples = []
            
            for i in range(0, len(terms), 2):
                if i + 1 < len(terms):
                    # Create a pair with adjacent terms (they serve as positives for each other)
                    train_examples.append(InputExample(texts=[terms[i], terms[i+1]]))
                else:
                    # If we have an odd number, pair the last term with itself
                    train_examples.append(InputExample(texts=[terms[i], terms[i]]))
                    
            logger.info(f"Created {len(train_examples)} training examples")
            
            if len(train_examples) == 0:
                logger.warning("No training examples created, skipping unsupervised training")
                return
                
            # Create data loader
            train_dataloader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)
            
            # MultipleNegativesRankingLoss uses in-batch negatives
            train_loss = MultipleNegativesRankingLoss(self.model)
            
            logger.info(f"Created dataloader with {len(train_examples)} examples, batch size {batch_size}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.model_save_path)), exist_ok=True)
            
            # Train the model
            warmup_steps = int(len(train_dataloader) * epochs * 0.1)
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=max(warmup_steps, 10),
                show_progress_bar=True
            )
            
            # Save the trained model
            self.model.save(self.model_save_path)
            logger.info(f"Model trained with unsupervised approach and saved to {self.model_save_path}")
        except Exception as e:
            logger.error(f"Error in unsupervised training: {str(e)}")
            raise
     
    def get_most_similar_terms(self, query_term, terms, top_k=15):
        """
        Find the most similar terms to a query term
        
        Args:
            query_term (str): The term to find similar skills for
            terms (list): List of candidate terms
            top_k (int): Number of similar terms to return
            
        Returns:
            list: Top k most similar terms with scores
        """
        if self.model is None:
            self.load_model()
        
        try:    
            # Filter out empty terms and the query term itself
            filtered_terms = [term for term in terms if term and term != query_term]
            
            if len(filtered_terms) == 0:
                logger.warning("No valid terms provided for comparison")
                return []
                
            # Encode query term and all terms
            query_embedding = self.model.encode(query_term, convert_to_tensor=True)
            term_embeddings = self.model.encode(filtered_terms, convert_to_tensor=True)
            
            # Calculate similarities
            similarities = []
            for i, term in enumerate(filtered_terms):
                similarity = torch.nn.functional.cosine_similarity(
                    query_embedding.unsqueeze(0), 
                    term_embeddings[i].unsqueeze(0)
                ).item()
                similarities.append((term, similarity))
            
            # Sort and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"Error finding similar terms: {str(e)}")
            return []
     
    def get_similarities_for_sample(self, sampled_items, top_k=15):
        """
        Calculate similarities between provided pairs of terms using sentence embeddings
        
        Args:
            sampled_items (list): Pre-sampled list of terms
            top_k (int): Number of top similar pairs to return
            
        Returns:
            list: Top k most similar term pairs with scores
        """
        if self.model is None:
            self.load_model()
            
        try:
            if len(sampled_items) < 2:
                logger.warning("Not enough terms for similarity calculation")
                return []
                
            # Encode all sampled terms at once (more efficient)
            embeddings = self.model.encode(sampled_items, convert_to_tensor=True)
            
            # Calculate cosine similarity between all pairs
            similarities = []
            for i in range(len(sampled_items)):
                for j in range(i+1, len(sampled_items)):
                    term1 = sampled_items[i]
                    term2 = sampled_items[j]
                    # Calculate cosine similarity using the pre-computed embeddings
                    similarity = torch.nn.functional.cosine_similarity(embeddings[i].unsqueeze(0), 
                                                                    embeddings[j].unsqueeze(0)).item()
                    similarities.append((term1, term2, similarity))
            
            # Sort by similarity score and get top k
            similarities.sort(key=lambda x: x[2], reverse=True)
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"Error calculating similarities: {str(e)}")
            return []

def load_hardskill_data(train_data_path):
    """
    Loads hard skill data from a file and strips whitespace.
    
    Args:
        train_data_path (str): Path to the training data file.
        
    Returns:
        np.ndarray: Array of technical skills data.
    """
    try:
        # Check if file exists
        if not os.path.exists(train_data_path):
            logger.error(f"Data file not found: {train_data_path}")
            raise FileNotFoundError(f"Data file not found: {train_data_path}")
            
        with open(train_data_path, 'r', encoding='utf-8') as f:
            train_data = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(train_data)} technical skills from {train_data_path}")
        return np.array(train_data, dtype=str)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def sample_random_terms(terms, n_samples=100):
    """
    Sample random terms from a list
    
    Args:
        terms (list): List of terms to sample from
        n_samples (int): Number of terms to sample
        
    Returns:
        list: Randomly sampled terms
    """
    # Ensure n_samples is not larger than the available terms
    n_samples = min(n_samples, len(terms))
    
    # Filter out any empty terms
    filtered_terms = [term for term in terms if term and len(term.strip()) > 0]
    
    if len(filtered_terms) < 2:
        logger.warning("Not enough valid terms for sampling")
        return filtered_terms
        
    # Sample random terms
    return np.random.choice(filtered_terms, size=n_samples, replace=False).tolist()

def main():
    try:
        # Example usage
        model_path = "../misc_data/hard_skills_model.bin"
        train_data_path = '../misc_data/all_technical_skills.txt'
        force_train_ft = False   # Set to True to force FastText training
        force_train_s2v = False  # Set to True to force Sentence2Vec training
        model_type = "both"     # Options: "fasttext", "sentence2vec", or "both"
        train_s2v = False        # Whether to train the sentence2vec model with unsupervised learning
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        
        # Check if training data exists
        if not os.path.exists(train_data_path):
            logger.error(f"Training data not found: {train_data_path}")
            return
    
        hardskill_data = load_hardskill_data(train_data_path)
        
        if len(hardskill_data) == 0:
            logger.error("No data loaded from training file")
            return
        
        # Initialize models
        ft_model = None
        s2v_model = None
        
        # Initialize and use models based on type
        if model_type == "fasttext" or model_type == "both":
            logger.info("=== FastText Model ===")
            ft_model = FastTextModule(model_path)
            
            # Train model if it doesn't exist or if force_train_ft is True
            if not os.path.exists(model_path) or force_train_ft:
                ft_model.train_model(train_data_path)
            else:
                ft_model.load_from_path()
                
        if model_type == "sentence2vec" or model_type == "both":
            logger.info("\n=== Sentence2Vec Model ===")
            # Use a pre-trained sentence transformer model
            s2v_model = SentenceTransformerModule()
            s2v_model.load_model()
            
            # Train if requested using unsupervised approach
            if train_s2v:
                s2v_model.unsupervised_train(hardskill_data, epochs=5, force_train=force_train_s2v)
            
            # Example query for a specific term
            if len(hardskill_data) > 0:
                query_term = hardskill_data[0]  # Use first term as example
                similar_terms = s2v_model.get_most_similar_terms(query_term, hardskill_data)
                print(f"\nMost similar skills to '{query_term}' (Sentence2Vec):")
                for term, score in similar_terms:
                    print(f"  {term}: {score:.3f}")
                
            # Get similarities between random pairs
            sampled_terms = sample_random_terms(hardskill_data, 100)
            s2v_similarities = s2v_model.get_similarities_for_sample(sampled_terms)
            print("\nSentence2Vec top similar pairs:")
            for term1, term2, score in s2v_similarities:
                print(f"{term1} - {term2}: {score:.3f}")
        
        # Generate similarity comparisons based on the same random sample
        if model_type == "both" and ft_model and s2v_model:
            # Sample random terms for comparison
            n_samples = 100
            sampled_terms = sample_random_terms(hardskill_data, n_samples)
            logger.info(f"Sampled {len(sampled_terms)} terms for model comparison")
            
            # Get similarities from both models using the same sample
            ft_similarities = ft_model.get_similarities_for_sample(sampled_terms)
            s2v_similarities = s2v_model.get_similarities_for_sample(sampled_terms)
            
            print("\n=== Model Comparison ===")
            print("\nFastText top similar pairs:")
            for term1, term2, score in ft_similarities:
                print(f"{term1} - {term2}: {score:.3f}")
                
            print("\nSentence2Vec top similar pairs:")
            for term1, term2, score in s2v_similarities:
                print(f"{term1} - {term2}: {score:.3f}")
            
            print("\nSide-by-side comparison of top pairs:")
            for i in range(min(10, len(ft_similarities), len(s2v_similarities))):
                ft_term1, ft_term2, ft_score = ft_similarities[i]
                s2v_term1, s2v_term2, s2v_score = s2v_similarities[i]
                
                print(f"FastText:     {ft_term1} - {ft_term2}: {ft_score:.3f}")
                print(f"Sentence2Vec: {s2v_term1} - {s2v_term2}: {s2v_score:.3f}")
                print()
        elif model_type == "fasttext" and ft_model:
            # Just show FastText similarities
            sampled_terms = sample_random_terms(hardskill_data, 100)
            ft_similarities = ft_model.get_similarities_for_sample(sampled_terms)
            
            print("\nFastText top similar pairs:")
            for term1, term2, score in ft_similarities:
                print(f"{term1} - {term2}: {score:.3f}")
        elif model_type == "sentence2vec" and s2v_model:
            # Just show Sentence2Vec similarities
            sampled_terms = sample_random_terms(hardskill_data, 100)
            s2v_similarities = s2v_model.get_similarities_for_sample(sampled_terms)
            
            print("\nSentence2Vec top similar pairs:")
            for term1, term2, score in s2v_similarities:
                print(f"{term1} - {term2}: {score:.3f}")
                
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise
        
if __name__ == "__main__":
    main()