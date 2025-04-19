import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Sentence2VecEncoder:
    """
    Utility class for loading and using the pretrained Sentence2Vec model for
    technical skills embeddings.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Sentence2Vec encoder
        
        Args:
            model_path (str, optional): Path to the trained model. If None, uses the default path.
        """
        self.model_path = model_path or "../misc_data/all-MiniLM-L6-v2-finetuned"
        self.model = None
        
    def load_model(self) -> None:
        """
        Load the pretrained Sentence2Vec model
        """
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model not found at {self.model_path}. Attempting to use base model name instead.")
                self.model = SentenceTransformer(os.path.basename(self.model_path))
            else:
                self.model = SentenceTransformer(self.model_path)
                logger.info(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for the given texts
        
        Args:
            texts (str or list): Single text or list of texts to encode
            normalize (bool): Whether to normalize the embeddings
            
        Returns:
            numpy.ndarray: Embeddings for the input texts
        """
        if self.model is None:
            self.load_model()
            
        try:
            # Handle single text input
            if isinstance(texts, str):
                texts = [texts]
                
            # Generate embeddings
            embeddings = self.model.encode(texts, normalize_embeddings=normalize)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise 