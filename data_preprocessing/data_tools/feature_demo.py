"""
Feature Transformation Pipeline Demo

This script demonstrates how to use the feature transformation pipeline 
to convert MongoDB job data into vector representations and visualize them.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from data_preprocessing.preprocessing_utils.feature_pipeline import FeatureTransformationPipeline
from data_preprocessing.preprocessing_utils.mongo_utils import JobIterator

def visualize_embeddings(embeddings, labels=None, method='tsne', title='Embedding Visualization'):
    """
    Visualize high-dimensional embeddings in 2D.
    
    Args:
        embeddings (np.ndarray): The embeddings to visualize
        labels (list): Optional labels for color-coding points
        method (str): Dimensionality reduction method ('tsne' or 'pca')
        title (str): Plot title
    """
    # Perform dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        # Convert labels to categorical for coloring
        unique_labels = list(set(labels))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        label_ids = [label_to_id[label] for label in labels]
        
        scatter = plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            c=label_ids, 
            cmap='viridis', 
            alpha=0.7
        )
        
        # Add legend
        if len(unique_labels) <= 20:  # Only show legend if not too many categories
            plt.legend(
                handles=scatter.legend_elements()[0], 
                labels=unique_labels,
                title="Job Categories",
                loc="best"
            )
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    plt.title(f"{title} ({method.upper()})")
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.tight_layout()
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f"plots/{title.lower().replace(' ', '_')}_{method}.png")
    plt.close()

def main():
    """Main demonstration function"""
    print("Initializing feature transformation pipeline...")
    pipeline = FeatureTransformationPipeline()
    
    # Get sample jobs for demonstration
    sample_size = 200  # Limit sample size for visualization
    print(f"Fetching {sample_size} sample jobs from MongoDB...")
    
    # Use JobIterator to get a batch of jobs
    job_iter = JobIterator(batch_size=sample_size)
    try:
        sample_jobs = next(job_iter)
    except StopIteration:
        print("No jobs found in database.")
        return
    
    # Extract job categories for visualization labels
    job_categories = []
    for job in sample_jobs:
        # Try to extract job category from metadata or title
        category = None
        if 'metadata' in job and 'category' in job['metadata']:
            category = job['metadata']['category']
        elif 'job_title' in job:
            # Simple heuristic to extract category from title
            title = job['job_title'].lower()
            if 'data' in title:
                category = 'Data'
            elif 'engineer' in title:
                category = 'Engineering'
            elif 'analyst' in title:
                category = 'Analyst'
            elif 'manager' in title or 'management' in title:
                category = 'Management'
            elif 'developer' in title or 'programmer' in title:
                category = 'Developer'
            else:
                category = 'Other'
        else:
            category = 'Unknown'
        
        job_categories.append(category)
    
    print("Processing sample jobs...")
    features = pipeline.process_job_batch(sample_jobs)
    
    # Visualize different feature representations
    print("Visualizing job title embeddings...")
    visualize_embeddings(
        features['title_embeddings'], 
        labels=job_categories,
        title='Job Title Embeddings'
    )
    
    print("Visualizing job description embeddings...")
    visualize_embeddings(
        features['description_embeddings'], 
        labels=job_categories,
        title='Job Description Embeddings'
    )
    
    print("Visualizing technical skills embeddings...")
    visualize_embeddings(
        features['technical_skill_vectors'], 
        labels=job_categories,
        title='Technical Skills Embeddings'
    )
    
    # Combine features using different methods
    print("Creating combined representations...")
    concatenated = pipeline.combine_features(features, method='concatenate')
    
    print("Visualizing combined embeddings...")
    visualize_embeddings(
        concatenated, 
        labels=job_categories,
        title='Combined Job Embeddings'
    )
    
    print("Demonstration complete. Visualizations saved to 'plots' directory.")

if __name__ == "__main__":
    main() 