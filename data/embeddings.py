
#embeddings.py - generate synthetic face embeddings for clustering

#simulates face embeddings w/ 128-d vectors. each "identity" is a cluster of similar vectors.



import numpy as np
import os

class EmbeddingGenerator:
    #Generates synthetic face embeddings with known identity clusters
    
    def __init__(self, seed=42):
        #random seed for reproducibility
        np.random.seed(seed)
    
    def generate_embeddings(self, n_samples=100000, n_identities=10, embedding_dim=128):
        
        #face-like embeddings w known identity clusters
        
        #Args:
           # n_samples: Total number of face embeddings 
           # n_identities: Number of different "people" (cluster centers)
           # embedding_dim: dimension of each embedding; 128 standard for faces
            
        #Returns:
            #embeddings: numpy array of shape (n_samples, embedding_dim)
            #labels: numpy array of shape (n_samples,) with ground truth identity labels
        
        print(f"Generating {n_samples} face embeddings with {n_identities} identities...")
        
        # create identity centers, one per person
        # These represent the "avg face" for each identity
        identity_centers = np.random.randn(n_identities, embedding_dim)
        
        #Normalize centers 
        identity_centers = identity_centers / np.linalg.norm(identity_centers, axis=1, keepdims=True)
        
        embeddings = []
        labels = []
        
        #generate samples for each identity
        samples_per_identity = n_samples // n_identities
        
        for identity_id in range(n_identities):
            # How many samples for this identity
            n = samples_per_identity
            if identity_id == n_identities - 1:  # last identity gets remaining samples
                n = n_samples - len(labels)
            
            # Generate variations around the identity center
            # different photos of the same person should be similar but not identical
            noise = np.random.randn(n, embedding_dim) * 0.15  # Small variation
            identity_samples = identity_centers[identity_id] + noise
            
            # Normalize all embeddings
            identity_samples = identity_samples / np.linalg.norm(identity_samples, axis=1, keepdims=True)
            
            embeddings.append(identity_samples)
            labels.extend([identity_id] * n)
        
        embeddings = np.vstack(embeddings)
        labels = np.array(labels)
        
        #shuffle them so identities arent grouped in order
        shuffle_idx = np.random.permutation(len(embeddings))
        embeddings = embeddings[shuffle_idx]
        labels = labels[shuffle_idx]
        
        print(f"Generated {len(embeddings)} embeddings")
        print(f"  - Shape: {embeddings.shape}")
        print(f"  - Identities: {len(np.unique(labels))}")
        
        return embeddings, labels
    
    def save_embeddings(self, embeddings, labels, output_dir='.'):
        
        #Save embeddings and labels to disk
        
        #Args:
            #embeddings: numpy array of embeddings
            #labels: numpy array of ground truth labels
            #output_dir: directory to save files
        
        os.makedirs(output_dir, exist_ok=True)
        
        embeddings_path = os.path.join(output_dir, 'embeddings.npy')
        labels_path = os.path.join(output_dir, 'labels.npy')
        
        np.save(embeddings_path, embeddings)
        np.save(labels_path, labels)
        
        print(f"Saved embeddings to {embeddings_path}")
        print(f"Saved labels to {labels_path}")
        
        return embeddings_path, labels_path
    
    def load_embeddings(self, data_dir='.'):
        #Load embeddings and labels from disk
        #Args:
            #data_dir: directory containing the .npy files
            
        #Returns:
            #embeddings: numpy array of embeddings
            #labels: numpy array of ground truth labels
        
        embeddings_path = os.path.join(data_dir, 'embeddings.npy')
        labels_path = os.path.join(data_dir, 'labels.npy')
        
        embeddings = np.load(embeddings_path)
        labels = np.load(labels_path)
        
        print(f"Loaded {len(embeddings)} embeddings from {data_dir}")
        print(f"  - Shape: {embeddings.shape}")
        print(f"  - Identities: {len(np.unique(labels))}")
        
        return embeddings, labels


def main():
    #Generate and save embeddings
    generator = EmbeddingGenerator(seed=42)
    
    #Generate 100k face embeddings with 10 identities
    embeddings, labels = generator.generate_embeddings(
        n_samples=100000,
        n_identities=10,
        embedding_dim=128
    )
    
    # Save to disk
    generator.save_embeddings(embeddings, labels)
    
    # Test loading
    print("\nTesting load...")
    loaded_embeddings, loaded_labels = generator.load_embeddings()
    
    print("\n Done, clustering algorithms can now use data/embeddings.npy")


if __name__ == '__main__':
    main()
