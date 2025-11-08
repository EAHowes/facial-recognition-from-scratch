"""
Facial Recognition Clustering CLI
Team JME - Jose Ortega, Mateo Bedoya, Ethan Howes

Main interface for running K-means and DBSCAN clustering algorithms
on synthetic facial embeddings.
"""

import numpy as np
import time
import sys
from pathlib import Path

#imports clustering algos
try:
    from k_means import Kmeans
    from dbscan import dbscan
except ImportError:
    print("Error: Could not import clustering algorithms.")
    print("Make sure k_means.py and dbscan.py are in the same directory.")
    sys.exit(1)


class FacialClusteringCLI:
    #command line interface
    
    def __init__(self): #we fill these after load/processing
        self.embeddings = None
        self.labels = None
        self.results = {} # store results from each algo to comp later
        
    def load_data(self):
        #load embewddings and ground truth tables

        #note: keeping it relative so we can run from src/ without installing a pkg
        try:
            embeddings_path = Path("../data/embeddings.npy")
            labels_path = Path("../data/labels.npy")
            
            print("\n" + "="*60)
            print("LOADING DATA")
            print("="*60)
            
            self.embeddings = np.load(embeddings_path)
            self.labels = np.load(labels_path)
            
            print(f"Loaded {len(self.embeddings)} embeddings")
            print(f"Embedding dimensions: {self.embeddings.shape[1]}")
            print(f"Ground truth identities: {len(np.unique(self.labels))}")
            print(f"Data type: {self.embeddings.dtype}")
            
            return True
            
        except FileNotFoundError:
            print("\nError: Embedding files not found!")
            print("Please run: cd ../data && python embeddings.py")
            return False
        except Exception as e: # a catch-all so that CLI doesnt crash without a msg
            print(f"\nError loading data: {e}")
            return False
    
    def run_kmeans(self, k=10, max_iters=20):
        #run kmeans algo
        print("\n" + "="*60)
        print("RUNNING K-MEANS CLUSTERING")
        print("="*60)
        print(f"Parameters: K={k}, max_iterations={max_iters}")
        print("Distance metric: Cosine distance")
        print("\nClustering in progress...\n")
        
        start_time = time.time()
        
        # Run K-means on full dataset
        km = Kmeans(K=k, max_iters=max_iters)
        km.fit_data(self.embeddings)
        
        runtime = time.time() - start_time
        
        # Calculate clustering accuracy
        accuracy = self._calculate_accuracy(km.labels, self.labels) #Evaluate against ground truths using majority vote per cluster
        
        # Store results for later without rerunning
        self.results['kmeans'] = {
            'labels': km.labels,
            'centroids': km.centroids,
            'iterations': km.n_iter_,
            'inertia': km.inertia,
            'runtime': runtime,
            'clusters': k,
            'accuracy': accuracy,
            'cluster_sizes': np.bincount(km.labels, minlength=k)
        }
        
        self._print_kmeans_results()
        
    def run_dbscan(self, eps=1.007, min_pts=10):
        # run dbscan algo
        print("\n" + "="*60)
        print("RUNNING DBSCAN CLUSTERING")
        print("="*60)
        print(f"Parameters: eps={eps}, minPts={min_pts}")
        print("Distance metric: Euclidean distance") # note: kmeans is using cosine on same data so results arent 1:1 comparable
        print("\nClustering in progress (this may take 30-90 minutes)...\n") #about 42 mins on my pc
        
        # Convert to float32 for performance
        # dbscan is On^2 so everything helps
        embeddings_f32 = self.embeddings.astype(np.float32, copy=False)
        
        start_time = time.time()
        
        # Run dbscan
        predicted_labels = dbscan(embeddings_f32, eps, min_pts) # Make sure its a numpy array so np ops work
        
        runtime = time.time() - start_time
        
        # Convert to numpy array
        predicted_labels = np.asarray(predicted_labels, dtype=int)
        
        # Calculate metrics.
        noise_points = int(np.sum(predicted_labels == -1))
        cluster_ids = [c for c in np.unique(predicted_labels) if c >= 1]
        n_clusters = len(cluster_ids)
        
        # Calculate accuracy, excluding noise points
        accuracy = self._calculate_dbscan_accuracy(predicted_labels, self.labels) #dbscan spits out -1 for noise so we dont want to punish accuracy for those
        
        # Store results
        self.results['dbscan'] = {
            'labels': predicted_labels,
            'runtime': runtime,
            'clusters': n_clusters,
            'noise_points': noise_points,
            'noise_percentage': (noise_points / len(predicted_labels)) * 100,
            'accuracy': accuracy,
            'total_points': len(predicted_labels)
        }
        
        self._print_dbscan_results()
    
    def _calculate_accuracy(self, predicted, ground_truth):# majority-vote accuracy per cluster, simple way to compare to ground truth
        
        if len(predicted) != len(ground_truth):
            return 0.0
        
        unique_clusters = np.unique(predicted)
        correct = 0
        
        for cluster_id in unique_clusters:
            # get all ground truth labels for this cluster
            cluster_mask = (predicted == cluster_id)
            cluster_gt_labels = ground_truth[cluster_mask]
            
            # Pick label that appears the most inside the cluster
            if len(cluster_gt_labels) > 0:
                labels, counts = np.unique(cluster_gt_labels, return_counts=True)
                correct += counts.max()
        
        return correct / len(predicted)
    
    def _calculate_dbscan_accuracy(self, predicted, ground_truth):
        # calculate clustering accuracy for dbscan, excluding noise
        # Ignore noise points, same idea above
        non_noise_mask = predicted != -1
        
        if np.sum(non_noise_mask) == 0: #corner case where dbscan says everything is noise
            return 0.0
        
        predicted_no_noise = predicted[non_noise_mask]
        gt_no_noise = ground_truth[non_noise_mask]
        
        unique_clusters = np.unique(predicted_no_noise)
        correct = 0
        
        for cluster_id in unique_clusters:
            cluster_mask = (predicted_no_noise == cluster_id)
            cluster_gt_labels = gt_no_noise[cluster_mask]
            
            if len(cluster_gt_labels) > 0:
                labels, counts = np.unique(cluster_gt_labels, return_counts=True)
                correct += counts.max()
        
        return correct / len(predicted_no_noise) # Accuracy over the non noise set
    
    def _print_kmeans_results(self):
        #Format the printing for kmeans
        r = self.results['kmeans']
        
        print("\n" + "-"*60)
        print("K-MEANS RESULTS")
        print("-"*60)
        print(f"Runtime:              {r['runtime']:.2f} seconds")
        print(f"Iterations:           {r['iterations']}")
        print(f"Clusters formed:      {r['clusters']}")
        print(f"Inertia:              {r['inertia']:.4f}")
        print(f"Clustering Accuracy:  {r['accuracy']*100:.2f}%")
        print(f"\nCluster sizes:")
        for i, size in enumerate(r['cluster_sizes']):
            print(f"  Cluster {i}: {size} faces")
        print("-"*60)
    
    def _print_dbscan_results(self):
        #format the printing for dbscan
        r = self.results['dbscan']
        
        print("\n" + "-"*60)
        print("DBSCAN RESULTS")
        print("-"*60)
        print(f"Runtime:              {r['runtime']:.2f} seconds ({r['runtime']/60:.1f} minutes)")
        print(f"Total points:         {r['total_points']}")
        print(f"Clusters formed:      {r['clusters']}")
        noise_pts = r['noise_points']
        noise_pct = r['noise_percentage']
        print(f"Noise points:         {noise_pts} ({noise_pct:.2f}%)")
        print(f"Clustered points:     {r['total_points'] - noise_pts}")
        print(f"Clustering Accuracy:  {r['accuracy']*100:.2f}% (excludes noise)")
        print("-"*60)
    
    def compare_algorithms(self):
        #print the side by side comparison of both algos
        if 'kmeans' not in self.results or 'dbscan' not in self.results:
            print("\nBoth algorithms must be run before comparison.")
            return
        
        km = self.results['kmeans']
        db = self.results['dbscan']
        
        print("\n" + "="*60)
        print("ALGORITHM COMPARISON")
        print("="*60)
        print(f"{'Metric':<25} {'K-Means':<20} {'DBSCAN':<20}")
        print("-"*60)
        
        km_runtime_str = f"{km['runtime']:.2f}s"
        db_runtime_str = f"{db['runtime']:.2f}s ({db['runtime']/60:.1f}m)"
        print(f"{'Runtime':<25} {km_runtime_str:<20} {db_runtime_str:<20}")
        print(f"{'Clusters Found':<25} {km['clusters']:<20} {db['clusters']:<20}")
        
        km_acc_str = f"{km['accuracy']*100:.2f}%"
        db_acc_str = f"{db['accuracy']*100:.2f}%"
        print(f"{'Accuracy':<25} {km_acc_str:<20} {db_acc_str:<20}")
        
        db_noise_str = f"Yes ({db['noise_points']} pts)"# dbscan has noise, kmeans doesnt
        print(f"{'Noise Handling':<25} {'No':<20} {db_noise_str:<20}")
        print(f"{'Distance Metric':<25} {'Cosine':<20} {'Euclidean':<20}")
        print(f"{'Iterations':<25} {km['iterations']:<20} {'N/A':<20}")
        print("="*60)
        
        # Determine winner
        print("\nKEY OBSERVATIONS:")
        if km['runtime'] < db['runtime']:
            speedup = db['runtime'] / km['runtime']
            print(f"K-Means is {speedup:.1f}x faster")
        else:
            speedup = km['runtime'] / db['runtime']
            print(f"DBSCAN is {speedup:.1f}x faster")
        
        if km['accuracy'] > db['accuracy']:
            acc_diff = (km['accuracy'] - db['accuracy']) * 100
            print(f"K-Means has {acc_diff:.2f}% higher accuracy")
        elif db['accuracy'] > km['accuracy']:
            acc_diff = (db['accuracy'] - km['accuracy']) * 100
            print(f"DBSCAN has {acc_diff:.2f}% higher accuracy")
        else:
            print("Both algorithms achieved similar accuracy")
        
        print(f"DBSCAN identified {db['noise_points']} noise/outlier points")
        print("="*60)
    
    
def print_banner():
    #Welcome banner
    print("\n" + "="*60)
    print("  FACIAL RECOGNITION CLUSTERING - Team JME")
    print("  Comparing K-Means vs. DBSCAN Algorithms")
    print("="*60)


def print_menu():
    # Main menu options
    print("\n" + "-"*60)
    print("MENU OPTIONS")
    print("-"*60)
    print("1. Run K-Means clustering")
    print("2. Run DBSCAN clustering")
    print("3. Run both algorithms")
    print("4. Compare results (requires both algorithms run)")
    print("5. Exit")
    print("-"*60)


def main():
    # main program loop
    print_banner()
    
    cli = FacialClusteringCLI()
    
    # Load data first
    if not cli.load_data(): #make sure we have data in memory first
        print("\nExiting due to data loading error.")
        return
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1': #let user override k/max_iters if they want
            k = input("Enter number of clusters K (default 10): ").strip()
            k = int(k) if k else 10
            max_iters = input("Enter max iterations (default 20): ").strip()
            max_iters = int(max_iters) if max_iters else 20
            cli.run_kmeans(k=k, max_iters=max_iters)
            
        elif choice == '2': #dbscan params are mroe sensitive so we expose them
            eps = input("Enter eps value (default 1.07): ").strip()
            eps = float(eps) if eps else 1.07
            min_pts = input("Enter minPts (default 10): ").strip()
            min_pts = int(min_pts) if min_pts else 10
            
            print("\nWarning: DBSCAN may take 30-90 minutes to run on 100k points")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                cli.run_dbscan(eps=eps, min_pts=min_pts)
            else:
                print("DBSCAN cancelled.")
                
        elif choice == '3': #run all at once
            print("\nWarning: This will run both algorithms sequentially")
            print("    Total time: ~30-90 minutes for DBSCAN + seconds for K-Means")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                cli.run_kmeans()
                cli.run_dbscan()
                cli.compare_algorithms()
            else:
                print("Operation cancelled.")
                
        elif choice == '4':
            cli.compare_algorithms()
            
            
        elif choice == '5':
            print("\n" + "="*60)
            print("Thank you for using Facial Recognition Clustering!")
            print("Team JME - Jose Ortega, Mateo Bedoya, Ethan Howes")
            print("="*60 + "\n")
            break
            
        else:
            print("\nInvalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()