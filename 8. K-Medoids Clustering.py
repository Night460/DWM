from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES

# Load data
sample = read_sample(SIMPLE_SAMPLES.SAMPLE_SIMPLE3)

# Initial medoids (indices of points in the dataset)
initial_medoids = [3, 12, 20]

# Create K-Medoids algorithm instance
kmedoids_instance = kmedoids(sample, initial_medoids)

# Run cluster analysis
kmedoids_instance.process()

# Extract clusters and medoids
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()

# Visualize results
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, sample)
visualizer.append_cluster(medoids, sample, marker='*', markersize=15)
visualizer.show()
