"""
================================================================================
Chapter 4: Python Implementation
Clustering, K-Means Algorithm and Applications
================================================================================

Table of Contents:
1. Part A: Implementation from Scratch (No Libraries except plotting)
2. Part B: Implementation with Scikit-learn
3. Part C: Visualization of K-Means Process
4. Part D: Real-World Applications
5. Part E: Advanced Topics (Elbow Method, Silhouette Score)

================================================================================
WHY DO WE NEED CLUSTERING?
================================================================================

The Problem:
------------
You have a bunch of data points but no labels. How do you find natural 
groupings or patterns?

Examples:
- Customer segmentation (who are my different customer types?)
- Image compression (group similar colors together)
- Document organization (which documents are about similar topics?)
- Anomaly detection (what's different from the main groups?)

Root Cause:
-----------
Data naturally forms groups, but we don't know them in advance!
We need an algorithm to DISCOVER these groups automatically.

Solution:
---------
K-Means Clustering!
- Partition data into K clusters
- Each cluster has a center (centroid)
- Points belong to nearest centroid

================================================================================
PART A: IMPLEMENTATION FROM SCRATCH
================================================================================
"""

import random

# First, let's implement helper functions we need
def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    
    Why we need this:
    -----------------
    To determine which cluster center is closest to each point!
    
    Formula: d = √((x₁-x₂)² + (y₁-y₂)² + ...)
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have same dimensions")
    
    sum_squared = 0
    for i in range(len(point1)):
        sum_squared += (point1[i] - point2[i]) ** 2
    
    return sum_squared ** 0.5


def calculate_centroid(points):
    """
    Calculate the centroid (mean position) of a group of points.
    
    Why we need this:
    -----------------
    The cluster center should be at the AVERAGE position of all points
    in that cluster!
    
    Formula: centroid = (mean of x-coords, mean of y-coords, ...)
    """
    if not points:
        return None
    
    num_points = len(points)
    num_dimensions = len(points[0])
    
    # Initialize centroid with zeros
    centroid = [0.0] * num_dimensions
    
    # Sum up all coordinates
    for point in points:
        for dim in range(num_dimensions):
            centroid[dim] += point[dim]
    
    # Divide by number of points to get average
    for dim in range(num_dimensions):
        centroid[dim] /= num_points
    
    return centroid


# =============================================================================
# THE K-MEANS ALGORITHM FROM SCRATCH
# =============================================================================

class KMeansFromScratch:
    """
    K-Means Clustering implemented from scratch.
    
    The Algorithm:
    --------------
    1. Initialize: Pick K random points as initial centroids
    2. Assignment: Assign each point to nearest centroid
    3. Update: Recalculate centroids as mean of assigned points
    4. Repeat steps 2-3 until convergence
    
    Why does this work?
    -------------------
    We're minimizing the total distance from points to their centroids!
    Each iteration improves the clustering.
    """
    
    def __init__(self, k=3, max_iterations=100, random_state=42):
        """
        Initialize K-Means.
        
        Parameters:
        -----------
        k : int
            Number of clusters
        max_iterations : int
            Maximum iterations before stopping
        random_state : int
            Random seed for reproducibility
        """
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.history = []  # Store history for visualization
    
    def fit(self, data):
        """
        Fit K-Means to the data.
        
        The Process:
        ------------
        Start → Initialize centroids → Assign points → Update centroids → 
        Repeat until convergence or max iterations
        
        Parameters:
        -----------
        data : list of lists
            Each inner list is a data point (can be any dimension)
        """
        # Set random seed for reproducibility
        random.seed(self.random_state)
        
        # Step 1: Initialize centroids randomly
        self.centroids = self._initialize_centroids(data)
        
        print(f"🎯 Starting K-Means with k={self.k}")
        print(f"📊 Data points: {len(data)}")
        print(f"📏 Dimensions: {len(data[0])}")
        print()
        
        # Store initial state
        self.history.append({
            'centroids': [c[:] for c in self.centroids],
            'labels': None
        })
        
        # Iterate until convergence or max iterations
        for iteration in range(self.max_iterations):
            # Step 2: Assign each point to nearest centroid
            old_labels = self.labels
            self.labels = self._assign_clusters(data)
            
            # Store iteration state
            self.history.append({
                'centroids': [c[:] for c in self.centroids],
                'labels': self.labels[:]
            })
            
            # Step 3: Update centroids
            old_centroids = [c[:] for c in self.centroids]
            self.centroids = self._update_centroids(data)
            
            # Check convergence (did centroids stop moving?)
            if self._has_converged(old_centroids, self.centroids):
                print(f"✅ Converged after {iteration + 1} iterations!")
                break
            
            if iteration % 10 == 0:
                inertia = self._calculate_inertia(data)
                print(f"Iteration {iteration}: Inertia = {inertia:.2f}")
        else:
            print(f"⚠️  Reached max iterations ({self.max_iterations})")
        
        # Final inertia
        final_inertia = self._calculate_inertia(data)
        print(f"🎉 Final Inertia: {final_inertia:.2f}")
        print()
        
        return self
    
    def _initialize_centroids(self, data):
        """
        Initialize centroids randomly from the data points.
        
        Why from data points?
        ---------------------
        Ensures centroids start in reasonable locations
        (within the range of actual data)
        """
        # Randomly select k data points as initial centroids
        indices = random.sample(range(len(data)), self.k)
        return [data[i][:] for i in indices]  # Copy to avoid reference issues
    
    def _assign_clusters(self, data):
        """
        Assign each point to the nearest centroid.
        
        Returns:
        --------
        list of int
            Cluster label (0 to k-1) for each point
        """
        labels = []
        
        for point in data:
            # Calculate distance to each centroid
            distances = [euclidean_distance(point, centroid) 
                        for centroid in self.centroids]
            
            # Assign to nearest centroid
            nearest_cluster = distances.index(min(distances))
            labels.append(nearest_cluster)
        
        return labels
    
    def _update_centroids(self, data):
        """
        Recalculate centroids as the mean of points in each cluster.
        
        Why recalculate?
        ----------------
        The centroid should be at the CENTER of its cluster!
        As points move between clusters, centroids must move too.
        """
        new_centroids = []
        
        for cluster_id in range(self.k):
            # Get all points in this cluster
            cluster_points = [data[i] for i in range(len(data)) 
                            if self.labels[i] == cluster_id]
            
            if cluster_points:
                # Calculate mean position
                new_centroid = calculate_centroid(cluster_points)
            else:
                # Empty cluster - reinitialize randomly
                new_centroid = random.choice(data)[:]
                print(f"⚠️  Cluster {cluster_id} is empty, reinitializing")
            
            new_centroids.append(new_centroid)
        
        return new_centroids
    
    def _has_converged(self, old_centroids, new_centroids, tolerance=1e-6):
        """
        Check if centroids have stopped moving (converged).
        
        Why check convergence?
        ----------------------
        No point continuing if nothing is changing!
        Saves computation.
        """
        for old, new in zip(old_centroids, new_centroids):
            if euclidean_distance(old, new) > tolerance:
                return False
        return True
    
    def _calculate_inertia(self, data):
        """
        Calculate inertia (sum of squared distances to nearest centroid).
        
        What is inertia?
        ----------------
        Measures how tight the clusters are.
        Lower inertia = tighter clusters = better fit
        
        Formula: Σ (distance from point to its centroid)²
        """
        inertia = 0
        for i, point in enumerate(data):
            cluster = self.labels[i]
            centroid = self.centroids[cluster]
            distance = euclidean_distance(point, centroid)
            inertia += distance ** 2
        
        return inertia
    
    def predict(self, data):
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        data : list of lists
            New data points to classify
            
        Returns:
        --------
        list of int
            Cluster label for each point
        """
        if self.centroids is None:
            raise ValueError("Model not fitted yet! Call fit() first.")
        
        labels = []
        for point in data:
            distances = [euclidean_distance(point, centroid) 
                        for centroid in self.centroids]
            labels.append(distances.index(min(distances)))
        
        return labels


# =============================================================================
# TEST THE IMPLEMENTATION
# =============================================================================

print("=" * 80)
print("TESTING K-MEANS FROM SCRATCH")
print("=" * 80)
print()

# Create simple 2D dataset with clear clusters
print("📊 Creating synthetic dataset with 3 clusters...")
print()

# Cluster 1: Points around (2, 2)
cluster1 = [[2 + random.gauss(0, 0.5), 2 + random.gauss(0, 0.5)] 
            for _ in range(30)]

# Cluster 2: Points around (8, 3)
cluster2 = [[8 + random.gauss(0, 0.5), 3 + random.gauss(0, 0.5)] 
            for _ in range(30)]

# Cluster 3: Points around (5, 8)
cluster3 = [[5 + random.gauss(0, 0.5), 8 + random.gauss(0, 0.5)] 
            for _ in range(30)]

# Combine all data
data = cluster1 + cluster2 + cluster3

# True labels (for comparison)
true_labels = [0] * 30 + [1] * 30 + [2] * 30

print(f"✓ Created {len(data)} points in 3 natural clusters")
print(f"  Cluster 1 (around 2,2): 30 points")
print(f"  Cluster 2 (around 8,3): 30 points")
print(f"  Cluster 3 (around 5,8): 30 points")
print()

# Fit K-Means
kmeans = KMeansFromScratch(k=3, max_iterations=50, random_state=42)
kmeans.fit(data)

# Show results
print("🎯 Final Centroids:")
for i, centroid in enumerate(kmeans.centroids):
    print(f"  Cluster {i}: ({centroid[0]:.2f}, {centroid[1]:.2f})")
print()

# Count points in each cluster
cluster_counts = [0, 0, 0]
for label in kmeans.labels:
    cluster_counts[label] += 1

print("📊 Cluster Sizes:")
for i, count in enumerate(cluster_counts):
    print(f"  Cluster {i}: {count} points")
print()


# =============================================================================
# DEMONSTRATION: STEP-BY-STEP VISUALIZATION
# =============================================================================

print("=" * 80)
print("STEP-BY-STEP: How K-Means Works")
print("=" * 80)
print()

# Create simpler 2D example for clear illustration
simple_data = [
    [1, 1], [1.5, 2], [2, 1.5],  # Group 1
    [8, 8], [8.5, 9], [9, 8.5],  # Group 2
]

print("📊 Simple dataset with 6 points:")
for i, point in enumerate(simple_data):
    print(f"  Point {i+1}: {point}")
print()

# Run K-Means with k=2
simple_kmeans = KMeansFromScratch(k=2, max_iterations=10, random_state=42)
simple_kmeans.fit(simple_data)

print("🔍 Let's trace through the first few iterations:")
print()

for iter_num in range(min(4, len(simple_kmeans.history))):
    state = simple_kmeans.history[iter_num]
    
    if iter_num == 0:
        print(f"INITIALIZATION:")
        print(f"  Initial centroids (randomly chosen from data):")
        for i, cent in enumerate(state['centroids']):
            print(f"    Centroid {i}: ({cent[0]:.2f}, {cent[1]:.2f})")
    else:
        print(f"\nITERATION {iter_num}:")
        print(f"  Current centroids:")
        for i, cent in enumerate(state['centroids']):
            print(f"    Centroid {i}: ({cent[0]:.2f}, {cent[1]:.2f})")
        
        print(f"  Point assignments:")
        if state['labels']:
            for i, (point, label) in enumerate(zip(simple_data, state['labels'])):
                print(f"    Point {i+1} {point} → Cluster {label}")
    
    print()


"""
================================================================================
PART B: IMPLEMENTATION WITH SCIKIT-LEARN
================================================================================

Why use Scikit-learn?
---------------------
- Highly optimized (faster)
- Battle-tested (reliable)
- More features (K-Means++, Mini-Batch K-Means, etc.)
- Industry standard

Our implementation is for LEARNING.
Scikit-learn is for PRODUCTION.
"""

from sklearn.cluster import KMeans
import numpy as np

print("=" * 80)
print("K-MEANS WITH SCIKIT-LEARN")
print("=" * 80)
print()

# Convert our data to numpy array (sklearn expects numpy)
data_np = np.array(data)

# Fit K-Means using sklearn
sklearn_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
sklearn_kmeans.fit(data_np)

print(f"✅ Sklearn K-Means fitted successfully")
print()

print("🎯 Sklearn Centroids:")
for i, centroid in enumerate(sklearn_kmeans.cluster_centers_):
    print(f"  Cluster {i}: ({centroid[0]:.2f}, {centroid[1]:.2f})")
print()

print("📊 Sklearn Cluster Sizes:")
unique, counts = np.unique(sklearn_kmeans.labels_, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} points")
print()

print(f"📉 Sklearn Inertia: {sklearn_kmeans.inertia_:.2f}")
print()

# Compare with our implementation
print("=" * 80)
print("COMPARISON: Our Implementation vs Sklearn")
print("=" * 80)
print()

our_inertia = kmeans._calculate_inertia(data)
sklearn_inertia = sklearn_kmeans.inertia_

print(f"Our Inertia:     {our_inertia:.2f}")
print(f"Sklearn Inertia: {sklearn_inertia:.2f}")
print(f"Difference:      {abs(our_inertia - sklearn_inertia):.2f}")
print()

print("💡 Note: Small differences are expected because:")
print("   - Different random initializations")
print("   - Sklearn uses K-Means++ (smarter initialization)")
print("   - Sklearn is more numerically precise")
print()


"""
================================================================================
PART C: VISUALIZATION (Using Matplotlib)
================================================================================
"""

import matplotlib.pyplot as plt

def plot_kmeans_result(data, labels, centroids, title="K-Means Clustering"):
    """
    Visualize K-Means clustering results.
    
    Parameters:
    -----------
    data : list or array
        Data points (2D for visualization)
    labels : list or array
        Cluster assignments
    centroids : list or array
        Cluster centers
    title : str
        Plot title
    """
    # Convert to numpy for easier indexing
    data_np = np.array(data)
    centroids_np = np.array(centroids)
    labels_np = np.array(labels)
    
    plt.figure(figsize=(10, 8))
    
    # Plot each cluster with different color
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for cluster_id in range(len(centroids)):
        # Get points in this cluster
        cluster_points = data_np[labels_np == cluster_id]
        
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[cluster_id % len(colors)], 
                       label=f'Cluster {cluster_id}',
                       alpha=0.6, s=50)
    
    # Plot centroids
    plt.scatter(centroids_np[:, 0], centroids_np[:, 1], 
               c='black', marker='X', s=200, 
               label='Centroids', edgecolors='white', linewidths=2)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_kmeans_animation(history, data, title="K-Means Iterations"):
    """
    Create subplots showing K-Means iterations.
    
    Parameters:
    -----------
    history : list of dict
        History of centroids and labels at each iteration
    data : list or array
        Data points
    title : str
        Main title
    """
    # Show first, middle, and last iterations
    n_iterations = len(history)
    indices_to_show = [0, n_iterations // 3, 2 * n_iterations // 3, n_iterations - 1]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    data_np = np.array(data)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for idx, iter_num in enumerate(indices_to_show):
        ax = axes[idx]
        state = history[iter_num]
        centroids = np.array(state['centroids'])
        labels = state['labels']
        
        if labels is None:
            # Initial state - just show centroids
            ax.scatter(data_np[:, 0], data_np[:, 1], 
                      c='gray', alpha=0.5, s=30)
            ax.scatter(centroids[:, 0], centroids[:, 1], 
                      c='black', marker='X', s=200, edgecolors='white', linewidths=2)
            ax.set_title(f'Initialization')
        else:
            # Show clusters
            for cluster_id in range(len(centroids)):
                cluster_points = data_np[np.array(labels) == cluster_id]
                if len(cluster_points) > 0:
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                              c=colors[cluster_id % len(colors)], 
                              alpha=0.6, s=30)
            
            ax.scatter(centroids[:, 0], centroids[:, 1], 
                      c='black', marker='X', s=200, edgecolors='white', linewidths=2)
            ax.set_title(f'Iteration {iter_num}')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


print("=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)
print()

print("📊 Creating plots...")
print("   1. Final clustering result")
print("   2. K-Means iterations (animation)")
print()

# Plot 1: Final result
plot_kmeans_result(data, kmeans.labels, kmeans.centroids, 
                   title="Final K-Means Clustering (Our Implementation)")

# Plot 2: Iterations
plot_kmeans_animation(kmeans.history, data, 
                     title="K-Means Algorithm: Watch Centroids Move!")


"""
================================================================================
PART D: REAL-WORLD APPLICATIONS
================================================================================
"""

print("\n")
print("=" * 80)
print("APPLICATION 1: CUSTOMER SEGMENTATION")
print("=" * 80)
print()

print("""
Business Problem:
-----------------
You have customer data: [Annual Income, Spending Score]
You want to segment customers into groups for targeted marketing.

Question: How many customer segments exist naturally?
""")

# Simulated customer data
# [Annual Income (k$), Spending Score (1-100)]
np.random.seed(42)

# Segment 1: Low income, low spending
seg1 = [[income, score] 
        for income, score in zip(
            np.random.normal(30, 5, 25),
            np.random.normal(35, 10, 25)
        )]

# Segment 2: Low income, high spending  
seg2 = [[income, score]
        for income, score in zip(
            np.random.normal(35, 5, 25),
            np.random.normal(75, 10, 25)
        )]

# Segment 3: High income, low spending
seg3 = [[income, score]
        for income, score in zip(
            np.random.normal(75, 5, 25),
            np.random.normal(30, 10, 25)
        )]

# Segment 4: High income, high spending
seg4 = [[income, score]
        for income, score in zip(
            np.random.normal(75, 5, 25),
            np.random.normal(75, 10, 25)
        )]

customer_data = seg1 + seg2 + seg3 + seg4

print(f"📊 Customer Data: {len(customer_data)} customers")
print(f"   Features: [Annual Income (k$), Spending Score (1-100)]")
print()

# Cluster customers
customer_kmeans = KMeansFromScratch(k=4, random_state=42)
customer_kmeans.fit(customer_data)

# Analyze segments
print("🎯 Customer Segments Discovered:")
print()

for cluster_id in range(4):
    cluster_customers = [customer_data[i] for i in range(len(customer_data))
                        if customer_kmeans.labels[i] == cluster_id]
    
    avg_income = sum(c[0] for c in cluster_customers) / len(cluster_customers)
    avg_spending = sum(c[1] for c in cluster_customers) / len(cluster_customers)
    
    print(f"Segment {cluster_id}: {len(cluster_customers)} customers")
    print(f"  Average Income: ${avg_income:.1f}k")
    print(f"  Average Spending Score: {avg_spending:.1f}")
    
    # Business interpretation
    if avg_income < 50 and avg_spending < 50:
        print(f"  💡 Type: Budget-conscious, lower income")
    elif avg_income < 50 and avg_spending >= 50:
        print(f"  💡 Type: High spenders despite lower income")
    elif avg_income >= 50 and avg_spending < 50:
        print(f"  💡 Type: High income but conservative spenders")
    else:
        print(f"  💡 Type: Premium customers - high income, high spending")
    print()

# Visualize
plot_kmeans_result(customer_data, customer_kmeans.labels, customer_kmeans.centroids,
                   title="Customer Segmentation")


print("=" * 80)
print("APPLICATION 2: IMAGE COMPRESSION")
print("=" * 80)
print()

print("""
Problem:
--------
Images have millions of colors, but we can approximate them with fewer colors!
Each pixel is a point in 3D RGB color space: (Red, Green, Blue)

Strategy:
---------
1. Treat each pixel as a 3D point
2. Cluster pixels into K color groups
3. Replace each pixel with its cluster center
4. Image now uses only K colors!

Example: Reduce 16 million colors → 16 colors
""")

# Simulate image pixels (we'll use a small example)
# In reality, you'd load actual image data
print("📸 Simulating image with 100 pixels...")
print()

# Create "image" with 3 dominant colors plus noise
red_pixels = [[200 + random.randint(-20, 20), 
               50 + random.randint(-20, 20), 
               50 + random.randint(-20, 20)] 
              for _ in range(40)]

green_pixels = [[50 + random.randint(-20, 20), 
                 200 + random.randint(-20, 20), 
                 50 + random.randint(-20, 20)] 
                for _ in range(30)]

blue_pixels = [[50 + random.randint(-20, 20), 
                50 + random.randint(-20, 20), 
                200 + random.randint(-20, 20)] 
               for _ in range(30)]

image_pixels = red_pixels + green_pixels + blue_pixels

print(f"Original: {len(image_pixels)} pixels, each with unique RGB value")
print(f"Goal: Compress to just 3 colors")
print()

# Cluster colors
color_kmeans = KMeansFromScratch(k=3, random_state=42)
color_kmeans.fit(image_pixels)

print("🎨 Compressed Color Palette (3 colors):")
for i, color in enumerate(color_kmeans.centroids):
    r, g, b = int(color[0]), int(color[1]), int(color[2])
    print(f"  Color {i}: RGB({r}, {g}, {b})")
print()

# Calculate compression ratio
original_size = len(image_pixels) * 3  # 3 bytes per pixel (RGB)
compressed_size = 3 * 3 + len(image_pixels) * 1  # 3 colors + 1 byte index per pixel

print(f"💾 Compression:")
print(f"  Original size: {original_size} bytes")
print(f"  Compressed size: {compressed_size} bytes")
print(f"  Compression ratio: {original_size / compressed_size:.1f}x")
print()


"""
================================================================================
PART E: ADVANCED TOPICS
================================================================================
"""

print("=" * 80)
print("ADVANCED TOPIC 1: ELBOW METHOD (Finding Optimal K)")
print("=" * 80)
print()

print("""
Problem:
--------
How do we choose K (number of clusters)?

The Elbow Method:
-----------------
1. Try different values of K (e.g., 1 to 10)
2. Calculate inertia for each K
3. Plot K vs Inertia
4. Look for "elbow" - point where adding more clusters doesn't help much

Why does it work?
-----------------
- More clusters → Lower inertia (better fit)
- But too many clusters → Overfitting!
- Elbow = sweet spot (good fit without overdoing it)
""")

# Test different K values
k_values = range(1, 11)
inertias = []

print("🔍 Testing different K values...")
print()

for k in k_values:
    kmeans_test = KMeansFromScratch(k=k, max_iterations=50, random_state=42)
    kmeans_test.fit(data)
    inertia = kmeans_test._calculate_inertia(data)
    inertias.append(inertia)
    print(f"K={k}: Inertia={inertia:.2f}")

print()

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.title('Elbow Method: Finding Optimal K', fontsize=14)
plt.grid(True, alpha=0.3)

# Highlight the "elbow" (K=3 in our case)
plt.axvline(x=3, color='red', linestyle='--', label='Elbow at K=3')
plt.legend()
plt.show()

print("💡 Interpretation:")
print("   The 'elbow' is around K=3")
print("   After K=3, adding more clusters doesn't reduce inertia much")
print("   This matches our data (we created 3 clusters!)")
print()


print("=" * 80)
print("ADVANCED TOPIC 2: SILHOUETTE SCORE (Cluster Quality)")
print("=" * 80)
print()

print("""
Problem:
--------
Inertia alone doesn't tell us if clusters are well-separated.

Silhouette Score:
-----------------
Measures how similar a point is to its own cluster vs other clusters.

Formula: s = (b - a) / max(a, b)
where:
- a = avg distance to points in same cluster
- b = avg distance to points in nearest other cluster

Score range: [-1, 1]
- 1: Perfect clustering
- 0: Overlapping clusters
- -1: Wrong cluster assignment
""")

def calculate_silhouette_score(data, labels, centroids):
    """
    Calculate average silhouette score for the clustering.
    
    Higher score = better defined clusters
    """
    n = len(data)
    silhouette_scores = []
    
    for i in range(n):
        point = data[i]
        own_cluster = labels[i]
        
        # Calculate a: avg distance to points in same cluster
        same_cluster_points = [data[j] for j in range(n) if labels[j] == own_cluster and j != i]
        
        if len(same_cluster_points) == 0:
            a = 0
        else:
            a = sum(euclidean_distance(point, p) for p in same_cluster_points) / len(same_cluster_points)
        
        # Calculate b: avg distance to nearest other cluster
        b = float('inf')
        
        for cluster_id in range(len(centroids)):
            if cluster_id == own_cluster:
                continue
            
            other_cluster_points = [data[j] for j in range(n) if labels[j] == cluster_id]
            
            if len(other_cluster_points) > 0:
                avg_dist = sum(euclidean_distance(point, p) for p in other_cluster_points) / len(other_cluster_points)
                b = min(b, avg_dist)
        
        # Calculate silhouette score for this point
        if max(a, b) == 0:
            s = 0
        else:
            s = (b - a) / max(a, b)
        
        silhouette_scores.append(s)
    
    # Return average
    return sum(silhouette_scores) / len(silhouette_scores)


# Calculate silhouette scores for different K
print("🔍 Calculating silhouette scores...")
print()

silhouette_scores = []

for k in range(2, 7):  # Start from 2 (need at least 2 clusters)
    kmeans_test = KMeansFromScratch(k=k, max_iterations=50, random_state=42)
    kmeans_test.fit(data)
    score = calculate_silhouette_score(data, kmeans_test.labels, kmeans_test.centroids)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score={score:.3f}")

print()

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 7), silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Analysis: Cluster Quality', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.show()

print("💡 Interpretation:")
print("   Higher silhouette score = better clustering")
print("   K with highest score is often the best choice")
print()


"""
================================================================================
SUMMARY AND KEY TAKEAWAYS
================================================================================
"""

print("\n")
print("=" * 80)
print("SUMMARY: What We've Learned")
print("=" * 80)
print()

print("""
✅ IMPLEMENTED K-MEANS FROM SCRATCH:
   1. Initialize K random centroids
   2. Assign each point to nearest centroid
   3. Update centroids as mean of assigned points
   4. Repeat until convergence

✅ UNDERSTOOD WHY IT WORKS:
   - Minimizes inertia (sum of squared distances)
   - Each iteration improves the clustering
   - Converges to local optimum

✅ LEARNED KEY CONCEPTS:
   - Centroid: Center of a cluster (mean position)
   - Inertia: Measure of cluster tightness
   - Convergence: When centroids stop moving
   - Elbow Method: Finding optimal K
   - Silhouette Score: Measuring cluster quality

✅ SAW REAL APPLICATIONS:
   - Customer segmentation (marketing)
   - Image compression (reduce colors)
   - Document clustering (organize text)
   - Anomaly detection (find outliers)

✅ COMPARED IMPLEMENTATIONS:
   - From scratch: Educational, full control
   - Scikit-learn: Production-ready, optimized

🎯 KEY INSIGHTS:
---------------
1. K-Means is simple but powerful
2. Works best with spherical clusters
3. Sensitive to initial centroids (run multiple times!)
4. Need to choose K carefully (Elbow method helps)
5. Not guaranteed to find global optimum

⚠️  LIMITATIONS TO REMEMBER:
---------------------------
- Assumes K is known (we must specify it)
- Assumes spherical clusters (struggles with complex shapes)
- Sensitive to outliers
- May converge to local optimum
- Doesn't work well with clusters of different sizes/densities

🚀 WHEN TO USE K-MEANS:
----------------------
✓ Data has clear, spherical clusters
✓ Number of clusters is roughly known
✓ Need fast, scalable algorithm
✓ Data is numeric (continuous features)

❌ WHEN NOT TO USE:
------------------
✗ Clusters have complex shapes (use DBSCAN instead)
✗ Don't know number of clusters (use hierarchical clustering)
✗ Clusters have very different densities
✗ Need deterministic results (K-Means is random)

📚 NEXT STEPS:
-------------
1. Try K-Means on real datasets
2. Experiment with different K values
3. Compare with other clustering algorithms (DBSCAN, Hierarchical)
4. Learn K-Means++ (smarter initialization)
5. Explore Mini-Batch K-Means (faster for large data)
6. Apply to your own problems!

🎉 CONGRATULATIONS!
------------------
You now understand K-Means deeply:
- WHY it works (minimizing inertia)
- HOW it works (iterative refinement)
- WHEN to use it (and when not to)
- How to IMPLEMENT it from scratch
- How to APPLY it to real problems

You're ready to use clustering in your ML projects! 🚀
""")

print("=" * 80)
print("END OF CHAPTER 4 IMPLEMENTATION")
print("=" * 80)
print()
print("💡 Remember: Understanding the algorithm deeply (from scratch)")
print("   makes you better at using it effectively (with libraries)!")
print()
print("=" * 80)
