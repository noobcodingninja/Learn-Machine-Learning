Linear Algebra for Machine Learning
Chapter 4: Clustering, K-means Algorithm, and Applications
A First-Principles Approach with Detailed Examples

Table of Contents

What Is Clustering?
The Clustering Objective
K-Means Algorithm
Detailed Examples and Walkthroughs
Choosing K and Evaluation
Practical Applications
Advanced Topics
Chapter Summary
Comprehensive Practice Problems


<a name="clustering"></a>
1. What Is Clustering?
The Core Problem: Finding Natural Groups in Data
Imagine you work at Netflix and have 1 million users. You want to send personalized recommendation emails.
Problem: You can't write 1 million different emails! That's impossible.
Insight: Many users probably have similar tastes. What if we could group users with similar preferences together?
Solution:

Find groups of similar users (clusters)
Write ONE email per group
Send each group their customized email

This is clustering!
What IS Clustering?
Clustering is the task of grouping similar data points together.
Given: n data points x₁, x₂, ..., xₙ
Goal: Partition them into k groups (clusters) such that:

Points in the SAME cluster are similar to each other
Points in DIFFERENT clusters are dissimilar from each other

Key insight: We define "similar" using distance!

Small distance = similar
Large distance = dissimilar

Why Is This Hard?
Question: Can't we just look at the data and see the groups?
Answer: Sometimes yes (in 2D/3D), but:

High dimensions: Real data has hundreds or thousands of dimensions

Can't visualize 1000-dimensional space!


Scale: Millions of data points

Can't manually inspect each one


Ambiguity: Groups aren't always clear-cut

Where does one group end and another begin?


Computation: Astronomical number of possible groupings

For 100 points into 3 clusters: ~10⁴⁷ possible ways!
Can't try them all!



We need an algorithm!
Types of Clustering
1. Partitional Clustering (K-Means)

Each point belongs to exactly ONE cluster
Creates non-overlapping partitions
Example: K-means, K-medoids

2. Hierarchical Clustering

Creates a tree of clusters
Can cut tree at any level for different number of clusters
Example: Agglomerative, Divisive

3. Density-Based Clustering

Finds regions of high density
Can find clusters of arbitrary shape
Example: DBSCAN, OPTICS

4. Model-Based Clustering

Assumes data comes from mixture of distributions
Example: Gaussian Mixture Models

This chapter focuses on K-Means (most popular and foundational)
Real-World Clustering Examples
Example 1.1: Customer Segmentation
E-commerce company has millions of customers.
Features: (purchases/month, avg_spend, website_visits, email_opens)
Cluster them into groups:

Cluster 1: High-value customers (5+ purchases, $200+ avg, frequent visits)
Cluster 2: Occasional shoppers (1-2 purchases, $50 avg, rare visits)
Cluster 3: Window shoppers (0 purchases, many visits, opens emails)
Cluster 4: Inactive (no purchases, no visits)

Use cases:

Different marketing strategies per cluster
Personalized promotions
Identify at-risk customers (Cluster 4)

Example 1.2: Image Compression
Photo has 1 million pixels, each with RGB color (256³ = 16M possible colors).
Goal: Reduce to 16 colors only (massive compression!)
Clustering approach:

Treat each pixel as a point in 3D color space (R, G, B)
Cluster into 16 groups
Replace each pixel with its cluster center color

Result: Image uses only 16 colors, much smaller file size!
Example 1.3: Document Organization
News website has 10,000 articles.
Features: Word frequency vectors (TF-IDF)
Cluster into topics:

Cluster 1: Politics (words: election, president, congress, ...)
Cluster 2: Technology (words: AI, software, startup, ...)
Cluster 3: Sports (words: game, team, championship, ...)
Cluster 4: Entertainment (words: movie, actor, music, ...)

Use cases:

Organize articles automatically
Recommend similar articles
Trending topic detection

Example 1.4: Anomaly Detection
Network traffic monitoring.
Features: (packet_size, frequency, duration, destination)
Clustering approach:

Cluster normal traffic patterns
New traffic that doesn't fit any cluster → ANOMALY!

Use cases:

Detect cyber attacks
Identify compromised machines
Flag suspicious behavior

Example 1.5: Gene Expression Analysis
Biologists measure gene activity across different conditions.
Features: Expression levels of 20,000 genes
Cluster genes by expression pattern:

Genes in same cluster may have related functions
Discover biological pathways
Identify disease markers

Visualizing Clustering: A Simple 2D Example
Imagine 9 points in 2D:
Points:
A: (1, 1)
B: (1, 2)
C: (2, 1)
D: (8, 8)
E: (8, 9)
F: (9, 8)
G: (5, 5)
H: (15, 2)
I: (16, 2)
Just by looking:

Group 1: {A, B, C} (bottom-left cluster)
Group 2: {D, E, F} (top-right cluster)
Group 3: {H, I} (bottom-right cluster)
Point G: Somewhat in-between (could go either way?)

This is easy in 2D because we can SEE the groups!
But in 100D? We need math!
What Makes a Good Clustering?
Property 1: Compactness
Points within a cluster should be CLOSE to each other.
Measure: Within-cluster sum of squared distances (small is good)
Property 2: Separation
Different clusters should be FAR from each other.
Measure: Between-cluster distances (large is good)
Property 3: Balance
Clusters shouldn't be too uneven in size.
Why: One cluster with 99% of points isn't useful!
Property 4: Interpretability
Clusters should make sense for your application.
Example: Customer segments should have actionable differences
Key Insight: Clustering Uses Everything We've Learned!
To cluster, we need:

✅ Distance (Chapter 3) - measure similarity
✅ Norm (Chapter 3) - measure cluster spread
✅ Linear combinations (Chapter 1) - compute cluster centers (means!)
✅ Vector operations (Chapter 1) - compare and update

Clustering brings together all the linear algebra concepts!
Practice Problems - Clustering Concepts
Problem 1.1: Identifying Clusters
Given 2D points: (1,1), (1,2), (2,1), (10,10), (10,11), (11,10)
a) How many natural clusters do you see?
b) Which points belong to each cluster?
c) What distance threshold separates the clusters?
d) Would this be obvious in 1000D?
Problem 1.2: Choosing Features
You're clustering customers for a retail store.
Available data:

Purchase amount ($)
Number of items bought
Time spent in store (minutes)
Age
Gender
Zip code
Date of visit

a) Which features would you use? Why?
b) Which would you exclude? Why?
c) Would you standardize features? Why?
Problem 1.3: Application Scenarios
For each scenario, explain how clustering could help:
a) Spotify wants to create "Daily Mix" playlists
b) Hospital wants to identify patient groups for treatment protocols
c) City wants to optimize bus routes
d) Social media wants to detect bot accounts
Problem 1.4: Good vs Bad Clustering
Dataset: Customer (purchases, spending)
Clustering A:

Cluster 1: 98% of customers
Cluster 2: 1.5% of customers
Cluster 3: 0.5% of customers

Clustering B:

Cluster 1: 40% of customers (low spend, low frequency)
Cluster 2: 35% of customers (medium spend, medium frequency)
Cluster 3: 25% of customers (high spend, high frequency)

a) Which clustering is better? Why?
b) What makes a clustering "useful"?
c) Could Clustering A ever be useful?
Problem 1.5: Hierarchical vs Flat
You have 1000 documents to organize.
Approach A (Flat): Cluster into exactly 10 categories
Approach B (Hierarchical): Create tree structure with subcategories
a) Pros/cons of each approach?
b) Which is easier to navigate?
c) Which is more flexible?
d) Which would you choose and why?

<a name="objective"></a>
2. The Clustering Objective
The Core Question: How Do We Measure "Good" Clustering?
We need a mathematical way to say: "This clustering is better than that one."
Why?

To compare different clusterings
To know when algorithm is improving
To know when to stop iterating

Building the Objective Function
The Intuition
Good clustering means:

Points in a cluster are CLOSE to the cluster center
The cluster is "compact" or "tight"

Bad clustering means:

Points are FAR from their cluster center
The cluster is "spread out" or "loose"

So we want to minimize the spread!
Step 1: Define Cluster Centers
For each cluster j, the centroid (center) is the mean of all points in that cluster.
μⱼ = (1/nⱼ) Σ {xᵢ : xᵢ in cluster j}
This is a linear combination with equal weights!
Example:
Cluster contains points: (1, 2), (3, 4), (5, 6)
μ = [(1,2) + (3,4) + (5,6)] / 3 = (9, 12) / 3 = (3, 4)
The centroid is the "average" point!
Step 2: Measure Distance to Center
For each point xᵢ in cluster j, measure:
distance = ||xᵢ - μⱼ||
Small distance → point is close to center → good!
Large distance → point is far from center → bad!
Step 3: Square the Distances
Why square? Same reasons as always:

Makes all values positive
Penalizes large distances more
Mathematically convenient (differentiable)

For point xᵢ in cluster j:
squared_distance = ||xᵢ - μⱼ||²
Step 4: Sum Over All Points
Within-Cluster Sum of Squares (WCSS) for cluster j:
WCSSⱼ = Σ {||xᵢ - μⱼ||² : xᵢ in cluster j}
This measures total spread within cluster j!
Step 5: Sum Over All Clusters
Total Within-Cluster Sum of Squares:
WCSS = Σⱼ₌₁ᵏ Σ {||xᵢ - μⱼ||² : xᵢ in cluster j}
This is our objective function!
Also called:

Inertia
Within-cluster variance
Distortion
Sum of Squared Errors (SSE)

The Formal Objective
Given:

n data points: x₁, ..., xₙ
k clusters with centroids: μ₁, ..., μₖ
Assignment: each xᵢ assigned to cluster cᵢ

Objective:
J = Σᵢ₌₁ⁿ ||xᵢ - μ_{cᵢ}||²
Goal: Minimize J
Read as: "Sum of squared distances from each point to its assigned cluster center"
Why This Objective Makes Sense
Minimizing WCSS means:

Compact clusters: Points close to their centers
Well-separated: Different clusters have different centers
Balanced: Large spread increases objective (bad)

Properties:

Always ≥ 0 (squared distances are non-negative)
= 0 only if: Each cluster has exactly one point (trivial, not useful!)
Decreases as: Clusters become tighter
Increases as: Clusters become looser

Detailed Example: Computing the Objective
Small Dataset
6 points in 2D:
A: (2, 10)
B: (2, 5)
C: (8, 4)
D: (5, 8)
E: (7, 5)
F: (6, 4)
Clustering 1: k=2
Assignment:

Cluster 1: {A, B, D}
Cluster 2: {C, E, F}

Step 1: Compute centroids
μ₁ = [(2,10) + (2,5) + (5,8)] / 3 = (9, 23) / 3 = (3, 7.67)
μ₂ = [(8,4) + (7,5) + (6,4)] / 3 = (21, 13) / 3 = (7, 4.33)
Step 2: Compute squared distances
Cluster 1:

A to μ₁: ||(2,10) - (3,7.67)||² = ||(-1, 2.33)||² = 1 + 5.43 = 6.43
B to μ₁: ||(2,5) - (3,7.67)||² = ||(-1, -2.67)||² = 1 + 7.13 = 8.13
D to μ₁: ||(5,8) - (3,7.67)||² = ||(2, 0.33)||² = 4 + 0.11 = 4.11

WCSS₁ = 6.43 + 8.13 + 4.11 = 18.67
Cluster 2:

C to μ₂: ||(8,4) - (7,4.33)||² = ||(1, -0.33)||² = 1 + 0.11 = 1.11
E to μ₂: ||(7,5) - (7,4.33)||² = ||(0, 0.67)||² = 0 + 0.45 = 0.45
F to μ₂: ||(6,4) - (7,4.33)||² = ||(-1, -0.33)||² = 1 + 0.11 = 1.11

WCSS₂ = 1.11 + 0.45 + 1.11 = 2.67
Total objective:
J = WCSS₁ + WCSS₂ = 18.67 + 2.67 = 21.34
Clustering 2: Different Assignment
Assignment:

Cluster 1: {A, B}
Cluster 2: {C, D, E, F}

New centroids:
μ₁ = [(2,10) + (2,5)] / 2 = (2, 7.5)
μ₂ = [(8,4) + (5,8) + (7,5) + (6,4)] / 4 = (6.5, 5.25)
Compute squared distances:
Cluster 1:

A to μ₁: ||(0, 2.5)||² = 6.25
B to μ₁: ||(0, -2.5)||² = 6.25

WCSS₁ = 12.5
Cluster 2:

C to μ₂: ||(1.5, -1.25)||² = 2.25 + 1.56 = 3.81
D to μ₂: ||(-1.5, 2.75)||² = 2.25 + 7.56 = 9.81
E to μ₂: ||(0.5, -0.25)||² = 0.25 + 0.06 = 0.31
F to μ₂: ||(-0.5, -1.25)||² = 0.25 + 1.56 = 1.81

WCSS₂ = 15.74
Total objective:
J = 12.5 + 15.74 = 28.24
Comparison:

Clustering 1: J = 21.34 ✓ Better!
Clustering 2: J = 28.24

Clustering 1 is better (lower objective)!
The Optimization Problem
Goal: Find the clustering that minimizes J
Formally:
minimize J = Σᵢ₌₁ⁿ ||xᵢ - μ_{cᵢ}||²
with respect to:

Cluster assignments {c₁, ..., cₙ}
Cluster centroids {μ₁, ..., μₖ}

Problem: This is NP-hard!

Can't try all possible clusterings (too many!)
Need an approximate algorithm

Solution: K-means algorithm (next section)!
What Happens with Different k?
Extreme case 1: k = 1 (one cluster)

All points in one cluster
Center = mean of all data
J = total variance of dataset (maximum!)

Extreme case 2: k = n (each point its own cluster)

Each point is its own center
Distance = 0 for all points
J = 0 (minimum!)

Trade-off:

Small k → Large J (loose clusters)
Large k → Small J (tight clusters, but not useful!)

Goal: Find the "right" k that gives useful, meaningful clusters
Practice Problems - Clustering Objective
Problem 2.1: Manual Calculation
Points: (1, 1), (2, 1), (1, 2), (10, 10), (11, 10), (10, 11)
Clustering:

Cluster 1: {(1,1), (2,1), (1,2)}
Cluster 2: {(10,10), (11,10), (10,11)}

a) Calculate centroid for each cluster
b) Calculate squared distance from each point to its centroid
c) Calculate WCSS for each cluster
d) Calculate total objective J
Problem 2.2: Comparing Clusterings
Same 6 points as above.
Clustering A: As given in Problem 2.1
Clustering B:

Cluster 1: {(1,1), (2,1)}
Cluster 2: {(1,2), (10,10), (11,10), (10,11)}

a) Calculate J for Clustering B
b) Which clustering is better?
c) Does this match your intuition?
Problem 2.3: Effect of k
Dataset: 100 points, well-separated into 4 natural clusters.
a) Estimate J for k=1 (one cluster)
b) Estimate J for k=4 (natural clusters)
c) Estimate J for k=100 (each point alone)
d) Sketch how J changes as k increases from 1 to 100
Problem 2.4: Centroid Properties
Prove that the centroid minimizes sum of squared distances.
That is, show: μ = argmin Σᵢ ||xᵢ - c||²
where the minimum is over all possible centers c.
Hint: Take derivative with respect to c and set to zero.
Problem 2.5: Understanding WCSS
Two clusters with same number of points:
Cluster A: Points tightly packed, WCSS = 10
Cluster B: Points spread out, WCSS = 100
a) Which cluster is "better" (more compact)?
b) If you had to split one cluster, which would you choose?
c) How does WCSS relate to cluster "quality"?

<a name="kmeans"></a>
3. The K-Means Algorithm
The Problem: We Can't Try Every Possible Clustering
Let's think about this carefully. We have n data points and we want to divide them into k clusters.
Question: How many ways can we do this?
For 100 points and 3 clusters, the number is approximately 10⁴⁷. That's more than the number of atoms in a human body!
So what's the problem?
Even with the world's fastest computer, we can't try all possible clusterings to find the one with minimum WCSS. It would take longer than the age of the universe!
What's the root cause?
The clustering problem is what computer scientists call "NP-hard" - meaning there's no known efficient algorithm that's guaranteed to find the absolute best solution for all possible inputs.
So how can we solve this problem?
We can't find the perfect solution efficiently, but we can find a pretty good solution! Instead of trying every possibility, we can use an iterative approach that keeps improving our clustering until we can't make it any better.
This is where K-means comes in!
The Big Idea: Iterative Improvement
Think of it like organizing a messy room:

You start by making some rough piles
You look at each item and move it to the pile it fits best
You reorganize the piles based on what's now in them
Repeat steps 2-3 until nothing needs to move anymore

K-means does exactly this with data points!
K-means strategy:

Start with k random cluster centers
Assign each point to nearest center
Recalculate centers (means) based on assignments
Repeat steps 2-3 until convergence

Key insight: Each step is guaranteed to decrease (or maintain) the objective!
Why Does This Work?
Question: If we can't find the optimal solution, how do we know K-means gives us something good?
Answer: Let's think about what happens in each step:
Assignment step: When we assign each point to its nearest center, we're minimizing the distance for that point. This can only decrease (or maintain) the total WCSS!
Update step: When we recalculate centers as the mean of assigned points, we're finding the optimal center for those points. This also can only decrease (or maintain) WCSS!
So each iteration makes things better (or keeps them the same)!
Question: Will it keep improving forever?
Answer: No! Since WCSS is always positive and decreasing, it must eventually stop decreasing. When no points want to change clusters and centers stop moving, we've reached a local minimum.
Important: It's a local minimum, not necessarily the global minimum. That's okay - it's still useful!
The Algorithm in Detail
Input:

Data points: x₁, ..., xₙ ∈ ℝᵈ
Number of clusters: k

Output:

Cluster assignments: c₁, ..., cₙ
Cluster centroids: μ₁, ..., μₖ

Step 0: Initialization
Choose k initial centroids μ₁, ..., μₖ
Question: How do we choose initial centers?
The Problem Without Good Initialization:
Imagine you're organizing books into fiction, non-fiction, and textbooks. If you start by randomly picking three cookbooks as your "centers," all books will initially be assigned based on how similar they are to cookbooks! This could lead to a weird clustering where the algorithm gets stuck.
What's the root cause?
Bad initial centers can cause the algorithm to converge to a poor local minimum. Since K-means is greedy (makes locally optimal choices), starting in the wrong place means we might never escape to a better solution.
So how can we solve this?
Common methods:
1. Random points (Forgy method):

Pick k random data points as initial centers
Simple and fast
But can lead to bad local minima

2. Random partition:

Randomly assign points to k clusters
Compute centers of these random clusters
Usually better than random points

3. K-means++ (Smart initialization):

First center: pick a random point
Next centers: pick points far from existing centers (with probability proportional to distance²)
Provably better than random!
This is the standard choice today

Example of K-means++ logic:
"I've picked my first center. Where should the second one be? If I pick a point close to the first center, I'm wasting an opportunity - those points would be assigned to the first center anyway! I should pick a point that's FAR away, so I can capture a different region of the data."
Step 1: Assignment Step
For each point xᵢ: Assign to nearest centroid
cᵢ = argmin_{j∈{1,...,k}} ||xᵢ - μⱼ||²
In words:

Calculate distance from xᵢ to each center
Assign to cluster with smallest distance

Why squared distance?
You might wonder: why ||xᵢ - μⱼ||² instead of ||xᵢ - μⱼ||?
Answer: Both give the same answer! If d₁² < d₂², then d₁ < d₂. Squaring preserves the ordering. But squared distances are easier to compute (no square root needed) and match our WCSS objective.
Example:
Point x = (5, 5)
Centers: μ₁ = (2, 2), μ₂ = (8, 8)
d₁² = ||(5,5) - (2,2)||² = ||(3,3)||² = 9 + 9 = 18
d₂² = ||(5,5) - (8,8)||² = ||(-3,-3)||² = 9 + 9 = 18
Tied! Assign to either (typically first one).
What happens geometrically?
Each center defines a region - all points closer to that center than any other. These regions are called Voronoi cells. The assignment step divides space into these cells!
Step 2: Update Step
For each cluster j: Recompute centroid as mean of assigned points
μⱼ = (1/nⱼ) Σ {xᵢ : cᵢ = j}
where nⱼ = number of points assigned to cluster j
In words:

Find all points assigned to cluster j
Average them (component-wise)
This is the new center!

Why the mean? Why not median or mode?
Great question! Let's think about what we're trying to minimize: WCSS = Σᵢ ||xᵢ - μⱼ||²
Question: For a fixed set of points, what center μⱼ minimizes their sum of squared distances?
Answer: The mean! This is a fundamental property. Let's see why:
Take the derivative of Σᵢ ||xᵢ - μⱼ||² with respect to μⱼ and set to zero:
d/dμⱼ Σᵢ ||xᵢ - μⱼ||² = -2Σᵢ (xᵢ - μⱼ) = 0
This gives: Σᵢ xᵢ = nⱼμⱼ
Therefore: μⱼ = (1/nⱼ)Σᵢ xᵢ (the mean!)
So using the mean is not arbitrary - it's the optimal center for minimizing squared distances!
Example:
Cluster 1 has points: (1, 2), (3, 4), (5, 6)
μ₁ = [(1,2) + (3,4) + (5,6)] / 3 = (9, 12) / 3 = (3, 4)
This is a linear combination with equal weights (1/3 each) - connecting back to Chapter 1!
Step 3: Check Convergence
Stop if:

Centroids don't change: μⱼ_new = μⱼ_old for all j
OR assignments don't change: c_new = c_old for all points
OR maximum iterations reached (safety check)

Why might we need a maximum iteration limit?
In theory, K-means always converges. In practice, due to numerical precision, centroids might keep making tiny adjustments forever. So we set a maximum (like 300 iterations) or check if changes are below a threshold (like 10⁻⁶).
Question: What if it hasn't converged yet?
Answer: Go back to Step 1 with the new centroids!
Otherwise: Repeat
If not converged, return to Step 1 with updated centroids.
Each iteration has two substeps:

E-step (Expectation): Assign points to clusters
M-step (Maximization): Update cluster centers

This is a special case of the EM algorithm (Expectation-Maximization)!
Why K-Means is Guaranteed to Converge
Let's understand this deeply:
Claim: K-means always converges (reaches a point where nothing changes).
Proof idea:

WCSS always decreases (or stays same): Each step either improves WCSS or leaves it unchanged
WCSS is bounded below: Since all distances are positive, WCSS ≥ 0
Finite number of possible assignments: With n points and k clusters, there are only finitely many ways to assign points (specifically, k^n ways, but k is fixed)
Can't repeat a configuration: If we return to a previous clustering, WCSS would be the same as before. But WCSS is always decreasing! So we can't cycle back.
Must eventually stop: Finite possibilities + no cycles + always decreasing → must reach a point where nothing changes!

Therefore, K-means always converges to a local minimum!
What Could Go Wrong?
Question: If K-means always converges, what's the problem?
Problem 1: Local Minima
K-means finds a local minimum, not necessarily the global minimum.
Analogy: Imagine you're blindfolded on a mountain range, trying to find the lowest point. You keep walking downhill until you can't go any lower. But you might be in a small valley, not the deepest valley!
Solution: Run K-means multiple times with different initializations and pick the best result!
Problem 2: Empty Clusters
During iteration, a cluster might have no points assigned to it.
What went wrong?
The centroid was so far from all points that no point chose it as nearest center.
Solutions:

Option 1: Remove the empty cluster (now you have k-1 clusters)
Option 2: Reinitialize that centroid to a random point
Option 3: Assign it to the point farthest from its current centroid

Most implementations use Option 2 or 3.
Problem 3: Sensitivity to Initialization
Different starting points lead to different final clusterings.
Example:
Run 1: Starts with centers near true clusters → Good result!
Run 2: Starts with all centers in one region → Bad result!
Solution:

Use K-means++ initialization
Run multiple times and
keep best result (based on lowest WCSS)

Typical: run 10-20 times, keep best

Problem 4: Choosing k
How many clusters should we use?
This is actually a separate, hard problem! We'll cover it in Section 5.
Detailed Walkthrough: K-Means Step-by-Step
Let's see K-means in action with a small example!
Dataset: 8 Points in 2D
Points:
P1: (2, 10)
P2: (2, 5)
P3: (8, 4)
P4: (5, 8)
P5: (7, 5)
P6: (6, 4)
P7: (1, 2)
P8: (4, 9)
Let's say k=2 (we want 2 clusters).
Iteration 0: Random Initialization
Random initial centers:

μ₁ = (2, 10) [picked P1]
μ₂ = (8, 4) [picked P3]

Iteration 1: First Assignment
For each point, calculate distances to both centers:
P1 (2, 10):

d₁² = ||(2,10) - (2,10)||² = 0
d₂² = ||(2,10) - (8,4)||² = 36 + 36 = 72
Assign to cluster 1 (d₁² < d₂²)

P2 (2, 5):

d₁² = ||(2,5) - (2,10)||² = 0 + 25 = 25
d₂² = ||(2,5) - (8,4)||² = 36 + 1 = 37
Assign to cluster 1

P3 (8, 4):

d₁² = ||(8,4) - (2,10)||² = 36 + 36 = 72
d₂² = ||(8,4) - (8,4)||² = 0
Assign to cluster 2

P4 (5, 8):

d₁² = ||(5,8) - (2,10)||² = 9 + 4 = 13
d₂² = ||(5,8) - (8,4)||² = 9 + 16 = 25
Assign to cluster 1

P5 (7, 5):

d₁² = ||(7,5) - (2,10)||² = 25 + 25 = 50
d₂² = ||(7,5) - (8,4)||² = 1 + 1 = 2
Assign to cluster 2

P6 (6, 4):

d₁² = ||(6,4) - (2,10)||² = 16 + 36 = 52
d₂² = ||(6,4) - (8,4)||² = 4 + 0 = 4
Assign to cluster 2

P7 (1, 2):

d₁² = ||(1,2) - (2,10)||² = 1 + 64 = 65
d₂² = ||(1,2) - (8,4)||² = 49 + 4 = 53
Assign to cluster 2 (slightly closer!)

P8 (4, 9):

d₁² = ||(4,9) - (2,10)||² = 4 + 1 = 5
d₂² = ||(4,9) - (8,4)||² = 16 + 25 = 41
Assign to cluster 1

Assignment after iteration 1:

Cluster 1: {P1, P2, P4, P8}
Cluster 2: {P3, P5, P6, P7}

Iteration 1: Update Centers
Cluster 1: {(2,10), (2,5), (5,8), (4,9)}
μ₁ = [(2,10) + (2,5) + (5,8) + (4,9)] / 4
= (13, 32) / 4
= (3.25, 8)
Cluster 2: {(8,4), (7,5), (6,4), (1,2)}
μ₂ = [(8,4) + (7,5) + (6,4) + (1,2)] / 4
= (22, 15) / 4
= (5.5, 3.75)
New centers:

μ₁ = (3.25, 8)
μ₂ = (5.5, 3.75)

Centers changed! Continue to next iteration.
Iteration 2: Second Assignment
Now using new centers: μ₁ = (3.25, 8), μ₂ = (5.5, 3.75)
P1 (2, 10):

d₁² = ||(2,10) - (3.25,8)||² = 1.5625 + 4 = 5.5625
d₂² = ||(2,10) - (5.5,3.75)||² = 12.25 + 39.0625 = 51.3125
Cluster 1

P2 (2, 5):

d₁² = ||(2,5) - (3.25,8)||² = 1.5625 + 9 = 10.5625
d₂² = ||(2,5) - (5.5,3.75)||² = 12.25 + 1.5625 = 13.8125
Cluster 1 (barely!)

P3 (8, 4):

d₁² = ||(8,4) - (3.25,8)||² = 22.5625 + 16 = 38.5625
d₂² = ||(8,4) - (5.5,3.75)||² = 6.25 + 0.0625 = 6.3125
Cluster 2

P4 (5, 8):

d₁² = ||(5,8) - (3.25,8)||² = 3.0625 + 0 = 3.0625
d₂² = ||(5,8) - (5.5,3.75)||² = 0.25 + 18.0625 = 18.3125
Cluster 1

P5 (7, 5):

d₁² = ||(7,5) - (3.25,8)||² = 14.0625 + 9 = 23.0625
d₂² = ||(7,5) - (5.5,3.75)||² = 2.25 + 1.5625 = 3.8125
Cluster 2

P6 (6, 4):

d₁² = ||(6,4) - (3.25,8)||² = 7.5625 + 16 = 23.5625
d₂² = ||(6,4) - (5.5,3.75)||² = 0.25 + 0.0625 = 0.3125
Cluster 2

P7 (1, 2):

d₁² = ||(1,2) - (3.25,8)||² = 5.0625 + 36 = 41.0625
d₂² = ||(1,2) - (5.5,3.75)||² = 20.25 + 3.0625 = 23.3125
Cluster 2

P8 (4, 9):

d₁² = ||(4,9) - (3.25,8)||² = 0.5625 + 1 = 1.5625
d₂² = ||(4,9) - (5.5,3.75)||² = 2.25 + 27.5625 = 29.8125
Cluster 1

Assignment after iteration 2:

Cluster 1: {P1, P2, P4, P8} (same as before!)
Cluster 2: {P3, P5, P6, P7} (same as before!)

Assignments didn't change! Converged!
Let's verify by updating centers anyway:
Iteration 2: Update Centers (Verification)
Cluster 1: {(2,10), (2,5), (5,8), (4,9)}
μ₁ = (3.25, 8) (same as before!)
Cluster 2: {(8,4), (7,5), (6,4), (1,2)}
μ₂ = (5.5, 3.75) (same as before!)
Centers unchanged! Definitely converged!
Final Result
Final clustering:

Cluster 1: {P1, P2, P4, P8} - Upper region
Cluster 2: {P3, P5, P6, P7} - Lower region

Final centers:

μ₁ = (3.25, 8)
μ₂ = (5.5, 3.75)

Final WCSS:
WCSS₁ = 5.5625 + 10.5625 + 3.0625 + 1.5625 = 20.75
WCSS₂ = 6.3125 + 3.8125 + 0.3125 + 23.3125 = 33.75
Total WCSS = 54.5
Algorithm converged in just 2 iterations!
Understanding Convergence Through WCSS
Let's track how WCSS changed:
After Iteration 0 (initial):
We'd need to calculate, but it would be large since initial centers were just two data points.
After Iteration 1:
WCSS improved significantly when centers moved to better positions.
After Iteration 2:
WCSS = 54.5, and assignments didn't change, so we stopped.
Key observation: WCSS decreased (or stayed same) at every step!
Computational Complexity
Question: How expensive is K-means?
Per iteration:

Assignment step: For each of n points, compute k distances

Cost: O(nkd) where d is number of dimensions


Update step: For each of k clusters, sum and average points

Cost: O(nd)



Total per iteration: O(nkd)
Number of iterations: Typically converges in 10-50 iterations
Overall: O(iterations × n × k × d)
This is very efficient! Linear in n, k, and d!
Comparison: Trying all clusterings would be O(k^n) - exponential! K-means is much faster!
Practical Implementation Tips
Tip 1: Normalize Features
Problem: Features with different scales dominate distance calculations.
Example:

Feature 1: Age (20-80, range ~60)
Feature 2: Income ($20k-$200k, range ~180k)

Income will dominate distance because its values are much larger!
Solution: Standardize features:
x'ᵢ = (xᵢ - mean) / std_dev
Now all features have mean 0 and standard deviation 1.
Tip 2: Use K-means++ Initialization
Always use K-means++ instead of random initialization:

Provably better worst-case performance
Typically faster convergence
Standard in most libraries (sklearn, etc.)

Tip 3: Run Multiple Times
Run K-means 10-20 times with different initializations:

Keep the result with lowest WCSS
This helps avoid bad local minima
Most libraries do this automatically

Tip 4: Check for Empty Clusters
Monitor for empty clusters during iteration:

Implement a strategy to handle them
Reinitialize empty centroids
Or reduce k by one

Tip 5: Set Maximum Iterations
Always set a maximum iteration limit:

Prevents infinite loops from numerical issues
Typical values: 300-1000 iterations
Can also set convergence threshold (like 10⁻⁴)

K-Means Variants
Mini-Batch K-Means
Problem: Standard K-means is slow for huge datasets (millions of points).
Solution: Use mini-batches!
How it works:

Each iteration, sample a small random subset (mini-batch)
Update centers based only on this subset
Converges faster (though to slightly different solution)

Trade-off: Speed vs. accuracy

10-100x faster on large datasets
Slightly worse WCSS (typically within 5%)
Good for exploratory analysis

K-Medoids (PAM)
Problem: K-means uses centroids that might not be actual data points.
Solution: Use medoids (actual data points) as centers!
How it works:

Centers must be actual data points
Update step: find data point that minimizes within-cluster distance
More robust to outliers!

Trade-off:

More robust
But slower (O(n²) vs O(n))
Used when centers need to be interpretable real examples

Fuzzy K-Means
Problem: Hard assignment (each point in exactly one cluster) is too rigid.
Solution: Soft assignment (each point has membership degree to each cluster)!
How it works:

Each point has probability of belonging to each cluster
Probabilities sum to 1
Centers weighted by membership probabilities

Use case: When cluster boundaries are unclear
When K-Means Works Well
Good scenarios:

Spherical clusters: Clusters are roughly round/spherical
Similar sizes: Clusters have roughly equal numbers of points
Well-separated: Clear gaps between clusters
Known k: You know how many clusters to expect
Continuous features: Works with Euclidean distance

Example: Customer segmentation where groups are naturally balanced
When K-Means Fails
Failure Case 1: Non-Spherical Clusters
Example: Two concentric circles
Inner circle: radius 1
Outer ring: radius 3
K-means with k=2 will NOT separate them correctly!
Why? K-means creates spherical decision boundaries. It can't handle this shape!
Solution: Use density-based clustering (DBSCAN) or spectral clustering
Failure Case 2: Different Sized Clusters
Example:

Cluster 1: 1000 points, compact
Cluster 2: 100 points, spread out

K-means might split cluster 1 to reduce its large WCSS, even though cluster 2 is worse!
Why? K-means minimizes total WCSS. Large clusters contribute more to WCSS even if tight!
Solution: Use mixture models or hierarchical clustering
Failure Case 3: Different Densities
Example:

Cluster 1: Very dense (points close together)
Cluster 2: Sparse (points spread out)

K-means treats both equally, might split dense cluster.
Solution: DBSCAN (finds arbitrary density-based clusters)
Failure Case 4: Outliers
Example:

Most points in nice clusters
A few points far away from everything

Outliers pull centroids toward them, distorting clusters!
Why? Squared distance heavily penalizes outliers.
Solution:

Remove outliers first
Use robust variants (K-medoids)
Use density-based methods

Practice Problems - K-Means Algorithm
Problem 3.1: Hand Execution
Points: A(1,1), B(1,2), C(2,1), D(8,9), E(9,8), F(9,9)
k=2, initial centers: μ₁ = (1,1), μ₂ = (9,9)
a) Iteration 1: Assign each point to nearest center
b) Iteration 1: Update centers
c) Iteration 2: Reassign points
d) Iteration 2: Update centers
e) Did it converge? If not, continue one more iteration
Problem 3.2: Calculating WCSS
From Problem 3.1, after convergence:
a) Calculate WCSS for cluster 1
b) Calculate WCSS for cluster 2
c) Calculate total WCSS
d) Is this the global minimum? How do you know?
Problem 3.3: Initialization Matters
Same points as Problem 3.1.
Now use: μ₁ = (1,1), μ₂ = (2,1)
a) Run K-means with these initial centers
b) Compare final WCSS to Problem 3.1
c) Did different initialization lead to different result?
d) Which is better?
Problem 3.4: Empty Cluster Problem
Points: (0,0), (0,1), (1,0), (1,1), (10,10)
k=3, initial centers: μ₁=(0,0), μ₂=(1,1), μ₃=(20,20)
a) First iteration: Which points get assigned where?
b) What happens to cluster 3?
c) How would you fix this?
Problem 3.5: Understanding Convergence
Consider K-means with n=100, k=5.
a) Maximum possible number of iterations before convergence?
b) Typical actual number of iterations?
c) Why the huge difference?
d) What guarantees convergence?
Problem 3.6: Feature Scaling Impact
Two features:

x₁: Age (20-70, mean=45, std=15)
x₂: Income ($20k-$200k, mean=$80k, std=$40k)

Point A: (25, $30k), Point B: (45, $80k)
a) Calculate Euclidean distance without scaling
b) Standardize both features: (x - mean)/std
c) Calculate distance after standardization
d) Which feature dominates before scaling?
e) Why is standardization important?
Problem 3.7: K-Means Limitations
Sketch or describe what happens when K-means (k=2) is applied to:
a) Two concentric circles
b) Two intertwined spirals
c) One dense cluster + one sparse cluster
d) Why does K-means fail in each case?
e) What alternatives would you suggest?

<a name="examples"></a>
4. Detailed Examples and Walkthroughs
Example 4.1: Customer Segmentation
The Business Problem
An e-commerce company wants to segment customers for targeted marketing.
Available data for each customer:

x₁: Number of purchases in past year
x₂: Average order value ($)
x₃: Days since last purchase

Dataset: 12 customers
CustomerPurchasesAvg OrderDays SinceC112855C215923C310787C4225120C5118150C633090C786515C875820C997012C10115180C11222140C1214884
Goal: Cluster into k=3 segments
Step 1: Feature Standardization
Why standardize?
Look at the ranges:

Purchases: 1-15 (range ~14)
Order value: $15-$92 (range ~$77)
Days since: 3-180 (range ~177)

Days since purchase would dominate distance calculations!
Standardize: z = (x - mean) / std
Calculate means:

Mean purchases: (12+15+10+2+1+3+8+7+9+1+2+14)/12 = 84/12 = 7
Mean order: (85+92+78+25+18+30+65+58+70+15+22+88)/12 = 646/12 ≈ 53.83
Mean days: (5+3+7+120+150+90+15+20+12+180+140+4)/12 = 746/12 ≈ 62.17

Calculate standard deviations:

Std purchases ≈ 4.97
Std order ≈ 30.05
Std days ≈ 64.09

Standardized features (rounded):
Customerz₁z₂z₃C11.011.04-0.89C21.611.27-0.92C30.600.80-0.86C4-1.01-0.960.90C5-1.21-1.191.37C6-0.81-0.790.43C70.200.37-0.74C80.000.14-0.66C90.400.54-0.78C10-1.21-1.291.84C11-1.01-1.061.21C121.411.14-0.91
Step 2: K-Means Initialization (K-means++)
First center (random): Pick C1
μ₁ = (1.01, 1.04, -0.89)
Second center: Pick point far from C1
Calculate distances from C1:
Largest distance: C10 at distance ≈ 4.5
μ₂ = C10 = (-1.21, -1.29, 1.84)
Third center: Pick point far from both C1 and C10
After calculation, C7 is roughly equidistant from both.
μ₃ = C7 = (0.20, 0.37, -0.74)
Step 3: First Iteration - Assignment
Calculate distance from each customer to each center:
C1 to centers:

To μ₁: 0 (it is μ₁!)
To μ₂: √[(1.01-(-1.21))² + (1.04-(-1.29))² + (-0.89-1.84)²] ≈ 4.5
To μ₃: √[(1.01-0.20)² + (1.04-0.37)² + (-0.89-(-0.74))²] ≈ 1.0

Assign C1 → Cluster 1 (μ₁ is closest)
After computing all distances:
Cluster 1: {C1, C2, C3, C12} - High value, active customers
Cluster 2: {C4, C5, C6, C10, C11} - Low value, inactive customers
Cluster 3: {C7, C8, C9} - Medium value, moderately active customers
Step 4: Update Centers
Cluster 1 centroid:
μ₁ = mean of {C1, C2, C3, C12}
= [(1.01, 1.04, -0.89) + (1.61, 1.27, -0.92) + (0.60, 0.80, -0.86) + (1.41, 1.14, -0.91)] / 4
= (1.16, 1.06, -0.90)
Cluster 2 centroid:
μ₂ = mean of {C4, C5, C6, C10, C11}
= [(-1.01, -0.96, 0.90) + (-1.21, -1.19, 1.37) + (-0.81, -0.79, 0.43) + (-1.21, -1.29, 1.84) + (-1.01, -1.06, 1.21)] / 5
= (-1.05, -1.06, 1.15)
Cluster 3 centroid:
μ₃ = mean of {C7, C8, C9}
= [(0.20, 0.37, -0.74) + (0.00, 0.14, -0.66) + (0.40, 0.54, -0.78)] / 3
= (0.20, 0.35, -0.73)
Step 5: Second Iteration - Check Convergence
Reassign points with new centers...
After recalculation, assignments don't change!
Converged after 2 iterations!
Final Result: Customer Segments
Segment 1 (Premium customers):

C1, C2, C3, C12
High purchase frequency (10-15 purchases)
High order values ($78-$92)
Recently active (3-7 days)
Marketing strategy: Loyalty rewards, premium products

Segment 2 (At-risk customers):

C4, C5, C6, C10, C11
Low purchase frequency (1-3 purchases)
Low order values ($15-$30)
Long time since purchase (90-180 days)
Marketing strategy: Win-back campaigns, discounts

Segment 3 (Regular customers):

C7, C8, C9
Medium purchase frequency (7-9 purchases)
Medium order values ($58-$70)
Moderately active (12-20 days)
Marketing strategy: Upselling, product recommendations

Business Impact
Before clustering:

One-size-fits-all marketing
2% email conversion rate
$50k marketing budget mostly wasted

After clustering:

Targeted campaigns per segment
8% conversion rate (4x improvement!)
Segments respond to different messages
Marketing spend optimized

Key insight: Not all customers are the same! Treating them differently (based on data) works better!
Example 4.2: Image Compression with K-Means
The Problem
You have a photo that's 256×256 pixels. Each pixel has an RGB color (3 bytes).
Current size: 256 × 256 × 3 bytes = 196,608 bytes ≈ 192 KB
Goal: Compress to use only 16 colors!
How K-Means Helps
Key insight: Most photos don't use all 16 million possible colors (256³). They use relatively few colors, with small variations!
Strategy:

Treat each pixel as a 3D point (R, G, B)
Cluster pixels into k=16 clusters
Replace each pixel with its cluster center color
Store: cluster centers + assignment for each pixel

Step-by-Step Process
Original Image Data (simplified example):
Let's say we have 9 pixels with these RGB values:
PixelRGBColor DescriptionP125500Bright redP225055Slightly different redP32451010Another redP402550Bright greenP552505Slightly different greenP61024510Another greenP700255Bright blueP855250Slightly different blueP91010245Another blue
Visual insight: We have 9 "different" colors, but really just 3 color families (red, green, blue) with slight variations!
Apply K-Means with k=3
Initial centers (K-means++):

μ₁ = (255, 0, 0) [P1]
μ₂ = (0, 255, 0) [P4]
μ₃ = (0, 0, 255) [P7]

Iteration 1: Assignment
P1 (255,0,0): Distance to μ₁ = 0 → Cluster 1
P2 (250,5,5): Distance to μ₁ = √(25+25+25) ≈ 8.7 → Cluster 1
P3 (245,10,10): Distance to μ₁ = √(100+100+100) ≈ 17.3 → Cluster 1
P4 (0,255,0): Distance to μ₂ = 0 → Cluster 2
P5 (5,250,5): Distance to μ₂ ≈ 8.7 → Cluster 2
P6 (10,245,10): Distance to μ₂ ≈ 17.3 → Cluster 2
P7 (0,0,255): Distance to μ₃ = 0 → Cluster 3
P8 (5,5,250): Distance to μ₃ ≈ 8.7 → Cluster 3
P9 (10,10,245): Distance to μ₃ ≈ 17.3 → Cluster 3
Iteration 1: Update Centers
Cluster 1 (Reds):
μ₁ = [(255,0,0) + (250,5,5) + (245,10,10)] / 3 = (250, 5, 5)
Cluster 2 (Greens):
μ₂ = [(0,255,0) + (5,250,5) + (10,245,10)] / 3 = (5, 250, 5)
Cluster 3 (Blues):
μ₃ = [(0,0,255) + (5,5,250) + (10,10,245)] / 3 = (5, 5, 250)
Iteration 2: Assignments don't change. Converged!
Compressed Image
Original storage:

9 pixels × 3 bytes each = 27 bytes
Each pixel stores full RGB value

Compressed storage:

3 cluster centers × 3 bytes = 9 bytes
9 pixel assignments (2 bits each to encode 0-2) = 2.25 bytes ≈ 3 bytes
Total: 12 bytes

Compression ratio: 27/12 = 2.25x compression!
Reconstruction:

P1 → Cluster 1 → (250, 5, 5)
P2 → Cluster 1 → (250, 5, 5)
P3 → Cluster 1 → (250, 5, 5)
... and so on

Visual result: Three red pixels become one shade of red. Slight variations removed, but overall image looks similar!
Scaling to Real Images
For 256×256 image with k=16 colors:
Original: 256 × 256 × 3 = 196,608 bytes

Compressed:

16 colors × 3 bytes = 48 bytes
65,536 pixels × 4 bits = 32,768 bytes
Total: 32,816 bytes ≈ 32 KB

Compression: 6x smaller!
Trade-off: Slight loss of color detail, but image still recognizable!
Why This Works
Question: Why does image compression with K-means work?
Answer: Natural images have spatial coherence - nearby pixels tend to have similar colors!

Sky: many shades of blue → one cluster
Grass: many shades of green → one cluster
Skin tones: variations → one cluster

We're exploiting redundancy in natural images!
Example 4.3: Document Clustering
The Problem
News website has 1000 articles. Want to organize them automatically into topics.
Simplified example: 8 articles
Articles (simplified text):

"President announces new economic policy for taxes"
"Congress votes on healthcare reform bill today"
"Election results show surprising voter turnout"
"Scientists discover new exoplanet in distant galaxy"
"AI breakthrough enables better medical diagnosis"
"Research reveals climate change affects ocean temperature"
"Football team wins championship in overtime thriller"
"Basketball playoffs begin with upset victory"

Step 1: Feature Representation (TF-IDF)
What's the problem with representing text?
K-means needs numerical vectors, but articles are text!
Solution: Convert text to vectors using word frequencies!
Simple approach: Count how many times each word appears.
Better approach: TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF formula:
TF-IDF(word, doc) = (count in doc) × log(total docs / docs containing word)
Why this formula?

TF (Term Frequency): Words that appear often in a document are important for that document
IDF (Inverse Document Frequency): Words that appear in many documents are less distinctive

Example:

Word "the" appears in all documents → low IDF → low importance
Word "exoplanet" appears in only 1 document → high IDF → high importance!

Step 2: Create Feature Vectors
Let's use a simplified vocabulary of 10 key words:
Words: [president, congress, election, science, AI, climate, football, basketball, team, medical]
Article 1 vector (simplified):
[1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
(only "president" has high TF-IDF)
Article 2 vector:
[0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0.8]
Article 3 vector:
[0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0]
Article 4 vector:
[0, 0, 0, 1.5, 0, 0, 0, 0, 0, 0]
Article 5 vector:
[0, 0, 0, 0, 1.5, 0, 0, 0, 0, 1.2]
Article 6 vector:
[0, 0, 0, 0.5, 0, 1.5, 0, 0, 0, 0]
Article 7 vector:
[0, 0, 0, 0, 0, 0, 1.5, 0, 1.2, 0]
Article 8 vector:
[0, 0, 0, 0, 0, 0, 0, 1.5, 1.2, 0]
Step 3: Apply K-Means with k=3
Initial centers (simplified):

μ₁ = Article 1 (politics cluster seed)
μ₂ = Article 4 (science cluster seed)
μ₃ = Article 7 (sports cluster seed)

Iteration 1: Assignment
Calculate cosine similarity (or Euclidean distance) to each center.
Articles 1, 2, 3: Closest to μ₁ → Cluster 1 (Politics)
Articles 4, 5, 6: Closest to μ₂ → Cluster 2 (Science)
Articles 7, 8: Closest to μ₃ → Cluster 3 (Sports)
Iteration 1: Update
μ₁ = mean of Articles {1, 2, 3}
= average of politics-related vectors
= high values for [president, congress, election]
μ₂ = mean of Articles {4, 5, 6}
= average of science-related vectors
= high values for [science, AI, climate, medical]
μ₃ = mean of Articles {7, 8}
= average of sports-related vectors
= high values for [football, basketball, team]
Iteration 2: Assignments don't change. Converged!
Final Clusters
Cluster 1 - Politics:

Articles 1, 2, 3
Key terms: president, congress, election, healthcare, policy
Can label this topic automatically as "Politics/Government"

Cluster 2 - Science/Technology:

Articles 4, 5, 6
Key terms: science, AI, climate, medical, research
Label: "Science & Technology"

Cluster 3 - Sports:

Articles 7, 8
Key terms: football, basketball, team, championship, playoffs
Label: "Sports"

Practical Applications
1. Automatic organization:

New articles automatically assigned to topics
Website navigation: browse by cluster

2. Recommendation system:

User reads Article 1 (Politics)
Recommend other articles from Cluster 1

3. Trending topics:

Large cluster suddenly forms → trending topic!
Can alert editors

4. Content gaps:

Some clusters very small → undercover topics
Suggests new content to create

Why This Works
Key insight: Articles about similar topics use similar words!
The vector representation captures semantic similarity:

Politics articles cluster because they share political vocabulary
Science articles cluster because they share scientific terms
Even without understanding meaning, word patterns reveal topics!

This is the foundation of:

Google News clustering
Topic modeling
Document search
Content recommendation

Example 4.4: Anomaly Detection in Network Traffic
The Problem
Company monitors network traffic to detect cyber attacks.
Normal traffic patterns:

Employees browsing websites
Email traffic
Cloud backups
API calls

Anomalous traffic:

Port scanning (hacker probing for vulnerabilities)
DDoS attacks (overwhelming traffic)
Data exfiltration (stealing data)
Malware communication

Challenge: Can't manually inspect millions of network packets per second!
The K-Means Approach
Key insight: Normal traffic follows predictable patterns. Anomalies don't fit any pattern!
Strategy:

Collect features for each network connection
Cluster normal traffic patterns
Any connection far from all clusters → ANOMALY!

Step 1: Feature Extraction
For each network connection, extract features:

x₁: Packet size (bytes)
x₂: Connection duration (seconds)
x₃: Number of packets
x₄: Source port
x₅: Destination port
x₆: Packets per second
x₇: Bytes per second

Step 2: Collect Training Data (Normal Traffic)
Sample 1000 connections during normal operations:
Connection types naturally present:

HTTP web browsing (port 80/443)
Email (port 25/587)
DNS queries (port 53)
SSH sessions (port 22)
Database queries (port 3306)

Step 3: Apply K-Means
Run K-means with k=5 (expecting 5 normal patterns).
Resulting clusters might be:
Cluster 1 - Web Browsing:

Centroid: (5000 bytes, 2 sec, 10 packets, random, 443, 5 pkt/sec, 2500 B/sec)
HTTPS traffic to various websites
Variable size, short duration

Cluster 2 - Email:

Centroid: (500 bytes, 1 sec, 3 packets, random, 587, 3 pkt/sec, 500 B/sec)
SMTP email sending
Small, quick connections

Cluster 3 - DNS:

Centroid: (100 bytes, 0.1 sec, 2 packets, random, 53, 20 pkt/sec, 1000 B/sec)
Domain name lookups
Very small, very fast

Cluster 4 - SSH:

Centroid: (10000 bytes, 300 sec, 1000 packets, random, 22, 3 pkt/sec, 33 B/sec)
Remote server administration
Long-lived, steady traffic

Cluster 5 - Database:

Centroid: (2000 bytes, 5 sec, 20 packets, random, 3306, 4 pkt/sec, 400 B/sec)
Application database queries
Medium size, quick connections

Step 4: Anomaly Detection
For each new connection, measure distance to nearest cluster center.
Define threshold: distance > threshold → ANOMALY
Example 1 - Normal Connection:
Connection: (4800 bytes, 1.8 sec, 9 packets, 55332, 443, 5 pkt/sec, 2667 B/sec)
Distance to Cluster 1 (Web): small → NORMAL
Example 2 - Port Scan:
Connection: (40 bytes, 0.01 sec, 1 packet, 55123, 8080, 100 pkt/sec, 4000 B/sec)

Tiny packets to unusual port
Very fast (trying many ports quickly)
Distance to all clusters: LARGE → ANOMALY!

Example 3 - Data Exfiltration:
Connection: (1000000 bytes, 60 sec, 5000 packets, 55444, 9999, 83 pkt/sec, 16667 B/sec)

Huge amount of data
To unusual port 9999
Much larger than any normal pattern
Distance to all clusters: LARGE → ANOMALY!

Example 4 - DDoS Attack:
Many connections: (64 bytes, 0.001 sec, 1 packet, random, 80, 1000 pkt/sec, 64000 B/sec)

Tiny SYN packets flooding server
Extremely high packet rate
Pattern different from normal web traffic
Distance to Cluster 1: LARGE → ANOMALY!

Step 5: Alert and Investigate
When anomaly detected:

Log the suspicious connection
Alert security team
Possibly block the source IP
Investigate if it's a real threat or false alarm

Why This Works
Key insight: Attackers behave differently from normal users!
Normal traffic:

Follows business application patterns
Predictable sizes, durations, ports
Falls into natural clusters

Malicious traffic:

Unusual patterns (port scanning)
Excessive volume (DDoS)
Unexpected destinations (data exfiltration)
Doesn't fit normal clusters!

K-means learns what "normal" looks like, without needing labeled attack examples!
This is called unsupervised anomaly detection - we don't need examples of attacks, just examples of normal behavior!
Practical Considerations
False Positives:

New legitimate application might look anomalous initially
Solution: Gradually update clusters with confirmed normal traffic

False Negatives:

Sophisticated attacks might mimic normal traffic
Solution: Combine with other detection methods (signatures, rules)

Scalability:

Real networks: millions of connections/second
Solution: Use mini-batch K-means, process samples

Adaptation:

Normal traffic patterns change over time
Solution: Retrain clusters periodically (daily/weekly)

Practice Problems - Detailed Examples
Problem 4.1: Customer Segmentation
You have 6 customers with (purchase_count, avg_spend):
C1: (10, 100), C2: (12, 110), C3: (2, 20), C4: (1, 15), C5: (11, 105), C6: (3, 25)
a) Standardize the features
b) Run K-means with k=2 (manual initialization: C1 and C3)
c) Interpret the two segments
d) What marketing strategy for each segment?
Problem 4.2: Image Compression
9 pixels with RGB colors:

6 pixels are shades of red: (200-255, 0-10, 0-10)
3 pixels are shades of blue: (0-10, 0-10, 200-255)

a) What k should you choose?
b) What will the cluster centers approximately be?
c) Calculate compression ratio
d) What does the compressed image lose?
Problem 4.3: Document Clustering
4 documents with word count vectors:

D1: (politics: 10, sports: 0, tech: 1)
D2: (politics: 1, sports: 15, tech: 0)
D3: (politics: 12, sports: 0, tech: 0)
D4: (politics: 0, sports: 13, tech: 1)

Run K-means with k=2:
a) Which documents cluster together?
b) Label each cluster
c) If D5 = (politics: 5, sports: 7, tech: 0), which cluster?
Problem 4.4: Anomaly Detection
Normal network connections cluster into 2 groups:

Cluster 1: web traffic (size: 5KB, duration: 1s)
Cluster 2: email (size: 10KB, duration: 5s)

Classify these new connections:
a) (size: 5.2KB, duration: 1.1s)
b) (size: 1000KB, duration: 0.1s)
c) (size: 8KB, duration: 4s)
d) Which are anomalies? Why?

<a name="choosing-k"></a>
5. Choosing K and Evaluation
The Fundamental Problem: How Many Clusters?
We've been assuming we know k (the number of clusters). But in reality, this is often the hardest question!
The problem:

Too few clusters (k too small): Different groups lumped together
Too many clusters (k too large): Natural groups split artificially
No "correct" answer: Depends on your goals and data!

Question: Can't we just try different values of k and pick the best?
Answer: Yes! But what does "best" mean?
The Elbow Method
The Intuition
Let's think about what happens as we increase k:
k=1: All points in one cluster

WCSS = total variance of data (maximum!)
Not useful (no segmentation)

k=2: Data split into 2 clusters

WCSS decreases (clusters more compact)
Some segmentation

k=3: Data split into 3 clusters

WCSS decreases more
More refined segmentation

k=n: Each point its own cluster

WCSS = 0 (perfect!)
But completely useless! No generalization!

Pattern: WCSS always decreases as k increases!
Question: So why not always use large k?
Answer: There's a diminishing return point! After the "natural" number of clusters, additional clusters don't help much.
The Elbow Plot
Create a plot:

x-axis: k (number of clusters)
y-axis: WCSS

What we see:

Sharp decrease initially (finding real structure)
Then gradual decrease (overfitting, splitting natural clusters)
The "elbow" (bend) is the optimal k!

Example with toy data:
k=1: WCSS = 1000
k=2: WCSS = 400  (60% reduction!)
k=3: WCSS = 150  (62.5% reduction from k=2!)
k=4: WCSS = 100  (33% reduction - slowing down)
k=5: WCSS = 80   (20% reduction - "elbow"!)
k=6: WCSS = 70   (12.5% reduction - diminishing)
k=7: WCSS = 63   (10% reduction)
k=8: WCSS = 58   (8% reduction)
Plot shape:
WCSS
1000|*
    |
 500|  *
    |    
 100|     *
    |       * * * * *
   0|_____________________ k
     1 2 3 4 5 6 7 8
         
           ↑
        "Elbow" at k=5!
The elbow at k=5 suggests the data has ~5 natural clusters!
Why The Elbow Works
Question: Why does the plot have an "elbow" shape?
Answer: It reflects the structure in the data!
Before the elbow (k < optimal):

Each new cluster captures real structure
Large reduction in WCSS
Steep decrease

At the elbow (k ≈ optimal):

Captured most natural structure
Transition point

After the elbow (k > optimal):

Splitting cohesive groups artificially
Small reduction in WCSS
Gradual decrease

The elbow marks the transition from "finding structure" to "overfitting"!
Limitations of Elbow Method
Problem 1: Ambiguous elbow
Sometimes there's no clear elbow! The curve is smooth with no obvious bend.
Example:
WCSS
1000|*
    | *
 500|  *
    |   *
 100|    *
    |     *
   0|_____________________ k
     1 2 3 4 5 6 7 8
Solution: Try other methods (silhouette, gap statistic)
Problem 2: Multiple elbows
Data might have hierarchical structure with elbows at multiple k values.
Example:

Elbow at k=3 (major categories)
Elbow at k=9 (sub categories)
Which to choose? Depends on your goal!

Problem 3: Subjective
Different people might identify different "elbows" in the same plot!
Solution: Complement with quantitative metrics
Silhouette Score
The Idea
The elbow method only looks at WCSS (within-cluster distance). But what about separation between clusters?
Good clustering should have:

Cohesion: Points close to their own cluster center (small within-cluster distance)
Separation: Points far from other cluster centers (large between-cluster distance)

Silhouette score measures both!
Silhouette Score for a Single Point
For each point xᵢ:
Step 1: Calculate a(i) = average distance to other points in same cluster
Step 2: Calculate b(i) = average distance to points in nearest other cluster
Step 3: Silhouette coefficient:
s(i) = [b(i) - a(i)] / max{a(i), b(i)}
Range: -1 to +1
Interpretation:

s(i) ≈ 1: Point well-matched to own cluster, far from others (good!)
s(i) ≈ 0: Point on border between clusters (ambiguous)
s(i) < 0: Point probably in wrong cluster (bad!)

Average Silhouette Score
For entire clustering:
Silhouette Score = (1/n) Σᵢ s(i)
Range: -1 to +1
Interpretation:

≈ 1: Excellent clustering (tight, well-separated clusters)
≈ 0: Overlapping clusters or arbitrary assignment
< 0: Many points in wrong clusters (bad clustering!)

Typical values:

> 0.7: Strong structure
0.5-0.7: Reasonable structure
0.25-0.5: Weak structure
< 0.25: No substantial structure

Using Silhouette for Choosing k
Calculate silhouette score for different k:
k=2: Silhouette = 0.65
k=3: Silhouette = 0.71  ← Maximum!
k=4: Silhouette = 0.62
k=5: Silhouette = 0.51
Choose k that maximizes average silhouette score!
In this example, k=3 is optimal!
Silhouette vs Elbow
Elbow method:

✓ Simple, intuitive
✓ Easy to implement
✗ Only considers within-cluster distance
✗ Subjective interpretation

Silhouette score:

✓ Considers both cohesion and separation
✓ Quantitative (not subjective)
✗ More expensive to compute (O(n²) for each k!)
✗ Can be misleading for some cluster shapes

Best practice: Use both! They complement each other.
Gap Statistic
The Problem with WCSS
Question: Why can't we just pick k with lowest WCSS?
Answer: WCSS always decreases with k! Even for random data with no structure!
The insight: We need to compare our WCSS to what we'd expect for random data.
The Gap Statistic Idea
Compare:

WCSS on our data
WCSS on random data (uniform distribution)

If our data has structure:

Our WCSS should be much smaller than random!
The "gap" between them is large!

If our data is random:

Our WCSS similar to random
Small gap

The optimal k maximizes this gap!
Gap Statistic Formula
Gap(k) = log(WCSS_random(k)) - log(WCSS_actual(k))
Why logarithms? To make the scale more interpretable and stabilize variance.
Procedure:

For each k:
a) Calculate WCSS on actual data
b) Generate B random datasets (uniform over same range)
c) Calculate WCSS on each random dataset
d) Average: WCSS_random(k)
e) Calculate Gap(k)
Choose k that maximizes Gap(k)

Typical B: 10-50 random datasets
Example
k=1: Gap = 0.2
k=2: Gap = 0.8
k=3: Gap = 1.2  ← Maximum!
k=4: Gap = 1.0
k=5: Gap = 0.9
Choose k=3!
Interpretation: At k=3, our clustering is much better than random, more so than at other k values!
Advantages of Gap Statistic
1. Principled: Based on statistical comparison to null hypothesis (random data)
2. Can detect "no structure": If Gap is small for all k, suggests data has no clear cluster structure!
3. Less subjective: Quantitative comparison
Disadvantages:
1. Computationally expensive: Need to cluster many random datasets
2. Assumes uniform null: Might not be appropriate for all data distributions
3. Can be conservative: Sometimes underestimates k
Domain Knowledge and Business Goals
The Often-Overlooked Factor
Mathematical methods give suggestions, but final choice should consider:

Business constraints
Operational feasibility
Interpretability
Actionability

Example: Customer Segmentation
Elbow method suggests: k=7 clusters
Silhouette suggests: k=5 clusters
Gap statistic suggests: k=6 clusters
Business team says: "We can only handle 3 different marketing campaigns!"
Final decision: k=3
Why this makes sense:

Mathematical optimality < practical utility
3 actionable segments > 7 unusable segments
Must be able to act on the insights!

Hierarchical Interpretation
Alternative approach: Use hierarchical clustering or run K-means with different k values.
Example with k=3 and k=9:
k=3 (High level):

Segment 1: High-value customers
Segment 2: Medium-value customers
Segment 3: Low-value customers

k=9 (Detailed):

Segment 1a: Premium loyalists
Segment 1b: Big spenders, infrequent
Segment 1c: Frequent, moderate spend
Segment 2a: Growing customers
Segment 2b: Stable regulars
Segment 2c: Declining interest
Segment 3a: New customers
Segment 3b: Occasional buyers
Segment 3c: At-risk churners

Use case: Executive dashboard shows k=3, detailed analysis uses k=9!
Comparing Multiple Clusterings
Scenario
You've run K-means with different k values. How do you compare them?
Metrics Summary
MetricRangeOptimal DirectionConsidersCostWCSS[0, ∞)LowerWithin-cluster onlyO(nk)Silhouette[-1, 1]HigherBoth cohesion & separationO(n²k)Gap Statistic(-∞, ∞)HigherComparison to randomO(Bnkd)Davies-Bouldin[0, ∞)LowerRatio of within/betweenO(nk²)Calinski-Harabasz[0, ∞)HigherRatio of between/withinO(nk)
Davies-Bouldin Index
Measures: Average similarity between each cluster and its most similar cluster
Formula:
DB = (1/k) Σⱼ max_{j≠i} [(σᵢ + σⱼ) / d(cᵢ, cⱼ)]
where:

σᵢ = average distance within cluster i
d(cᵢ, cⱼ) = distance between cluster centers

Lower is better!
Intuition:

Numerator: within-cluster spread
Denominator: between-cluster separation
Lower ratio → better clustering!

Calinski-Harabasz Index (Variance Ratio)
Measures: Ratio of between-cluster variance to within-cluster variance
Formula:
CH = [Between-cluster variance / (k-1)] / [Within-cluster variance / (n-k)]
Higher is better!
Intuition:

Like F-statistic in ANOVA
High between-cluster variance = well-separated
Low within-cluster variance = compact
Higher ratio → better clustering!

Practical Workflow for Choosing K
Step-by-Step Process
Step 1: Start with Domain Knowledge

Do you expect a certain number of groups?
Any business constraints on k?
What's the purpose of clustering?

Step 2: Try Elbow Method

Plot WCSS vs k for k=1 to 10 (or higher)
Look for obvious elbow
Note potential k values

Step 3: Calculate Silhouette Scores

For k values near the elbow
Plot silhouette score vs k
Note k with highest score

Step 4: (Optional) Gap Statistic

If previous methods disagree
Or if you need statistical justification
Note k with highest gap

Step 5: Visualize Results

If possible (2D/3D), plot clusterings
Use dimensionality reduction (PCA) for high-D data
Do clusters look sensible?

Step 6: Interpret

Can you describe each cluster?
Are they actionable/meaningful?
Do they align with domain knowledge?

Step 7: Make Decision

Weigh all evidence
Consider practical constraints
Choose k that balances statistical evidence and utility

Example Workflow
Problem: Segment customers for marketing
Step 1:

Marketing team can handle 3-5 campaigns
Expecting "high/medium/low value" at minimum

Step 2 - Elbow:

Elbow appears around k=4 or k=5

Step 3 - Silhouette:

k=3: 0.65
k=4: 0.71 ← highest!
k=5: 0.68
k=6: 0.61

Step 4 - Gap (optional):

k=4 has highest gap

Step 5 - Visualization:

k=4 shows clear separation
Clusters interpretable

Step 6 - Interpretation:

Cluster 1: VIP (high frequency, high value)
Cluster 2: Regulars (medium frequency, medium value)
Cluster 3: Occasional (low frequency, variable value)
Cluster 4: At-risk (low frequency, decreasing over time)

Step 7 - Decision:

Choose k=4
All methods agree (roughly)
Within operational constraints
Clusters are interpretable and actionable!

Practice Problems - Choosing K
Problem 5.1: Elbow Method
WCSS values for different k:
k=1: 500, k=2: 250, k=3: 150, k=4: 100, k=5: 80, k=6: 70, k=7: 65
a) Plot WCSS vs k (sketch)
b) Where is the elbow?
c) Calculate percent reduction in WCSS from k to k+1 for each k
d) Which k would you choose? Why?
Problem 5.2: Silhouette Interpretation
Point A has:

Average distance to own cluster: 2.0
Average distance to nearest other cluster: 8.0

a) Calculate silhouette coefficient for A
b) Is A well-clustered?
c) What if distances were 5.0 and 5.5?
d) What if distances were 8.0 and 2.0?
Problem 5.3: Method Comparison
Three methods give different recommendations:

Elbow: k=4
Silhouette: k=5
Gap: k=3

Business constraint: Can handle 3-4 segments
a) Which k would you choose?
b) What additional information would help decide?
c) Could you use multiple k values? How?
Problem 5.4: No Clear Structure
All metrics show weak evidence:

WCSS decreases gradually (no elbow)
Silhouette scores all around 0.3
Gap statistic small for all k

a) What does this suggest about the data?
b) Should you still cluster?
c) What alternatives might work better?
Problem 5.5: Hierarchical Decisions
Company wants to segment customers.
CEO needs: 3-4 strategic segments
Marketing needs: 8-10 operational segments
a) How can you satisfy both?
b) What's the relationship between strategic and operational segments?
c) How would you present this to stakeholders?

<a name="applications"></a>
6. Practical Applications
Application 1: Recommendation Systems
The Netflix Problem
Netflix has millions of users and thousands of movies. How do they recommend movies?
Naive approach: Recommend most popular movies to everyone

Problem: Ignores personal preferences!
Everyone sees the same recommendations

Better approach: Find similar users, recommend what they liked!

If User A and User B have similar taste
User A liked Movie X
Recommend Movie X to User B!

This is collaborative filtering using clustering!
Implementation with K-Means
Step 1: Represent users as vectors
For each user, create a vector of movie ratings:
User 1: [5, ?, 3, 4, ?, 1, 5, ...]
(rated movies on scale 1-5, ? = not rated)
Problem: Sparse vectors (most movies unrated)!
Solution: Use dimensionality reduction (matrix factorization, PCA) first, then cluster
Step 2: Cluster users
Apply K-means to find user clusters:

Cluster 1: Action movie fans
Cluster 2: Romance movie fans
Cluster 3: Documentary enthusiasts

Cluster 4: Comedy lovers
Cluster 5: Horror fans
etc.

Step 3: Make recommendations
For a new user or existing user:

Assign to nearest cluster
Find what cluster members rated highly
Recommend top-rated movies from cluster (that user hasn't seen)

Example:
User Alice:

Rated: Avengers (5), Iron Man (4), Terminator (5)
Cluster assignment: Cluster 1 (Action fans)

Other Cluster 1 members also rated highly:

Mad Max (avg 4.7)
John Wick (avg 4.5)
Mission Impossible (avg 4.3)

Recommend these to Alice!
Why This Works
Key insight: Users with similar past preferences likely have similar future preferences!
Advantages:

Scalable (cluster once, use for all recommendations)
Discovers hidden patterns (users who like A and B also like C)
Works without understanding movie content

Limitations:

Cold start problem (new users with no ratings)
Popularity bias (recommends popular items in cluster)
Filter bubble (never exposed to other genres)

Modern solutions combine:

Collaborative filtering (user clustering)
Content-based filtering (movie feature similarity)
Hybrid approaches

Application 2: Market Segmentation
The Real Estate Problem
Real estate company has data on neighborhoods:
Features:

x₁: Median home price
x₂: School quality (1-10)
x₃: Crime rate (per 1000)
x₄: Distance to downtown (miles)
x₅: Average lot size (sqft)
x₆: Population density (per sq mile)
x₇: Median income

Goal: Segment neighborhoods for targeted marketing
Clustering Analysis
After K-means with k=5:
Cluster 1 - Urban Professional:

High price ($800k+)
Good schools (8+)
Low crime
Very close to downtown (<5 miles)
Small lots
High density
High income

Marketing strategy:

Target young professionals, DINKs (Dual Income No Kids)
Emphasize walkability, restaurants, nightlife
Highlight career opportunities nearby
Price point: premium condos, modern apartments

Cluster 2 - Family Suburban:

Medium-high price ($500-700k)
Excellent schools (9+)
Very low crime
Moderate distance (10-15 miles)
Large lots
Low density
Upper-middle income

Marketing strategy:

Target families with children
Emphasize school quality, safety, space
Highlight family amenities (parks, community centers)
Price point: single-family homes, 4-5 bedrooms

Cluster 3 - Affordable Starter:

Lower price ($200-350k)
Moderate schools (5-7)
Moderate crime
Far from downtown (15-25 miles)
Medium lots
Medium density
Middle income

Marketing strategy:

Target first-time homebuyers, young families
Emphasize affordability, potential appreciation
Highlight community feel, value
Price point: townhomes, smaller single-family homes

Cluster 4 - Luxury Estate:

Very high price ($1M+)
Good schools (7-9)
Very low crime
Moderate distance (8-12 miles)
Very large lots (acres)
Very low density
Very high income

Marketing strategy:

Target wealthy families, executives
Emphasize privacy, space, exclusivity
Highlight custom homes, acreage
Price point: estates, luxury homes

Cluster 5 - Urban Affordable:

Lower price ($150-300k)
Poor schools (3-5)
Higher crime
Close to downtown (3-8 miles)
Small lots
Very high density
Lower income

Marketing strategy:

Target budget-conscious buyers, investors
Emphasize location, potential gentrification
Highlight rental income potential
Price point: condos, small homes, investment properties

Business Impact
Before clustering:

Generic "homes for sale" messaging
Wasted ad spend on irrelevant audiences
Low conversion rates

After clustering:

Targeted ads for each segment
Relevant messaging (schools for Cluster 2, walkability for Cluster 1)
3x higher conversion rates
40% more efficient marketing spend

Additional insights:

Cluster 5 gentrifying → opportunity for investors
Cluster 2 expanding → build more family homes
Cluster 1 highly competitive → premium marketing needed

Application 3: Genomic Data Analysis
The Medical Research Problem
Researchers have gene expression data from cancer patients.
Data:

100 patients
Expression levels of 20,000 genes for each patient
Very high dimensional!

Goal: Find patient subgroups with similar genetic profiles
Why this matters:

Different subgroups may respond to different treatments
Can lead to personalized medicine
May discover new cancer subtypes

Approach
Step 1: Dimensionality Reduction
Can't cluster in 20,000 dimensions directly!
Solution:

Use PCA to reduce to ~50 dimensions
Or select most variable genes (~1000)
Or use specialized methods (t-SNE, UMAP)

Step 2: Cluster patients
Apply K-means in reduced space with k=4
Result: 4 patient subtypes discovered
Subtype 1 (25 patients):

High expression of immune genes
Low expression of cell division genes
Clinical correlation: Respond well to immunotherapy!
Treatment: Checkpoint inhibitors

Subtype 2 (30 patients):

High expression of hormone receptors
Moderate cell division
Clinical correlation: Hormone-sensitive tumors
Treatment: Hormone therapy (Tamoxifen, etc.)

Subtype 3 (20 patients):

Very high expression of cell division genes
Mutations in DNA repair genes
Clinical correlation: Aggressive, fast-growing
Treatment: Aggressive chemotherapy + radiation

Subtype 4 (25 patients):

Intermediate expression patterns
Mix of characteristics
Clinical correlation: Unclear, needs more research
Treatment: Standard combination therapy

Medical Impact
Before clustering:

One-size-fits-all treatment protocols
40% response rate
Unnecessary side effects for non-responders

After clustering:

Personalized treatment based on subtype
65% response rate (cluster-matched therapy)
Reduced side effects
Better patient outcomes

Research insights:

Subtype 3 needs new targeted therapies
Subtype 4 may actually be mix of other subtypes
Gene signatures for each subtype identified

Future directions:

Predict subtype from initial biopsy
Match patients to clinical trials
Drug development for specific subtypes

Ethical Considerations
Benefits:

Better outcomes for patients
Reduced unnecessary treatments
More efficient resource use

Challenges:

Access to genomic testing (cost, availability)
Privacy concerns (genetic data)
Health insurance implications
Need for diverse patient populations in research

Application 4: Social Network Analysis
The Social Media Problem
Social media platform wants to detect communities.
Data:

Who follows whom
Who interacts with whom (likes, comments, shares)
Millions of users

Goal: Find communities (groups of tightly connected users)
Graph Clustering Approach
Represent network as graph:

Nodes = users
Edges = connections (follow/interact)

Convert to feature vectors:
For each user, features could be:

Number of followers
Number of following
Interaction rate
Topics of interest (hashtags used)
Activity level (posts per day)
Connection patterns

Alternative: Use graph embedding methods

Node2Vec: Convert graph to vectors
DeepWalk: Random walks to learn representations
Then cluster these vectors!

Discovered Communities
After K-means with k=8:
Community 1 - Tech Enthusiasts:

50,000 users
High interaction with tech news
Follow tech influencers
Active in #AI, #startup, #coding discussions

Community 2 - Fitness/Health:

35,000 users
Share workout routines, healthy recipes
Follow fitness influencers
Active in #fitness, #health, #wellness

Community 3 - Gaming:

80,000 users
Discuss video games, esports
Follow gaming streamers
Active in #gaming, #esports, #twitch

Community 4 - Fashion/Beauty:

45,000 users
Share outfit ideas, makeup tutorials
Follow fashion influencers
Active in #fashion, #beauty, #style

Community 5 - Politics:

30,000 users
Discuss current events, policy
Follow politicians, journalists
Active in political hashtags

Community 6 - Parenting:

25,000 users
Share parenting tips, advice
Follow parenting accounts
Active in #momlife, #parenting

Community 7 - Travel:

40,000 users
Share travel photos, tips
Follow travel bloggers
Active in #travel, #wanderlust

Community 8 - Food:

55,000 users
Share recipes, restaurant reviews
Follow chefs, food bloggers
Active in #foodie, #cooking, #recipes

Applications
1. Content Recommendation:

User in Community 1 → show tech content
User in Community 2 → show fitness content
Increases engagement!

2. Targeted Advertising:

Tech brands advertise to Community 1
Sportswear to Community 2
Higher conversion rates!

3. Influencer Identification:

Find central nodes in each community
These are key influencers
Partner with them for marketing

4. Trend Detection:

Sudden activity spike in community → emerging trend
Early detection of viral content
Content moderation priorities

5. Network Health:

Communities too isolated → echo chambers
Promote cross-community connections
Suggest diverse content

Why This Works
Key insight: People cluster by shared interests naturally!
Network effects:

Similar people follow similar accounts
Create reinforcing connections
Form tight communities

Clustering reveals:

Hidden community structure
Interest-based segments
Connection patterns

Application 5: Fraud Detection in Banking
The Banking Problem
Bank processes millions of transactions daily.
Challenge: Detect fraudulent transactions in real-time
Traditional approach: Rule-based systems

Flag transactions over $10,000
Flag international transactions
Many false positives!

Better approach: Learn patterns of normal behavior, flag anomalies!
Feature Engineering
For each transaction:

x₁: Amount
x₂: Time of day (0-23)
x₃: Day of week (0-6)
x₄: Merchant category
x₅: Distance from usual locations
x₆: Time since last transaction (seconds)
x₇: Ratio to average transaction
x₈: Account age (days)
x₉: Recent velocity (transactions per day)

Clustering Normal Behavior
Step 1: Cluster on normal transactions only
After K-means with k=6:
Cluster 1 - Regular Purchases:

Small amounts ($10-100)
Daytime (9am-6pm)
Weekdays
Local stores
Frequent (daily)

Cluster 2 - Bill Payments:

Fixed amounts
Automatic transactions
Beginning/end of month
Utility companies
Regular (monthly)

Cluster 3 - Weekend Entertainment:

Medium amounts ($50-200)
Evening/night (6pm-midnight)
Weekends
Restaurants, bars, entertainment
Occasional (weekly)

Cluster 4 - Online Shopping:

Variable amounts ($20-500)
Any time of day
E-commerce merchants
Delivered to home address
Intermittent

Cluster 5 - Large Purchases:

Large amounts ($500-5000)
Rare (monthly/yearly)
Specific merchants (electronics, furniture, etc.)
Often followed by smaller transactions

Cluster 6 - Travel:

Variable amounts
Foreign merchants
Different timezones
Clustered in time (trip duration)
Preceded by travel bookings

Fraud Detection Strategy
For each new transaction:
Step 1: Calculate distance to nearest cluster
Step 2: If distance > threshold → POTENTIAL FRAUD
Step 3: Additional checks:

Does it match user's typical pattern?
Is it an unusual merchant?
Geographic anomaly?
Velocity check (too many transactions too fast)

Step 4: Decide:

Low risk → Approve automatically
Medium risk → Approve but monitor
High risk → Decline, require verification

Example Scenarios
Scenario 1 - Normal:
Transaction: $45 at grocery store, 10am, Tuesday, local

Matches Cluster 1 (Regular Purchases)
Distance to cluster center: small
Decision: APPROVE

Scenario 2 - Normal but unusual:
Transaction: $2000 at jewelry store, 3pm, Saturday, local

Somewhat matches Cluster 5 (Large Purchases)
Distance moderate
User has history of occasional large purchases
Decision: APPROVE but MONITOR

Scenario 3 - Suspicious:
Transaction: $500 at electronics store, 3am, Monday, foreign country

Doesn't match any cluster well
Distance to all clusters: large
User has no travel history
Time is unusual (3am in foreign country)
Decision: DECLINE, request verification

Scenario 4 - Clearly fraudulent:
Rapid sequence:

$300 at store A, 2am
$400 at store B, 2:05am
$500 at store C, 2:10am
(All in different countries!)
Impossible velocity (can't be in multiple countries simultaneously)
Far from all normal patterns
Decision: DECLINE ALL, freeze card, alert customer

Results
Before clustering-based fraud detection:

Rule-based system
60% of fraud caught
5% false positive rate (legitimate transactions declined)
Customer frustration

After clustering-based fraud detection:

85% of fraud caught
0.5% false positive rate
Better customer experience
Adaptive (learns new patterns)

Additional benefits:

Faster detection (real-time clustering)
Fewer customer service calls
Reduced fraud losses ($10M+ saved per year)

Adaptive Learning
Key advantage: System adapts over time
As new transactions come in:

Periodically retrain clusters (weekly/monthly)
Capture evolving spending patterns
Detect new fraud techniques

Example evolution:

COVID-19 pandemic: more online purchases
Cluster 4 (Online Shopping) grew
New cluster emerged: Video streaming services
System adapted automatically!

Practice Problems - Applications
Problem 6.1: Recommendation System
You cluster 1000 users into 4 groups based on movie ratings:

Cluster 1: Action fans (300 users)
Cluster 2: Drama fans (250 users)
Cluster 3: Comedy fans (350 users)
Cluster 4: Horror fans (100 users)

New user rates: Die Hard (5), Inception (4), John Wick (5)
a) Which cluster should they be assigned to?
b) What movies should you recommend?
c) User later rates The Notebook (2). Does this change anything?
d) What's the cold start problem? How could you handle it?
Problem 6.2: Market Segmentation
You cluster neighborhoods and find:

Cluster A: Expensive, great schools, far from downtown
Cluster B: Cheap, poor schools, far from downtown
Cluster C: Expensive, moderate schools, close to downtown

New neighborhood: Moderate price, great schools, moderate distance
a) Which cluster is it closest to?
b) What if it's equally distant from all clusters?
c) Should you create a new cluster?
d) How does this inform pricing strategy?
Problem 6.3: Medical Subtyping
Cancer patients clustered by gene expression:

Subtype A: 40% survival at 5 years
Subtype B: 80% survival at 5 years
Subtype C: 60% survival at 5 years

New patient's genes cluster with Subtype A.
a) What's the prognosis?
b) How should this inform treatment?
c) What if the patient is on the border between A and B?
d) Ethical considerations?
Problem 6.4: Fraud Detection
Normal transaction clusters:

Cluster 1: Small purchases ($5-50), local, frequent
Cluster 2: Large purchases ($500+), rare
Cluster 3: Online purchases, moderate amounts

Classify these transactions:
a) $30 at local grocery, 10am
b) $5000 at jewelry store, 3am, foreign country
c) $100 online purchase, 2pm
d) Rapid sequence: $200, $250, $300, all different countries in 5 minutes
Problem 6.5: Social Network Communities
You've clustered users into communities.
a) How would you identify influencers in each community?
b) How would you suggest new connections?
c) What if a user belongs to multiple communities?
d) How do you detect emerging communities?

<a name="advanced"></a>
7. Advanced Topics
K-Means Limitations and Solutions
Limitation 1: Assumes Spherical Clusters
Problem: K-means creates circular/spherical decision boundaries
Fails on:

Elongated clusters
Irregular shapes
Concentric patterns

Example failure: Two moons dataset
    ***
  **   **
 *       *
*         *    ***
 *       *   **   **
  **   **   *       *
    ***     *        *
             *      *
              **  **
                **
K-means will split each moon in half instead of separating the two moons!
Why? It uses Euclidean distance to spherical centroids. Can't capture curved shapes!
Solution 1: Kernel K-Means

Map data to higher dimensional space
Apply K-means there
Like kernel SVM!

Solution 2: Spectral Clustering

Build similarity graph
Use graph structure to find clusters
Can handle arbitrary shapes!

Solution 3: DBSCAN

Density-based clustering
Groups points in dense regions
Naturally handles arbitrary shapes!

Limitation 2: Sensitive to Outliers
Problem: Outliers pull centroids toward them
Example:
Main cluster: 100 points at (0,0) with std 1
Outlier: 1 point at (100, 100)

Without outlier: centroid at (0,0)
With outlier: centroid pulled to (~1, ~1)
Why? Squared distance heavily penalizes outliers, centroid moves toward them!
Solution 1: Remove outliers first

Pre-process data
Remove points far from dense regions
Then cluster

Solution 2: K-Medoids (PAM)

Use actual data points as centers
Less sensitive to outliers
More robust!

Solution 3: Robust K-Means

Use robust statistics
Trimmed K-means: ignore farthest points
Use L₁ norm instead of L₂

Limitation 3: Need to Specify k
Problem: Don't always know k in advance!
Solutions discussed:

Elbow method
Silhouette score
Gap statistic
Domain knowledge

Alternative approaches:
DBSCAN:

Doesn't require k!
Automatically determines number of clusters
Based on density parameters (ε, minPts)

Hierarchical Clustering:

Creates full hierarchy
Cut at any level for different k
Explore multiple granularities!

Gaussian Mixture Models:

Model-based clustering
Can use BIC/AIC to select number of components
More flexible than K-means!

Limitation 4: Local Minima
Problem: K-means finds local minimum, not global
Example:
True clusters:
  AAA       BBB
  AAA       BBB
  AAA       BBB

Bad initialization leads to:
  AAB       BAB
  AAB       BAB
  AAB       BAB
Converges to suboptimal solution!
Solutions:
1. Multiple random restarts (standard)

Run 10-50 times
Keep best result
Most effective!

2. K-means++ (smart initialization)

Provably better initialization
First center: random
Next centers: far from existing ones
Default in most libraries!

3. Hierarchical initialization

Run hierarchical clustering first
Use resulting centers to initialize K-means
Slower but more reliable

4. Deterministic Annealing

Start with k=1, gradually increase
Each k initialized from previous solution
More stable convergence

Limitation 5: Assumes Equal Cluster Sizes
Problem: K-means biased toward equal-sized clusters
Example:
True clusters:
Cluster A: 1000 points, compact
Cluster B: 100 points, spread out

K-means might split A to balance sizes!
Why? Minimizing total WCSS favors splitting large clusters even if they're cohesive!
Solution 1: Weighted K-means

Add cluster size penalty to objective
Prefer balanced clusters explicitly

Solution 2: Gaussian Mixture Models

Model each cluster as Gaussian distribution
Different clusters can have different sizes/shapes
More flexible!

Solution 3: Hierarchical Clustering

Doesn't assume equal sizes
Bottom-up or top-down approaches
Can handle imbalanced clusters

K-Means Variants and Extensions
Mini-Batch K-Means
Problem: Standard K-means slow on huge datasets (millions of points)
Idea: Use only a random subset (mini-batch) each iteration!
Algorithm:

Sample random mini-batch (e.g., 1000 points)
Assign points in mini-batch to nearest centers
Update centers using only mini-batch
Repeat

Trade-offs:

✓ Much faster (10-100x)
✓ Scales to very large datasets
✗ Slightly worse clustering quality (~5%)
✗ More iterations needed

When to use:

Dataset too large for memory
Need fast approximate solution
Exploratory analysis
Online/streaming data

Implementation note: scikit-learn provides MiniBatchKMeans
Fuzzy C-Means
Problem: Hard assignment (point belongs to ONE cluster) too rigid
Idea: Soft assignment (point has membership degree to ALL clusters)!
Membership:
Each point xᵢ has membership uᵢⱼ to cluster j

0 ≤ uᵢⱼ ≤ 1
Σⱼ uᵢⱼ = 1 (memberships sum to 1)

Example:
Point on boundary between clusters:

Cluster 1: u = 0.6
Cluster 2: u = 0.4

Point clearly in cluster 1:

Cluster 1: u = 0.95
Cluster 2: u = 0.05

Update equations:
Centers weighted by membership:
μⱼ = Σᵢ uᵢⱼᵐ xᵢ / Σᵢ uᵢⱼᵐ
where m > 1 is fuzziness parameter
When to use:

Clusters overlap significantly
Need probability of membership
Soft decision boundaries needed

K-Medoids (PAM - Partitioning Around Medoids)
Key difference: Centers must be actual data points!
Why?

More robust to outliers
Centers are interpretable (real examples)
Works with any distance metric (not just Euclidean)

Algorithm:

Initialize: Select k random points as medoids
Assignment: Assign each point to nearest medoid
Update: For each cluster, try all points as potential medoid

Choose point that minimizes total distance


Repeat until convergence

Complexity: O(n²) per iteration (vs O(n) for K-means)
Trade-off:

✓ More robust
✓ Interpretable centers
✓ Works with any distance
✗ Much slower

When to use:

Small to medium datasets (n < 10,000)
Outliers present
Need interpretable centers
Non-Euclidean distance needed

K-Modes and K-Prototypes
Problem: K-means only works with numerical data!
K-Modes: For categorical data

Uses mode instead of mean
Uses matching dissimilarity instead of Euclidean distance
Dissimilarity = number of mismatches

Example:
Point A: (red, large, yes)
Point B: (red, small, yes)
Dissimilarity = 1 (differ in size only)
K-Prototypes: For mixed data (numerical + categorical)

Combines K-means and K-modes
Distance = numerical distance + categorical dissimilarity

When to use:

Categorical features (color, type, yes/no)
Mixed feature types
Survey data, demographic data

Hierarchical Clustering: An Alternative
The Idea
Instead of flat partitions, create a hierarchy of clusters!
Tree structure (dendrogram):
                    All Data
                   /        \
              Group A      Group B
              /    \        /    \
            A1     A2     B1     B2
           / \    / \    / \    / \
          ...   ...   ...   ... ...
Advantages:

Don't need to specify k upfront!
Can explore multiple granularities
Reveals hierarchical structure
No random initialization (deterministic)

Types:
1. Agglomerative (Bottom-up):

Start: each point its own cluster
Repeatedly merge closest clusters
Stop: all points in one cluster

2. Divisive (Top-down):

Start: all points in one cluster
Repeatedly split clusters
Stop: each point its own cluster

Agglomerative is more common (easier to implement)
Agglomerative Clustering Algorithm
Input: n data points, distance metric, linkage criterion
Algorithm:

Start with n clusters (each point alone)
While more than 1 cluster:
a) Find pair of closest clusters
b) Merge them
c) Record merge in dendrogram
Output: dendrogram tree

Key question: How to measure distance between clusters?
Linkage Criteria
Single Linkage:
distance(A, B) = min {d(xᵢ, xⱼ) : xᵢ ∈ A, xⱼ ∈ B}
Closest points between clusters
Complete Linkage:
distance(A, B) = max {d(xᵢ, xⱼ) : xᵢ ∈ A, xⱼ ∈ B}
Farthest points between clusters
Average Linkage:
distance(A, B) = average of all pairwise distances
Ward's Linkage:
distance(A, B) = increase in WCSS if clusters merged
Most similar to K-means objective!
Example:
Cluster A: {(1,1), (2,1)}
Cluster B: {(5,5), (6,5)}
Single:

min distance = ||(2,1) - (5,5)|| = 5
Tends to form long chains

Complete:

max distance = ||(1,1) - (6,5)|| = 6.4
Tends to form compact clusters

Average:

avg of all 4 distances ≈ 5.6
Balanced approach

Ward's:

Measures WCSS increase
Most similar to K-means
Generally works well

Reading a Dendrogram
Height
  |
10|            ___
  |           |   |___
 8|      _____|       |___
  |     |                 |
 6|  ___|                 |
  |  |  |                 |
 4|  |  |             ____|
  |  |  |       _____|    
 2|  |  |  _____|         
  |  |  |  |              
 0|__|__|__|_______________
    1  2  3  4  5  6  7  8
How to read:

x-axis: data points
y-axis: distance/height at which clusters merge
Horizontal lines: clusters
Cut at any height to get different k!

Example:

Cut at height 5: get 3 clusters
Cut at height 7: get 2 clusters
Cut at height 3: get 6 clusters

K-Means vs Hierarchical
AspectK-MeansHierarchicalNeed k?YesNo (choose later)SpeedFast O(nkd)Slow O(n²logn) or O(n³)ScalabilityMillions of pointsThousands of pointsDeterministicNo (random init)YesCluster shapeSphericalAny (depends on linkage)Multiple granularitiesNoYes (dendogram)OutliersSensitiveDepends on linkage
When to use K-means:

Large datasets
Know approximate k
Need speed
Spherical clusters

When to use Hierarchical:

Small/medium datasets
Don't know k
Want to explore hierarchy
Need deterministic results

DBSCAN: Density-Based Clustering
The Core Idea
K-means asks: Which center is each point closest to?
DBSCAN asks: Which points are in dense regions together?
Key insight: Clusters are regions of high density, separated by regions of low density!
Advantages:

Handles arbitrary shapes!
Doesn't require k!
Finds outliers automatically!
Robust to noise

DBSCAN Parameters
ε (epsilon): Maximum distance for neighborhood
minPts: Minimum points to form dense region
Point classifications:
Core point: Has ≥ minPts points within distance ε
Border point: Within ε of core point, but not core itself
Noise point: Neither core nor border
DBSCAN Algorithm
Input: Data points, ε, minPts
Algorithm:

For each unvisited point p:
a) Find all points within distance ε (neighborhood)
b) If |neighborhood| < minPts:

Mark p as noise (for now)
c) Else:
Create new cluster
Add p and neighborhood to cluster
For each neighbor q:

If q is core point, add its neighbors to cluster
Continue expanding (depth-first search)




Points not in any cluster = noise/outliers

Example:
Points:
     A  B  C
     
D  E  F  G  H

        I  J  K
        
            L (outlier)
With ε=2, minPts=3:

A, B, C: core points (each has 2+ neighbors)
Form Cluster 1
D, E, F, G, H: core points
Form Cluster 2
I, J, K: core points
Form Cluster 3
L: noise (no neighbors)

When DBSCAN Works Well
Good for:

Arbitrary cluster shapes (crescents, circles, etc.)
Clusters of varying densities (with careful parameter tuning)
Spatial data (geographic clustering)
Outlier detection

Example success: Two moons dataset
Remember K-means failed here! DBSCAN succeeds:

Each moon is a dense region
Gap between moons has low density
DBSCAN correctly separates them!

DBSCAN Challenges
Challenge 1: Choosing ε and minPts
No automatic way to choose!
Heuristics:

Plot k-distance graph (distance to k-th nearest neighbor)
Look for "elbow" in plot
Domain knowledge

Challenge 2: Varying densities
If clusters have very different densities, single ε doesn't work!
Solution: Use HDBSCAN (Hierarchical DBSCAN)

Builds hierarchy of clusters at different densities
Automatically selects best density for each cluster

Challenge 3: High dimensions
Distance becomes less meaningful in high dimensions (curse of dimensionality)

Comparison with K-Means
K-Means: Fast, simple, but limited
DBSCAN: Flexible, but parameter-sensitive
Use K-Means when:

Large dataset
Spherical clusters
Know approximate k
Need speed

Use DBSCAN when:

Arbitrary shapes
Don't know k
Want outlier detection
Small/medium dataset

Gaussian Mixture Models (GMM)
The Probabilistic Approach
K-means says: Each point belongs to one cluster
GMM says: Each point has a probability of belonging to each cluster!
Model: Data generated from mixture of Gaussian distributions
Each cluster:

Center: μⱼ
Shape/spread: Covariance matrix Σⱼ
Weight: πⱼ (proportion of data from this cluster)

The Generative Story
GMM assumes this process generated the data:
For each point:

Choose cluster j with probability πⱼ
Generate point from Gaussian(μⱼ, Σⱼ)

Our task: Reverse engineer the parameters (μⱼ, Σⱼ, πⱼ) from observed data!
EM Algorithm for GMM
Problem: Don't know which cluster generated each point!
Solution: Expectation-Maximization (EM) algorithm
E-step (Expectation):
For each point xᵢ, calculate probability it came from each cluster j:
γᵢⱼ = P(cluster j | xᵢ)
M-step (Maximization):
Update parameters using these probabilities:

μⱼ = weighted mean (weights = γᵢⱼ)
Σⱼ = weighted covariance
πⱼ = average γᵢⱼ

Iterate until convergence!
This is exactly like K-means, but with soft assignments (γᵢⱼ) instead of hard assignments!
Advantages Over K-Means
1. Soft clustering:
Points can partially belong to multiple clusters
2. Cluster shape:
Can be elliptical (via covariance matrix), not just spherical!
3. Probabilistic:
Get probability distributions, not just assignments
4. Model selection:
Can use BIC/AIC to choose number of clusters!
5. Confidence:
Can quantify uncertainty in assignments
When to Use GMM
Use GMM when:

Clusters have different shapes/sizes
Need probabilities, not just assignments
Want statistical model
Moderate dataset size

Use K-means when:

Very large dataset
Simpler model preferred
Spherical clusters OK
Need speed

GMM is more powerful but slower and more complex!
Practice Problems - Advanced Topics
Problem 7.1: Identifying Limitations
For each dataset, explain why K-means would fail:
a) Two concentric circles
b) Three clusters: one with 1000 points, two with 50 points each
c) Crescent moon shapes
d) Data with 10% outliers
Problem 7.2: Method Selection
Choose the best clustering method (K-means, Hierarchical, DBSCAN, GMM):
a) 1 million customer records, want 5 segments
b) 100 proteins, want to explore hierarchy
c) Geographic data with irregular boundaries
d) 10,000 patients, clusters may overlap
Problem 7.3: DBSCAN Parameters
Points in 2D:

Dense region A: 20 points in 1×1 square
Dense region B: 30 points in 1×1 square
5 scattered outliers

What ε and minPts would work?
Problem 7.4: Hierarchical Clustering
Given dendrogram, cutting at different heights gives:

Height 10: 2 clusters
Height 5: 4 clusters
Height 2: 8 clusters

a) Which cut to choose for market segmentation?
b) How to validate the choice?
c) Could you use multiple cuts? How?
Problem 7.5: GMM vs K-Means
Dataset has 3 clusters:

Cluster 1: Spherical, 100 points
Cluster 2: Elongated (elliptical), 100 points
Cluster 3: Spherical but small, 100 points

a) How would K-means handle this?
b) How would GMM handle this?
c) Which is better? Why?

<a name="summary"></a>
Chapter 4 Summary
The Big Picture
Clustering is unsupervised learning: finding structure in unlabeled data!
Core problem: Group similar data points together
Solution: K-means algorithm (and variants)
Key Concepts
1. What Is Clustering?

Goal: Partition n points into k groups
Similarity: Measured by distance (small = similar)
Applications: Customer segmentation, image compression, anomaly detection, genomics, social networks, fraud detection

2. The Clustering Objective
Within-Cluster Sum of Squares (WCSS):
WCSS = Σⱼ Σᵢ in cluster j ||xᵢ - μⱼ||²
Minimize WCSS = Find compact clusters!
Properties:

Always ≥ 0
Decreases with more clusters
Trade-off: too many clusters not useful

3. K-Means Algorithm
Iterative approach:

Initialize k centers
Assign points to nearest center
Update centers (compute means)
Repeat until convergence

Guarantees:

Always converges
Finds local minimum
Each step decreases WCSS

Complexity: O(iterations × n × k × d)
Limitations:

Needs k specified
Sensitive to initialization
Assumes spherical clusters
Finds local minima

4. Choosing K
Elbow Method:

Plot WCSS vs k
Look for "elbow" (diminishing returns)
Subjective but intuitive

Silhouette Score:

Measures cohesion + separation
Range [-1, 1], higher better
More rigorous than elbow

Gap Statistic:

Compare to random data
Statistical test
Most principled

Domain Knowledge:

Consider business constraints
Operational feasibility
Often decisive!

5. Applications
Customer Segmentation:

Features: purchases, spending, recency
Outcome: Targeted marketing campaigns

Image Compression:

Cluster pixels by color
Replace with cluster centers
Massive compression!

Document Clustering:

Features: TF-IDF vectors
Outcome: Automatic topic organization

Anomaly Detection:

Cluster normal patterns
Points far from all clusters = anomalies
Network security, fraud detection

Medical Research:

Cluster patients by gene expression
Discover disease subtypes
Personalized treatment

6. Advanced Topics
Variants:

Mini-batch K-means (speed)
Fuzzy C-means (soft clustering)
K-medoids (robustness)

Alternatives:

Hierarchical clustering (explores hierarchy)
DBSCAN (arbitrary shapes, finds outliers)
GMM (probabilistic, flexible shapes)

Choosing method:

K-means: fast, simple, large data
Hierarchical: small data, explore structure
DBSCAN: arbitrary shapes, outliers
GMM: probabilistic, flexible

Connecting to Previous Chapters
Chapter 1 (Vectors):

Data points are vectors
Cluster centers are linear combinations (means!)

Chapter 2 (Linear Functions, Regression):

Clustering finds structure
Regression predicts values
Both learn from data!

Chapter 3 (Norm, Distance):

Distance measures similarity
Norm measures spread
Foundation of clustering!

Chapter 4 (Clustering):

Brings everything together!
Unsupervised learning
Finding natural structure in data

Key Formulas
Centroid:
μⱼ = (1/nⱼ) Σᵢ in cluster j xᵢ
WCSS:
WCSS = Σⱼ Σᵢ in cluster j ||xᵢ - μⱼ||²
Assignment:
cᵢ = argmin_j ||xᵢ - μⱼ||²
Silhouette:
s(i) = [b(i) - a(i)] / max{a(i), b(i)}
Common Pitfalls
1. Not standardizing features

Features with different scales dominate distance
Always standardize!

2. Choosing k arbitrarily

Use elbow, silhouette, gap statistic
Consider domain knowledge

3. Running K-means once

Can get stuck in local minimum
Run multiple times, keep best!

4. Expecting global optimum

K-means finds local minimum
That's OK! Still useful!

5. Using K-means for all data

Not all data is clusterable
Check silhouette scores
Consider if clustering appropriate

Practical Tips
1. Always preprocess:

Remove outliers (or use robust method)
Standardize features
Handle missing values

2. Use K-means++:

Much better than random initialization
Default in most libraries

3. Run multiple times:

Try 10-20 different initializations
Keep result with lowest WCSS

4. Validate results:

Calculate silhouette scores
Visualize (PCA for high-D data)
Check interpretability

5. Iterate:

Try different k values
Compare methods
Refine based on domain feedback

What's Next?
Chapter 5 will cover:

Linear independence
Basis and dimension
Orthogonality
Gram-Schmidt process

These concepts enable:

Understanding vector spaces deeply
Principal Component Analysis (PCA)
Dimensionality reduction
Advanced ML techniques

The journey continues!

<a name="practice"></a>
Comprehensive Practice Problems
Section 1: Core Concepts
Problem 8.1: Understanding WCSS
Six points: (0,0), (1,0), (0,1), (10,10), (11,10), (10,11)
a) Intuitively, how many clusters?
b) Calculate WCSS for k=1 (all one cluster)
c) Calculate WCSS for k=2 with assignment: {(0,0), (1,0), (0,1)} and {(10,10), (11,10), (10,11)}
d) Calculate WCSS for k=6 (each point its own cluster)
e) Verify WCSS decreases as k increases
Problem 8.2: Centroid Calculation
Cluster contains: (2,3,1), (4,1,3), (0,5,1), (2,1,5)
a) Calculate centroid by averaging each dimension
b) This is a linear combination with what weights?
c) Verify centroid minimizes sum of squared distances
d) What if cluster had (2,3,1) twice? How does centroid change?
Problem 8.3: Assignment Step
Points: A(1,1), B(5,5), C(9,9)
Centers: μ₁(0,0), μ₂(10,10)
a) Calculate distance from A to each center
b) Calculate distance from B to each center
c) Calculate distance from C to each center
d) Assign each point to nearest center
e) Which point is ambiguous (close to both)?
Problem 8.4: Convergence Check
Iteration 1: Centers at (2,2) and (8,8), Assignments: WCSS = 50
Iteration 2: Centers at (2.5,2.5) and (7.5,7.5), Assignments unchanged, WCSS = 48
Iteration 3: Centers at (2.5,2.5) and (7.5,7.5), Assignments unchanged, WCSS = 48
a) Did assignments change from iteration 1 to 2?
b) Did centers change from iteration 2 to 3?
c) Did WCSS change from iteration 2 to 3?
d) Has algorithm converged? Why or why not?
e) Should you continue iterating?
Problem 8.5: Why Squared Distance?
Point at (3,4), center at (0,0)
a) Calculate distance: ||x - μ||
b) Calculate squared distance: ||x - μ||²
c) Why does K-means use squared distance?
d) Would K-means work with absolute distance |x-μ|? What would change?
Section 2: Running K-Means
Problem 8.6: Complete K-Means Execution
Points: (1,1), (1,2), (2,1), (8,8), (8,9), (9,8)
k=2, initial centers: μ₁(1,1), μ₂(9,8)
a) Iteration 1: Assign each point
b) Iteration 1: Update centers
c) Calculate WCSS after iteration 1
d) Iteration 2: Reassign points
e) Iteration 2: Update centers
f) Has it converged? If not, continue until convergence
g) Final WCSS
Problem 8.7: Effect of Initialization
Same points as Problem 8.6.
New initialization: μ₁(1,1), μ₂(2,1)
a) Run K-means with this initialization
b) Compare final result to Problem 8.6
c) Compare final WCSS values
d) Are they the same? Why or why not?
e) What does this tell you about initialization?
Problem 8.8: Standardization Impact
Two features: Age (20-80) and Income ($20k-$200k)
Points: A(25, $30k), B(30, $180k), C(75, $40k)
a) Calculate distance between A and B without standardization
b) Calculate distance between A and C without standardization
c) Which is "closer"?
d) Standardize features (z-score)
e) Recalculate distances after standardization
f) Which is closer now? Why did it change?
Problem 8.9: Empty Cluster Problem
Points: (0,0), (0,1), (1,0), (1,1), (20,20)
k=3, initial centers: (0,0), (1,1), (30,30)
a) First iteration: Assign points
b) Which cluster is empty?
c) How would you handle this?
d) Suggest new center for empty cluster
e) Continue K-means with your fix
Problem 8.10: Tie Breaking
Point at (5,5), centers at (2,2) and (8,8)
a) Calculate distance to each center
b) Are distances equal?
c) How do implementations typically break ties?
d) Does it matter which cluster you choose?
Section 3: Choosing K
Problem 8.11: Elbow Method
WCSS values:
k=1: 1000
k=2: 400
k=3: 200
k=4: 140
k=5: 120
k=6: 110
k=7: 105
k=8: 102
a) Plot WCSS vs k (sketch)
b) Calculate % decrease for each k
c) Where is the elbow?
d) What k would you choose?
e) Is the elbow clear or ambiguous?
Problem 8.12: Silhouette Calculation
Point A in cluster 1:

Average distance to other points in cluster 1: a(A) = 2.0
Average distance to points in cluster 2: 6.0
Average distance to points in cluster 3: 8.0

a) What is b(A) (distance to nearest other cluster)?
b) Calculate silhouette coefficient s(A)
c) Is A well-clustered?
d) If a(A) = 5.0 and b(A) = 5.5, calculate new s(A)
e) What if a(A) = 7.0 and b(A) = 4.0?
Problem 8.13: Multiple Metrics
For k=3, 4, 5:

k=3: WCSS=150, Silhouette=0.65, Gap=0.8
k=4: WCSS=110, Silhouette=0.72, Gap=1.1
k=5: WCSS=90, Silhouette=0.68, Gap=0.9

a) Which k does WCSS prefer? (Not applicable - always prefer higher!)
b) Which k does Silhouette prefer?
c) Which k does Gap prefer?
d) Do they agree?
e) Which k would you choose? Why?
Problem 8.14: Business Constraints
Elbow at k=7, but marketing team can only handle 3 campaigns.
a) What k should you use?
b) How do you justify this to technical team?
c) Could you provide both k=3 and k=7 results?
d) What's the trade-off?
Problem 8.15: Hierarchical K
Data has hierarchical structure:

3 major types
Each type has 3 subtypes

a) What k for major types?
b) What k for subtypes?
c) How could you model both levels?
d) Which is more actionable?
Section 4: Applications
Problem 8.16: Customer Segmentation Design
E-commerce company has:

Purchase frequency
Average order value
Recency (days since last purchase)
Total lifetime value
Return rate

a) Which features would you use?
b) Should you include all? Why or why not?
c) How would you standardize?
d) What k would you expect?
e) How would you name/describe segments?
Problem 8.17: Image Compression Trade-offs
Image: 100×100 pixels, RGB (3 bytes each)
Original size: 30,000 bytes
a) With k=16 colors, calculate compressed size
b) With k=256 colors, calculate compressed size
c) Calculate compression ratios
d) Trade-off between k and quality?
e) How to choose optimal k?
Problem 8.18: Document Clustering Evaluation
Clustered 100 news articles into 5 topics:

Cluster 1 (25): Politics articles
Cluster 2 (30): Sports articles
Cluster 3 (20): Technology articles
Cluster 4 (15): Entertainment articles
Cluster 5 (10): Mixed articles

a) Are clusters good? How can you tell?
b) What does Cluster 5 suggest?
c) Should it be k=4 instead?
d) How to evaluate without labels?
Problem 8.19: Anomaly Detection Threshold
Normal patterns cluster into 3 groups.
New transactions:

Transaction A: distance 0.5 from nearest cluster
Transaction B: distance 2.0 from nearest cluster
Transaction C: distance 5.0 from nearest cluster

a) If threshold is 3.0, which are anomalies?
b) How do you choose the threshold?
c) Trade-off between false positives and false negatives?
d) Should threshold be same for all clusters?
Problem 8.20: Cold Start Problem
Recommendation system clusters users by viewing history.
New user: no viewing history yet!
a) Why can't you assign to cluster?
b) How could you handle this?
c) Could you use demographic features?
d) When can you start using collaborative filtering?
Section 5: Advanced Topics
Problem 8.21: Identifying Failure Cases
For each scenario, explain why K-means fails:
a) Dataset: Two spiral patterns
b) Dataset: One large cluster (1000 points), one small (50 points)
c) Dataset: Clusters of very different densities
d) Dataset: 30% outliers scattered randomly
Problem 8.22: Method Selection Decision Tree
For each dataset, choose best method (K-means, Hierarchical, DBSCAN, GMM):
a) 10 million customers, want 5 segments, fast execution needed
b) 500 genes, want to explore relationships, no time pressure
c) Geographic crime data, irregular boundaries, outliers present
d) 50,000 transactions, clusters may overlap, need probabilities
Problem 8.23: DBSCAN Parameter Intuition
Dense region: 100 points in 2×2 square
Sparse region: 10 points in 10×10 square
Outliers: 5 random scattered points
a) What ε would capture dense region?
b) What minPts would work?
c) Would these parameters capture sparse region too?
d) What's the fundamental limitation?
Problem 8.24: Hierarchical Clustering Decision
Dendrogram shows:

Clear split into 2 clusters at height 10
Each splits into 2 (total 4) at height 5
Further splits to 8 at height 2

a) For C-suite presentation, which cut?
b) For operational teams, which cut?
c) For detailed analysis, which cut?
d) How to present all three views?
Problem 8.25: Mini-Batch K-Means
Dataset: 1 million points
Standard K-means: 10 hours
Mini-batch K-means with batch size 1000: 30 minutes
a) Why is mini-batch so much faster?
b) Would results be identical?
c) How much worse is quality typically?
d) When is speed/quality trade-off worth it?
Section 6: Integration and Synthesis
Problem 8.26: End-to-End Pipeline
Task: Segment 10,000 retail customers
a) Design feature set
b) Preprocessing steps?
c) How to choose k?
d) How to validate clustering?
e) How to present to business team?
f) How to maintain over time?
Problem 8.27: Comparing Clusterings
Two analysts cluster same data:

Analyst A: k=3, Silhouette=0.70, Interpretable clusters
Analyst B: k=5, Silhouette=0.75, Less clear interpretation

a) Which is "better" mathematically?
b) Which is more useful?
c) How to decide?
d) Could both be valuable?
Problem 8.28: Temporal Clustering
Customer behavior changes over time.

January clusters: {A, B, C}
June clusters: {D, E, F}

a) Why did clusters change?
b) Should you retrain monthly?
c) How to track cluster evolution?
d) How to handle customers switching clusters?
Problem 8.29: Multi-View Clustering
Customers described by:

Demographics (age, income, location)
Behavior (purchases, browsing, clicks)
Preferences (survey responses)

a) Should you cluster each view separately?
b) Should you combine all features?
c) What's best approach?
d) How to integrate multiple clusterings?
Problem 8.30: Evaluation Without Ground Truth
You've clustered data but have no labels to check against.
a) How do you know if clustering is good?
b) What metrics can you use?
c) What qualitative checks?
d) How to get stakeholder feedback?
e) When to declare clustering "successful"?

Answer Key (Selected Problems)
Problem 8.1:
b) k=1: μ=(5.5, 5.5), WCSS ≈ 221
c) k=2: μ₁=(0.33, 0.33), μ₂=(10.33, 10.33), WCSS ≈ 4
d) k=6: WCSS = 0 (each point is its own cluster)
e) WCSS: 221 → 4 → 0 ✓
Problem 8.6:
Final clustering:

Cluster 1: {(1,1), (1,2), (2,1)}
Cluster 2: {(8,8), (8,9), (9,8)}
Converges in 1-2 iterations
Final WCSS ≈ 4

Problem 8.11:
c) Elbow around k=4 or k=5
d) Choose k=4 (good balance)
% decreases: 60%, 50%, 30%, 14%, 8%, 4.5%, 2.9%
Problem 8.13:
b) Silhouette prefers k=4 (0.72)
c) Gap prefers k=4 (1.1)
d) Yes, both suggest k=4!
e) Choose k=4 (consensus)
Problem 8.22:
a) K-means (scalability)
b) Hierarchical (exploration)
c) DBSCAN (irregular shapes)
d) GMM (probabilities)

Congratulations!
You've completed Chapter 4! You now understand:

✅ What clustering is and why it matters
✅ The K-means algorithm and how it works
✅ How to choose the number of clusters (k)
✅ Real-world applications across many domains
✅ Advanced methods and when to use them
✅ How to evaluate and validate clustering results

You're ready for advanced machine learning topics!
Next steps:

Apply clustering to your own data
Experiment with different methods
Learn dimensionality reduction (PCA)
Explore deep learning


End of Chapter 4
