# Linear Algebra for Machine Learning
## Chapter 3: Norm, Distance, Standard Deviation, and Angles

### A First-Principles Approach with Detailed Examples

---

# Table of Contents

1. [Norm (Vector Length/Magnitude)](#norm)
2. [Distance Between Vectors](#distance)
3. [Standard Deviation and Spread](#standard-deviation)
4. [Angle and Cosine Similarity](#angle)
5. [Complexity Analysis](#complexity)
6. [Chapter Summary](#summary)
7. [Comprehensive Practice Problems](#practice)

---

<a name="norm"></a>
# 1. Norm (Vector Length/Magnitude)

## The Fundamental Question: How "Big" Is a Vector?

Imagine you're hiking and your GPS shows your displacement from the starting point:
- 3 km east
- 4 km north

Your displacement vector: **v** = (3, 4)

**Question:** How far did you travel from your starting point in a straight line?

You didn't walk 3 + 4 = 7 km (that's the Manhattan distance, walking along streets).

You want the **straight-line distance** - the length of the arrow from start to your current position.

## Building the Solution: Pythagorean Theorem

Remember the Pythagorean theorem from geometry?

For a right triangle with sides a and b, and hypotenuse c:
c² = a² + b²

**Your displacement forms a right triangle!**
- Horizontal side: 3 km
- Vertical side: 4 km
- Hypotenuse: your straight-line distance

Distance = √(3² + 4²) = √(9 + 16) = √25 = 5 km

**This distance is the NORM (or length, or magnitude) of the vector!**

## Formal Definition

The **norm** (or **length** or **magnitude**) of a vector **v** = (v₁, v₂, ..., vₙ) is:

||**v**|| = √(v₁² + v₂² + ... + vₙ²)

**Notation:** 
- ||**v**|| means "the norm of v"
- Also written as |**v**| or ‖**v**‖
- Sometimes called L2 norm or Euclidean norm

**Key insight:** The norm is always a **non-negative number** (scalar), not a vector!

## Why Square and Then Square Root?

**Question:** Why this specific formula? Why not just add components or add absolute values?

### Bad Idea 1: Just Add Components

||**v**|| = v₁ + v₂ + ... + vₙ

**Problem:** What if components are negative?

**v** = (3, -4)
||**v**|| = 3 + (-4) = -1 (Negative length?! Makes no sense!)

### Bad Idea 2: Add Absolute Values (L1 Norm)

||**v**||₁ = |v₁| + |v₂| + ... + |vₙ|

**v** = (3, 4)
||**v**||₁ = |3| + |4| = 7

This actually works and is called the **L1 norm** or **Manhattan distance**!

**But:** This gives you the "city block" distance (walking on a grid), not straight-line distance.

**Think about it:** To go from (0,0) to (3,4), you walk 3 blocks east + 4 blocks north = 7 blocks total.

### Why Squares Work Best (L2 Norm / Euclidean Norm)

||**v**||₂ = √(v₁² + v₂² + ... + vₙ²)

**Advantages:**
1. **Squaring makes everything positive:** Both 3² and (-3)² give 9
2. **Geometric meaning:** Matches Pythagorean theorem (straight-line distance)
3. **Smooth and differentiable:** Important for optimization (gradients exist everywhere)
4. **Penalizes large values more:** Errors of 2 count as 4, errors of 10 count as 100
5. **Nice mathematical properties:** Works beautifully with inner products

**This is the "standard" norm in ML** unless specified otherwise!

## Connection to Inner Product

**Beautiful relationship:** The norm is the square root of the inner product of a vector with itself!

||**v**||² = **v · v** = v₁² + v₂² + ... + vₙ²

**Therefore:**
||**v**|| = √(**v · v**)

**This connects everything we've learned!**

## Examples in Different Dimensions

### Example 1.1: 1D Vectors (Just Numbers)

**v** = (5)
||**v**|| = √(5²) = √25 = 5

**v** = (-3)
||**v**|| = √((-3)²) = √9 = 3

**Key point:** Length is always positive, even for negative numbers!

### Example 1.2: 2D Vectors

**v** = (3, 4)
||**v**|| = √(3² + 4²) = √(9 + 16) = √25 = 5

**v** = (-1, -1)
||**v**|| = √((-1)² + (-1)²) = √(1 + 1) = √2 ≈ 1.414

**v** = (0, 5)
||**v**|| = √(0² + 5²) = √25 = 5 (pointing straight up)

**v** = (5, 0)
||**v**|| = √(5² + 0²) = 5 (pointing straight right)

### Example 1.3: 3D Vectors

**v** = (1, 2, 2)
||**v**|| = √(1² + 2² + 2²) = √(1 + 4 + 4) = √9 = 3

**v** = (2, -3, 6)
||**v**|| = √(4 + 9 + 36) = √49 = 7

**Think about it:** This is the distance from the origin (0, 0, 0) to the point (2, -3, 6) in 3D space!

### Example 1.4: High-Dimensional Vectors

**v** = (1, 1, 1, 1, 1) ∈ ℝ⁵
||**v**|| = √(1 + 1 + 1 + 1 + 1) = √5 ≈ 2.236

**v** = (15, 3, 4, 1, 1) (our spam email features!)
||**v**|| = √(15² + 3² + 4² + 1² + 1²)
= √(225 + 9 + 16 + 1 + 1)
= √252 ≈ 15.87

**Even though we can't visualize 5D space, the math works the same!**

## Properties of the Norm

These properties define what makes something a "norm":

### Property 1: Non-negativity
||**v**|| ≥ 0 for all **v**

**And:** ||**v**|| = 0 **if and only if** **v** = **0**

**Interpretation:** Length is never negative, and only the zero vector has zero length.

### Property 2: Homogeneity (Scaling)
||α**v**|| = |α| · ||**v**|| for any scalar α

**Example:**
**v** = (3, 4), ||**v**|| = 5

**2v** = (6, 8)
||**2v**|| = √(36 + 64) = √100 = 10 = 2 · 5 ✓

**-v** = (-3, -4)
||-**v**|| = √(9 + 16) = 5 = |-1| · 5 ✓

**Interpretation:** If you double the vector, you double its length. If you flip direction (multiply by -1), length stays the same.

### Property 3: Triangle Inequality
||**u + v**|| ≤ ||**u**|| + ||**v**||

**Interpretation:** The direct path is never longer than taking a detour!

**Think about it:** 
- Walk from A to B directly: ||**u + v**||
- Walk from A to C to B: ||**u**|| + ||**v**||
- Direct is always shorter or equal!

**Example:**
**u** = (1, 0), **v** = (0, 1)

||**u**|| = 1
||**v**|| = 1
||**u + v**|| = ||(1, 1)|| = √2 ≈ 1.414

Check: 1.414 ≤ 1 + 1 = 2 ✓

### Property 4: Relationship with Inner Product (Cauchy-Schwarz)
|**u · v**| ≤ ||**u**|| · ||**v**||

**This will be crucial when we discuss angles!**

## Unit Vectors: Vectors with Length 1

A **unit vector** is a vector with norm exactly equal to 1.

**To create a unit vector from any non-zero vector:**
**v̂** = **v** / ||**v**||

(Read as "v-hat" - the hat notation means "unit vector")

**This process is called normalization.**

### Example 1.5: Normalizing Vectors

**v** = (3, 4)
||**v**|| = 5

**v̂** = (3, 4) / 5 = (3/5, 4/5) = (0.6, 0.8)

**Verify:**
||**v̂**|| = √(0.6² + 0.8²) = √(0.36 + 0.64) = √1 = 1 ✓

**Key insight:** **v̂** points in the **same direction** as **v**, but has **length 1**!

### Example 1.6: Standard Basis Vectors Are Unit Vectors

**e₁** = (1, 0, 0)
||**e₁**|| = √(1² + 0² + 0²) = 1 ✓

**e₂** = (0, 1, 0)
||**e₂**|| = 1 ✓

**e₃** = (0, 0, 1)
||**e₃**|| = 1 ✓

**All standard basis vectors are unit vectors!**

## Why Machine Learning Cares About Norms

Norms are **everywhere** in ML!

### 1. Regularization: Preventing Overfitting

**Problem:** Large weights can cause models to overfit (memorize training data instead of learning patterns).

**Solution:** Add penalty for large weights!

**Ridge Regression (L2 regularization):**
Loss = MSE + λ||**w**||²

Where:
- MSE = prediction error
- ||**w**||² = sum of squared weights
- λ = regularization strength

**Effect:** Model prefers smaller weights → simpler, more generalizable models

**Lasso Regression (L1 regularization):**
Loss = MSE + λ||**w**||₁

Where ||**w**||₁ = sum of absolute values of weights

**Effect:** Can force some weights to exactly zero → automatic feature selection!

### 2. Normalization: Standardizing Inputs

**Problem:** Features with different scales can dominate learning.

**Example:**
- Feature 1: Income ($20k - $200k)
- Feature 2: Age (20 - 65 years)

Income dominates just because numbers are bigger!

**Solution:** Normalize feature vectors to have same scale

**Unit normalization:**
**x_normalized** = **x** / ||**x**||

Now all feature vectors have length 1!

### 3. Gradient Clipping: Stabilizing Training

**Problem:** Sometimes gradients become very large (exploding gradients in deep learning).

**Solution:** If ||∇L|| > threshold, scale it down!

if ||∇L|| > max_norm:
    ∇L = ∇L · (max_norm / ||∇L||)

**Effect:** Gradient direction preserved, but magnitude limited.

### 4. Measuring Prediction Confidence

In neural networks, the norm of output vectors can indicate confidence.

**Example:** Image classification
- Output: **y** = (0.1, 0.1, 0.7, 0.1) (probabilities for 4 classes)
- ||**y**|| = √(0.01 + 0.01 + 0.49 + 0.01) = √0.52 ≈ 0.72

Large norm → confident prediction
Small norm → uncertain

### 5. Batch Normalization

Normalize activations in neural networks:
**a_normalized** = (**a** - mean) / std

Helps training converge faster and more stably!

## Norm Squared: A Useful Shortcut

Often in ML, we use **||v||²** instead of ||**v**| because:

**Advantages:**
1. **No square root needed** → computationally faster
2. **Easier to differentiate** → simpler gradients
3. **Still monotonic** → bigger ||**v**|| means bigger ||**v**||²

||**v**||² = **v · v** = v₁² + v₂² + ... + vₙ²

**When comparing lengths, ||v||² works just as well as ||v||!**

### Example 1.7: Comparing Distances

Which is closer to origin: **u** = (3, 4) or **v** = (2, 5)?

**Method 1: Using norm**
||**u**|| = √25 = 5
||**v**|| = √29 ≈ 5.39
**u** is closer

**Method 2: Using squared norm (faster!)**
||**u**||² = 25
||**v**||² = 29
**u** is closer (same answer, no square roots!)

## Different Types of Norms

### L0 "Norm" (Not Really a Norm)
||**v**||₀ = number of non-zero components

**v** = (0, 3, 0, 5, 0)
||**v**||₀ = 2 (two non-zero entries)

**Use:** Counting sparsity (how many features are active)

### L1 Norm (Manhattan Distance)
||**v**||₁ = |v₁| + |v₂| + ... + |vₙ|

**v** = (3, -4)
||**v**||₁ = |3| + |-4| = 3 + 4 = 7

**Use:** Lasso regularization, robust to outliers

### L2 Norm (Euclidean Distance) - The Standard!
||**v**||₂ = √(v₁² + v₂² + ... + vₙ²)

**This is what we mean by "norm" unless specified otherwise!**

### L∞ Norm (Maximum Norm)
||**v**||∞ = max(|v₁|, |v₂|, ..., |vₙ|)

**v** = (3, -7, 2)
||**v**||∞ = max(3, 7, 2) = 7

**Use:** Measuring worst-case deviation

### Comparison Example

**v** = (3, -4, 0)

||**v**||₀ = 2 (two non-zero)
||**v**||₁ = 3 + 4 + 0 = 7
||**v**||₂ = √(9 + 16 + 0) = √25 = 5
||**v**||∞ = max(3, 4, 0) = 4

**General relationship:** ||**v**||∞ ≤ ||**v**||₂ ≤ ||**v**||₁

## Detailed Worked Examples

### Example 1.8: GPS Navigation

Your hiking trail:
- Start: Origin (0, 0)
- Checkpoint 1: (3, 4) km
- Checkpoint 2: (8, 6) km
- End: (10, 2) km

**Calculate distances from origin:**

To Checkpoint 1:
**v₁** = (3, 4)
||**v₁**|| = √(9 + 16) = 5 km

To Checkpoint 2:
**v₂** = (8, 6)
||**v₂**|| = √(64 + 36) = √100 = 10 km

To End:
**v₃** = (10, 2)
||**v₃**|| = √(100 + 4) = √104 ≈ 10.2 km

**Which checkpoint is furthest?** Checkpoint 2 (10 km)

### Example 1.9: Feature Vector Magnitude

Customer behavior vector:
**c** = (purchases, avg_spend, days_active, reviews)
= (23, 150, 365, 12)

||**c**|| = √(23² + 150² + 365² + 12²)
= √(529 + 22500 + 133225 + 144)
= √156398 ≈ 395.47

**Interpretation:** This is the "magnitude" of customer engagement.

**Compare two customers:**
- Customer A: (23, 150, 365, 12), ||**cₐ**|| ≈ 395.47
- Customer B: (50, 200, 730, 25), ||**cᵦ**|| ≈ 762.76

Customer B has higher engagement magnitude!

### Example 1.10: Neural Network Weight Initialization

Initialize weights with small random values:

**w** = (0.01, -0.02, 0.015, -0.008, 0.012)

||**w**|| = √(0.0001 + 0.0004 + 0.000225 + 0.000064 + 0.000144)
= √0.001033 ≈ 0.032

**Check:** ||**w**|| < 0.1 ✓ (good initialization - small weights)

### Example 1.11: Normalizing Image Pixels

Image pixel vector (simplified, 3 pixels):
**img** = (128, 200, 64) (pixel brightness 0-255)

||**img**|| = √(16384 + 40000 + 4096) = √60480 ≈ 245.93

**Normalized:**
**img_normalized** = **img** / ||**img**||
= (128, 200, 64) / 245.93
= (0.520, 0.813, 0.260)

Now ||**img_normalized**|| = 1 ✓

### Example 1.12: Gradient Magnitude

Loss gradient: ∇L = (2.5, -1.8, 3.2, -0.9)

||∇L|| = √(6.25 + 3.24 + 10.24 + 0.81)
= √20.54 ≈ 4.53

**If gradient is too large (>10), clip it:**

Since 4.53 < 10, no clipping needed.

But if ||∇L|| = 15, then:
∇L_clipped = ∇L · (10 / 15) = 0.667 · ∇L

### Example 1.13: Portfolio Volatility

Stock returns vector (5 days):
**r** = (0.02, -0.01, 0.03, -0.02, 0.01) (daily returns as decimals)

||**r**|| = √(0.0004 + 0.0001 + 0.0009 + 0.0004 + 0.0001)
= √0.0019 ≈ 0.0436

**Interpretation:** This measures the magnitude of price movements (volatility indicator).

## Practice Problems - Norm

**Problem 1.1: Basic Norm Calculations**

Calculate ||**v**|| for:
a) **v** = (5, 12)
b) **v** = (-3, 4)
c) **v** = (1, 1, 1)
d) **v** = (2, -2, 1, -1)
e) **v** = (0, 0, 0)

**Problem 1.2: Pythagorean Triples**

Verify these are Pythagorean triples by calculating norms:
a) (3, 4) should have norm 5
b) (5, 12) should have norm 13
c) (8, 15) should have norm 17
d) (7, 24) should have norm 25

**Problem 1.3: Normalization**

Normalize these vectors (find unit vectors):
a) **v** = (3, 4)
b) **v** = (1, 1)
c) **v** = (0, 5)
d) **v** = (2, -2, 1)

Verify each normalized vector has norm 1.

**Problem 1.4: Comparing Magnitudes**

Which vector has larger norm?
a) **u** = (3, 4) vs **v** = (5, 2)
b) **u** = (1, 1, 1, 1) vs **v** = (2, 0, 0, 0)
c) **u** = (10, 1) vs **v** = (1, 10)

**Problem 1.5: Properties Verification**

For **u** = (3, 4) and scalar α = 2:
a) Calculate ||**u**||
b) Calculate ||α**u**||
c) Verify ||α**u**|| = |α| · ||**u**||
d) What if α = -3? Verify the property still holds.

**Problem 1.6: Triangle Inequality**

For **u** = (1, 2) and **v** = (3, 1):
a) Calculate ||**u**||, ||**v**||, and ||**u + v**||
b) Verify ||**u + v**|| ≤ ||**u**|| + ||**v**||
c) When does equality hold in triangle inequality?

**Problem 1.7: Different Norms**

For **v** = (3, -4, 5):
a) Calculate L1 norm: ||**v**||₁
b) Calculate L2 norm: ||**v**||₂
c) Calculate L∞ norm: ||**v**||∞
d) Which is largest? Why?

**Problem 1.8: Sparse Vectors**

Vector **v** = (0, 5, 0, 0, 3, 0, 0, 7, 0)
a) Calculate ||**v**||₀ (count non-zeros)
b) Calculate ||**v**||₁
c) Calculate ||**v**||₂
d) Why is this vector called "sparse"?

**Problem 1.9: Feature Scaling**

Two features with different scales:
- **f₁** = (1000, 2000, 1500) (income in $)
- **f₂** = (25, 35, 30) (age in years)

a) Calculate ||**f₁**|| and ||**f₂**||
b) Normalize both to unit vectors
c) Now calculate norms. What do you notice?
d) Why is this normalization useful in ML?

**Problem 1.10: Gradient Clipping**

Gradient: ∇L = (8, -6, 12, -4)
Max allowed norm: 10

a) Calculate ||∇L||
b) Is clipping needed?
c) If yes, calculate clipped gradient
d) Verify clipped gradient has norm ≤ 10

---

<a name="distance"></a>
# 2. Distance Between Vectors

## The Core Question: How Far Apart Are Two Things?

You have two movie preference vectors:
- **Alice:** (5, 2, 1, 4) = ratings for (action, comedy, drama, horror)
- **Bob:** (4, 3, 1, 3)

**Question:** How similar are their movie tastes?

To answer this, we need to measure how "far apart" their preference vectors are!

## Building the Solution: The Difference Vector

**First insight:** To measure distance between two points, find how they differ!

**Bob's preferences - Alice's preferences:**
**b - a** = (4, 3, 1, 3) - (5, 2, 1, 4) = (-1, 1, 0, -1)

**This difference vector tells us:**
- Action: Bob rates 1 point lower
- Comedy: Bob rates 1 point higher  
- Drama: Same!
- Horror: Bob rates 1 point lower

**Second insight:** The LENGTH of this difference vector is the distance!

distance(**a**, **b**) = ||**b - a**|| = ||(-1, 1, 0, -1)||
= √(1 + 1 + 0 + 1) = √3 ≈ 1.73

## Formal Definition

The **Euclidean distance** between vectors **u** and **v** is:

d(**u**, **v**) = ||**u - v**|| = √[(u₁-v₁)² + (u₂-v₂)² + ... + (uₙ-vₙ)²]

**Alternative formula using inner product:**
d(**u**, **v**) = √[(**u - v**) · (**u - v**)]

**Key properties:**
- Always non-negative: d(**u**, **v**) ≥ 0
- Zero iff identical: d(**u**, **v**) = 0 ⟺ **u** = **v**
- Symmetric: d(**u**, **v**) = d(**v**, **u**)
- Triangle inequality: d(**u**, **w**) ≤ d(**u**, **v**) + d(**v**, **w**)

## Geometric Interpretation in 2D

Two points in a plane:
- **p₁** = (1, 2)
- **p₂** = (4, 6)

**Visualize:** Plot these points. The distance is the length of the straight line connecting them!

d(**p₁**, **p₂**) = ||**p₂ - p₁**|| = ||(3, 4)|| = 5

**You can literally measure this with a ruler on graph paper!**

## Why This Formula Works

The distance formula comes directly from the Pythagorean theorem!

**Think about moving from **p₁** to **p₂**:**
- Horizontal change: Δx = 4 - 1 = 3
- Vertical change: Δy = 6 - 2 = 4
- These form a right triangle!
- Hypotenuse (distance): √(3² + 4²) = 5

**This extends to any dimension!**

## Properties of Distance

### Property 1: Non-negativity
d(**u**, **v**) ≥ 0

Distance is never negative!

### Property 2: Identity of Indiscernibles  
d(**u**, **v**) = 0 if and only if **u** = **v**

Zero distance means the vectors are identical.

### Property 3: Symmetry
d(**u**, **v**) = d(**v**, **u**)

Distance from A to B equals distance from B to A.

**Proof:**
d(**u**, **v**) = ||**u - v**||
d(**v**, **u**) = ||**v - u**|| = ||**-(u - v)**|| = |-1| · ||**u - v**|| = ||**u - v**|| ✓

### Property 4: Triangle Inequality
d(**u**, **w**) ≤ d(**u**, **v**) + d(**v**, **w**)

**Interpretation:** Going directly from **u** to **w** is never longer than going via **v**!

Think about driving:
- Direct route from home to store
- vs going home → friend's house → store

Direct is always shorter (or equal if friend is on the way)!

## Why Machine Learning Cares About Distance

Distance is **absolutely fundamental** to ML!

### 1. K-Nearest Neighbors (KNN)

**Algorithm:** To classify a new point:
1. Find the K closest training examples (smallest distances)
2. Use their labels to vote

**Example: Email Spam Detection**
- New email: **x** = (5, 20, 1, 50)
- Known spam: **s** = (12, 25, 1, 30)
- Known ham: **h** = (1, 3, 0, 200)

d(**x**, **s**) = ||(5, 20, 1, 50) - (12, 25, 1, 30)||
= ||(-7, -5, 0, 20)||
= √(49 + 25 + 0 + 400) = √474 ≈ 21.8

d(**x**, **h**) = ||(4, 17, 1, -150)||
= √(16 + 289 + 1 + 22500) = √22806 ≈ 151

**x** is MUCH closer to spam example → Classify as spam!

### 2. Clustering (K-Means)

**Goal:** Group similar data points together

**How:** Points are "similar" if distance is small!

**Algorithm:**
1. Assign each point to nearest cluster center
2. Update centers (mean of assigned points)
3. Repeat until convergence

**All based on distance calculations!**

### 3. Anomaly Detection

**Question:** Is this data point unusual?

**Answer:** If it has large distance from all normal examples → Anomaly!

**Example: Fraud Detection**
- Normal transaction: **t_normal** ≈ (50, 1, 10)
- New transaction: **t_new** = (10000, 5, 1000)

d(**t_new**, **t_normal**) = very large → Suspicious!

### 4. Recommendation Systems

**Find users with similar preferences:**

- Your ratings: **you** = (5, 1, 4, 2, 5)
- User A: **a** = (5, 2, 4, 1, 5)
- User B: **b** = (1, 5, 1, 5, 2)

d(**you**, **a**) = small → Similar tastes!
d(**you**, **b**) = large → Different tastes!

**Recommendation:** Show what similar users liked!

### 5. Loss Functions

**Mean Squared Error** is based on distance!

MSE = (1/n) Σᵢ ||**yᵢ** - **ŷᵢ**||²

Average squared distance between predictions and true values!

## Distance vs. Similarity

**Key insight:** Small distance = high similarity!

Often we convert distance to similarity:

**Similarity metrics:**
1. sim = 1 / (1 + distance)
2. sim = e^(-distance)
3. sim = 1 - (distance / max_distance)

**Example:**
- d = 0 → sim = 1 (identical)
- d = 1 → sim = 0.5 (somewhat similar)
- d = ∞ → sim = 0 (completely different)

## Different Distance Metrics

### Euclidean Distance (L2) - The Standard!
d₂(**u**, **v**) = ||**u - v**||₂ = √[Σ(uᵢ - vᵢ)²]

**Use:** General purpose, most common

### Manhattan Distance (L1)
d₁(**u**, **v**) = ||**u - v**||₁ = Σ|uᵢ - vᵢ|

**Use:** When you can only move along axes (like city blocks)

### Chebyshev Distance (L∞)
d∞(**u**, **v**) = ||**u - v**||∞ = max|uᵢ

---

### Chebyshev Distance (L∞)
d∞(**u**, **v**) = ||**u - v**||∞ = max|uᵢ - vᵢ|

**Use:** Measuring worst-case difference

### Comparison Example

**u** = (2, 5, 1)
**v** = (5, 3, 4)

**Difference:** **u - v** = (-3, 2, -3)

**Euclidean (L2):**
d₂ = √(9 + 4 + 9) = √22 ≈ 4.69

**Manhattan (L1):**
d₁ = |−3| + |2| + |−3| = 3 + 2 + 3 = 8

**Chebyshev (L∞):**
d∞ = max(3, 2, 3) = 3

**General relationship:** d∞ ≤ d₂ ≤ d₁

## Squared Distance: A Computational Shortcut

Just like with norms, we often use **squared distance**:

d²(**u**, **v**) = ||**u - v**||²

**Advantages:**
1. No square root → faster computation
2. Preserves ordering (if d₁ < d₂, then d₁² < d₂²)
3. Easier to differentiate

**When comparing distances, squared distance works just as well!**

### Example: Finding Nearest Neighbor

Which is closer to **x** = (0, 0)?
- **a** = (3, 4)
- **b** = (5, 1)

**Method 1: Using distance**
d(**x**, **a**) = √25 = 5
d(**x**, **b**) = √26 ≈ 5.1
**a** is closer

**Method 2: Using squared distance (faster!)**
d²(**x**, **a**) = 25
d²(**x**, **b**) = 26
**a** is closer (same answer, no square roots!)

## Detailed Worked Examples

### Example 2.1: Customer Segmentation

Two customer profiles:
- **Customer A:** (age=25, income=50k, purchases=10, satisfaction=8)
- **Customer B:** (age=28, income=55k, purchases=12, satisfaction=7)

**A** = (25, 50, 10, 8)
**B** = (28, 55, 12, 7)

d(**A**, **B**) = ||(28, 55, 12, 7) - (25, 50, 10, 8)||
= ||(3, 5, 2, -1)||
= √(9 + 25 + 4 + 1)
= √39 ≈ 6.24

**Note:** Different features have different scales! 
- Age differs by 3 years
- Income differs by $5k

**Better approach:** Standardize features first!

After standardization (z-scores):
**A_std** = (0.2, 0.1, 0.3, 0.5)
**B_std** = (0.4, 0.3, 0.5, 0.3)

d(**A_std**, **B_std**) = ||(0.2, 0.2, 0.2, -0.2)||
= √(0.04 + 0.04 + 0.04 + 0.04)
= √0.16 = 0.4

**Much better!** Now all features contribute fairly.

### Example 2.2: Image Similarity

Two 2×2 grayscale images (flattened to vectors):

**Image 1:** [100, 120, 110, 130] (brightness values)
**Image 2:** [105, 125, 115, 135]

d(**img1**, **img2**) = ||(5, 5, 5, 5)||
= √(25 + 25 + 25 + 25)
= √100 = 10

**Small distance → images are similar!**

**Image 3:** [200, 50, 180, 70]

d(**img1**, **img3**) = ||(100, -70, 70, -60)||
= √(10000 + 4900 + 4900 + 3600)
= √23400 ≈ 153

**Large distance → images are very different!**

### Example 2.3: Document Similarity

Word count vectors (vocabulary: ["cat", "dog", "bird", "fish"]):

**Doc 1:** (5, 2, 0, 1) - mentions cats a lot
**Doc 2:** (6, 1, 0, 2) - also about cats
**Doc 3:** (0, 0, 8, 5) - about birds and fish

d(**doc1**, **doc2**) = ||(1, -1, 0, 1)||
= √(1 + 1 + 0 + 1) = √3 ≈ 1.73 (similar!)

d(**doc1**, **doc3**) = ||(-5, -2, 8, 4)||
= √(25 + 4 + 64 + 16) = √109 ≈ 10.44 (very different!)

**Doc 1 and Doc 2 are about similar topics!**

### Example 2.4: Time Series Comparison

Temperature readings over 5 hours:

**City A:** (20, 21, 23, 24, 25)°C
**City B:** (22, 23, 24, 25, 26)°C

d(**A**, **B**) = ||(2, 2, 1, 1, 1)||
= √(4 + 4 + 1 + 1 + 1)
= √11 ≈ 3.32

**City B is consistently ~2°C warmer**

**City C:** (20, 15, 25, 18, 28)°C

d(**A**, **C**) = ||(0, -6, 2, -6, 3)||
= √(0 + 36 + 4 + 36 + 9)
= √85 ≈ 9.22

**City C has more variable weather (larger distance from A)**

### Example 2.5: K-Nearest Neighbors

Training data (2D for visualization):
- Point A: (1, 2), Label: Red
- Point B: (2, 1), Label: Red
- Point C: (5, 6), Label: Blue
- Point D: (6, 5), Label: Blue

New point to classify: **x** = (3, 3)

**Calculate distances:**
d(**x**, **A**) = ||(2, 1)|| = √5 ≈ 2.24
d(**x**, **B**) = ||(1, 2)|| = √5 ≈ 2.24
d(**x**, **C**) = ||(2, 3)|| = √13 ≈ 3.61
d(**x**, **D**) = ||(3, 2)|| = √13 ≈ 3.61

**For K=2 (2 nearest neighbors):**
Nearest: A and B (both Red)
**Prediction: Red!**

### Example 2.6: Anomaly Detection

Normal system metrics:
- **Normal 1:** (CPU=30%, Memory=40%, Disk=50%, Network=20%)
- **Normal 2:** (32%, 38%, 52%, 18%)
- **Normal 3:** (28%, 42%, 48%, 22%)

Average normal: **avg** ≈ (30, 40, 50, 20)

New reading: **new** = (85, 90, 95, 15)

d(**new**, **avg**) = ||(55, 50, 45, -5)||
= √(3025 + 2500 + 2025 + 25)
= √7575 ≈ 87.0

**Very large distance from normal → ANOMALY!**

### Example 2.7: Clustering Quality

Two clusters with centers:
- **C1** = (2, 3)
- **C2** = (8, 7)

Distance between cluster centers:
d(**C1**, **C2**) = ||(6, 4)|| = √52 ≈ 7.21

**Points in Cluster 1:**
- **p1** = (2, 3), d(**p1**, **C1**) = 0
- **p2** = (3, 4), d(**p2**, **C1**) = √2 ≈ 1.41
- **p3** = (1, 2), d(**p3**, **C1**) = √2 ≈ 1.41

**Average distance within cluster:** (0 + 1.41 + 1.41) / 3 ≈ 0.94

**Good clustering:** 
- Small within-cluster distances (0.94)
- Large between-cluster distances (7.21)
- Ratio: 7.21 / 0.94 ≈ 7.7 (well-separated!)

## Practice Problems - Distance

**Problem 2.1: Basic Distance Calculations**

Calculate d(**u**, **v**) for:
a) **u** = (1, 2), **v** = (4, 6)
b) **u** = (0, 0), **v** = (3, 4)
c) **u** = (5, 2, 1), **v** = (1, 2, 5)
d) **u** = (1, 1, 1, 1), **v** = (2, 2, 2, 2)

**Problem 2.2: Distance Properties**

For **u** = (1, 2), **v** = (4, 6), **w** = (7, 10):
a) Calculate d(**u**, **v**)
b) Calculate d(**v**, **u**)
c) Verify d(**u**, **v**) = d(**v**, **u**) (symmetry)
d) Calculate d(**u**, **w**) and d(**v**, **w**)
e) Verify triangle inequality: d(**u**, **w**) ≤ d(**u**, **v**) + d(**v**, **w**)

**Problem 2.3: Different Distance Metrics**

For **u** = (2, 5, 1) and **v** = (5, 3, 4):
a) Calculate Euclidean distance (L2)
b) Calculate Manhattan distance (L1)
c) Calculate Chebyshev distance (L∞)
d) Which is largest? Smallest?
e) When would you use each metric?

**Problem 2.4: Nearest Neighbor**

Given points and a query:
- **a** = (1, 1)
- **b** = (2, 4)
- **c** = (4, 2)
- **query** = (3, 3)

a) Calculate distance from query to each point
b) Which point is nearest?
c) If these have labels: a=Red, b=Blue, c=Red, what's the KNN prediction (K=1)?
d) What if K=3?

**Problem 2.5: Customer Similarity**

Two customers with features (purchases, avg_spend, days_active):
- **Customer A:** (10, 50, 100)
- **Customer B:** (12, 55, 95)
- **Customer C:** (50, 200, 300)

a) Calculate d(**A**, **B**)
b) Calculate d(**A**, **C**)
c) Which customer is more similar to A?
d) Why might raw distances be misleading here?

**Problem 2.6: Standardization Impact**

Before standardization:
- **u** = (1000, 5, 20) (income=$1000, age=5... wait, that's weird!)
- **v** = (1100, 6, 22)

After standardization (subtract mean, divide by std):
- **u_std** = (0.2, 0.3, 0.4)
- **v_std** = (0.4, 0.5, 0.6)

a) Calculate d(**u**, **v**) before standardization
b) Calculate d(**u_std**, **v_std**) after
c) Which is more meaningful? Why?

**Problem 2.7: Time Series Distance**

Stock prices over 5 days:
- **Stock A:** (100, 102, 101, 103, 105)
- **Stock B:** (100, 101, 100, 102, 104)
- **Stock C:** (50, 51, 50, 52, 54)

a) Calculate d(**A**, **B**)
b) Calculate d(**A**, **C**)
c) Why is C far from A even though patterns are similar?
d) How could you make comparison fairer?

**Problem 2.8: Clustering Assignment**

Cluster centers:
- **C1** = (2, 2)
- **C2** = (8, 8)

New points to assign:
- **p1** = (3, 3)
- **p2** = (7, 6)
- **p3** = (5, 5)

a) Calculate distances from each point to each center
b) Assign each point to nearest cluster
c) Point p3 is equidistant-ish. Why might this be a problem?

**Problem 2.9: Anomaly Threshold**

Normal data has distances from center:
- Point 1: d = 2.1
- Point 2: d = 2.3
- Point 3: d = 1.8
- Point 4: d = 2.5

Average normal distance: ~2.2

New point: d = 8.5

a) How many standard deviations away is the new point?
b) Set threshold at mean + 2×std. Is new point anomaly?
c) What happens if threshold is too low? Too high?

**Problem 2.10: Image Similarity**

Three images represented as vectors (simplified):
- **img1** = (100, 120, 110, 130)
- **img2** = (105, 125, 115, 135) (slightly brighter)
- **img3** = (50, 60, 55, 65) (much darker)

a) Calculate d(**img1**, **img2**)
b) Calculate d(**img1**, **img3**)
c) Are img1 and img2 similar?
d) Could img1 and img3 be the same image with different lighting?

---

<a name="standard-deviation"></a>
# 3. Standard Deviation and Spread

## The Core Question: How Spread Out Is the Data?

You measure heights of students in two classes:

**Class A:** 170, 171, 169, 170, 170 cm (very consistent!)
**Class B:** 150, 160, 170, 180, 190 cm (very varied!)

**Both have the same average:** 170 cm

**But Class B is much more "spread out"!**

**Question:** How do we measure this spread mathematically?

## Why Average Isn't Enough

The **mean** (average) tells you the center, but nothing about variability.

**Example: Test Scores**
- Student A: 70, 70, 70, 70, 70 → Average: 70
- Student B: 0, 50, 70, 90, 140 → Average: 70

**Same average, completely different patterns!**

Student A is consistent.
Student B is all over the place!

**We need a measure of spread!**

## Building the Solution: Step by Step

### Step 1: Find the Average (Mean)

For Class A heights: [170, 171, 169, 170, 170]

Mean = (170 + 171 + 169 + 170 + 170) / 5 = 850 / 5 = 170

### Step 2: Find Deviations from Mean

**How far is each point from the center?**

Class A deviations:
- 170 - 170 = 0
- 171 - 170 = +1
- 169 - 170 = -1
- 170 - 170 = 0
- 170 - 170 = 0

Class B deviations:
- 150 - 170 = -20
- 160 - 170 = -10
- 170 - 170 = 0
- 180 - 170 = +10
- 190 - 170 = +20

**Class B has much larger deviations!**

### Step 3: Why We Square the Deviations

**Bad Idea:** Just average the deviations

Class A: (0 + 1 + (-1) + 0 + 0) / 5 = 0
Class B: (-20 + (-10) + 0 + 10 + 20) / 5 = 0

**Problem:** Positive and negative cancel out! Both give 0!

**Solution:** Square the deviations!

Class A squared deviations:
0², 1², (-1)², 0², 0² = 0, 1, 1, 0, 0

Class B squared deviations:
(-20)², (-10)², 0², 10², 20² = 400, 100, 0, 100, 400

**Now positive and negative both contribute positively!**

### Step 4: Average the Squared Deviations (Variance)

**Variance** = average of squared deviations

Class A variance:
σ²_A = (0 + 1 + 1 + 0 + 0) / 5 = 2/5 = 0.4

Class B variance:
σ²_B = (400 + 100 + 0 + 100 + 400) / 5 = 1000/5 = 200

**Class B has much higher variance!**

### Step 5: Take Square Root (Standard Deviation)

**Problem with variance:** Units are squared! (cm² instead of cm)

**Solution:** Take square root to get back to original units!

**Standard deviation** = √(variance)

Class A: σ_A = √0.4 ≈ 0.63 cm
Class B: σ_B = √200 ≈ 14.14 cm

**Perfect!** Now we have a measure of spread in the same units as the data.

## Formal Definition

For data x₁, x₂, ..., xₙ with mean μ:

**Variance:**
σ² = (1/n) Σᵢ₌₁ⁿ (xᵢ - μ)²

**Standard Deviation:**
σ = √[variance] = √[(1/n) Σᵢ₌₁ⁿ (xᵢ - μ)²]

**Alternative notation:**
- Variance: Var(X) or σ²
- Standard deviation: SD(X) or σ or s

## Connection to Vectors and Norms!

Think of your data as a vector: **x** = (x₁, x₂, ..., xₙ)

Create a **centered vector** (subtract mean from each):
**x_centered** = (x₁ - μ, x₂ - μ, ..., xₙ - μ)

**Standard deviation is related to the norm of this centered vector!**

σ = ||**x_centered**|| / √n

**This connects standard deviation to everything we learned about norms!**

## Population vs Sample Standard Deviation

**Population** (entire group):
σ = √[(1/n) Σ(xᵢ - μ)²]

**Sample** (subset of group):
s = √[(1/(n-1)) Σ(xᵢ - x̄)²]

**Note the (n-1) instead of n!** This is **Bessel's correction** - adjusts for bias when estimating from a sample.

**When to use which:**
- Use n if you have ALL the data (entire population)
- Use (n-1) if you have a sample and want to estimate population std

**In ML:** We usually use n (treating our dataset as the population of interest).

## Interpreting Standard Deviation

**Small σ:** Data tightly clustered around mean
**Large σ:** Data widely spread out

**Rule of thumb for normal distributions:**
- ~68% of data within 1σ of mean
- ~95% within 2σ
- ~99.7% within 3σ

**Example:** 
- Mean = 170 cm, σ = 10 cm
- 68% of heights between 160-180 cm
- 95% between 150-190 cm
- 99.7% between 140-200 cm

## Why Machine Learning Cares About Standard Deviation

### 1. Feature Standardization (Z-Score Normalization)

**Problem:** Features with different scales

- Feature 1: Income ($20k - $200k), σ ≈ $50k
- Feature 2: Age (20 - 65), σ ≈ 15 years

**Solution:** Standardize using z-scores!

z = (x - μ) / σ

After standardization:
- Mean = 0
- Standard deviation = 1
- All features on same scale!

**This helps:**
- Gradient descent converge faster
- Features contribute fairly
- Compare importance across features

### 2. Outlier Detection

**Question:** Is a data point unusual?

**Method:** Check how many standard deviations from mean

**Example:**
- Heights: mean = 170 cm, σ = 10 cm
- New person: 210 cm
- Z-score: (210 - 170) / 10 = 4

**4 standard deviations away → very unusual! (> 99.99% of data)**

**Typical threshold:** |z| > 3 for outliers

### 3. Measuring Model Uncertainty

**Prediction intervals** use standard deviation:

prediction ± 2σ gives ~95% confidence interval

**Example:** House price prediction
- Predicted: $500k
- σ of predictions: $50k
- 95% interval: $400k - $600k

### 4. Variance Explained (PCA)

**Principal Component Analysis:**
- Finds directions of maximum variance (σ²)
- Projects data onto these directions
- Keeps components with high variance

**High variance = important information!**

### 5. Batch Normalization

Normalize activations in neural networks:

x_normalized = (x - μ_batch) / σ_batch

**Helps training stability and speed!**

## Detailed Worked Examples

### Example 3.1: Test Score Variance

Class test scores: [65, 70, 75, 80, 85]

**Step 1: Mean**
μ = (65 + 70 + 75 + 80 + 85) / 5 = 375 / 5 = 75

**Step 2: Deviations**
- 65 - 75 = -10
- 70 - 75 = -5
- 75 - 75 = 0
- 80 - 75 = +5
- 85 - 75 = +10

**Step 3: Squared deviations**
100, 25, 0, 25, 100

**Step 4: Variance**
σ² = (100 + 25 + 0 + 25 + 100) / 5 = 250 / 5 = 50

**Step 5: Standard deviation**
σ = √50 ≈ 7.07 points

**Interpretation:** Typical deviation from mean is ~7 points

### Example 3.2: Income Standardization

Incomes: [$30k, $40k, $50k, $60k, $70k]

**Mean:** μ = $50k
**Std:** σ = √200 ≈ $14.14k

**Standardize (z-scores):**
- $30k: z = (30 - 50) / 14.14 ≈ -1.41
- $40k: z = (40 - 50) / 14.14 ≈ -0.71
- $50k: z = (50 - 50) / 14.14 = 0
- $60k: z = (60 - 50) / 14.14 ≈ +0.71
- $70k: z = (70 - 50) / 14.14 ≈ +1.41

**After standardization:**
- Mean of z-scores = 0
- Std of z-scores = 1
- All on standard scale!

### Example 3.3: Outlier Detection

Daily website visitors: [1000, 1100, 950, 1050, 1020, 980, 5000]

**Mean:** μ ≈ 1586
**Std:** σ ≈ 1410

**Check last point (5000):**
z = (5000 - 1586) / 1410 ≈ 2.42

**2.42 standard deviations away → Unusual but not extreme**

**If threshold is |z| > 3, this wouldn't be flagged**
**If threshold is |z| > 2, this would be flagged as outlier**

### Example 3.4: Comparing Variability

**Stock A returns (%):** [2, 3, 2.5, 3.5, 3]
Mean = 2.8%, σ_A ≈ 0.55%

**Stock B returns (%):** [-5, 10, -3, 12, 1]
Mean = 3%, σ_B ≈ 6.96%

**Stock B is much more volatile (higher σ)!**

**Risk-adjusted return (Sharpe ratio):**
Sharpe = Mean / σ

Stock A: 2.8 / 0.55 ≈ 5.09
Stock B: 3.0 / 6.96 ≈ 0.43

**Stock A has better risk-adjusted return!**

### Example 3.5: Feature Scaling Impact

**Before scaling:**
- Feature 1 (income): [20000, 40000, 60000], σ ≈ 16330
- Feature 2 (age): [25, 35, 45], σ ≈ 8.16

**Feature 1 dominates purely by scale!**

**After standardization (z-scores):**
- Feature 1: [-1.22, 0, 1.22], σ = 1
- Feature 2: [-1.22, 0, 1.22], σ = 1

**Now equal contribution!**

### Example 3.6: Normal Distribution Properties

Heights: μ = 170 cm, σ = 10 cm

**68% within 1σ:**
170 ± 10 = [160, 180] cm

**95% within 2σ:**
170 ± 20 = [150, 190] cm

**99.7% within 3σ:**
170 ± 30 = [140, 200] cm

**Someone 195 cm tall:**
z = (195 - 170) / 10 = 2.5

**Between 2σ and 3σ → in top ~1%!**

### Example 3.7: Prediction Confidence

Linear regression predicts house price:
- Prediction: $500k
- Residual std: $50k (from training errors)

**95% confidence interval:**
$500k ± 2($50k) = [$400k, $600k]

**Interpretation:** 95% confident true price is in this range

### Example 3.8: Variance in PCA

Data projections onto principal components:
- PC1: σ² = 45 (explains most variance)
- PC2: σ² = 15
- PC3: σ² = 5

**Total variance:** 45 + 15 + 5 = 65

**Variance explained:**
- PC1: 45/65 ≈ 69%
- PC2: 15/65 ≈ 23%
- PC3: 5/65 ≈ 8%

**Keep PC1 and PC2 → retain 92% of variance!**

## Practice Problems - Standard Deviation

**Problem 3.1: Basic Calculations**

Calculate mean, variance, and standard deviation:
a) Data: [2, 4, 6, 8, 10]
b) Data: [5, 5, 5, 5, 5]
c) Data: [0, 10, 0, 10, 0]
d) Which has highest variance? Lowest?

**Problem 3.2: Deviations**

For data [10, 15, 20, 25, 30]:
a) Calculate mean
b) List all deviations from mean
c) Verify deviations sum to zero
d) Why must deviations always sum to zero?

**Problem 3.3: Effect of Transformations**

Original data: [1, 2, 3, 4, 5], μ = 3, σ = √2

a) Add 10 to each value. New mean? New σ?
b) Multiply each by 2. New mean? New σ?
c) What's the rule for how transformations affect μ and σ?

**Problem 3.4: Z-Scores**

Data: [100, 110, 120, 130, 140], μ = 120, σ = √200

Calculate z-scores for:
a) 100
b) 120
c) 150
d) Which z-score indicates outlier (|z| > 2)?

**Problem 3.5: Comparing Datasets**

**Dataset A:** [10, 11, 12, 13, 14] (small range)
**Dataset B:** [5, 10, 15, 20, 25] (large range)

Both have mean = 12.

a) Calculate σ for each
b) Which is more spread out?
c) Does this match intuition?

**Problem 3.6: Outlier Impact**

Data without outlier: [10, 12, 11, 13, 12] → μ = 11.6, σ ≈ 1.02
Data with outlier: [10, 12, 11, 13, 100] → μ = 29.2, σ ≈ 37.

a) How much did outlier change mean?
b) How much did outlier change σ?
c) Which is more sensitive to outliers?

**Problem 3.7: Feature Standardization**

Two features:
- Income: [30k, 40k, 50k, 60k], μ = 45k, σ = 11.2k
- Age: [25, 35, 45, 55], μ = 40, σ = 11.2

a) Why do they have same σ but different ranges?
b) Standardize both using z-scores
c) Verify both have mean=0, σ=1 after
d) Why is this useful for ML?

**Problem 3.8: Stock Volatility**

**Stock X returns (%):** [2, 3, 2, 3, 2] → σ_X = 0.45
**Stock Y returns (%):** [-10, 20, -5, 15, -10] → σ_Y = 13.04

a) Which stock is more volatile?
b) If mean returns are equal, which would you prefer?
c) How does σ relate to risk?

**Problem 3.9: Test Score Interpretation**

Class: μ = 75, σ = 10

a) A student scores 85. What percentile (assuming normal)?
b) Another scores 65. What percentile?
c) Score needed to be in top 16% (1σ above mean)?
d) Range containing middle 68% of students?

**Problem 3.10: Population vs Sample**

Full class (population): [70, 75, 80, 85, 90]

a) Calculate population σ (divide by n)
b) Sample: just first 3 values [70, 75, 80]
c) Calculate sample s (divide by n-1)
d) Why do we use n-1 for samples?

---

<a name="angle"></a>
# 4. Angle and Cos# Linear Algebra for Machine Learning
## Chapter 3: Norm, Distance, Standard Deviation, and Angles

### A First-Principles Approach with Detailed Examples

---


1. [Norm (Vector Length/Magnitude)](#norm)
2. [Distance Between Vectors](#distance)
3. [Standard Deviation and Spread](#standard-deviation)
4. [Angle and Cosine Similarity](#angle)
5. [Complexity Analysis](#complexity)
6. [Chapter Summary](#summary)
7. [Comprehensive Practice Problems](#practice)

---

<a name="norm"></a>

## The Fundamental Question: How "Big" Is a Vector?

Imagine you're hiking and your GPS shows your displacement from the starting point:
- 3 km east
- 4 km north

Your displacement vector: **v** = (3, 4)

**Question:** How far did you travel from your starting point in a straight line?

You didn't walk 3 + 4 = 7 km (that's the Manhattan distance, walking along streets).

You want the **straight-line distance** - the length of the arrow from start to your current position.

## Building the Solution: Pythagorean Theorem

Remember the Pythagorean theorem from geometry?

For a right triangle with sides a and b, and hypotenuse c:
c² = a² + b²

**Your displacement forms a right triangle!**
- Horizontal side: 3 km
- Vertical side: 4 km
- Hypotenuse: your straight-line distance

Distance = √(3² + 4²) = √(9 + 16) = √25 = 5 km

**This distance is the NORM (or length, or magnitude) of the vector!**

## Formal Definition

The **norm** (or **length** or **magnitude**) of a vector **v** = (v₁, v₂, ..., vₙ) is:

||**v**|| = √(v₁² + v₂² + ... + vₙ²)

**Notation:** 
- ||**v**|| means "the norm of v"
- Also written as |**v**| or ‖**v**‖
- Sometimes called L2 norm or Euclidean norm

**Key insight:** The norm is always a **non-negative number** (scalar), not a vector!

## Why Square and Then Square Root?

**Question:** Why this specific formula? Why not just add components or add absolute values?

### Bad Idea 1: Just Add Components

||**v**|| = v₁ + v₂ + ... + vₙ

**Problem:** What if components are negative?

**v** = (3, -4)
||**v**|| = 3 + (-4) = -1 (Negative length?! Makes no sense!)

### Bad Idea 2: Add Absolute Values (L1 Norm)

||**v**||₁ = |v₁| + |v₂| + ... + |vₙ|

**v** = (3, 4)
||**v**||₁ = |3| + |4| = 7

This actually works and is called the **L1 norm** or **Manhattan distance**!

**But:** This gives you the "city block" distance (walking on a grid), not straight-line distance.

**Think about it:** To go from (0,0) to (3,4), you walk 3 blocks east + 4 blocks north = 7 blocks total.

### Why Squares Work Best (L2 Norm / Euclidean Norm)

||**v**||₂ = √(v₁² + v₂² + ... + vₙ²)

**Advantages:**
1. **Squaring makes everything positive:** Both 3² and (-3)² give 9
2. **Geometric meaning:** Matches Pythagorean theorem (straight-line distance)
3. **Smooth and differentiable:** Important for optimization (gradients exist everywhere)
4. **Penalizes large values more:** Errors of 2 count as 4, errors of 10 count as 100
5. **Nice mathematical properties:** Works beautifully with inner products

**This is the "standard" norm in ML** unless specified otherwise!

## Connection to Inner Product

**Beautiful relationship:** The norm is the square root of the inner product of a vector with itself!

||**v**||² = **v · v** = v₁² + v₂² + ... + vₙ²

**Therefore:**
||**v**|| = √(**v · v**)

**This connects everything we've learned!**

## Examples in Different Dimensions

### Example 1.1: 1D Vectors (Just Numbers)

**v** = (5)
||**v**|| = √(5²) = √25 = 5

**v** = (-3)
||**v**|| = √((-3)²) = √9 = 3

**Key point:** Length is always positive, even for negative numbers!

### Example 1.2: 2D Vectors

**v** = (3, 4)
||**v**|| = √(3² + 4²) = √(9 + 16) = √25 = 5

**v** = (-1, -1)
||**v**|| = √((-1)² + (-1)²) = √(1 + 1) = √2 ≈ 1.414

**v** = (0, 5)
||**v**|| = √(0² + 5²) = √25 = 5 (pointing straight up)

**v** = (5, 0)
||**v**|| = √(5² + 0²) = 5 (pointing straight right)

### Example 1.3: 3D Vectors

**v** = (1, 2, 2)
||**v**|| = √(1² + 2² + 2²) = √(1 + 4 + 4) = √9 = 3

**v** = (2, -3, 6)
||**v**|| = √(4 + 9 + 36) = √49 = 7

**Think about it:** This is the distance from the origin (0, 0, 0) to the point (2, -3, 6) in 3D space!

### Example 1.4: High-Dimensional Vectors

**v** = (1, 1, 1, 1, 1) ∈ ℝ⁵
||**v**|| = √(1 + 1 + 1 + 1 + 1) = √5 ≈ 2.236

**v** = (15, 3, 4, 1, 1) (our spam email features!)
||**v**|| = √(15² + 3² + 4² + 1² + 1²)
= √(225 + 9 + 16 + 1 + 1)
= √252 ≈ 15.87

**Even though we can't visualize 5D space, the math works the same!**

## Properties of the Norm

These properties define what makes something a "norm":

### Property 1: Non-negativity
||**v**|| ≥ 0 for all **v**

**And:** ||**v**|| = 0 **if and only if** **v** = **0**

**Interpretation:** Length is never negative, and only the zero vector has zero length.

### Property 2: Homogeneity (Scaling)
||α**v**|| = |α| · ||**v**|| for any scalar α

**Example:**
**v** = (3, 4), ||**v**|| = 5

**2v** = (6, 8)
||**2v**|| = √(36 + 64) = √100 = 10 = 2 · 5 ✓

**-v** = (-3, -4)
||-**v**|| = √(9 + 16) = 5 = |-1| · 5 ✓

**Interpretation:** If you double the vector, you double its length. If you flip direction (multiply by -1), length stays the same.

### Property 3: Triangle Inequality
||**u + v**|| ≤ ||**u**|| + ||**v**||

**Interpretation:** The direct path is never longer than taking a detour!

**Think about it:** 
- Walk from A to B directly: ||**u + v**||
- Walk from A to C to B: ||**u**|| + ||**v**||
- Direct is always shorter or equal!

**Example:**
**u** = (1, 0), **v** = (0, 1)

||**u**|| = 1
||**v**|| = 1
||**u + v**|| = ||(1, 1)|| = √2 ≈ 1.414

Check: 1.414 ≤ 1 + 1 = 2 ✓

### Property 4: Relationship with Inner Product (Cauchy-Schwarz)
|**u · v**| ≤ ||**u**|| · ||**v**||

**This will be crucial when we discuss angles!**

## Unit Vectors: Vectors with Length 1

A **unit vector** is a vector with norm exactly equal to 1.

**To create a unit vector from any non-zero vector:**
**v̂** = **v** / ||**v**||

(Read as "v-hat" - the hat notation means "unit vector")

**This process is called normalization.**

### Example 1.5: Normalizing Vectors

**v** = (3, 4)
||**v**|| = 5

**v̂** = (3, 4) / 5 = (3/5, 4/5) = (0.6, 0.8)

**Verify:**
||**v̂**|| = √(0.6² + 0.8²) = √(0.36 + 0.64) = √1 = 1 ✓

**Key insight:** **v̂** points in the **same direction** as **v**, but has **length 1**!

### Example 1.6: Standard Basis Vectors Are Unit Vectors

**e₁** = (1, 0, 0)
||**e₁**|| = √(1² + 0² + 0²) = 1 ✓

**e₂** = (0, 1, 0)
||**e₂**|| = 1 ✓

**e₃** = (0, 0, 1)
||**e₃**|| = 1 ✓

**All standard basis vectors are unit vectors!**

## Why Machine Learning Cares About Norms

Norms are **everywhere** in ML!

### 1. Regularization: Preventing Overfitting

**Problem:** Large weights can cause models to overfit (memorize training data instead of learning patterns).

**Solution:** Add penalty for large weights!

**Ridge Regression (L2 regularization):**
Loss = MSE + λ||**w**||²

Where:
- MSE = prediction error
- ||**w**||² = sum of squared weights
- λ = regularization strength

**Effect:** Model prefers smaller weights → simpler, more generalizable models

**Lasso Regression (L1 regularization):**
Loss = MSE + λ||**w**||₁

Where ||**w**||₁ = sum of absolute values of weights

**Effect:** Can force some weights to exactly zero → automatic feature selection!

### 2. Normalization: Standardizing Inputs

**Problem:** Features with different scales can dominate learning.

**Example:**
- Feature 1: Income ($20k - $200k)
- Feature 2: Age (20 - 65 years)

Income dominates just because numbers are bigger!

**Solution:** Normalize feature vectors to have same scale

**Unit normalization:**
**x_normalized** = **x** / ||**x**||

Now all feature vectors have length 1!

### 3. Gradient Clipping: Stabilizing Training

**Problem:** Sometimes gradients become very large (exploding gradients in deep learning).

**Solution:** If ||∇L|| > threshold, scale it down!

if ||∇L|| > max_norm:
    ∇L = ∇L · (max_norm / ||∇L||)

**Effect:** Gradient direction preserved, but magnitude limited.

### 4. Measuring Prediction Confidence

In neural networks, the norm of output vectors can indicate confidence.

**Example:** Image classification
- Output: **y** = (0.1, 0.1, 0.7, 0.1) (probabilities for 4 classes)
- ||**y**|| = √(0.01 + 0.01 + 0.49 + 0.01) = √0.52 ≈ 0.72

Large norm → confident prediction
Small norm → uncertain

### 5. Batch Normalization

Normalize activations in neural networks:
**a_normalized** = (**a** - mean) / std

Helps training converge faster and more stably!

## Norm Squared: A Useful Shortcut

Often in ML, we use **||v||²** instead of ||**v**| because:

**Advantages:**
1. **No square root needed** → computationally faster
2. **Easier to differentiate** → simpler gradients
3. **Still monotonic** → bigger ||**v**|| means bigger ||**v**||²

||**v**||² = **v · v** = v₁² + v₂² + ... + vₙ²

**When comparing lengths, ||v||² works just as well as ||v||!**

### Example 1.7: Comparing Distances

Which is closer to origin: **u** = (3, 4) or **v** = (2, 5)?

**Method 1: Using norm**
||**u**|| = √25 = 5
||**v**|| = √29 ≈ 5.39
**u** is closer

**Method 2: Using squared norm (faster!)**
||**u**||² = 25
||**v**||² = 29
**u** is closer (same answer, no square roots!)

## Different Types of Norms

### L0 "Norm" (Not Really a Norm)
||**v**||₀ = number of non-zero components

**v** = (0, 3, 0, 5, 0)
||**v**||₀ = 2 (two non-zero entries)

**Use:** Counting sparsity (how many features are active)

### L1 Norm (Manhattan Distance)
||**v**||₁ = |v₁| + |v₂| + ... + |vₙ|

**v** = (3, -4)
||**v**||₁ = |3| + |-4| = 3 + 4 = 7

**Use:** Lasso regularization, robust to outliers

### L2 Norm (Euclidean Distance) - The Standard!
||**v**||₂ = √(v₁² + v₂² + ... + vₙ²)

**This is what we mean by "norm" unless specified otherwise!**

### L∞ Norm (Maximum Norm)
||**v**||∞ = max(|v₁|, |v₂|, ..., |vₙ|)

**v** = (3, -7, 2)
||**v**||∞ = max(3, 7, 2) = 7

**Use:** Measuring worst-case deviation

### Comparison Example

**v** = (3, -4, 0)

||**v**||₀ = 2 (two non-zero)
||**v**||₁ = 3 + 4 + 0 = 7
||**v**||₂ = √(9 + 16 + 0) = √25 = 5
||**v**||∞ = max(3, 4, 0) = 4

**General relationship:** ||**v**||∞ ≤ ||**v**||₂ ≤ ||**v**||₁

## Detailed Worked Examples

### Example 1.8: GPS Navigation

Your hiking trail:
- Start: Origin (0, 0)
- Checkpoint 1: (3, 4) km
- Checkpoint 2: (8, 6) km
- End: (10, 2) km

**Calculate distances from origin:**

To Checkpoint 1:
**v₁** = (3, 4)
||**v₁**|| = √(9 + 16) = 5 km

To Checkpoint 2:
**v₂** = (8, 6)
||**v₂**|| = √(64 + 36) = √100 = 10 km

To End:
**v₃** = (10, 2)
||**v₃**|| = √(100 + 4) = √104 ≈ 10.2 km

**Which checkpoint is furthest?** Checkpoint 2 (10 km)

### Example 1.9: Feature Vector Magnitude

Customer behavior vector:
**c** = (purchases, avg_spend, days_active, reviews)
= (23, 150, 365, 12)

||**c**|| = √(23² + 150² + 365² + 12²)
= √(529 + 22500 + 133225 + 144)
= √156398 ≈ 395.47

**Interpretation:** This is the "magnitude" of customer engagement.

**Compare two customers:**
- Customer A: (23, 150, 365, 12), ||**cₐ**|| ≈ 395.47
- Customer B: (50, 200, 730, 25), ||**cᵦ**|| ≈ 762.76

Customer B has higher engagement magnitude!

### Example 1.10: Neural Network Weight Initialization

Initialize weights with small random values:

**w** = (0.01, -0.02, 0.015, -0.008, 0.012)

||**w**|| = √(0.0001 + 0.0004 + 0.000225 + 0.000064 + 0.000144)
= √0.001033 ≈ 0.032

**Check:** ||**w**|| < 0.1 ✓ (good initialization - small weights)

### Example 1.11: Normalizing Image Pixels

Image pixel vector (simplified, 3 pixels):
**img** = (128, 200, 64) (pixel brightness 0-255)

||**img**|| = √(16384 + 40000 + 4096) = √60480 ≈ 245.93

**Normalized:**
**img_normalized** = **img** / ||**img**||
= (128, 200, 64) / 245.93
= (0.520, 0.813, 0.260)

Now ||**img_normalized**|| = 1 ✓

### Example 1.12: Gradient Magnitude

Loss gradient: ∇L = (2.5, -1.8, 3.2, -0.9)

||∇L|| = √(6.25 + 3.24 + 10.24 + 0.81)
= √20.54 ≈ 4.53

**If gradient is too large (>10), clip it:**

Since 4.53 < 10, no clipping needed.

But if ||∇L|| = 15, then:
∇L_clipped = ∇L · (10 / 15) = 0.667 · ∇L

### Example 1.13: Portfolio Volatility

Stock returns vector (5 days):
**r** = (0.02, -0.01, 0.03, -0.02, 0.01) (daily returns as decimals)

||**r**|| = √(0.0004 + 0.0001 + 0.0009 + 0.0004 + 0.0001)
= √0.0019 ≈ 0.0436

**Interpretation:** This measures the magnitude of price movements (volatility indicator).

## Practice Problems - Norm

**Problem 1.1: Basic Norm Calculations**

Calculate ||**v**|| for:
a) **v** = (5, 12)
b) **v** = (-3, 4)
c) **v** = (1, 1, 1)
d) **v** = (2, -2, 1, -1)
e) **v** = (0, 0, 0)

**Problem 1.2: Pythagorean Triples**

Verify these are Pythagorean triples by calculating norms:
a) (3, 4) should have norm 5
b) (5, 12) should have norm 13
c) (8, 15) should have norm 17
d) (7, 24) should have norm 25

**Problem 1.3: Normalization**

Normalize these vectors (find unit vectors):
a) **v** = (3, 4)
b) **v** = (1, 1)
c) **v** = (0, 5)
d) **v** = (2, -2, 1)

Verify each normalized vector has norm 1.

**Problem 1.4: Comparing Magnitudes**

Which vector has larger norm?
a) **u** = (3, 4) vs **v** = (5, 2)
b) **u** = (1, 1, 1, 1) vs **v** = (2, 0, 0, 0)
c) **u** = (10, 1) vs **v** = (1, 10)

**Problem 1.5: Properties Verification**

For **u** = (3, 4) and scalar α = 2:
a) Calculate ||**u**||
b) Calculate ||α**u**||
c) Verify ||α**u**|| = |α| · ||**u**||
d) What if α = -3? Verify the property still holds.

**Problem 1.6: Triangle Inequality**

For **u** = (1, 2) and **v** = (3, 1):
a) Calculate ||**u**||, ||**v**||, and ||**u + v**||
b) Verify ||**u + v**|| ≤ ||**u**|| + ||**v**||
c) When does equality hold in triangle inequality?

**Problem 1.7: Different Norms**

For **v** = (3, -4, 5):
a) Calculate L1 norm: ||**v**||₁
b) Calculate L2 norm: ||**v**||₂
c) Calculate L∞ norm: ||**v**||∞
d) Which is largest? Why?

**Problem 1.8: Sparse Vectors**

Vector **v** = (0, 5, 0, 0, 3, 0, 0, 7, 0)
a) Calculate ||**v**||₀ (count non-zeros)
b) Calculate ||**v**||₁
c) Calculate ||**v**||₂
d) Why is this vector called "sparse"?

**Problem 1.9: Feature Scaling**

Two features with different scales:
- **f₁** = (1000, 2000, 1500) (income in $)
- **f₂** = (25, 35, 30) (age in years)

a) Calculate ||**f₁**|| and ||**f₂**||
b) Normalize both to unit vectors
c) Now calculate norms. What do you notice?
d) Why is this normalization useful in ML?

**Problem 1.10: Gradient Clipping**

Gradient: ∇L = (8, -6, 12, -4)
Max allowed norm: 10

a) Calculate ||∇L||
b) Is clipping needed?
c) If yes, calculate clipped gradient
d) Verify clipped gradient has norm ≤ 10

---

<a name="distance"></a>

## The Core Question: How Far Apart Are Two Things?

You have two movie preference vectors:
- **Alice:** (5, 2, 1, 4) = ratings for (action, comedy, drama, horror)
- **Bob:** (4, 3, 1, 3)

**Question:** How similar are their movie tastes?

To answer this, we need to measure how "far apart" their preference vectors are!

## Building the Solution: The Difference Vector

**First insight:** To measure distance between two points, find how they differ!

**Bob's preferences - Alice's preferences:**
**b - a** = (4, 3, 1, 3) - (5, 2, 1, 4) = (-1, 1, 0, -1)

**This difference vector tells us:**
- Action: Bob rates 1 point lower
- Comedy: Bob rates 1 point higher  
- Drama: Same!
- Horror: Bob rates 1 point lower

**Second insight:** The LENGTH of this difference vector is the distance!

distance(**a**, **b**) = ||**b - a**|| = ||(-1, 1, 0, -1)||
= √(1 + 1 + 0 + 1) = √3 ≈ 1.73

## Formal Definition

The **Euclidean distance** between vectors **u** and **v** is:

d(**u**, **v**) = ||**u - v**|| = √[(u₁-v₁)² + (u₂-v₂)² + ... + (uₙ-vₙ)²]

**Alternative formula using inner product:**
d(**u**, **v**) = √[(**u - v**) · (**u - v**)]

**Key properties:**
- Always non-negative: d(**u**, **v**) ≥ 0
- Zero iff identical: d(**u**, **v**) = 0 ⟺ **u** = **v**
- Symmetric: d(**u**, **v**) = d(**v**, **u**)
- Triangle inequality: d(**u**, **w**) ≤ d(**u**, **v**) + d(**v**, **w**)

## Geometric Interpretation in 2D

Two points in a plane:
- **p₁** = (1, 2)
- **p₂** = (4, 6)

**Visualize:** Plot these points. The distance is the length of the straight line connecting them!

d(**p₁**, **p₂**) = ||**p₂ - p₁**|| = ||(3, 4)|| = 5

**You can literally measure this with a ruler on graph paper!**

## Why This Formula Works

The distance formula comes directly from the Pythagorean theorem!

**Think about moving from **p₁** to **p₂**:**
- Horizontal change: Δx = 4 - 1 = 3
- Vertical change: Δy = 6 - 2 = 4
- These form a right triangle!
- Hypotenuse (distance): √(3² + 4²) = 5

**This extends to any dimension!**

## Properties of Distance

### Property 1: Non-negativity
d(**u**, **v**) ≥ 0

Distance is never negative!

### Property 2: Identity of Indiscernibles  
d(**u**, **v**) = 0 if and only if **u** = **v**

Zero distance means the vectors are identical.

### Property 3: Symmetry
d(**u**, **v**) = d(**v**, **u**)

Distance from A to B equals distance from B to A.

**Proof:**
d(**u**, **v**) = ||**u - v**||
d(**v**, **u**) = ||**v - u**|| = ||**-(u - v)**|| = |-1| · ||**u - v**|| = ||**u - v**|| ✓

### Property 4: Triangle Inequality
d(**u**, **w**) ≤ d(**u**, **v**) + d(**v**, **w**)

**Interpretation:** Going directly from **u** to **w** is never longer than going via **v**!

Think about driving:
- Direct route from home to store
- vs going home → friend's house → store

Direct is always shorter (or equal if friend is on the way)!

## Why Machine Learning Cares About Distance

Distance is **absolutely fundamental** to ML!

### 1. K-Nearest Neighbors (KNN)

**Algorithm:** To classify a new point:
1. Find the K closest training examples (smallest distances)
2. Use their labels to vote

**Example: Email Spam Detection**
- New email: **x** = (5, 20, 1, 50)
- Known spam: **s** = (12, 25, 1, 30)
- Known ham: **h** = (1, 3, 0, 200)

d(**x**, **s**) = ||(5, 20, 1, 50) - (12, 25, 1, 30)||
= ||(-7, -5, 0, 20)||
= √(49 + 25 + 0 + 400) = √474 ≈ 21.8

d(**x**, **h**) = ||(4, 17, 1, -150)||
= √(16 + 289 + 1 + 22500) = √22806 ≈ 151

**x** is MUCH closer to spam example → Classify as spam!

### 2. Clustering (K-Means)

**Goal:** Group similar data points together

**How:** Points are "similar" if distance is small!

**Algorithm:**
1. Assign each point to nearest cluster center
2. Update centers (mean of assigned points)
3. Repeat until convergence

**All based on distance calculations!**

### 3. Anomaly Detection

**Question:** Is this data point unusual?

**Answer:** If it has large distance from all normal examples → Anomaly!

**Example: Fraud Detection**
- Normal transaction: **t_normal** ≈ (50, 1, 10)
- New transaction: **t_new** = (10000, 5, 1000)

d(**t_new**, **t_normal**) = very large → Suspicious!

### 4. Recommendation Systems

**Find users with similar preferences:**

- Your ratings: **you** = (5, 1, 4, 2, 5)
- User A: **a** = (5, 2, 4, 1, 5)
- User B: **b** = (1, 5, 1, 5, 2)

d(**you**, **a**) = small → Similar tastes!
d(**you**, **b**) = large → Different tastes!

**Recommendation:** Show what similar users liked!

### 5. Loss Functions

**Mean Squared Error** is based on distance!

MSE = (1/n) Σᵢ ||**yᵢ** - **ŷᵢ**||²

Average squared distance between predictions and true values!

## Distance vs. Similarity

**Key insight:** Small distance = high similarity!

Often we convert distance to similarity:

**Similarity metrics:**
1. sim = 1 / (1 + distance)
2. sim = e^(-distance)
3. sim = 1 - (distance / max_distance)

**Example:**
- d = 0 → sim = 1 (identical)
- d = 1 → sim = 0.5 (somewhat similar)
- d = ∞ → sim = 0 (completely different)

## Different Distance Metrics

### Euclidean Distance (L2) - The Standard!
d₂(**u**, **v**) = ||**u - v**||₂ = √[Σ(uᵢ - vᵢ)²]

**Use:** General purpose, most common

### Manhattan Distance (L1)
d₁(**u**, **v**) = ||**u - v**||₁ = Σ|uᵢ - vᵢ|

**Use:** When you can only move along axes (like city blocks)

### Chebyshev Distance (L∞)
d∞(**u**, **v**) = ||**u - v**||∞ = max|uᵢ

---

---

<a name="angle"></a>
# 4. Angle and Cosine Similarity

## The Core Question: Do Two Vectors Point in Similar Directions?

Imagine two students rating movies:

**Alice:** (5, 1, 4, 2, 5) = (loves action, hates comedy, likes drama, neutral horror, loves sci-fi)
**Bob:** (10, 2, 8, 4, 10) = exactly 2× Alice's ratings!

**Notice:** Bob's ratings are just Alice's scaled by 2!

**They have the SAME preferences**, just on a different scale!

**Geometrically:** Their vectors point in the **same direction**, but Bob's is longer.

**Question:** How do we measure if vectors point in similar directions, independent of their lengths?

**Answer:** Using the **angle** between them!

## Why Direction Matters More Than Magnitude

### Example: Movie Recommendations

**User A:** (5, 1, 4, 2) - Uses full 1-5 scale
**User B:** (3, 0, 2, 1) - More conservative, uses 0-3 scale

**Even though magnitudes differ, the *pattern* is the same!**

Both users:
- Love genre 1 (highest rating)
- Hate genre 2 (lowest rating)
- Like genre 3 (medium-high)
- Neutral on genre 4 (low-medium)

**Distance would say they're different** (different magnitudes)
**Angle says they're similar** (same direction/pattern)

**For recommendations, direction matters more than magnitude!**

## Building the Solution: The Inner Product Formula

Remember from geometry: For two 2D vectors with angle θ between them:

**u · v** = ||**u**|| · ||**v**|| · cos(θ)

**Rearranging to solve for angle:**

cos(θ) = (**u · v**) / (||**u**|| · ||**v**||)

**This works in ANY dimension!**

**Cosine similarity** = cos(θ) = (**u · v**) / (||**u**|| · ||**v**||)

## Understanding Cosine Values

**cos(θ) = 1** → θ = 0° → vectors point **same direction** (perfectly similar!)
**cos(θ) = 0** → θ = 90° → vectors **perpendicular** (unrelated!)
**cos(θ) = -1** → θ = 180° → vectors point **opposite directions** (perfectly dissimilar!)

**In between:**
- cos(θ) close to 1 → small angle → similar directions
- cos(θ) close to 0 → ~90° → unrelated directions
- cos(θ) close to -1 → ~180° → opposite directions

**Key advantage:** Cosine similarity is **normalized** - always between -1 and 1!

## Examples of Different Angles

### Example 4.1: Same Direction (0°)

**u** = (3, 4)
**v** = (6, 8) = 2**u**

**u · v** = 3(6) + 4(8) = 18 + 32 = 50
||**u**|| = √(9 + 16) = 5
||**v**|| = √(36 + 64) = 10

cos(θ) = 50 / (5 × 10) = 50 / 50 = 1

**θ = arccos(1) = 0°** ✓

**Vectors point in exactly the same direction!**

### Example 4.2: Perpendicular (90°)

**u** = (1, 0) - pointing east
**v** = (0, 1) - pointing north

**u · v** = 1(0) + 0(1) = 0
||**u**|| = 1
||**v**|| = 1

cos(θ) = 0 / (1 × 1) = 0

**θ = arccos(0) = 90°** ✓

**Vectors are perpendicular (orthogonal)!**

### Example 4.3: Opposite Directions (180°)

**u** = (1, 1)
**v** = (-1, -1) = -**u**

**u · v** = 1(-1) + 1(-1) = -1 - 1 = -2
||**u**|| = √2
||**v**|| = √2

cos(θ) = -2 / (√2 × √2) = -2 / 2 = -1

**θ = arccos(-1) = 180°** ✓

**Vectors point in opposite directions!**

### Example 4.4: 45° Angle

**u** = (1, 0)
**v** = (1, 1)

**u · v** = 1(1) + 0(1) = 1
||**u**|| = 1
||**v**|| = √2

cos(θ) = 1 / (1 × √2) = 1/√2 ≈ 0.707

**θ = arccos(0.707) ≈ 45°** ✓

**Exactly halfway between same direction and perpendicular!**

## Why Orthogonality (90°) Is Special

When **u · v** = 0, we say vectors are **orthogonal** (perpendicular).

**Why special?**

**1. Independent Information**
If features are orthogonal, they provide completely independent information!

**Example:** Image features
- **h** = horizontal edge detector
- **v** = vertical edge detector

**h · v** = 0 (orthogonal!)

Horizontal edges don't affect vertical edge detection - perfect independence!

**2. No Correlation**
Orthogonal features are uncorrelated - no redundancy!

**3. Simplifies Mathematics**
Many formulas become much simpler with orthogonal vectors.

**4. Optimal for Machine Learning**
- PCA finds orthogonal directions
- Neural network layers often initialized with orthogonal weights
- Orthogonal features are ideal (no multicollinearity)

## Cosine Similarity vs Euclidean Distance

**When to use cosine similarity (angle):**
- Direction/pattern matters more than magnitude
- Data has different scales
- Text analysis (TF-IDF vectors)
- Recommendations (rating patterns)
- High-dimensional sparse data

**When to use Euclidean distance:**
- Absolute differences matter
- Data on same scale
- Clustering in physical space
- K-nearest neighbors (often)

### Example: The Difference

**User A:** (5, 1, 5)
**User B:** (10, 2, 10) = 2 × **A**

**Euclidean distance:**
d(**A**, **B**) = ||(5, 1, 5)|| = √(25 + 1 + 25) = √51 ≈ 7.14
(Considers them different due to magnitude)

**Cosine similarity:**
cos(θ) = **A · B** / (||**A**|| × ||**B**||)
= (50 + 2 + 50) / (√51 × √204)
= 102 / 102 = 1
(Perfect similarity - same direction!)

**For recommendations, cosine is better here!**

## Why Machine Learning Cares About Angles

### 1. Document Similarity (Text Analysis)

**Bag of words representation:**

**Doc 1:** "cat cat dog" → (2, 1) for vocab ["cat", "dog"]
**Doc 2:** "cat cat cat dog dog dog" → (3, 3)

**Distance** would say they're different (different magnitudes)
**Cosine similarity:**

cos(θ) = (2×3 + 1×3) / (√5 × √18) = 9 / √90 ≈ 0.949

**Very high similarity! Same topic, different lengths.**

**This is why cosine similarity is standard for text!**

### 2. Recommendation Systems

**User A:** (5, 1, 4, 2, 5)
**User B:** (4, 0, 3, 1, 4)

cos(θ) = (20 + 0 + 12 + 2 + 20) / (√51 × √42)
= 54 / √2142 ≈ 0.984

**High similarity → recommend what User A liked to User B!**

**Works even though User B rates more conservatively.**

### 3. Face Recognition

Face embeddings are vectors in high-dimensional space.

**Same person, different photos:**
- Photo 1: **f₁** (certain lighting)
- Photo 2: **f₂** (different lighting)

Lighting changes magnitude, but face identity (direction) stays similar!

**Cosine similarity > threshold → same person!**

### 4. Word Embeddings (Word2Vec, GloVe)

Words are vectors where direction encodes meaning.

**Famous example:**
- king - man + woman ≈ queen

**Similarity by cosine:**
- cos(**king**, **queen**) = high (both royalty)
- cos(**king**, **table**) = low (unrelated)

**Direction captures semantic relationships!**

### 5. Attention Mechanisms (Transformers, GPT)

**Attention score** between query and key:

attention(**Q**, **K**) ∝ **Q · K** / (||**Q**|| × ||**K**||)

**This is cosine similarity!**

High cosine → words should "pay attention" to each other.

**The entire attention mechanism is based on measuring directional similarity!**

### 6. Clustering with Cosine Similarity

**K-means with cosine distance:**

Instead of Euclidean distance, use:
distance = 1 - cos(θ)

**Advantages:**
- Handles different magnitudes
- Better for sparse, high-dimensional data
- Standard for document clustering

## Connection Between All Concepts

**Beautiful unified formula:**

cos(θ) = (**u · v**) / (||**u**|| × ||**v**||)

This connects:
- **Inner product** (**u · v**) - measures alignment
- **Norm** (||**u**||, ||**v**||) - measures length
- **Angle** (θ) - measures direction similarity

**Everything we've learned comes together!**

## Detailed Worked Examples

### Example 4.5: Movie Recommendations

**Alice:** (5, 1, 4, 2, 5) - genres: action, comedy, drama, horror, sci-fi
**Bob:** (4, 2, 3, 2, 4)
**Carol:** (1, 5, 2, 5, 1)

**Compare Alice to Bob:**

**A · B** = 5(4) + 1(2) + 4(3) + 2(2) + 5(4) = 20 + 2 + 12 + 4 + 20 = 58
||**A**|| = √(25 + 1 + 16 + 4 + 25) = √71 ≈ 8.43
||**B**|| = √(16 + 4 + 9 + 4 + 16) = √49 = 7

cos(θ_AB) = 58 / (8.43 × 7) = 58 / 59.01 ≈ 0.983

**Very high! Alice and Bob have similar tastes.**

**Compare Alice to Carol:**

**A · C** = 5(1) + 1(5) + 4(2) + 2(5) + 5(1) = 5 + 5 + 8 + 10 + 5 = 33
||**C**|| = √(1 + 25 + 4 + 25 + 1) = √56 ≈ 7.48

cos(θ_AC) = 33 / (8.43 × 7.48) = 33 / 63.06 ≈ 0.523

**Moderate similarity. Some overlap but different preferences.**

**Recommendation:** Show Alice what Bob liked (high similarity), not what Carol liked.

### Example 4.6: Document Similarity

Vocabulary: ["machine", "learning", "deep", "neural", "network"]

**Doc 1** (about ML): (5, 8, 2, 1, 3) - mostly "machine" and "learning"
**Doc 2** (about deep learning): (3, 5, 10, 8, 7) - balanced, emphasizes "deep" and "neural"
**Doc 3** (about cooking): (0, 0, 1, 0, 0) - mentioned "deep" (deep frying!)

**Doc 1 vs Doc 2:**

**D₁ · D₂** = 5(3) + 8(5) + 2(10) + 1(8) + 3(7) = 15 + 40 + 20 + 8 + 21 = 104
||**D₁**|| = √(25 + 64 + 4 + 1 + 9) = √103 ≈ 10.15
||**D₂**|| = √(9 + 25 + 100 + 64 + 49) = √247 ≈ 15.72

cos(θ) = 104 / (10.15 × 15.72) ≈ 0.652

**Moderately similar - both about machine learning!**

**Doc 1 vs Doc 3:**

**D₁ · D₃** = 5(0) + 8(0) + 2(1) + 1(0) + 3(0) = 2
||**D₃**|| = 1

cos(θ) = 2 / (10.15 × 1) ≈ 0.197

**Very low similarity - different topics!**

### Example 4.7: Perpendicular Features

**Feature 1** (horizontal edges in image): (1, 0, 0, 0)
**Feature 2** (vertical edges): (0, 1, 0, 0)
**Feature 3** (diagonal edges): (0, 0, 1, 0)
**Feature 4** (brightness): (0, 0, 0, 1)

**Check if F1 and F2 are orthogonal:**

**F₁ · F₂** = 1(0) + 0(1) + 0(0) + 0(0) = 0 ✓

cos(θ) = 0 / (1 × 1) = 0
θ = 90°

**Perfectly orthogonal! They provide independent information.**

**Check all pairs:**
All pairs have inner product = 0 → all orthogonal!

**This is an orthonormal basis** (orthogonal + length 1)!

### Example 4.8: Stock Correlation

Stock returns over 5 days:

**Stock A:** (2, -1, 3, -2, 1)%
**Stock B:** (3, -2, 4, -3, 2)% = 1.5 × **A** (moves together!)
**Stock C:** (-2, 1, -3, 2, -1)% = -1 × **A** (moves opposite!)

**A vs B:**

**A · B** = 2(3) + (-1)(-2) + 3(4) + (-2)(-3) + 1(2) = 6 + 2 + 12 + 6 + 2 = 28
||**A**|| = √(4 + 1 + 9 + 4 + 1) = √19 ≈ 4.36
||**B**|| = √(9 + 4 + 16 + 9 + 4) = √42 ≈ 6.48

cos(θ_AB) = 28 / (4.36 × 6.48) ≈ 0.992

**Nearly perfect correlation! They move together.**

**A vs C:**

**A · C** = 2(-2) + (-1)(1) + 3(-3) + (-2)(2) + 1(-1) = -4 - 1 - 9 - 4 - 1 = -19

cos(θ_AC) = -19 / (4.36 × 4.36) ≈ -1.0

**Perfect negative correlation! They move opposite.**

**Portfolio diversification:** Combine A and C to reduce volatility!

### Example 4.9: Plagiarism Detection

**Student A's essay** (word frequencies): (10, 5, 3, 8, 12, 2, ...)
**Student B's essay:** (30, 15, 9, 24, 36, 6, ...) = 3 × **A**

cos(θ) = **A · B** / (||**A**|| × ||**B**||)

Since **B** = 3**A**:
**A · B** = **A** · (3**A**) = 3(**A · A**) = 3||**A**||²
||**B**|| = 3||**A**||

cos(θ) = 3||**A**||² / (||**A**|| × 3||**A**||) = 3||**A**||² / 3||**A**||² = 1

**Perfect similarity! Likely plagiarism** (even though lengths differ).

### Example 4.10: Search Engine Ranking

Query: "machine learning python"
Query vector: (1, 1, 1) for ["machine", "learning", "python"]

**Document A:** (3, 2, 5) - tutorial with lots of code
**Document B:** (10, 10, 1) - theory-heavy, less code
**Document C:** (5, 4, 6) - balanced

**Rank by cosine similarity:**

**Query · DocA** = 1(3) + 1(2) + 1(5) = 10
||Query|| = √3 ≈ 1.73
||DocA|| = √(9 + 4 + 25) = √38 ≈ 6.16
cos(θ_A) = 10 / (1.73 × 6.16) ≈ 0.938

**Query · DocB** = 1(10) + 1(10) + 1(1) = 21
||DocB|| = √(100 + 100 + 1) = √201 ≈ 14.18
cos(θ_B) = 21 / (1.73 × 14.18) ≈ 0.856

**Query · DocC** = 1(5) + 1(4) + 1(6) = 15
||DocC|| = √(25 + 16 + 36) = √77 ≈ 8.77
cos(θ_C) = 15 / (1.73 × 8.77) ≈ 0.988

**Ranking:** DocC (0.988) > DocA (0.938) > DocB (0.856)

**Show Doc C first!**

### Example 4.11: Image Similarity

Two images represented as pixel vectors (simplified):

**Image 1:** (100, 150, 80, 200) - certain brightness
**Image 2:** (50, 75, 40, 100) = 0.5 × **Image1** - darker version!

cos(θ) = **I₁ · I₂** / (||**I₁**|| × ||**I₂**||)

Since **I₂** = 0.5**I₁**:
cos(θ) = 1

**Perfect similarity! Same image, different brightness.**

**This is why cosine similarity is used in image retrieval!**

### Example 4.12: Neural Network Attention

Query vector: **Q** = (1, 2, 3)
Key vectors:
- **K₁** = (1, 2, 3) - exactly matches query
- **K₂** = (0, 1, 0) - somewhat related
- **K₃** = (-1, -2, -3) - opposite

**Attention scores (simplified, using cosine):**

cos(**Q**, **K₁**) = (1 + 4 + 9) / (√14 × √14) = 14/14 = 1.0 ← High attention!
cos(**Q**, **K₂**) = (0 + 2 + 0) / (√14 × 1) = 2/√14 ≈ 0.53 ← Medium
cos(**Q**, **K₃**) = (-1 - 4 - 9) / (√14 × √14) = -14/14 = -1.0 ← Negative

**K₁ gets most attention (highest cosine similarity)!**

## Practice Problems - Angle and Cosine Similarity

**Problem 4.1: Basic Angle Calculations**

Calculate cos(θ) for these pairs:
a) **u** = (1, 0), **v** = (0, 1)
b) **u** = (3, 4), **v** = (6, 8)
c) **u** = (1, 1), **v** = (-1, -1)
d) **u** = (1, 1), **v** = (1, -1)

For each, state whether they're: same direction, perpendicular, or opposite.

**Problem 4.2: Orthogonality Check**

Which pairs are orthogonal (perpendicular)?
a) **u** = (2, 3), **v** = (3, -2)
b) **u** = (1, 2, 3), **v** = (3, -2, 0)
c) **u** = (1, 1, 1), **v** = (1, -1, 0)
d) **u** = (4, 0, 0), **v** = (0, 5, 0)

**Problem 4.3: Movie Preferences**

Three users rate 4 genres (action, comedy, drama, horror):
- **Alice:** (5, 2, 4, 1)
- **Bob:** (4, 3, 3, 2)
- **Carol:** (1, 5, 2, 5)

a) Calculate cos(θ) for Alice vs Bob
b) Calculate cos(θ) for Alice vs Carol
c) Who has more similar taste to Alice?
d) Should you recommend Bob's favorites to Alice?

**Problem 4.4: Document Clustering**

Word counts for vocabulary ["data", "science", "art", "paint"]:
- **Doc 1:** (10, 8, 0, 0)
- **Doc 2:** (8, 10, 0, 0)
- **Doc 3:** (0, 0, 7, 9)

a) Calculate cosine similarity for all pairs
b) Which documents should be clustered together?
c) What's the angle between Doc 1 and Doc 3?

**Problem 4.5: Magnitude Independence**

**User A:** (5, 1, 5)
**User B:** (10, 2, 10) = 2 × **A**
**User C:** (5, 5, 5)

a) Calculate Euclidean distance between A and B
b) Calculate cosine similarity between A and B
c) Which measure shows they have same preferences?
d) Calculate cosine between A and C. Are they similar?

**Problem 4.6: Correlation**

Stock returns over 3 days:
- **Stock X:** (2, -1, 3)
- **Stock Y:** (3, -1.5, 4.5) = 1.5 × **X**
- **Stock Z:** (-2, 1, -3) = -1 × **X**

a) cos(**X**, **Y**) = ?
b) cos(**X**, **Z**) = ?
c) Are X and Y positively correlated?
d) Are X and Z negatively correlated?
e) Which pair would diversify a portfolio?

**Problem 4.7: Feature Independence**

Image features:
- **Horizontal edges:** (1, 0, 0)
- **Vertical edges:** (0, 1, 0)
- **Brightness:** (0, 0, 1)

a) Show all three pairs are orthogonal
b) What does this mean for information content?
c) Is this basis orthonormal? Check.

**Problem 4.8: Search Relevance**

Query: "python programming"
Query vector: (1, 1) for ["python", "programming"]

Documents:
- **Doc A:** (5, 5) - balanced content
- **Doc B:** (10, 2) - mostly "python", less "programming"
- **Doc C:** (3, 7) - more "programming", less "python"

a) Calculate cosine similarity for each document
b) Rank documents by relevance
c) Would Euclidean distance give same ranking?

**Problem 4.9: Plagiarism Score**

**Original:** (10, 5, 8, 3, 12)
**Submission 1:** (30, 15, 24, 9, 36) = 3 × **Original**
**Submission 2:** (11, 6, 7, 4, 13)

a) cos(Original, Submission1) = ?
b) cos(Original, Submission2) = ?
c) Which is more suspicious?
d) Why is cosine better than distance for plagiarism?

**Problem 4.10: Understanding Negative Cosine**

**u** = (1, 2, 3)
**v** = (-2, -4, -6) = -2**u**

a) Calculate **u · v**
b) Calculate cos(θ)
c) What does negative cosine mean?
d) What's the angle θ?

---

<a name="complexity"></a>
# 5. Complexity Analysis

## Why Care About Computational Complexity?

In real ML applications:
- Vectors can have millions of dimensions (image pixels, word vocabularies)
- Datasets can have billions of examples (user interactions, sensor readings)
- Operations are repeated millions of times (training iterations)

**Understanding complexity helps us:**
1. Estimate runtime
2. Choose efficient algorithms
3. Scale to larger problems
4. Identify bottlenecks

## Complexity of Vector Operations

Assume vectors have dimension n.

### Norm Calculation: ||**v**||

**Operation:** Compute √(v₁² + v₂² + ... + vₙ²)

**Steps:**
1. Square each component: n multiplications
2. Sum all squares: n-1 additions
3. Take square root: 1 operation

**Complexity:** **O(n)**

**Squared norm:** ||**v**||² skips square root → still O(n) but faster constant

### Distance Calculation: d(**u**, **v**)

**Operation:** ||**u** - **v**|| = √[(u₁-v₁)² + ... + (uₙ-vₙ)²]

**Steps:**
1. Subtract components: n subtractions
2. Square differences: n multiplications
3. Sum: n-1 additions
4. Square root: 1 operation

**Complexity:** **O(n)**

**Squared distance:** d²(**u**, **v**) skips square root → still O(n)

### Cosine Similarity

**Operation:** cos(θ) = (**u · v**) / (||**u**|| × ||**v**||)

**Steps:**
1. Inner product: O(n)
2. Norm of **u**: O(n)
3. Norm of **v**: O(n)
4. Division: O(1)

**Total:** O(n) + O(n) + O(n) = **O(n)**

**All basic vector operations are linear in dimension!**

## Complexity for Multiple Vectors

### K-Nearest Neighbors

**Input:** N training points, 1 query point, dimension n

**For each training point:**
- Calculate distance: O(n)

**Total:** N × O(n) = **O(Nn)**

**For K-nearest:** Also need to sort/track K smallest → O(N log K)

**Full complexity:** **O(Nn + N log K)** ≈ **O(Nn)** for small K

### K-Means (One Iteration)

**Input:** N points, K clusters, dimension n

**Assignment step:**
- For each point: calculate K distances
- Each distance: O(n)
- Total: N × K × O(n) = **O(NKn)**

**Update step:**
- For each cluster: sum assigned points
- Total: O(Nn)

**One iteration:** **O(NKn)**

**Full algorithm:** I iterations → **O(INKn)**

### Cosine Similarity Matrix

**Input:** N documents, dimension n

**Need:** Similarity between all pairs

**Pairs:** N(N-1)/2 ≈ N²/2

**Each pair:** O(n)

**Total:** **O(N²n)**

**This gets expensive fast!**
- N = 1,000: ~500,000 comparisons
- N = 10,000: ~50,000,000 comparisons
- N = 100,000: ~5,000,000,000 comparisons

## Memory Complexity

**Single vector:** O(n) memory

**Matrix (N×n):** O(Nn) memory

**Distance matrix (N×N):** O(N²) memory

**Example:**
- 1,000,000 documents
- 10,000 dimensions
- 4 bytes per float

**Data matrix:** 1M × 10K × 4 bytes = 40 GB
**Distance matrix:** 1M × 1M × 4 bytes = 4 TB!

**Distance matrix often too large to store!**

## Practical Implications

### 1. Feature Selection Matters

**Before:** n = 10,000 features
**After:** n = 100 features (selected best)

**Speedup:** 100× faster for all operations!

### 2. Mini-Batches for Large N

Instead of processing all N points:
- Process m << N at a time
- Update iteratively

**Per batch:** O(mn) instead of O(Nn)

### 3. Sparse Vectors

If vector has only k non-zero entries:
- Dense: O(n) operations
- Sparse: O(k) operations

**Example:** Text vectors
- Vocabulary: n = 100,000
- Non-zero words per document: k = 100
- Speedup: 1000×!

### 4. Approximation Methods

For very large N:
- Locality Sensitive Hashing (LSH)
- Approximate nearest neighbors
- Random projections

Trade accuracy for massive speedup!

### 5. GPU Acceleration

**GPU advantage:** Parallel computation

**Vector operations are highly parallelizable!**

- CPU: O(n) time sequentially
- GPU: O(log n) time with n processors

**Practical speedup:** 10-100× for large vectors

## Comparison Table

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Norm ||**v**|| | O(n) | Linear in dimension |
| Distance d(**u**, **v**) | O(n) | Linear in dimension |
| Inner product **u · v** | O(n) | Linear in dimension |
| Cosine similarity | O(n) | Three O(n) operations |
| KNN (1 query) | O(Nn) | N comparisons |
| K-means iteration | O(NKn) | All pairwise distances |
| Pairwise distances | O(N²n) | All pairs |
| Standard deviation | O(n) | One pass through data |

## Practice Problems - Complexity

**Problem 5.1: Operation Counting**

Vector dimension n = 1,000. Estimate operations for:
a) Calculate norm ||**v**||
b) Calculate distance d(**u**, **v**)
c) Calculate cosine similarity
d) Which is slowest?

**Problem 5.2: Scaling Analysis**

KNN with N = 10,000 points, n = 100 dimensions.

a) Operations for one query?
b) Double dimensions (n = 200). New cost?
c) Double data (N = 20,000). New cost?
d) Which has bigger impact?

**Problem 5.3: Memory Requirements**

Store N = 1,000,000 vectors, n = 1,000 dimensions,### Chebyshev Distance (L∞)
d∞(**u**, **v**) = ||**u - v**||∞ = max|uᵢ - vᵢ|

**Use:** Measuring worst-case difference

### Comparison Example

**u** = (2, 5, 1)
**v** = (5, 3, 4)

**Difference:** **u - v** = (-3, 2, -3)

**Euclidean (L2):**
d₂ = √(9 + 4 + 9) = √22 ≈ 4.69

**Manhattan (L1):**
d₁ = |−3| + |2| + |−3| = 3 + 2 + 3 = 8

**Chebyshev (L∞):**
d∞ = max(3, 2, 3) = 3

**General relationship:** d∞ ≤ d₂ ≤ d₁

## Squared Distance: A Computational Shortcut

Just like with norms, we often use **squared distance**:

d²(**u**, **v**) = ||**u - v**||²

**Advantages:**
1. No square root → faster computation
2. Preserves ordering (if d₁ < d₂, then d₁² < d₂²)
3. Easier to differentiate

**When comparing distances, squared distance works just as well!**

### Example: Finding Nearest Neighbor

Which is closer to **x** = (0, 0)?
- **a** = (3, 4)
- **b** = (5, 1)

**Method 1: Using distance**
d(**x**, **a**) = √25 = 5
d(**x**, **b**) = √26 ≈ 5.1
**a** is closer

**Method 2: Using squared distance (faster!)**
d²(**x**, **a**) = 25
d²(**x**, **b**) = 26
**a** is closer (same answer, no square roots!)

## Detailed Worked Examples

### Example 2.1: Customer Segmentation

Two customer profiles:
- **Customer A:** (age=25, income=50k, purchases=10, satisfaction=8)
- **Customer B:** (age=28, income=55k, purchases=12, satisfaction=7)

**A** = (25, 50, 10, 8)
**B** = (28, 55, 12, 7)

d(**A**, **B**) = ||(28, 55, 12, 7) - (25, 50, 10, 8)||
= ||(3, 5, 2, -1)||
= √(9 + 25 + 4 + 1)
= √39 ≈ 6.24

**Note:** Different features have different scales! 
- Age differs by 3 years
- Income differs by $5k

**Better approach:** Standardize features first!

After standardization (z-scores):
**A_std** = (0.2, 0.1, 0.3, 0.5)
**B_std** = (0.4, 0.3, 0.5, 0.3)

d(**A_std**, **B_std**) = ||(0.2, 0.2, 0.2, -0.2)||
= √(0.04 + 0.04 + 0.04 + 0.04)
= √0.16 = 0.4

**Much better!** Now all features contribute fairly.

### Example 2.2: Image Similarity

Two 2×2 grayscale images (flattened to vectors):

**Image 1:** [100, 120, 110, 130] (brightness values)
**Image 2:** [105, 125, 115, 135]

d(**img1**, **img2**) = ||(5, 5, 5, 5)||
= √(25 + 25 + 25 + 25)
= √100 = 10

**Small distance → images are similar!**

**Image 3:** [200, 50, 180, 70]

d(**img1**, **img3**) = ||(100, -70, 70, -60)||
= √(10000 + 4900 + 4900 + 3600)
= √23400 ≈ 153

**Large distance → images are very different!**

### Example 2.3: Document Similarity

Word count vectors (vocabulary: ["cat", "dog", "bird", "fish"]):

**Doc 1:** (5, 2, 0, 1) - mentions cats a lot
**Doc 2:** (6, 1, 0, 2) - also about cats
**Doc 3:** (0, 0, 8, 5) - about birds and fish

d(**doc1**, **doc2**) = ||(1, -1, 0, 1)||
= √(1 + 1 + 0 + 1) = √3 ≈ 1.73 (similar!)

d(**doc1**, **doc3**) = ||(-5, -2, 8, 4)||
= √(25 + 4 + 64 + 16) = √109 ≈ 10.44 (very different!)

**Doc 1 and Doc 2 are about similar topics!**

### Example 2.4: Time Series Comparison

Temperature readings over 5 hours:

**City A:** (20, 21, 23, 24, 25)°C
**City B:** (22, 23, 24, 25, 26)°C

d(**A**, **B**) = ||(2, 2, 1, 1, 1)||
= √(4 + 4 + 1 + 1 + 1)
= √11 ≈ 3.32

**City B is consistently ~2°C warmer**

**City C:** (20, 15, 25, 18, 28)°C

d(**A**, **C**) = ||(0, -6, 2, -6, 3)||
= √(0 + 36 + 4 + 36 + 9)
= √85 ≈ 9.22

**City C has more variable weather (larger distance from A)**

### Example 2.5: K-Nearest Neighbors

Training data (2D for visualization):
- Point A: (1, 2), Label: Red
- Point B: (2, 1), Label: Red
- Point C: (5, 6), Label: Blue
- Point D: (6, 5), Label: Blue

New point to classify: **x** = (3, 3)

**Calculate distances:**
d(**x**, **A**) = ||(2, 1)|| = √5 ≈ 2.24
d(**x**, **B**) = ||(1, 2)|| = √5 ≈ 2.24
d(**x**, **C**) = ||(2, 3)|| = √13 ≈ 3.61
d(**x**, **D**) = ||(3, 2)|| = √13 ≈ 3.61

**For K=2 (2 nearest neighbors):**
Nearest: A and B (both Red)
**Prediction: Red!**

### Example 2.6: Anomaly Detection

Normal system metrics:
- **Normal 1:** (CPU=30%, Memory=40%, Disk=50%, Network=20%)
- **Normal 2:** (32%, 38%, 52%, 18%)
- **Normal 3:** (28%, 42%, 48%, 22%)

Average normal: **avg** ≈ (30, 40, 50, 20)

New reading: **new** = (85, 90, 95, 15)

d(**new**, **avg**) = ||(55, 50, 45, -5)||
= √(3025 + 2500 + 2025 + 25)
= √7575 ≈ 87.0

**Very large distance from normal → ANOMALY!**

### Example 2.7: Clustering Quality

Two clusters with centers:
- **C1** = (2, 3)
- **C2** = (8, 7)

Distance between cluster centers:
d(**C1**, **C2**) = ||(6, 4)|| = √52 ≈ 7.21

**Points in Cluster 1:**
- **p1** = (2, 3), d(**p1**, **C1**) = 0
- **p2** = (3, 4), d(**p2**, **C1**) = √2 ≈ 1.41
- **p3** = (1, 2), d(**p3**, **C1**) = √2 ≈ 1.41

**Average distance within cluster:** (0 + 1.41 + 1.41) / 3 ≈ 0.94

**Good clustering:** 
- Small within-cluster distances (0.94)
- Large between-cluster distances (7.21)
- Ratio: 7.21 / 0.94 ≈ 7.7 (well-separated!)

## Practice Problems - Distance

**Problem 2.1: Basic Distance Calculations**

Calculate d(**u**, **v**) for:
a) **u** = (1, 2), **v** = (4, 6)
b) **u** = (0, 0), **v** = (3, 4)
c) **u** = (5, 2, 1), **v** = (1, 2, 5)
d) **u** = (1, 1, 1, 1), **v** = (2, 2, 2, 2)

**Problem 2.2: Distance Properties**

For **u** = (1, 2), **v** = (4, 6), **w** = (7, 10):
a) Calculate d(**u**, **v**)
b) Calculate d(**v**, **u**)
c) Verify d(**u**, **v**) = d(**v**, **u**) (symmetry)
d) Calculate d(**u**, **w**) and d(**v**, **w**)
e) Verify triangle inequality: d(**u**, **w**) ≤ d(**u**, **v**) + d(**v**, **w**)

**Problem 2.3: Different Distance Metrics**

For **u** = (2, 5, 1) and **v** = (5, 3, 4):
a) Calculate Euclidean distance (L2)
b) Calculate Manhattan distance (L1)
c) Calculate Chebyshev distance (L∞)
d) Which is largest? Smallest?
e) When would you use each metric?

**Problem 2.4: Nearest Neighbor**

Given points and a query:
- **a** = (1, 1)
- **b** = (2, 4)
- **c** = (4, 2)
- **query** = (3, 3)

a) Calculate distance from query to each point
b) Which point is nearest?
c) If these have labels: a=Red, b=Blue, c=Red, what's the KNN prediction (K=1)?
d) What if K=3?

**Problem 2.5: Customer Similarity**

Two customers with features (purchases, avg_spend, days_active):
- **Customer A:** (10, 50, 100)
- **Customer B:** (12, 55, 95)
- **Customer C:** (50, 200, 300)

a) Calculate d(**A**, **B**)
b) Calculate d(**A**, **C**)
c) Which customer is more similar to A?
d) Why might raw distances be misleading here?

**Problem 2.6: Standardization Impact**

Before standardization:
- **u** = (1000, 5, 20) (income=$1000, age=5... wait, that's weird!)
- **v** = (1100, 6, 22)

After standardization (subtract mean, divide by std):
- **u_std** = (0.2, 0.3, 0.4)
- **v_std** = (0.4, 0.5, 0.6)

a) Calculate d(**u**, **v**) before standardization
b) Calculate d(**u_std**, **v_std**) after
c) Which is more meaningful? Why?

**Problem 2.7: Time Series Distance**

Stock prices over 5 days:
- **Stock A:** (100, 102, 101, 103, 105)
- **Stock B:** (100, 101, 100, 102, 104)
- **Stock C:** (50, 51, 50, 52, 54)

a) Calculate d(**A**, **B**)
b) Calculate d(**A**, **C**)
c) Why is C far from A even though patterns are similar?
d) How could you make comparison fairer?

**Problem 2.8: Clustering Assignment**

Cluster centers:
- **C1** = (2, 2)
- **C2** = (8, 8)

New points to assign:
- **p1** = (3, 3)
- **p2** = (7, 6)
- **p3** = (5, 5)

a) Calculate distances from each point to each center
b) Assign each point to nearest cluster
c) Point p3 is equidistant-ish. Why might this be a problem?

**Problem 2.9: Anomaly Threshold**

Normal data has distances from center:
- Point 1: d = 2.1
- Point 2: d = 2.3
- Point 3: d = 1.8
- Point 4: d = 2.5

Average normal distance: ~2.2

New point: d = 8.5

a) How many standard deviations away is the new point?
b) Set threshold at mean + 2×std. Is new point anomaly?
c) What happens if threshold is too low? Too high?

**Problem 2.10: Image Similarity**

Three images represented as vectors (simplified):
- **img1** = (100, 120, 110, 130)
- **img2** = (105, 125, 115, 135) (slightly brighter)
- **img3** = (50, 60, 55, 65) (much darker)

a) Calculate d(**img1**, **img2**)
b) Calculate d(**img1**, **img3**)
c) Are img1 and img2 similar?
d) Could img1 and img3 be the same image with different lighting?

---

<a name="standard-deviation"></a>

## The Core Question: How Spread Out Is the Data?

You measure heights of students in two classes:

**Class A:** 170, 171, 169, 170, 170 cm (very consistent!)
**Class B:** 150, 160, 170, 180, 190 cm (very varied!)

**Both have the same average:** 170 cm

**But Class B is much more "spread out"!**

**Question:** How do we measure this spread mathematically?

## Why Average Isn't Enough

The **mean** (average) tells you the center, but nothing about variability.

**Example: Test Scores**
- Student A: 70, 70, 70, 70, 70 → Average: 70
- Student B: 0, 50, 70, 90, 140 → Average: 70

**Same average, completely different patterns!**

Student A is consistent.
Student B is all over the place!

**We need a measure of spread!**

## Building the Solution: Step by Step

### Step 1: Find the Average (Mean)

For Class A heights: [170, 171, 169, 170, 170]

Mean = (170 + 171 + 169 + 170 + 170) / 5 = 850 / 5 = 170

### Step 2: Find Deviations from Mean

**How far is each point from the center?**

Class A deviations:
- 170 - 170 = 0
- 171 - 170 = +1
- 169 - 170 = -1
- 170 - 170 = 0
- 170 - 170 = 0

Class B deviations:
- 150 - 170 = -20
- 160 - 170 = -10
- 170 - 170 = 0
- 180 - 170 = +10
- 190 - 170 = +20

**Class B has much larger deviations!**

### Step 3: Why We Square the Deviations

**Bad Idea:** Just average the deviations

Class A: (0 + 1 + (-1) + 0 + 0) / 5 = 0
Class B: (-20 + (-10) + 0 + 10 + 20) / 5 = 0

**Problem:** Positive and negative cancel out! Both give 0!

**Solution:** Square the deviations!

Class A squared deviations:
0², 1², (-1)², 0², 0² = 0, 1, 1, 0, 0

Class B squared deviations:
(-20)², (-10)², 0², 10², 20² = 400, 100, 0, 100, 400

**Now positive and negative both contribute positively!**

### Step 4: Average the Squared Deviations (Variance)

**Variance** = average of squared deviations

Class A variance:
σ²_A = (0 + 1 + 1 + 0 + 0) / 5 = 2/5 = 0.4

Class B variance:
σ²_B = (400 + 100 + 0 + 100 + 400) / 5 = 1000/5 = 200

**Class B has much higher variance!**

### Step 5: Take Square Root (Standard Deviation)

**Problem with variance:** Units are squared! (cm² instead of cm)

**Solution:** Take square root to get back to original units!

**Standard deviation** = √(variance)

Class A: σ_A = √0.4 ≈ 0.63 cm
Class B: σ_B = √200 ≈ 14.14 cm

**Perfect!** Now we have a measure of spread in the same units as the data.

## Formal Definition

For data x₁, x₂, ..., xₙ with mean μ:

**Variance:**
σ² = (1/n) Σᵢ₌₁ⁿ (xᵢ - μ)²

**Standard Deviation:**
σ = √[variance] = √[(1/n) Σᵢ₌₁ⁿ (xᵢ - μ)²]

**Alternative notation:**
- Variance: Var(X) or σ²
- Standard deviation: SD(X) or σ or s

## Connection to Vectors and Norms!

Think of your data as a vector: **x** = (x₁, x₂, ..., xₙ)

Create a **centered vector** (subtract mean from each):
**x_centered** = (x₁ - μ, x₂ - μ, ..., xₙ - μ)

**Standard deviation is related to the norm of this centered vector!**

σ = ||**x_centered**|| / √n

**This connects standard deviation to everything we learned about norms!**

## Population vs Sample Standard Deviation

**Population** (entire group):
σ = √[(1/n) Σ(xᵢ - μ)²]

**Sample** (subset of group):
s = √[(1/(n-1)) Σ(xᵢ - x̄)²]

**Note the (n-1) instead of n!** This is **Bessel's correction** - adjusts for bias when estimating from a sample.

**When to use which:**
- Use n if you have ALL the data (entire population)
- Use (n-1) if you have a sample and want to estimate population std

**In ML:** We usually use n (treating our dataset as the population of interest).

## Interpreting Standard Deviation

**Small σ:** Data tightly clustered around mean
**Large σ:** Data widely spread out

**Rule of thumb for normal distributions:**
- ~68% of data within 1σ of mean
- ~95% within 2σ
- ~99.7% within 3σ

**Example:** 
- Mean = 170 cm, σ = 10 cm
- 68% of heights between 160-180 cm
- 95% between 150-190 cm
- 99.7% between 140-200 cm

## Why Machine Learning Cares About Standard Deviation

### 1. Feature Standardization (Z-Score Normalization)

**Problem:** Features with different scales

- Feature 1: Income ($20k - $200k), σ ≈ $50k
- Feature 2: Age (20 - 65), σ ≈ 15 years

**Solution:** Standardize using z-scores!

z = (x - μ) / σ

After standardization:
- Mean = 0
- Standard deviation = 1
- All features on same scale!

**This helps:**
- Gradient descent converge faster
- Features contribute fairly
- Compare importance across features

### 2. Outlier Detection

**Question:** Is a data point unusual?

**Method:** Check how many standard deviations from mean

**Example:**
- Heights: mean = 170 cm, σ = 10 cm
- New person: 210 cm
- Z-score: (210 - 170) / 10 = 4

**4 standard deviations away → very unusual! (> 99.99% of data)**

**Typical threshold:** |z| > 3 for outliers

### 3. Measuring Model Uncertainty

**Prediction intervals** use standard deviation:

prediction ± 2σ gives ~95% confidence interval

**Example:** House price prediction
- Predicted: $500k
- σ of predictions: $50k
- 95% interval: $400k - $600k

### 4. Variance Explained (PCA)

**Principal Component Analysis:**
- Finds directions of maximum variance (σ²)
- Projects data onto these directions
- Keeps components with high variance

**High variance = important information!**

### 5. Batch Normalization

Normalize activations in neural networks:

x_normalized = (x - μ_batch) / σ_batch

**Helps training stability and speed!**

## Detailed Worked Examples

### Example 3.1: Test Score Variance

Class test scores: [65, 70, 75, 80, 85]

**Step 1: Mean**
μ = (65 + 70 + 75 + 80 + 85) / 5 = 375 / 5 = 75

**Step 2: Deviations**
- 65 - 75 = -10
- 70 - 75 = -5
- 75 - 75 = 0
- 80 - 75 = +5
- 85 - 75 = +10

**Step 3: Squared deviations**
100, 25, 0, 25, 100

**Step 4: Variance**
σ² = (100 + 25 + 0 + 25 + 100) / 5 = 250 / 5 = 50

**Step 5: Standard deviation**
σ = √50 ≈ 7.07 points

**Interpretation:** Typical deviation from mean is ~7 points

### Example 3.2: Income Standardization

Incomes: [$30k, $40k, $50k, $60k, $70k]

**Mean:** μ = $50k
**Std:** σ = √200 ≈ $14.14k

**Standardize (z-scores):**
- $30k: z = (30 - 50) / 14.14 ≈ -1.41
- $40k: z = (40 - 50) / 14.14 ≈ -0.71
- $50k: z = (50 - 50) / 14.14 = 0
- $60k: z = (60 - 50) / 14.14 ≈ +0.71
- $70k: z = (70 - 50) / 14.14 ≈ +1.41

**After standardization:**
- Mean of z-scores = 0
- Std of z-scores = 1
- All on standard scale!

### Example 3.3: Outlier Detection

Daily website visitors: [1000, 1100, 950, 1050, 1020, 980, 5000]

**Mean:** μ ≈ 1586
**Std:** σ ≈ 1410

**Check last point (5000):**
z = (5000 - 1586) / 1410 ≈ 2.42

**2.42 standard deviations away → Unusual but not extreme**

**If threshold is |z| > 3, this wouldn't be flagged**
**If threshold is |z| > 2, this would be flagged as outlier**

### Example 3.4: Comparing Variability

**Stock A returns (%):** [2, 3, 2.5, 3.5, 3]
Mean = 2.8%, σ_A ≈ 0.55%

**Stock B returns (%):** [-5, 10, -3, 12, 1]
Mean = 3%, σ_B ≈ 6.96%

**Stock B is much more volatile (higher σ)!**

**Risk-adjusted return (Sharpe ratio):**
Sharpe = Mean / σ

Stock A: 2.8 / 0.55 ≈ 5.09
Stock B: 3.0 / 6.96 ≈ 0.43

**Stock A has better risk-adjusted return!**

### Example 3.5: Feature Scaling Impact

**Before scaling:**
- Feature 1 (income): [20000, 40000, 60000], σ ≈ 16330
- Feature 2 (age): [25, 35, 45], σ ≈ 8.16

**Feature 1 dominates purely by scale!**

**After standardization (z-scores):**
- Feature 1: [-1.22, 0, 1.22], σ = 1
- Feature 2: [-1.22, 0, 1.22], σ = 1

**Now equal contribution!**

### Example 3.6: Normal Distribution Properties

Heights: μ = 170 cm, σ = 10 cm

**68% within 1σ:**
170 ± 10 = [160, 180] cm

**95% within 2σ:**
170 ± 20 = [150, 190] cm

**99.7% within 3σ:**
170 ± 30 = [140, 200] cm

**Someone 195 cm tall:**
z = (195 - 170) / 10 = 2.5

**Between 2σ and 3σ → in top ~1%!**

### Example 3.7: Prediction Confidence

Linear regression predicts house price:
- Prediction: $500k
- Residual std: $50k (from training errors)

**95% confidence interval:**
$500k ± 2($50k) = [$400k, $600k]

**Interpretation:** 95% confident true price is in this range

### Example 3.8: Variance in PCA

Data projections onto principal components:
- PC1: σ² = 45 (explains most variance)
- PC2: σ² = 15
- PC3: σ² = 5

**Total variance:** 45 + 15 + 5 = 65

**Variance explained:**
- PC1: 45/65 ≈ 69%
- PC2: 15/65 ≈ 23%
- PC3: 5/65 ≈ 8%

**Keep PC1 and PC2 → retain 92% of variance!**

## Practice Problems - Standard Deviation

**Problem 3.1: Basic Calculations**

Calculate mean, variance, and standard deviation:
a) Data: [2, 4, 6, 8, 10]
b) Data: [5, 5, 5, 5, 5]
c) Data: [0, 10, 0, 10, 0]
d) Which has highest variance? Lowest?

**Problem 3.2: Deviations**

For data [10, 15, 20, 25, 30]:
a) Calculate mean
b) List all deviations from mean
c) Verify deviations sum to zero
d) Why must deviations always sum to zero?

**Problem 3.3: Effect of Transformations**

Original data: [1, 2, 3, 4, 5], μ = 3, σ = √2

a) Add 10 to each value. New mean? New σ?
b) Multiply each by 2. New mean? New σ?
c) What's the rule for how transformations affect μ and σ?

**Problem 3.4: Z-Scores**

Data: [100, 110, 120, 130, 140], μ = 120, σ = √200

Calculate z-scores for:
a) 100
b) 120
c) 150
d) Which z-score indicates outlier (|z| > 2)?

**Problem 3.5: Comparing Datasets**

**Dataset A:** [10, 11, 12, 13, 14] (small range)
**Dataset B:** [5, 10, 15, 20, 25] (large range)

Both have mean = 12.

a) Calculate σ for each
b) Which is more spread out?
c) Does this match intuition?

**Problem 3.6: Outlier Impact**

Data without outlier: [10, 12, 11, 13, 12] → μ = 11.6, σ ≈ 1.02
Data with outlier: [10, 12, 11, 13, 100] → μ = 29.2, σ ≈ 37.

a) How much did outlier change mean?
b) How much did outlier change σ?
c) Which is more sensitive to outliers?

**Problem 3.7: Feature Standardization**

Two features:
- Income: [30k, 40k, 50k, 60k], μ = 45k, σ = 11.2k
- Age: [25, 35, 45, 55], μ = 40, σ = 11.2

a) Why do they have same σ but different ranges?
b) Standardize both using z-scores
c) Verify both have mean=0, σ=1 after
d) Why is this useful for ML?

**Problem 3.8: Stock Volatility**

**Stock X returns (%):** [2, 3, 2, 3, 2] → σ_X = 0.45
**Stock Y returns (%):** [-10, 20, -5, 15, -10] → σ_Y = 13.04

a) Which stock is more volatile?
b) If mean returns are equal, which would you prefer?
c) How does σ relate to risk?

**Problem 3.9: Test Score Interpretation**

Class: μ = 75, σ = 10

a) A student scores 85. What percentile (assuming normal)?
b) Another scores 65. What percentile?
c) Score needed to be in top 16% (1σ above mean)?
d) Range containing middle 68% of students?

**Problem 3.10: Population vs Sample**

Full class (population): [70, 75, 80, 85, 90]

a) Calculate population σ (divide by n)
b) Sample: just first 3 values [70, 75, 80]
c) Calculate sample s (divide by n-1)
d) Why do we use n-1 for samples?

---

<a name="angle"></a>
## Chapter 3: Norm, Distance, Standard Deviation, and Angles

### A First-Principles Approach with Detailed Examples

---


1. [Norm (Vector Length/Magnitude)](#norm)
2. [Distance Between Vectors](#distance)
3. [Standard Deviation and Spread](#standard-deviation)
4. [Angle and Cosine Similarity](#angle)
5. [Complexity Analysis](#complexity)
6. [Chapter Summary](#summary)
7. [Comprehensive Practice Problems](#practice)

---

<a name="norm"></a>

## The Fundamental Question: How "Big" Is a Vector?

Imagine you're hiking and your GPS shows your displacement from the starting point:
- 3 km east
- 4 km north

Your displacement vector: **v** = (3, 4)

**Question:** How far did you travel from your starting point in a straight line?

You didn't walk 3 + 4 = 7 km (that's the Manhattan distance, walking along streets).

You want the **straight-line distance** - the length of the arrow from start to your current position.

## Building the Solution: Pythagorean Theorem

Remember the Pythagorean theorem from geometry?

For a right triangle with sides a and b, and hypotenuse c:
c² = a² + b²

**Your displacement forms a right triangle!**
- Horizontal side: 3 km
- Vertical side: 4 km
- Hypotenuse: your straight-line distance

Distance = √(3² + 4²) = √(9 + 16) = √25 = 5 km

**This distance is the NORM (or length, or magnitude) of the vector!**

## Formal Definition

The **norm** (or **length** or **magnitude**) of a vector **v** = (v₁, v₂, ..., vₙ) is:

||**v**|| = √(v₁² + v₂² + ... + vₙ²)

**Notation:** 
- ||**v**|| means "the norm of v"
- Also written as |**v**| or ‖**v**‖
- Sometimes called L2 norm or Euclidean norm

**Key insight:** The norm is always a **non-negative number** (scalar), not a vector!

## Why Square and Then Square Root?

**Question:** Why this specific formula? Why not just add components or add absolute values?

### Bad Idea 1: Just Add Components

||**v**|| = v₁ + v₂ + ... + vₙ

**Problem:** What if components are negative?

**v** = (3, -4)
||**v**|| = 3 + (-4) = -1 (Negative length?! Makes no sense!)

### Bad Idea 2: Add Absolute Values (L1 Norm)

||**v**||₁ = |v₁| + |v₂| + ... + |vₙ|

**v** = (3, 4)
||**v**||₁ = |3| + |4| = 7

This actually works and is called the **L1 norm** or **Manhattan distance**!

**But:** This gives you the "city block" distance (walking on a grid), not straight-line distance.

**Think about it:** To go from (0,0) to (3,4), you walk 3 blocks east + 4 blocks north = 7 blocks total.

### Why Squares Work Best (L2 Norm / Euclidean Norm)

||**v**||₂ = √(v₁² + v₂² + ... + vₙ²)

**Advantages:**
1. **Squaring makes everything positive:** Both 3² and (-3)² give 9
2. **Geometric meaning:** Matches Pythagorean theorem (straight-line distance)
3. **Smooth and differentiable:** Important for optimization (gradients exist everywhere)
4. **Penalizes large values more:** Errors of 2 count as 4, errors of 10 count as 100
5. **Nice mathematical properties:** Works beautifully with inner products

**This is the "standard" norm in ML** unless specified otherwise!

## Connection to Inner Product

**Beautiful relationship:** The norm is the square root of the inner product of a vector with itself!

||**v**||² = **v · v** = v₁² + v₂² + ... + vₙ²

**Therefore:**
||**v**|| = √(**v · v**)

**This connects everything we've learned!**

## Examples in Different Dimensions

### Example 1.1: 1D Vectors (Just Numbers)

**v** = (5)
||**v**|| = √(5²) = √25 = 5

**v** = (-3)
||**v**|| = √((-3)²) = √9 = 3

**Key point:** Length is always positive, even for negative numbers!

### Example 1.2: 2D Vectors

**v** = (3, 4)
||**v**|| = √(3² + 4²) = √(9 + 16) = √25 = 5

**v** = (-1, -1)
||**v**|| = √((-1)² + (-1)²) = √(1 + 1) = √2 ≈ 1.414

**v** = (0, 5)
||**v**|| = √(0² + 5²) = √25 = 5 (pointing straight up)

**v** = (5, 0)
||**v**|| = √(5² + 0²) = 5 (pointing straight right)

### Example 1.3: 3D Vectors

**v** = (1, 2, 2)
||**v**|| = √(1² + 2² + 2²) = √(1 + 4 + 4) = √9 = 3

**v** = (2, -3, 6)
||**v**|| = √(4 + 9 + 36) = √49 = 7

**Think about it:** This is the distance from the origin (0, 0, 0) to the point (2, -3, 6) in 3D space!

### Example 1.4: High-Dimensional Vectors

**v** = (1, 1, 1, 1, 1) ∈ ℝ⁵
||**v**|| = √(1 + 1 + 1 + 1 + 1) = √5 ≈ 2.236

**v** = (15, 3, 4, 1, 1) (our spam email features!)
||**v**|| = √(15² + 3² + 4² + 1² + 1²)
= √(225 + 9 + 16 + 1 + 1)
= √252 ≈ 15.87

**Even though we can't visualize 5D space, the math works the same!**

## Properties of the Norm

These properties define what makes something a "norm":

### Property 1: Non-negativity
||**v**|| ≥ 0 for all **v**

**And:** ||**v**|| = 0 **if and only if** **v** = **0**

**Interpretation:** Length is never negative, and only the zero vector has zero length.

### Property 2: Homogeneity (Scaling)
||α**v**|| = |α| · ||**v**|| for any scalar α

**Example:**
**v** = (3, 4), ||**v**|| = 5

**2v** = (6, 8)
||**2v**|| = √(36 + 64) = √100 = 10 = 2 · 5 ✓

**-v** = (-3, -4)
||-**v**|| = √(9 + 16) = 5 = |-1| · 5 ✓

**Interpretation:** If you double the vector, you double its length. If you flip direction (multiply by -1), length stays the same.

### Property 3: Triangle Inequality
||**u + v**|| ≤ ||**u**|| + ||**v**||

**Interpretation:** The direct path is never longer than taking a detour!

**Think about it:** 
- Walk from A to B directly: ||**u + v**||
- Walk from A to C to B: ||**u**|| + ||**v**||
- Direct is always shorter or equal!

**Example:**
**u** = (1, 0), **v** = (0, 1)

||**u**|| = 1
||**v**|| = 1
||**u + v**|| = ||(1, 1)|| = √2 ≈ 1.414

Check: 1.414 ≤ 1 + 1 = 2 ✓

### Property 4: Relationship with Inner Product (Cauchy-Schwarz)
|**u · v**| ≤ ||**u**|| · ||**v**||

**This will be crucial when we discuss angles!**

## Unit Vectors: Vectors with Length 1

A **unit vector** is a vector with norm exactly equal to 1.

**To create a unit vector from any non-zero vector:**
**v̂** = **v** / ||**v**||

(Read as "v-hat" - the hat notation means "unit vector")

**This process is called normalization.**

### Example 1.5: Normalizing Vectors

**v** = (3, 4)
||**v**|| = 5

**v̂** = (3, 4) / 5 = (3/5, 4/5) = (0.6, 0.8)

**Verify:**
||**v̂**|| = √(0.6² + 0.8²) = √(0.36 + 0.64) = √1 = 1 ✓

**Key insight:** **v̂** points in the **same direction** as **v**, but has **length 1**!

### Example 1.6: Standard Basis Vectors Are Unit Vectors

**e₁** = (1, 0, 0)
||**e₁**|| = √(1² + 0² + 0²) = 1 ✓

**e₂** = (0, 1, 0)
||**e₂**|| = 1 ✓

**e₃** = (0, 0, 1)
||**e₃**|| = 1 ✓

**All standard basis vectors are unit vectors!**

## Why Machine Learning Cares About Norms

Norms are **everywhere** in ML!

### 1. Regularization: Preventing Overfitting

**Problem:** Large weights can cause models to overfit (memorize training data instead of learning patterns).

**Solution:** Add penalty for large weights!

**Ridge Regression (L2 regularization):**
Loss = MSE + λ||**w**||²

Where:
- MSE = prediction error
- ||**w**||² = sum of squared weights
- λ = regularization strength

**Effect:** Model prefers smaller weights → simpler, more generalizable models

**Lasso Regression (L1 regularization):**
Loss = MSE + λ||**w**||₁

Where ||**w**||₁ = sum of absolute values of weights

**Effect:** Can force some weights to exactly zero → automatic feature selection!

### 2. Normalization: Standardizing Inputs

**Problem:** Features with different scales can dominate learning.

**Example:**
- Feature 1: Income ($20k - $200k)
- Feature 2: Age (20 - 65 years)

Income dominates just because numbers are bigger!

**Solution:** Normalize feature vectors to have same scale

**Unit normalization:**
**x_normalized** = **x** / ||**x**||

Now all feature vectors have length 1!

### 3. Gradient Clipping: Stabilizing Training

**Problem:** Sometimes gradients become very large (exploding gradients in deep learning).

**Solution:** If ||∇L|| > threshold, scale it down!

if ||∇L|| > max_norm:
    ∇L = ∇L · (max_norm / ||∇L||)

**Effect:** Gradient direction preserved, but magnitude limited.

### 4. Measuring Prediction Confidence

In neural networks, the norm of output vectors can indicate confidence.

**Example:** Image classification
- Output: **y** = (0.1, 0.1, 0.7, 0.1) (probabilities for 4 classes)
- ||**y**|| = √(0.01 + 0.01 + 0.49 + 0.01) = √0.52 ≈ 0.72

Large norm → confident prediction
Small norm → uncertain

### 5. Batch Normalization

Normalize activations in neural networks:
**a_normalized** = (**a** - mean) / std

Helps training converge faster and more stably!

## Norm Squared: A Useful Shortcut

Often in ML, we use **||v||²** instead of ||**v**| because:

**Advantages:**
1. **No square root needed** → computationally faster
2. **Easier to differentiate** → simpler gradients
3. **Still monotonic** → bigger ||**v**|| means bigger ||**v**||²

||**v**||² = **v · v** = v₁² + v₂² + ... + vₙ²

**When comparing lengths, ||v||² works just as well as ||v||!**

### Example 1.7: Comparing Distances

Which is closer to origin: **u** = (3, 4) or **v** = (2, 5)?

**Method 1: Using norm**
||**u**|| = √25 = 5
||**v**|| = √29 ≈ 5.39
**u** is closer

**Method 2: Using squared norm (faster!)**
||**u**||² = 25
||**v**||² = 29
**u** is closer (same answer, no square roots!)

## Different Types of Norms

### L0 "Norm" (Not Really a Norm)
||**v**||₀ = number of non-zero components

**v** = (0, 3, 0, 5, 0)
||**v**||₀ = 2 (two non-zero entries)

**Use:** Counting sparsity (how many features are active)

### L1 Norm (Manhattan Distance)
||**v**||₁ = |v₁| + |v₂| + ... + |vₙ|

**v** = (3, -4)
||**v**||₁ = |3| + |-4| = 3 + 4 = 7

**Use:** Lasso regularization, robust to outliers

### L2 Norm (Euclidean Distance) - The Standard!
||**v**||₂ = √(v₁² + v₂² + ... + vₙ²)

**This is what we mean by "norm" unless specified otherwise!**

### L∞ Norm (Maximum Norm)
||**v**||∞ = max(|v₁|, |v₂|, ..., |vₙ|)

**v** = (3, -7, 2)
||**v**||∞ = max(3, 7, 2) = 7

**Use:** Measuring worst-case deviation

### Comparison Example

**v** = (3, -4, 0)

||**v**||₀ = 2 (two non-zero)
||**v**||₁ = 3 + 4 + 0 = 7
||**v**||₂ = √(9 + 16 + 0) = √25 = 5
||**v**||∞ = max(3, 4, 0) = 4

**General relationship:** ||**v**||∞ ≤ ||**v**||₂ ≤ ||**v**||₁

## Detailed Worked Examples

### Example 1.8: GPS Navigation

Your hiking trail:
- Start: Origin (0, 0)
- Checkpoint 1: (3, 4) km
- Checkpoint 2: (8, 6) km
- End: (10, 2) km

**Calculate distances from origin:**

To Checkpoint 1:
**v₁** = (3, 4)
||**v₁**|| = √(9 + 16) = 5 km

To Checkpoint 2:
**v₂** = (8, 6)
||**v₂**|| = √(64 + 36) = √100 = 10 km

To End:
**v₃** = (10, 2)
||**v₃**|| = √(100 + 4) = √104 ≈ 10.2 km

**Which checkpoint is furthest?** Checkpoint 2 (10 km)

### Example 1.9: Feature Vector Magnitude

Customer behavior vector:
**c** = (purchases, avg_spend, days_active, reviews)
= (23, 150, 365, 12)

||**c**|| = √(23² + 150² + 365² + 12²)
= √(529 + 22500 + 133225 + 144)
= √156398 ≈ 395.47

**Interpretation:** This is the "magnitude" of customer engagement.

**Compare two customers:**
- Customer A: (23, 150, 365, 12), ||**cₐ**|| ≈ 395.47
- Customer B: (50, 200, 730, 25), ||**cᵦ**|| ≈ 762.76

Customer B has higher engagement magnitude!

### Example 1.10: Neural Network Weight Initialization

Initialize weights with small random values:

**w** = (0.01, -0.02, 0.015, -0.008, 0.012)

||**w**|| = √(0.0001 + 0.0004 + 0.000225 + 0.000064 + 0.000144)
= √0.001033 ≈ 0.032

**Check:** ||**w**|| < 0.1 ✓ (good initialization - small weights)

### Example 1.11: Normalizing Image Pixels

Image pixel vector (simplified, 3 pixels):
**img** = (128, 200, 64) (pixel brightness 0-255)

||**img**|| = √(16384 + 40000 + 4096) = √60480 ≈ 245.93

**Normalized:**
**img_normalized** = **img** / ||**img**||
= (128, 200, 64) / 245.93
= (0.520, 0.813, 0.260)

Now ||**img_normalized**|| = 1 ✓

### Example 1.12: Gradient Magnitude

Loss gradient: ∇L = (2.5, -1.8, 3.2, -0.9)

||∇L|| = √(6.25 + 3.24 + 10.24 + 0.81)
= √20.54 ≈ 4.53

**If gradient is too large (>10), clip it:**

Since 4.53 < 10, no clipping needed.

But if ||∇L|| = 15, then:
∇L_clipped = ∇L · (10 / 15) = 0.667 · ∇L

### Example 1.13: Portfolio Volatility

Stock returns vector (5 days):
**r** = (0.02, -0.01, 0.03, -0.02, 0.01) (daily returns as decimals)

||**r**|| = √(0.0004 + 0.0001 + 0.0009 + 0.0004 + 0.0001)
= √0.0019 ≈ 0.0436

**Interpretation:** This measures the magnitude of price movements (volatility indicator).

## Practice Problems - Norm

**Problem 1.1: Basic Norm Calculations**

Calculate ||**v**|| for:
a) **v** = (5, 12)
b) **v** = (-3, 4)
c) **v** = (1, 1, 1)
d) **v** = (2, -2, 1, -1)
e) **v** = (0, 0, 0)

**Problem 1.2: Pythagorean Triples**

Verify these are Pythagorean triples by calculating norms:
a) (3, 4) should have norm 5
b) (5, 12) should have norm 13
c) (8, 15) should have norm 17
d) (7, 24) should have norm 25

**Problem 1.3: Normalization**

Normalize these vectors (find unit vectors):
a) **v** = (3, 4)
b) **v** = (1, 1)
c) **v** = (0, 5)
d) **v** = (2, -2, 1)

Verify each normalized vector has norm 1.

**Problem 1.4: Comparing Magnitudes**

Which vector has larger norm?
a) **u** = (3, 4) vs **v** = (5, 2)
b) **u** = (1, 1, 1, 1) vs **v** = (2, 0, 0, 0)
c) **u** = (10, 1) vs **v** = (1, 10)

**Problem 1.5: Properties Verification**

For **u** = (3, 4) and scalar α = 2:
a) Calculate ||**u**||
b) Calculate ||α**u**||
c) Verify ||α**u**|| = |α| · ||**u**||
d) What if α = -3? Verify the property still holds.

**Problem 1.6: Triangle Inequality**

For **u** = (1, 2) and **v** = (3, 1):
a) Calculate ||**u**||, ||**v**||, and ||**u + v**||
b) Verify ||**u + v**|| ≤ ||**u**|| + ||**v**||
c) When does equality hold in triangle inequality?

**Problem 1.7: Different Norms**

For **v** = (3, -4, 5):
a) Calculate L1 norm: ||**v**||₁
b) Calculate L2 norm: ||**v**||₂
c) Calculate L∞ norm: ||**v**||∞
d) Which is largest? Why?

**Problem 1.8: Sparse Vectors**

Vector **v** = (0, 5, 0, 0, 3, 0, 0, 7, 0)
a) Calculate ||**v**||₀ (count non-zeros)
b) Calculate ||**v**||₁
c) Calculate ||**v**||₂
d) Why is this vector called "sparse"?

**Problem 1.9: Feature Scaling**

Two features with different scales:
- **f₁** = (1000, 2000, 1500) (income in $)
- **f₂** = (25, 35, 30) (age in years)

a) Calculate ||**f₁**|| and ||**f₂**||
b) Normalize both to unit vectors
c) Now calculate norms. What do you notice?
d) Why is this normalization useful in ML?

**Problem 1.10: Gradient Clipping**

Gradient: ∇L = (8, -6, 12, -4)
Max allowed norm: 10

a) Calculate ||∇L||
b) Is clipping needed?
c) If yes, calculate clipped gradient
d) Verify clipped gradient has norm ≤ 10

---

<a name="distance"></a>

## The Core Question: How Far Apart Are Two Things?

You have two movie preference vectors:
- **Alice:** (5, 2, 1, 4) = ratings for (action, comedy, drama, horror)
- **Bob:** (4, 3, 1, 3)

**Question:** How similar are their movie tastes?

To answer this, we need to measure how "far apart" their preference vectors are!

## Building the Solution: The Difference Vector

**First insight:** To measure distance between two points, find how they differ!

**Bob's preferences - Alice's preferences:**
**b - a** = (4, 3, 1, 3) - (5, 2, 1, 4) = (-1, 1, 0, -1)

**This difference vector tells us:**
- Action: Bob rates 1 point lower
- Comedy: Bob rates 1 point higher  
- Drama: Same!
- Horror: Bob rates 1 point lower

**Second insight:** The LENGTH of this difference vector is the distance!

distance(**a**, **b**) = ||**b - a**|| = ||(-1, 1, 0, -1)||
= √(1 + 1 + 0 + 1) = √3 ≈ 1.73

## Formal Definition

The **Euclidean distance** between vectors **u** and **v** is:

d(**u**, **v**) = ||**u - v**|| = √[(u₁-v₁)² + (u₂-v₂)² + ... + (uₙ-vₙ)²]

**Alternative formula using inner product:**
d(**u**, **v**) = √[(**u - v**) · (**u - v**)]

**Key properties:**
- Always non-negative: d(**u**, **v**) ≥ 0
- Zero iff identical: d(**u**, **v**) = 0 ⟺ **u** = **v**
- Symmetric: d(**u**, **v**) = d(**v**, **u**)
- Triangle inequality: d(**u**, **w**) ≤ d(**u**, **v**) + d(**v**, **w**)

## Geometric Interpretation in 2D

Two points in a plane:
- **p₁** = (1, 2)
- **p₂** = (4, 6)

**Visualize:** Plot these points. The distance is the length of the straight line connecting them!

d(**p₁**, **p₂**) = ||**p₂ - p₁**|| = ||(3, 4)|| = 5

**You can literally measure this with a ruler on graph paper!**

## Why This Formula Works

The distance formula comes directly from the Pythagorean theorem!

**Think about moving from **p₁** to **p₂**:**
- Horizontal change: Δx = 4 - 1 = 3
- Vertical change: Δy = 6 - 2 = 4
- These form a right triangle!
- Hypotenuse (distance): √(3² + 4²) = 5

**This extends to any dimension!**

## Properties of Distance

### Property 1: Non-negativity
d(**u**, **v**) ≥ 0

Distance is never negative!

### Property 2: Identity of Indiscernibles  
d(**u**, **v**) = 0 if and only if **u** = **v**

Zero distance means the vectors are identical.

### Property 3: Symmetry
d(**u**, **v**) = d(**v**, **u**)

Distance from A to B equals distance from B to A.

**Proof:**
d(**u**, **v**) = ||**u - v**||
d(**v**, **u**) = ||**v - u**|| = ||**-(u - v)**|| = |-1| · ||**u - v**|| = ||**u - v**|| ✓

### Property 4: Triangle Inequality
d(**u**, **w**) ≤ d(**u**, **v**) + d(**v**, **w**)

**Interpretation:** Going directly from **u** to **w** is never longer than going via **v**!

Think about driving:
- Direct route from home to store
- vs going home → friend's house → store

Direct is always shorter (or equal if friend is on the way)!

## Why Machine Learning Cares About Distance

Distance is **absolutely fundamental** to ML!

### 1. K-Nearest Neighbors (KNN)

**Algorithm:** To classify a new point:
1. Find the K closest training examples (smallest distances)
2. Use their labels to vote

**Example: Email Spam Detection**
- New email: **x** = (5, 20, 1, 50)
- Known spam: **s** = (12, 25, 1, 30)
- Known ham: **h** = (1, 3, 0, 200)

d(**x**, **s**) = ||(5, 20, 1, 50) - (12, 25, 1, 30)||
= ||(-7, -5, 0, 20)||
= √(49 + 25 + 0 + 400) = √474 ≈ 21.8

d(**x**, **h**) = ||(4, 17, 1, -150)||
= √(16 + 289 + 1 + 22500) = √22806 ≈ 151

**x** is MUCH closer to spam example → Classify as spam!

### 2. Clustering (K-Means)

**Goal:** Group similar data points together

**How:** Points are "similar" if distance is small!

**Algorithm:**
1. Assign each point to nearest cluster center
2. Update centers (mean of assigned points)
3. Repeat until convergence

**All based on distance calculations!**

### 3. Anomaly Detection

**Question:** Is this data point unusual?

**Answer:** If it has large distance from all normal examples → Anomaly!

**Example: Fraud Detection**
- Normal transaction: **t_normal** ≈ (50, 1, 10)
- New transaction: **t_new** = (10000, 5, 1000)

d(**t_new**, **t_normal**) = very large → Suspicious!

### 4. Recommendation Systems

**Find users with similar preferences:**

- Your ratings: **you** = (5, 1, 4, 2, 5)
- User A: **a** = (5, 2, 4, 1, 5)
- User B: **b** = (1, 5, 1, 5, 2)

d(**you**, **a**) = small → Similar tastes!
d(**you**, **b**) = large → Different tastes!

**Recommendation:** Show what similar users liked!

### 5. Loss Functions

**Mean Squared Error** is based on distance!

MSE = (1/n) Σᵢ ||**yᵢ** - **ŷᵢ**||²

Average squared distance between predictions and true values!

## Distance vs. Similarity

**Key insight:** Small distance = high similarity!

Often we convert distance to similarity:

**Similarity metrics:**
1. sim = 1 / (1 + distance)
2. sim = e^(-distance)
3. sim = 1 - (distance / max_distance)

**Example:**
- d = 0 → sim = 1 (identical)
- d = 1 → sim = 0.5 (somewhat similar)
- d = ∞ → sim = 0 (completely different)

## Different Distance Metrics

### Euclidean Distance (L2) - The Standard!
d₂(**u**, **v**) = ||**u - v**||₂ = √[Σ(uᵢ - vᵢ)²]

**Use:** General purpose, most common

### Manhattan Distance (L1)
d₁(**u**, **v**) = ||**u - v**||₁ = Σ|uᵢ - vᵢ|

**Use:** When you can only move along axes (like city blocks)

### Chebyshev Distance (L∞)
d∞(**u**, **v**) = ||**u - v**||∞ = max|uᵢ

---

## Practice Problems - Complexity

**Problem 5.1: Operation Counting**

Vector dimension n = 1,000. Estimate operations for:
a) Calculate norm ||**v**||
b) Calculate distance d(**u**, **v**)
c) Calculate cosine similarity
d) Which is slowest?

**Problem 5.2: Scaling Analysis**

KNN with N = 10,000 points, n = 100 dimensions.

a) Operations for one query?
b) Double dimensions (n = 200). New cost?
c) Double data (N = 20,000). New cost?
d) Which has bigger impact?

**Problem 5.3: Memory Requirements**

Store N = 1,000,000 vectors, n = 1,000 dimensions, 4 bytes/float.

a) Memory for data matrix?
b) Memory for distance matrix (if we computed all pairs)?
c) Which is feasible to store?

**Problem 5.4: Sparse vs Dense**

Document vector: n = 100,000 vocabulary, k = 100 non-zero words.

a) Dense inner product cost?
b) Sparse inner product cost?
c) Speedup ratio?

**Problem 5.5: Batch Processing**

N = 1,000,000 examples, n = 1,000 features.

Full batch: Process all N at once
Mini-batch: m = 1,000 at a time

a) Cost per full batch update?
b) Cost per mini-batch update?
c) How many mini-batches for one epoch?
d) Total cost for one epoch with mini-batches?

---

<a name="summary"></a>
# Chapter Summary

## Key Concepts We've Learned

### 1. Norm (Vector Length)
- **Definition:** ||**v**|| = √(v₁² + v₂² + ... + vₙ²)
- **Measures:** Size/magnitude of a vector
- **ML uses:** Regularization, normalization, gradient clipping
- **Complexity:** O(n)
- **Key insight:** ||**v**||² = **v · v** (connects to inner product!)

### 2. Distance
- **Definition:** d(**u**, **v**) = ||**u - v**||
- **Measures:** How far apart two vectors are
- **ML uses:** KNN, clustering, anomaly detection, recommendations
- **Complexity:** O(n)
- **Key insight:** Distance squared avoids expensive square root

### 3. Standard Deviation
- **Definition:** σ = √[(1/n)Σ(xᵢ - μ)²]
- **Measures:** Spread/variability of data
- **ML uses:** Standardization (z-scores), outlier detection, confidence intervals
- **Complexity:** O(n)
- **Key insight:** Related to norm of centered data vector

### 4. Angle and Cosine Similarity
- **Definition:** cos(θ) = (**u · v**) / (||**u**|| ||**v**||)
- **Measures:** Directional similarity (independent of magnitude)
- **ML uses:** Text similarity, recommendations, attention, face recognition
- **Complexity:** O(n)
- **Key insight:** -1 (opposite) to +1 (same direction), 0 = perpendicular

## How Everything Connects

```
Inner Product (**u · v**)
       ↓
    Norm (||**v**|| = √(**v · v**))
       ↓
    Distance (d = ||**u - v||)
       ↓
    Cosine Similarity (cos θ = (**u · v**) / (||**u**|| ||**v**||))
```

**The inner product is the foundation of everything!**

## When to Use Each Measure

**Use Norm when:**
- Measuring vector "size"
- Regularizing model complexity
- Normalizing data
- Checking gradient magnitude

**Use Distance when:**
- Finding nearest neighbors
- Measuring how different things are
- Clustering by proximity
- Absolute differences matter

**Use Standard Deviation when:**
- Understanding data spread
- Standardizing features
- Detecting outliers
- Measuring uncertainty

**Use Cosine Similarity when:**
- Direction matters more than magnitude
- Different scales in data
- Text/document similarity
- Recommendations based on patterns
- High-dimensional sparse data

## Key Formulas to Remember

**Norm:**
```
||v|| = √(v₁² + v₂² + ... + vₙ²)
||v||² = v · v
```

**Distance:**
```
d(u, v) = ||u - v|| = √[Σ(uᵢ - vᵢ)²]
```

**Standard Deviation:**
```
σ = √[(1/n)Σ(xᵢ - μ)²]
z-score = (x - μ) / σ
```

**Cosine Similarity:**
```
cos(θ) = (u · v) / (||u|| ||v||)
-1 ≤ cos(θ) ≤ 1
```

## Common Pitfalls and Tips

**Pitfall 1: Forgetting to normalize**
- ❌ Comparing distances with different scales
- ✓ Standardize features first!

**Pitfall 2: Using distance when direction matters**
- ❌ Euclidean distance for user ratings (magnitude varies)
- ✓ Cosine similarity for patterns!

**Pitfall 3: Computing unnecessary square roots**
- ❌ If only comparing, use ||v||²
- ✓ Squared distances preserve ordering!

**Pitfall 4: Storing full distance matrices**
- ❌ O(N²) memory - too large!
- ✓ Compute on-the-fly or use approximations

**Pitfall 5: Ignoring feature scales**
- ❌ Income ($1000s) dominates age (years)
- ✓ Standardize using z-scores!

---

<a name="practice"></a>
# Comprehensive Practice Problems

## Section 1: Integrated Concepts

**Problem 6.1: Complete Data Analysis**

Customer data: [purchases, spending, days_active]
- Customer A: (10, 500, 100)
- Customer B: (12, 600, 120)
- Customer C: (50, 2000, 400)

a) Calculate mean customer vector
b) Calculate standard deviation for each feature
c) Standardize all customers using z-scores
d) Calculate distances between standardized customers
e) Calculate cosine similarities
f) Which measure better captures similarity?

**Problem 6.2: Movie Recommendation System**

5 users, 5 movies, ratings 1-5:
- User 1: (5, 1, 4, 2, 5)
- User 2: (4, 2, 3, 2, 4)
- User 3: (1, 5, 2, 5, 1)
- User 4: (5, 1, 5, 1, 5)
- User 5: (10, 2, 8, 4, 10)

a) Find most similar user to User 1 (by cosine)
b) Notice User 5 = 2 × User 1. What's their cosine similarity?
c) What's their Euclidean distance?
d) Which metric is better for recommendations?
e) Create recommendation for User 1

**Problem 6.3: Document Clustering**

10 documents, represent as word count vectors.
Vocabulary: ["machine", "learning", "cooking", "recipe"]

- Docs 1-5: ML documents with high counts for "machine", "learning"
- Docs 6-10: Cooking documents with high counts for "cooking", "recipe"

a) What would cosine similarity matrix look like? (sketch pattern)
b) Should ML docs cluster together?
c) What's expected cosine between Doc1 and Doc6?
d) How would you implement this efficiently for 1M documents?

**Problem 6.4: Anomaly Detection Pipeline**

Normal transactions: [amount, frequency, location_distance]
- Normal 1: (50, 2, 5)
- Normal 2: (45, 3, 4)
- Normal 3: (55, 2, 6)
- New: (500, 10, 100)

a) Calculate mean and std for each feature
b) Calculate z-scores for new transaction
c) Calculate distance from new to mean
d) Is this an anomaly? (use both z-scores and distance)
e) Which features contribute most to anomaly score?

**Problem 6.5: Feature Engineering Impact**

Original features: [income, age]
- Person A: (100000, 30)
- Person B: (120000, 35)

a) Calculate Euclidean distance
b) Calculate cosine similarity
c) Standardize both features
d) Recalculate distance and cosine on standardized
e) Which changed more? Why?

## Section 2: Real-World Applications

**Problem 6.6: Face Verification**

Face embedding vectors (simplified to 5D):
- Person1_Photo1: (0.8, 0.2, 0.5, 0.3, 0.9)
- Person1_Photo2: (0.7, 0.3, 0.4, 0.3, 0.8)
- Person2_Photo1: (0.1, 0.9, 0.2, 0.8, 0.1)

a) Cosine similarity between Person1's two photos
b) Cosine between Person1_Photo1 and Person2_Photo1
c) Set threshold at 0.8. Would system verify Person1?
d) Why cosine instead of distance?

**Problem 6.7: Stock Portfolio Optimization**

5 stocks, returns over 10 days.
Want to find stocks that move differently (diversify).

a) Calculate standard deviation for each stock
b) Calculate pairwise correlations (cosine similarities)
c) Which pair is most negatively correlated?
d) Recommend portfolio combining negatively correlated stocks
e) Why is standard deviation important here?

**Problem 6.8: Text Search Engine**

Query: "python machine learning"
Query vector: (1, 1, 1) for ["python", "machine", "learning"]

1000 documents to search.

a) What's complexity of ranking all documents?
b) Each doc has ~100 non-zero words out of 10,000 vocabulary. How does sparsity help?
c) Describe efficient implementation
d) Would you use cosine or distance? Why?

**Problem 6.9: Image Similarity**

Compare images by histogram vectors (256 bins).

Image1: Bright outdoor photo
Image2: Same photo, reduced brightness
Image3: Different scene entirely

a) What would distance measure show?
b) What would cosine similarity show?
c) Which correctly identifies Image1 and Image2 as same scene?
d) How would you handle brightness normalization?

**Problem 6.10: Collaborative Filtering**

User-item rating matrix (missing values exist).
1000 users, 5000 items.

a) Calculate similarity between all user pairs. What's complexity?
b) Too expensive! Propose approximation method.
c) Why cosine similarity standard for collaborative filtering?
d) Handle missing ratings in similarity calculation

## Section 3: Theoretical Understanding

**Problem 6.11: Proving Properties**

Prove using the definitions:
a) ||α**v**|| = |α| · ||**v**||
b) d(**u**, **v**) = d(**v**, **u**)
c) If **u** ⊥ **v**, then ||**u** + **v**||² = ||**u**||² + ||**v**||²
d) For unit vectors, cos(θ) = **u · v**

**Problem 6.12: Cauchy-Schwarz Inequality**

The inequality states: |**u · v**| ≤ ||**u**|| · ||**v**||

a) Explain what this means geometrically
b) When does equality hold?
c) How does this relate to cosine similarity?
d) Verify for **u** = (3, 4), **v** = (5, 12)

**Problem 6.13: Triangle Inequality**

For any vectors: ||**u** + **v**|| ≤ ||**u**|| + ||**v**||

a) Draw diagram showing this in 2D
b) When does equality hold?
c) Verify for **u** = (3, 4), **v** = (1, 1)
d) Why is this called "triangle" inequality?

**Problem 6.14: Orthogonal Decomposition**

Given **v** and **u**, decompose **v** = **v**‖ + **v**⊥

where **v**‖ is parallel to **u** and **v**⊥ is perpendicular.

a) Formula: **v**‖ = [(**v · u**) / ||**u**||²] **u**
b) Show **v**⊥ = **v** - **v**‖
c) Verify **v**‖ · **v**⊥ = 0
d) Apply to **v** = (5, 5), **u** = (1, 0)

**Problem 6.15: Different Norms**

For **v** = (3, -4, 5):
a) L1 norm: ||**v**||₁ = Σ|vᵢ|
b) L2 norm: ||**v**||₂ = √(Σvᵢ²)
c) L∞ norm: ||**v**||∞ = max|vᵢ|
d) Prove: ||**v**||∞ ≤ ||**v**||₂ ≤ ||**v**||₁

## Section 4: Algorithmic Thinking

**Problem 6.16: Efficient KNN**

Given N = 1,000,000 points, n = 100 dimensions, K = 10.

a) Naive approach complexity?
b) Using squared distances instead of distances, what's saved?
c) Early termination: stop if found K points with distance < threshold. Best/worst case?
d) Using approximate methods (LSH), can reduce to O(log N). Worth the tradeoff?

**Problem 6.17: Online Mean and Std**

Calculate mean and std as data arrives (streaming).

a) Naive: store all values, recalculate. Memory?
b) Welford's algorithm: update mean and M₂ incrementally. Describe.
c) What's memory complexity?
d) Implement for data: 2, 4, 6, 8, 10

**Problem 6.18: Matrix-Free Distance**

Avoid creating full N×N distance matrix.

a) Compute distances on-the-fly in KNN. What's saved?
b) For clustering, need distances to K centers. Complexity?
c) Trade-off: recomputing vs storing
d) When is each approach better?

**Problem 6.19: Approximate Nearest Neighbors**

LSH (Locality Sensitive Hashing) for cosine similarity.

a) Basic idea: similar vectors → same hash
b) Complexity reduction: O(Nn) → O(log N)
c) Cost: may miss true nearest neighbor
d) When is approximation acceptable?

**Problem 6.20: GPU Parallelization**

Inner product on GPU with 1000 cores.

a) Each core computes one multiplication
b) Then reduce (tree-based sum): O(log n) steps
c) Compare to sequential O(n)
d) For what vector sizes is GPU worth overhead?

---

# Answer Key (Selected Problems)

**Problem 6.1:**
c) After standardization, all features have mean=0, std=1
f) Cosine better captures pattern similarity; distance affected by scales

**Problem 6.2:**
b) cos = 1.0 (perfect similarity - same direction)
c) distance = ||v||₁ = √(1² + 0² + 1² + 0² + 1²) × 5 = √75 ≈ 8.66
d) Cosine better - captures rating pattern regardless of scale

**Problem 6.13:**
b) Equality when **v** = α**u** (parallel vectors)

**Problem 6.14:**
d) **v**‖ = (5, 0), **v**⊥ = (0, 5), verify perpendicular ✓

---

# Congratulations!

You've completed Chapter 3! You now understand:
- ✅ How to measure vector size (norm)
- ✅ How to measure separation (distance)
- ✅ How to measure spread (standard deviation)
- ✅ How to measure direction (angle/cosine)
- ✅ When to use each measure
- ✅ How they all connect through inner products
- ✅ Computational complexity considerations

**These are the fundamental measurement tools of machine learning!**

**You're ready for Chapter 4: Clustering Algorithms!**

---

*End of Chapter 3*---

<a name="angle"></a>

## The Core Question: Do Two Vectors Point in Similar Directions?

Imagine two students rating movies:

**Alice:** (5, 1, 4, 2, 5) = (loves action, hates comedy, likes drama, neutral horror, loves sci-fi)
**Bob:** (10, 2, 8, 4, 10) = exactly 2× Alice's ratings!

**Notice:** Bob's ratings are just Alice's scaled by 2!

**They have the SAME preferences**, just on a different scale!

**Geometrically:** Their vectors point in the **same direction**, but Bob's is longer.

**Question:** How do we measure if vectors point in similar directions, independent of their lengths?

**Answer:** Using the **angle** between them!

## Why Direction Matters More Than Magnitude

### Example: Movie Recommendations

**User A:** (5, 1, 4, 2) - Uses full 1-5 scale
**User B:** (3, 0, 2, 1) - More conservative, uses 0-3 scale

**Even though magnitudes differ, the *pattern* is the same!**

Both users:
- Love genre 1 (highest rating)
- Hate genre 2 (lowest rating)
- Like genre 3 (medium-high)
- Neutral on genre 4 (low-medium)

**Distance would say they're different** (different magnitudes)
**Angle says they're similar** (same direction/pattern)

**For recommendations, direction matters more than magnitude!**

## Building the Solution: The Inner Product Formula

Remember from geometry: For two 2D vectors with angle θ between them:

**u · v** = ||**u**|| · ||**v**|| · cos(θ)

**Rearranging to solve for angle:**

cos(θ) = (**u · v**) / (||**u**|| · ||**v**||)

**This works in ANY dimension!**

**Cosine similarity** = cos(θ) = (**u · v**) / (||**u**|| · ||**v**||)

## Understanding Cosine Values

**cos(θ) = 1** → θ = 0° → vectors point **same direction** (perfectly similar!)
**cos(θ) = 0** → θ = 90° → vectors **perpendicular** (unrelated!)
**cos(θ) = -1** → θ = 180° → vectors point **opposite directions** (perfectly dissimilar!)

**In between:**
- cos(θ) close to 1 → small angle → similar directions
- cos(θ) close to 0 → ~90° → unrelated directions
- cos(θ) close to -1 → ~180° → opposite directions

**Key advantage:** Cosine similarity is **normalized** - always between -1 and 1!

## Examples of Different Angles

### Example 4.1: Same Direction (0°)

**u** = (3, 4)
**v** = (6, 8) = 2**u**

**u · v** = 3(6) + 4(8) = 18 + 32 = 50
||**u**|| = √(9 + 16) = 5
||**v**|| = √(36 + 64) = 10

cos(θ) = 50 / (5 × 10) = 50 / 50 = 1

**θ = arccos(1) = 0°** ✓

**Vectors point in exactly the same direction!**

### Example 4.2: Perpendicular (90°)

**u** = (1, 0) - pointing east
**v** = (0, 1) - pointing north

**u · v** = 1(0) + 0(1) = 0
||**u**|| = 1
||**v**|| = 1

cos(θ) = 0 / (1 × 1) = 0

**θ = arccos(0) = 90°** ✓

**Vectors are perpendicular (orthogonal)!**

### Example 4.3: Opposite Directions (180°)

**u** = (1, 1)
**v** = (-1, -1) = -**u**

**u · v** = 1(-1) + 1(-1) = -1 - 1 = -2
||**u**|| = √2
||**v**|| = √2

cos(θ) = -2 / (√2 × √2) = -2 / 2 = -1

**θ = arccos(-1) = 180°** ✓

**Vectors point in opposite directions!**

### Example 4.4: 45° Angle

**u** = (1, 0)
**v** = (1, 1)

**u · v** = 1(1) + 0(1) = 1
||**u**|| = 1
||**v**|| = √2

cos(θ) = 1 / (1 × √2) = 1/√2 ≈ 0.707

**θ = arccos(0.707) ≈ 45°** ✓

**Exactly halfway between same direction and perpendicular!**

## Why Orthogonality (90°) Is Special

When **u · v** = 0, we say vectors are **orthogonal** (perpendicular).

**Why special?**

**1. Independent Information**
If features are orthogonal, they provide completely independent information!

**Example:** Image features
- **h** = horizontal edge detector
- **v** = vertical edge detector

**h · v** = 0 (orthogonal!)

Horizontal edges don't affect vertical edge detection - perfect independence!

**2. No Correlation**
Orthogonal features are uncorrelated - no redundancy!

**3. Simplifies Mathematics**
Many formulas become much simpler with orthogonal vectors.

**4. Optimal for Machine Learning**
- PCA finds orthogonal directions
- Neural network layers often initialized with orthogonal weights
- Orthogonal features are ideal (no multicollinearity)

## Cosine Similarity vs Euclidean Distance

**When to use cosine similarity (angle):**
- Direction/pattern matters more than magnitude
- Data has different scales
- Text analysis (TF-IDF vectors)
- Recommendations (rating patterns)
- High-dimensional sparse data

**When to use Euclidean distance:**
- Absolute differences matter
- Data on same scale
- Clustering in physical space
- K-nearest neighbors (often)

### Example: The Difference

**User A:** (5, 1, 5)
**User B:** (10, 2, 10) = 2 × **A**

**Euclidean distance:**
d(**A**, **B**) = ||(5, 1, 5)|| = √(25 + 1 + 25) = √51 ≈ 7.14
(Considers them different due to magnitude)

**Cosine similarity:**
cos(θ) = **A · B** / (||**A**|| × ||**B**||)
= (50 + 2 + 50) / (√51 × √204)
= 102 / 102 = 1
(Perfect similarity - same direction!)

**For recommendations, cosine is better here!**

## Why Machine Learning Cares About Angles

### 1. Document Similarity (Text Analysis)

**Bag of words representation:**

**Doc 1:** "cat cat dog" → (2, 1) for vocab ["cat", "dog"]
**Doc 2:** "cat cat cat dog dog dog" → (3, 3)

**Distance** would say they're different (different magnitudes)
**Cosine similarity:**

cos(θ) = (2×3 + 1×3) / (√5 × √18) = 9 / √90 ≈ 0.949

**Very high similarity! Same topic, different lengths.**

**This is why cosine similarity is standard for text!**

### 2. Recommendation Systems

**User A:** (5, 1, 4, 2, 5)
**User B:** (4, 0, 3, 1, 4)

cos(θ) = (20 + 0 + 12 + 2 + 20) / (√51 × √42)
= 54 / √2142 ≈ 0.984

**High similarity → recommend what User A liked to User B!**

**Works even though User B rates more conservatively.**

### 3. Face Recognition

Face embeddings are vectors in high-dimensional space.

**Same person, different photos:**
- Photo 1: **f₁** (certain lighting)
- Photo 2: **f₂** (different lighting)

Lighting changes magnitude, but face identity (direction) stays similar!

**Cosine similarity > threshold → same person!**

### 4. Word Embeddings (Word2Vec, GloVe)

Words are vectors where direction encodes meaning.

**Famous example:**
- king - man + woman ≈ queen

**Similarity by cosine:**
- cos(**king**, **queen**) = high (both royalty)
- cos(**king**, **table**) = low (unrelated)

**Direction captures semantic relationships!**

### 5. Attention Mechanisms (Transformers, GPT)

**Attention score** between query and key:

attention(**Q**, **K**) ∝ **Q · K** / (||**Q**|| × ||**K**||)

**This is cosine similarity!**

High cosine → words should "pay attention" to each other.

**The entire attention mechanism is based on measuring directional similarity!**

### 6. Clustering with Cosine Similarity

**K-means with cosine distance:**

Instead of Euclidean distance, use:
distance = 1 - cos(θ)

**Advantages:**
- Handles different magnitudes
- Better for sparse, high-dimensional data
- Standard for document clustering

## Connection Between All Concepts

**Beautiful unified formula:**

cos(θ) = (**u · v**) / (||**u**|| × ||**v**||)

This connects:
- **Inner product** (**u · v**) - measures alignment
- **Norm** (||**u**||, ||**v**||) - measures length
- **Angle** (θ) - measures direction similarity

**Everything we've learned comes together!**

## Detailed Worked Examples

### Example 4.5: Movie Recommendations

**Alice:** (5, 1, 4, 2, 5) - genres: action, comedy, drama, horror, sci-fi
**Bob:** (4, 2, 3, 2, 4)
**Carol:** (1, 5, 2, 5, 1)

**Compare Alice to Bob:**

**A · B** = 5(4) + 1(2) + 4(3) + 2(2) + 5(4) = 20 + 2 + 12 + 4 + 20 = 58
||**A**|| = √(25 + 1 + 16 + 4 + 25) = √71 ≈ 8.43
||**B**|| = √(16 + 4 + 9 + 4 + 16) = √49 = 7

cos(θ_AB) = 58 / (8.43 × 7) = 58 / 59.01 ≈ 0.983

**Very high! Alice and Bob have similar tastes.**

**Compare Alice to Carol:**

**A · C** = 5(1) + 1(5) + 4(2) + 2(5) + 5(1) = 5 + 5 + 8 + 10 + 5 = 33
||**C**|| = √(1 + 25 + 4 + 25 + 1) = √56 ≈ 7.48

cos(θ_AC) = 33 / (8.43 × 7.48) = 33 / 63.06 ≈ 0.523

**Moderate similarity. Some overlap but different preferences.**

**Recommendation:** Show Alice what Bob liked (high similarity), not what Carol liked.

### Example 4.6: Document Similarity

Vocabulary: ["machine", "learning", "deep", "neural", "network"]

**Doc 1** (about ML): (5, 8, 2, 1, 3) - mostly "machine" and "learning"
**Doc 2** (about deep learning): (3, 5, 10, 8, 7) - balanced, emphasizes "deep" and "neural"
**Doc 3** (about cooking): (0, 0, 1, 0, 0) - mentioned "deep" (deep frying!)

**Doc 1 vs Doc 2:**

**D₁ · D₂** = 5(3) + 8(5) + 2(10) + 1(8) + 3(7) = 15 + 40 + 20 + 8 + 21 = 104
||**D₁**|| = √(25 + 64 + 4 + 1 + 9) = √103 ≈ 10.15
||**D₂**|| = √(9 + 25 + 100 + 64 + 49) = √247 ≈ 15.72

cos(θ) = 104 / (10.15 × 15.72) ≈ 0.652

**Moderately similar - both about machine learning!**

**Doc 1 vs Doc 3:**

**D₁ · D₃** = 5(0) + 8(0) + 2(1) + 1(0) + 3(0) = 2
||**D₃**|| = 1

cos(θ) = 2 / (10.15 × 1) ≈ 0.197

**Very low similarity - different topics!**

### Example 4.7: Perpendicular Features

**Feature 1** (horizontal edges in image): (1, 0, 0, 0)
**Feature 2** (vertical edges): (0, 1, 0, 0)
**Feature 3** (diagonal edges): (0, 0, 1, 0)
**Feature 4** (brightness): (0, 0, 0, 1)

**Check if F1 and F2 are orthogonal:**

**F₁ · F₂** = 1(0) + 0(1) + 0(0) + 0(0) = 0 ✓

cos(θ) = 0 / (1 × 1) = 0
θ = 90°

**Perfectly orthogonal! They provide independent information.**

**Check all pairs:**
All pairs have inner product = 0 → all orthogonal!

**This is an orthonormal basis** (orthogonal + length 1)!

### Example 4.8: Stock Correlation

Stock returns over 5 days:

**Stock A:** (2, -1, 3, -2, 1)%
**Stock B:** (3, -2, 4, -3, 2)% = 1.5 × **A** (moves together!)
**Stock C:** (-2, 1, -3, 2, -1)% = -1 × **A** (moves opposite!)

**A vs B:**

**A · B** = 2(3) + (-1)(-2) + 3(4) + (-2)(-3) + 1(2) = 6 + 2 + 12 + 6 + 2 = 28
||**A**|| = √(4 + 1 + 9 + 4 + 1) = √19 ≈ 4.36
||**B**|| = √(9 + 4 + 16 + 9 + 4) = √42 ≈ 6.48

cos(θ_AB) = 28 / (4.36 × 6.48) ≈ 0.992

**Nearly perfect correlation! They move together.**

**A vs C:**

**A · C** = 2(-2) + (-1)(1) + 3(-3) + (-2)(2) + 1(-1) = -4 - 1 - 9 - 4 - 1 = -19

cos(θ_AC) = -19 / (4.36 × 4.36) ≈ -1.0

**Perfect negative correlation! They move opposite.**

**Portfolio diversification:** Combine A and C to reduce volatility!

### Example 4.9: Plagiarism Detection

**Student A's essay** (word frequencies): (10, 5, 3, 8, 12, 2, ...)
**Student B's essay:** (30, 15, 9, 24, 36, 6, ...) = 3 × **A**

cos(θ) = **A · B** / (||**A**|| × ||**B**||)

Since **B** = 3**A**:
**A · B** = **A** · (3**A**) = 3(**A · A**) = 3||**A**||²
||**B**|| = 3||**A**||

cos(θ) = 3||**A**||² / (||**A**|| × 3||**A**||) = 3||**A**||² / 3||**A**||² = 1

**Perfect similarity! Likely plagiarism** (even though lengths differ).

### Example 4.10: Search Engine Ranking

Query: "machine learning python"
Query vector: (1, 1, 1) for ["machine", "learning", "python"]

**Document A:** (3, 2, 5) - tutorial with lots of code
**Document B:** (10, 10, 1) - theory-heavy, less code
**Document C:** (5, 4, 6) - balanced

**Rank by cosine similarity:**

**Query · DocA** = 1(3) + 1(2) + 1(5) = 10
||Query|| = √3 ≈ 1.73
||DocA|| = √(9 + 4 + 25) = √38 ≈ 6.16
cos(θ_A) = 10 / (1.73 × 6.16) ≈ 0.938

**Query · DocB** = 1(10) + 1(10) + 1(1) = 21
||DocB|| = √(100 + 100 + 1) = √201 ≈ 14.18
cos(θ_B) = 21 / (1.73 × 14.18) ≈ 0.856

**Query · DocC** = 1(5) + 1(4) + 1(6) = 15
||DocC|| = √(25 + 16 + 36) = √77 ≈ 8.77
cos(θ_C) = 15 / (1.73 × 8.77) ≈ 0.988

**Ranking:** DocC (0.988) > DocA (0.938) > DocB (0.856)

**Show Doc C first!**

### Example 4.11: Image Similarity

Two images represented as pixel vectors (simplified):

**Image 1:** (100, 150, 80, 200) - certain brightness
**Image 2:** (50, 75, 40, 100) = 0.5 × **Image1** - darker version!

cos(θ) = **I₁ · I₂** / (||**I₁**|| × ||**I₂**||)

Since **I₂** = 0.5**I₁**:
cos(θ) = 1

**Perfect similarity! Same image, different brightness.**

**This is why cosine similarity is used in image retrieval!**

### Example 4.12: Neural Network Attention

Query vector: **Q** = (1, 2, 3)
Key vectors:
- **K₁** = (1, 2, 3) - exactly matches query
- **K₂** = (0, 1, 0) - somewhat related
- **K₃** = (-1, -2, -3) - opposite

**Attention scores (simplified, using cosine):**

cos(**Q**, **K₁**) = (1 + 4 + 9) / (√14 × √14) = 14/14 = 1.0 ← High attention!
cos(**Q**, **K₂**) = (0 + 2 + 0) / (√14 × 1) = 2/√14 ≈ 0.53 ← Medium
cos(**Q**, **K₃**) = (-1 - 4 - 9) / (√14 × √14) = -14/14 = -1.0 ← Negative

**K₁ gets most attention (highest cosine similarity)!**

## Practice Problems - Angle and Cosine Similarity

**Problem 4.1: Basic Angle Calculations**

Calculate cos(θ) for these pairs:
a) **u** = (1, 0), **v** = (0, 1)
b) **u** = (3, 4), **v** = (6, 8)
c) **u** = (1, 1), **v** = (-1, -1)
d) **u** = (1, 1), **v** = (1, -1)

For each, state whether they're: same direction, perpendicular, or opposite.

**Problem 4.2: Orthogonality Check**

Which pairs are orthogonal (perpendicular)?
a) **u** = (2, 3), **v** = (3, -2)
b) **u** = (1, 2, 3), **v** = (3, -2, 0)
c) **u** = (1, 1, 1), **v** = (1, -1, 0)
d) **u** = (4, 0, 0), **v** = (0, 5, 0)

**Problem 4.3: Movie Preferences**

Three users rate 4 genres (action, comedy, drama, horror):
- **Alice:** (5, 2, 4, 1)
- **Bob:** (4, 3, 3, 2)
- **Carol:** (1, 5, 2, 5)

a) Calculate cos(θ) for Alice vs Bob
b) Calculate cos(θ) for Alice vs Carol
c) Who has more similar taste to Alice?
d) Should you recommend Bob's favorites to Alice?

**Problem 4.4: Document Clustering**

Word counts for vocabulary ["data", "science", "art", "paint"]:
- **Doc 1:** (10, 8, 0, 0)
- **Doc 2:** (8, 10, 0, 0)
- **Doc 3:** (0, 0, 7, 9)

a) Calculate cosine similarity for all pairs
b) Which documents should be clustered together?
c) What's the angle between Doc 1 and Doc 3?

**Problem 4.5: Magnitude Independence**

**User A:** (5, 1, 5)
**User B:** (10, 2, 10) = 2 × **A**
**User C:** (5, 5, 5)

a) Calculate Euclidean distance between A and B
b) Calculate cosine similarity between A and B
c) Which measure shows they have same preferences?
d) Calculate cosine between A and C. Are they similar?

**Problem 4.6: Correlation**

Stock returns over 3 days:
- **Stock X:** (2, -1, 3)
- **Stock Y:** (3, -1.5, 4.5) = 1.5 × **X**
- **Stock Z:** (-2, 1, -3) = -1 × **X**

a) cos(**X**, **Y**) = ?
b) cos(**X**, **Z**) = ?
c) Are X and Y positively correlated?
d) Are X and Z negatively correlated?
e) Which pair would diversify a portfolio?

**Problem 4.7: Feature Independence**

Image features:
- **Horizontal edges:** (1, 0, 0)
- **Vertical edges:** (0, 1, 0)
- **Brightness:** (0, 0, 1)

a) Show all three pairs are orthogonal
b) What does this mean for information content?
c) Is this basis orthonormal? Check.

**Problem 4.8: Search Relevance**

Query: "python programming"
Query vector: (1, 1) for ["python", "programming"]

Documents:
- **Doc A:** (5, 5) - balanced content
- **Doc B:** (10, 2) - mostly "python", less "programming"
- **Doc C:** (3, 7) - more "programming", less "python"

a) Calculate cosine similarity for each document
b) Rank documents by relevance
c) Would Euclidean distance give same ranking?

**Problem 4.9: Plagiarism Score**

**Original:** (10, 5, 8, 3, 12)
**Submission 1:** (30, 15, 24, 9, 36) = 3 × **Original**
**Submission 2:** (11, 6, 7, 4, 13)

a) cos(Original, Submission1) = ?
b) cos(Original, Submission2) = ?
c) Which is more suspicious?
d) Why is cosine better than distance for plagiarism?

**Problem 4.10: Understanding Negative Cosine**

**u** = (1, 2, 3)
**v** = (-2, -4, -6) = -2**u**

a) Calculate **u · v**
b) Calculate cos(θ)
c) What does negative cosine mean?
d) What's the angle θ?

---

<a name="complexity"></a>

## Why Care About Computational Complexity?

In real ML applications:
- Vectors can have millions of dimensions (image pixels, word vocabularies)
- Datasets can have billions of examples (user interactions, sensor readings)
- Operations are repeated millions of times (training iterations)

**Understanding complexity helps us:**
1. Estimate runtime
2. Choose efficient algorithms
3. Scale to larger problems
4. Identify bottlenecks

## Complexity of Vector Operations

Assume vectors have dimension n.

### Norm Calculation: ||**v**||

**Operation:** Compute √(v₁² + v₂² + ... + vₙ²)

**Steps:**
1. Square each component: n multiplications
2. Sum all squares: n-1 additions
3. Take square root: 1 operation

**Complexity:** **O(n)**

**Squared norm:** ||**v**||² skips square root → still O(n) but faster constant

### Distance Calculation: d(**u**, **v**)

**Operation:** ||**u** - **v**|| = √[(u₁-v₁)² + ... + (uₙ-vₙ)²]

**Steps:**
1. Subtract components: n subtractions
2. Square differences: n multiplications
3. Sum: n-1 additions
4. Square root: 1 operation

**Complexity:** **O(n)**

**Squared distance:** d²(**u**, **v**) skips square root → still O(n)

### Cosine Similarity

**Operation:** cos(θ) = (**u · v**) / (||**u**|| × ||**v**||)

**Steps:**
1. Inner product: O(n)
2. Norm of **u**: O(n)
3. Norm of **v**: O(n)
4. Division: O(1)

**Total:** O(n) + O(n) + O(n) = **O(n)**

**All basic vector operations are linear in dimension!**

## Complexity for Multiple Vectors

### K-Nearest Neighbors

**Input:** N training points, 1 query point, dimension n

**For each training point:**
- Calculate distance: O(n)

**Total:** N × O(n) = **O(Nn)**

**For K-nearest:** Also need to sort/track K smallest → O(N log K)

**Full complexity:** **O(Nn + N log K)** ≈ **O(Nn)** for small K

### K-Means (One Iteration)

**Input:** N points, K clusters, dimension n

**Assignment step:**
- For each point: calculate K distances
- Each distance: O(n)
- Total: N × K × O(n) = **O(NKn)**

**Update step:**
- For each cluster: sum assigned points
- Total: O(Nn)

**One iteration:** **O(NKn)**

**Full algorithm:** I iterations → **O(INKn)**

### Cosine Similarity Matrix

**Input:** N documents, dimension n

**Need:** Similarity between all pairs

**Pairs:** N(N-1)/2 ≈ N²/2

**Each pair:** O(n)

**Total:** **O(N²n)**

**This gets expensive fast!**
- N = 1,000: ~500,000 comparisons
- N = 10,000: ~50,000,000 comparisons
- N = 100,000: ~5,000,000,000 comparisons

## Memory Complexity

**Single vector:** O(n) memory

**Matrix (N×n):** O(Nn) memory

**Distance matrix (N×N):** O(N²) memory

**Example:**
- 1,000,000 documents
- 10,000 dimensions
- 4 bytes per float

**Data matrix:** 1M × 10K × 4 bytes = 40 GB
**Distance matrix:** 1M × 1M × 4 bytes = 4 TB!

**Distance matrix often too large to store!**

## Practical Implications

### 1. Feature Selection Matters

**Before:** n = 10,000 features
**After:** n = 100 features (selected best)

**Speedup:** 100× faster for all operations!

### 2. Mini-Batches for Large N

Instead of processing all N points:
- Process m << N at a time
- Update iteratively

**Per batch:** O(mn) instead of O(Nn)

### 3. Sparse Vectors

If vector has only k non-zero entries:
- Dense: O(n) operations
- Sparse: O(k) operations

**Example:** Text vectors
- Vocabulary: n = 100,000
- Non-zero words per document: k = 100
- Speedup: 1000×!

### 4. Approximation Methods

For very large N:
- Locality Sensitive Hashing (LSH)
- Approximate nearest neighbors
- Random projections

Trade accuracy for massive speedup!

### 5. GPU Acceleration

**GPU advantage:** Parallel computation

**Vector operations are highly parallelizable!**

- CPU: O(n) time sequentially
- GPU: O(log n) time with n processors

**Practical speedup:** 10-100× for large vectors

## Comparison Table

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Norm ||**v**|| | O(n) | Linear in dimension |
| Distance d(**u**, **v**) | O(n) | Linear in dimension |
| Inner product **u · v** | O(n) | Linear in dimension |
| Cosine similarity | O(n) | Three O(n) operations |
| KNN (1 query) | O(Nn) | N comparisons |
| K-means iteration | O(NKn) | All pairwise distances |
| Pairwise distances | O(N²n) | All pairs |
| Standard deviation | O(n) | One pass through data |

## Practice Problems - Complexity

**Problem 5.1: Operation Counting**

Vector dimension n = 1,000. Estimate operations for:
a) Calculate norm ||**v**||
b) Calculate distance d(**u**, **v**)
c) Calculate cosine similarity
d) Which is slowest?

**Problem 5.2: Scaling Analysis**

KNN with N = 10,000 points, n = 100 dimensions.

a) Operations for one query?
b) Double dimensions (n = 200). New cost?
c) Double data (N = 20,000). New cost?
d) Which has bigger impact?

**Problem 5.3: Memory Requirements**

Store N = 1,000,000 vectors, n = 1,000 dimensions,### Chebyshev Distance (L∞)
d∞(**u**, **v**) = ||**u - v**||∞ = max|uᵢ - vᵢ|

**Use:** Measuring worst-case difference

### Comparison Example

**u** = (2, 5, 1)
**v** = (5, 3, 4)

**Difference:** **u - v** = (-3, 2, -3)

**Euclidean (L2):**
d₂ = √(9 + 4 + 9) = √22 ≈ 4.69

**Manhattan (L1):**
d₁ = |−3| + |2| + |−3| = 3 + 2 + 3 = 8

**Chebyshev (L∞):**
d∞ = max(3, 2, 3) = 3

**General relationship:** d∞ ≤ d₂ ≤ d₁

## Squared Distance: A Computational Shortcut

Just like with norms, we often use **squared distance**:

d²(**u**, **v**) = ||**u - v**||²

**Advantages:**
1. No square root → faster computation
2. Preserves ordering (if d₁ < d₂, then d₁² < d₂²)
3. Easier to differentiate

**When comparing distances, squared distance works just as well!**

### Example: Finding Nearest Neighbor

Which is closer to **x** = (0, 0)?
- **a** = (3, 4)
- **b** = (5, 1)

**Method 1: Using distance**
d(**x**, **a**) = √25 = 5
d(**x**, **b**) = √26 ≈ 5.1
**a** is closer

**Method 2: Using squared distance (faster!)**
d²(**x**, **a**) = 25
d²(**x**, **b**) = 26
**a** is closer (same answer, no square roots!)

## Detailed Worked Examples

### Example 2.1: Customer Segmentation

Two customer profiles:
- **Customer A:** (age=25, income=50k, purchases=10, satisfaction=8)
- **Customer B:** (age=28, income=55k, purchases=12, satisfaction=7)

**A** = (25, 50, 10, 8)
**B** = (28, 55, 12, 7)

d(**A**, **B**) = ||(28, 55, 12, 7) - (25, 50, 10, 8)||
= ||(3, 5, 2, -1)||
= √(9 + 25 + 4 + 1)
= √39 ≈ 6.24

**Note:** Different features have different scales! 
- Age differs by 3 years
- Income differs by $5k

**Better approach:** Standardize features first!

After standardization (z-scores):
**A_std** = (0.2, 0.1, 0.3, 0.5)
**B_std** = (0.4, 0.3, 0.5, 0.3)

d(**A_std**, **B_std**) = ||(0.2, 0.2, 0.2, -0.2)||
= √(0.04 + 0.04 + 0.04 + 0.04)
= √0.16 = 0.4

**Much better!** Now all features contribute fairly.

### Example 2.2: Image Similarity

Two 2×2 grayscale images (flattened to vectors):

**Image 1:** [100, 120, 110, 130] (brightness values)
**Image 2:** [105, 125, 115, 135]

d(**img1**, **img2**) = ||(5, 5, 5, 5)||
= √(25 + 25 + 25 + 25)
= √100 = 10

**Small distance → images are similar!**

**Image 3:** [200, 50, 180, 70]

d(**img1**, **img3**) = ||(100, -70, 70, -60)||
= √(10000 + 4900 + 4900 + 3600)
= √23400 ≈ 153

**Large distance → images are very different!**

### Example 2.3: Document Similarity

Word count vectors (vocabulary: ["cat", "dog", "bird", "fish"]):

**Doc 1:** (5, 2, 0, 1) - mentions cats a lot
**Doc 2:** (6, 1, 0, 2) - also about cats
**Doc 3:** (0, 0, 8, 5) - about birds and fish

d(**doc1**, **doc2**) = ||(1, -1, 0, 1)||
= √(1 + 1 + 0 + 1) = √3 ≈ 1.73 (similar!)

d(**doc1**, **doc3**) = ||(-5, -2, 8, 4)||
= √(25 + 4 + 64 + 16) = √109 ≈ 10.44 (very different!)

**Doc 1 and Doc 2 are about similar topics!**

### Example 2.4: Time Series Comparison

Temperature readings over 5 hours:

**City A:** (20, 21, 23, 24, 25)°C
**City B:** (22, 23, 24, 25, 26)°C

d(**A**, **B**) = ||(2, 2, 1, 1, 1)||
= √(4 + 4 + 1 + 1 + 1)
= √11 ≈ 3.32

**City B is consistently ~2°C warmer**

**City C:** (20, 15, 25, 18, 28)°C

d(**A**, **C**) = ||(0, -6, 2, -6, 3)||
= √(0 + 36 + 4 + 36 + 9)
= √85 ≈ 9.22

**City C has more variable weather (larger distance from A)**

### Example 2.5: K-Nearest Neighbors

Training data (2D for visualization):
- Point A: (1, 2), Label: Red
- Point B: (2, 1), Label: Red
- Point C: (5, 6), Label: Blue
- Point D: (6, 5), Label: Blue

New point to classify: **x** = (3, 3)

**Calculate distances:**
d(**x**, **A**) = ||(2, 1)|| = √5 ≈ 2.24
d(**x**, **B**) = ||(1, 2)|| = √5 ≈ 2.24
d(**x**, **C**) = ||(2, 3)|| = √13 ≈ 3.61
d(**x**, **D**) = ||(3, 2)|| = √13 ≈ 3.61

**For K=2 (2 nearest neighbors):**
Nearest: A and B (both Red)
**Prediction: Red!**

### Example 2.6: Anomaly Detection

Normal system metrics:
- **Normal 1:** (CPU=30%, Memory=40%, Disk=50%, Network=20%)
- **Normal 2:** (32%, 38%, 52%, 18%)
- **Normal 3:** (28%, 42%, 48%, 22%)

Average normal: **avg** ≈ (30, 40, 50, 20)

New reading: **new** = (85, 90, 95, 15)

d(**new**, **avg**) = ||(55, 50, 45, -5)||
= √(3025 + 2500 + 2025 + 25)
= √7575 ≈ 87.0

**Very large distance from normal → ANOMALY!**

### Example 2.7: Clustering Quality

Two clusters with centers:
- **C1** = (2, 3)
- **C2** = (8, 7)

Distance between cluster centers:
d(**C1**, **C2**) = ||(6, 4)|| = √52 ≈ 7.21

**Points in Cluster 1:**
- **p1** = (2, 3), d(**p1**, **C1**) = 0
- **p2** = (3, 4), d(**p2**, **C1**) = √2 ≈ 1.41
- **p3** = (1, 2), d(**p3**, **C1**) = √2 ≈ 1.41

**Average distance within cluster:** (0 + 1.41 + 1.41) / 3 ≈ 0.94

**Good clustering:** 
- Small within-cluster distances (0.94)
- Large between-cluster distances (7.21)
- Ratio: 7.21 / 0.94 ≈ 7.7 (well-separated!)

## Practice Problems - Distance

**Problem 2.1: Basic Distance Calculations**

Calculate d(**u**, **v**) for:
a) **u** = (1, 2), **v** = (4, 6)
b) **u** = (0, 0), **v** = (3, 4)
c) **u** = (5, 2, 1), **v** = (1, 2, 5)
d) **u** = (1, 1, 1, 1), **v** = (2, 2, 2, 2)

**Problem 2.2: Distance Properties**

For **u** = (1, 2), **v** = (4, 6), **w** = (7, 10):
a) Calculate d(**u**, **v**)
b) Calculate d(**v**, **u**)
c) Verify d(**u**, **v**) = d(**v**, **u**) (symmetry)
d) Calculate d(**u**, **w**) and d(**v**, **w**)
e) Verify triangle inequality: d(**u**, **w**) ≤ d(**u**, **v**) + d(**v**, **w**)

**Problem 2.3: Different Distance Metrics**

For **u** = (2, 5, 1) and **v** = (5, 3, 4):
a) Calculate Euclidean distance (L2)
b) Calculate Manhattan distance (L1)
c) Calculate Chebyshev distance (L∞)
d) Which is largest? Smallest?
e) When would you use each metric?

**Problem 2.4: Nearest Neighbor**

Given points and a query:
- **a** = (1, 1)
- **b** = (2, 4)
- **c** = (4, 2)
- **query** = (3, 3)

a) Calculate distance from query to each point
b) Which point is nearest?
c) If these have labels: a=Red, b=Blue, c=Red, what's the KNN prediction (K=1)?
d) What if K=3?

**Problem 2.5: Customer Similarity**

Two customers with features (purchases, avg_spend, days_active):
- **Customer A:** (10, 50, 100)
- **Customer B:** (12, 55, 95)
- **Customer C:** (50, 200, 300)

a) Calculate d(**A**, **B**)
b) Calculate d(**A**, **C**)
c) Which customer is more similar to A?
d) Why might raw distances be misleading here?

**Problem 2.6: Standardization Impact**

Before standardization:
- **u** = (1000, 5, 20) (income=$1000, age=5... wait, that's weird!)
- **v** = (1100, 6, 22)

After standardization (subtract mean, divide by std):
- **u_std** = (0.2, 0.3, 0.4)
- **v_std** = (0.4, 0.5, 0.6)

a) Calculate d(**u**, **v**) before standardization
b) Calculate d(**u_std**, **v_std**) after
c) Which is more meaningful? Why?

**Problem 2.7: Time Series Distance**

Stock prices over 5 days:
- **Stock A:** (100, 102, 101, 103, 105)
- **Stock B:** (100, 101, 100, 102, 104)
- **Stock C:** (50, 51, 50, 52, 54)

a) Calculate d(**A**, **B**)
b) Calculate d(**A**, **C**)
c) Why is C far from A even though patterns are similar?
d) How could you make comparison fairer?

**Problem 2.8: Clustering Assignment**

Cluster centers:
- **C1** = (2, 2)
- **C2** = (8, 8)

New points to assign:
- **p1** = (3, 3)
- **p2** = (7, 6)
- **p3** = (5, 5)

a) Calculate distances from each point to each center
b) Assign each point to nearest cluster
c) Point p3 is equidistant-ish. Why might this be a problem?

**Problem 2.9: Anomaly Threshold**

Normal data has distances from center:
- Point 1: d = 2.1
- Point 2: d = 2.3
- Point 3: d = 1.8
- Point 4: d = 2.5

Average normal distance: ~2.2

New point: d = 8.5

a) How many standard deviations away is the new point?
b) Set threshold at mean + 2×std. Is new point anomaly?
c) What happens if threshold is too low? Too high?

**Problem 2.10: Image Similarity**

Three images represented as vectors (simplified):
- **img1** = (100, 120, 110, 130)
- **img2** = (105, 125, 115, 135) (slightly brighter)
- **img3** = (50, 60, 55, 65) (much darker)

a) Calculate d(**img1**, **img2**)
b) Calculate d(**img1**, **img3**)
c) Are img1 and img2 similar?
d) Could img1 and img3 be the same image with different lighting?

---

<a name="standard-deviation"></a>

## The Core Question: How Spread Out Is the Data?

You measure heights of students in two classes:

**Class A:** 170, 171, 169, 170, 170 cm (very consistent!)
**Class B:** 150, 160, 170, 180, 190 cm (very varied!)

**Both have the same average:** 170 cm

**But Class B is much more "spread out"!**

**Question:** How do we measure this spread mathematically?

## Why Average Isn't Enough

The **mean** (average) tells you the center, but nothing about variability.

**Example: Test Scores**
- Student A: 70, 70, 70, 70, 70 → Average: 70
- Student B: 0, 50, 70, 90, 140 → Average: 70

**Same average, completely different patterns!**

Student A is consistent.
Student B is all over the place!

**We need a measure of spread!**

## Building the Solution: Step by Step

### Step 1: Find the Average (Mean)

For Class A heights: [170, 171, 169, 170, 170]

Mean = (170 + 171 + 169 + 170 + 170) / 5 = 850 / 5 = 170

### Step 2: Find Deviations from Mean

**How far is each point from the center?**

Class A deviations:
- 170 - 170 = 0
- 171 - 170 = +1
- 169 - 170 = -1
- 170 - 170 = 0
- 170 - 170 = 0

Class B deviations:
- 150 - 170 = -20
- 160 - 170 = -10
- 170 - 170 = 0
- 180 - 170 = +10
- 190 - 170 = +20

**Class B has much larger deviations!**

### Step 3: Why We Square the Deviations

**Bad Idea:** Just average the deviations

Class A: (0 + 1 + (-1) + 0 + 0) / 5 = 0
Class B: (-20 + (-10) + 0 + 10 + 20) / 5 = 0

**Problem:** Positive and negative cancel out! Both give 0!

**Solution:** Square the deviations!

Class A squared deviations:
0², 1², (-1)², 0², 0² = 0, 1, 1, 0, 0

Class B squared deviations:
(-20)², (-10)², 0², 10², 20² = 400, 100, 0, 100, 400

**Now positive and negative both contribute positively!**

### Step 4: Average the Squared Deviations (Variance)

**Variance** = average of squared deviations

Class A variance:
σ²_A = (0 + 1 + 1 + 0 + 0) / 5 = 2/5 = 0.4

Class B variance:
σ²_B = (400 + 100 + 0 + 100 + 400) / 5 = 1000/5 = 200

**Class B has much higher variance!**

### Step 5: Take Square Root (Standard Deviation)

**Problem with variance:** Units are squared! (cm² instead of cm)

**Solution:** Take square root to get back to original units!

**Standard deviation** = √(variance)

Class A: σ_A = √0.4 ≈ 0.63 cm
Class B: σ_B = √200 ≈ 14.14 cm

**Perfect!** Now we have a measure of spread in the same units as the data.

## Formal Definition

For data x₁, x₂, ..., xₙ with mean μ:

**Variance:**
σ² = (1/n) Σᵢ₌₁ⁿ (xᵢ - μ)²

**Standard Deviation:**
σ = √[variance] = √[(1/n) Σᵢ₌₁ⁿ (xᵢ - μ)²]

**Alternative notation:**
- Variance: Var(X) or σ²
- Standard deviation: SD(X) or σ or s

## Connection to Vectors and Norms!

Think of your data as a vector: **x** = (x₁, x₂, ..., xₙ)

Create a **centered vector** (subtract mean from each):
**x_centered** = (x₁ - μ, x₂ - μ, ..., xₙ - μ)

**Standard deviation is related to the norm of this centered vector!**

σ = ||**x_centered**|| / √n

**This connects standard deviation to everything we learned about norms!**

## Population vs Sample Standard Deviation

**Population** (entire group):
σ = √[(1/n) Σ(xᵢ - μ)²]

**Sample** (subset of group):
s = √[(1/(n-1)) Σ(xᵢ - x̄)²]

**Note the (n-1) instead of n!** This is **Bessel's correction** - adjusts for bias when estimating from a sample.

**When to use which:**
- Use n if you have ALL the data (entire population)
- Use (n-1) if you have a sample and want to estimate population std

**In ML:** We usually use n (treating our dataset as the population of interest).

## Interpreting Standard Deviation

**Small σ:** Data tightly clustered around mean
**Large σ:** Data widely spread out

**Rule of thumb for normal distributions:**
- ~68% of data within 1σ of mean
- ~95% within 2σ
- ~99.7% within 3σ

**Example:** 
- Mean = 170 cm, σ = 10 cm
- 68% of heights between 160-180 cm
- 95% between 150-190 cm
- 99.7% between 140-200 cm

## Why Machine Learning Cares About Standard Deviation

### 1. Feature Standardization (Z-Score Normalization)

**Problem:** Features with different scales

- Feature 1: Income ($20k - $200k), σ ≈ $50k
- Feature 2: Age (20 - 65), σ ≈ 15 years

**Solution:** Standardize using z-scores!

z = (x - μ) / σ

After standardization:
- Mean = 0
- Standard deviation = 1
- All features on same scale!

**This helps:**
- Gradient descent converge faster
- Features contribute fairly
- Compare importance across features

### 2. Outlier Detection

**Question:** Is a data point unusual?

**Method:** Check how many standard deviations from mean

**Example:**
- Heights: mean = 170 cm, σ = 10 cm
- New person: 210 cm
- Z-score: (210 - 170) / 10 = 4

**4 standard deviations away → very unusual! (> 99.99% of data)**

**Typical threshold:** |z| > 3 for outliers

### 3. Measuring Model Uncertainty

**Prediction intervals** use standard deviation:

prediction ± 2σ gives ~95% confidence interval

**Example:** House price prediction
- Predicted: $500k
- σ of predictions: $50k
- 95% interval: $400k - $600k

### 4. Variance Explained (PCA)

**Principal Component Analysis:**
- Finds directions of maximum variance (σ²)
- Projects data onto these directions
- Keeps components with high variance

**High variance = important information!**

### 5. Batch Normalization

Normalize activations in neural networks:

x_normalized = (x - μ_batch) / σ_batch

**Helps training stability and speed!**

## Detailed Worked Examples

### Example 3.1: Test Score Variance

Class test scores: [65, 70, 75, 80, 85]

**Step 1: Mean**
μ = (65 + 70 + 75 + 80 + 85) / 5 = 375 / 5 = 75

**Step 2: Deviations**
- 65 - 75 = -10
- 70 - 75 = -5
- 75 - 75 = 0
- 80 - 75 = +5
- 85 - 75 = +10

**Step 3: Squared deviations**
100, 25, 0, 25, 100

**Step 4: Variance**
σ² = (100 + 25 + 0 + 25 + 100) / 5 = 250 / 5 = 50

**Step 5: Standard deviation**
σ = √50 ≈ 7.07 points

**Interpretation:** Typical deviation from mean is ~7 points

### Example 3.2: Income Standardization

Incomes: [$30k, $40k, $50k, $60k, $70k]

**Mean:** μ = $50k
**Std:** σ = √200 ≈ $14.14k

**Standardize (z-scores):**
- $30k: z = (30 - 50) / 14.14 ≈ -1.41
- $40k: z = (40 - 50) / 14.14 ≈ -0.71
- $50k: z = (50 - 50) / 14.14 = 0
- $60k: z = (60 - 50) / 14.14 ≈ +0.71
- $70k: z = (70 - 50) / 14.14 ≈ +1.41

**After standardization:**
- Mean of z-scores = 0
- Std of z-scores = 1
- All on standard scale!

### Example 3.3: Outlier Detection

Daily website visitors: [1000, 1100, 950, 1050, 1020, 980, 5000]

**Mean:** μ ≈ 1586
**Std:** σ ≈ 1410

**Check last point (5000):**
z = (5000 - 1586) / 1410 ≈ 2.42

**2.42 standard deviations away → Unusual but not extreme**

**If threshold is |z| > 3, this wouldn't be flagged**
**If threshold is |z| > 2, this would be flagged as outlier**

### Example 3.4: Comparing Variability

**Stock A returns (%):** [2, 3, 2.5, 3.5, 3]
Mean = 2.8%, σ_A ≈ 0.55%

**Stock B returns (%):** [-5, 10, -3, 12, 1]
Mean = 3%, σ_B ≈ 6.96%

**Stock B is much more volatile (higher σ)!**

**Risk-adjusted return (Sharpe ratio):**
Sharpe = Mean / σ

Stock A: 2.8 / 0.55 ≈ 5.09
Stock B: 3.0 / 6.96 ≈ 0.43

**Stock A has better risk-adjusted return!**

### Example 3.5: Feature Scaling Impact

**Before scaling:**
- Feature 1 (income): [20000, 40000, 60000], σ ≈ 16330
- Feature 2 (age): [25, 35, 45], σ ≈ 8.16

**Feature 1 dominates purely by scale!**

**After standardization (z-scores):**
- Feature 1: [-1.22, 0, 1.22], σ = 1
- Feature 2: [-1.22, 0, 1.22], σ = 1

**Now equal contribution!**

### Example 3.6: Normal Distribution Properties

Heights: μ = 170 cm, σ = 10 cm

**68% within 1σ:**
170 ± 10 = [160, 180] cm

**95% within 2σ:**
170 ± 20 = [150, 190] cm

**99.7% within 3σ:**
170 ± 30 = [140, 200] cm

**Someone 195 cm tall:**
z = (195 - 170) / 10 = 2.5

**Between 2σ and 3σ → in top ~1%!**

### Example 3.7: Prediction Confidence

Linear regression predicts house price:
- Prediction: $500k
- Residual std: $50k (from training errors)

**95% confidence interval:**
$500k ± 2($50k) = [$400k, $600k]

**Interpretation:** 95% confident true price is in this range

### Example 3.8: Variance in PCA

Data projections onto principal components:
- PC1: σ² = 45 (explains most variance)
- PC2: σ² = 15
- PC3: σ² = 5

**Total variance:** 45 + 15 + 5 = 65

**Variance explained:**
- PC1: 45/65 ≈ 69%
- PC2: 15/65 ≈ 23%
- PC3: 5/65 ≈ 8%

**Keep PC1 and PC2 → retain 92% of variance!**

## Practice Problems - Standard Deviation

**Problem 3.1: Basic Calculations**

Calculate mean, variance, and standard deviation:
a) Data: [2, 4, 6, 8, 10]
b) Data: [5, 5, 5, 5, 5]
c) Data: [0, 10, 0, 10, 0]
d) Which has highest variance? Lowest?

**Problem 3.2: Deviations**

For data [10, 15, 20, 25, 30]:
a) Calculate mean
b) List all deviations from mean
c) Verify deviations sum to zero
d) Why must deviations always sum to zero?

**Problem 3.3: Effect of Transformations**

Original data: [1, 2, 3, 4, 5], μ = 3, σ = √2

a) Add 10 to each value. New mean? New σ?
b) Multiply each by 2. New mean? New σ?
c) What's the rule for how transformations affect μ and σ?

**Problem 3.4: Z-Scores**

Data: [100, 110, 120, 130, 140], μ = 120, σ = √200

Calculate z-scores for:
a) 100
b) 120
c) 150
d) Which z-score indicates outlier (|z| > 2)?

**Problem 3.5: Comparing Datasets**

**Dataset A:** [10, 11, 12, 13, 14] (small range)
**Dataset B:** [5, 10, 15, 20, 25] (large range)

Both have mean = 12.

a) Calculate σ for each
b) Which is more spread out?
c) Does this match intuition?

**Problem 3.6: Outlier Impact**

Data without outlier: [10, 12, 11, 13, 12] → μ = 11.6, σ ≈ 1.02
Data with outlier: [10, 12, 11, 13, 100] → μ = 29.2, σ ≈ 37.

a) How much did outlier change mean?
b) How much did outlier change σ?
c) Which is more sensitive to outliers?

**Problem 3.7: Feature Standardization**

Two features:
- Income: [30k, 40k, 50k, 60k], μ = 45k, σ = 11.2k
- Age: [25, 35, 45, 55], μ = 40, σ = 11.2

a) Why do they have same σ but different ranges?
b) Standardize both using z-scores
c) Verify both have mean=0, σ=1 after
d) Why is this useful for ML?

**Problem 3.8: Stock Volatility**

**Stock X returns (%):** [2, 3, 2, 3, 2] → σ_X = 0.45
**Stock Y returns (%):** [-10, 20, -5, 15, -10] → σ_Y = 13.04

a) Which stock is more volatile?
b) If mean returns are equal, which would you prefer?
c) How does σ relate to risk?

**Problem 3.9: Test Score Interpretation**

Class: μ = 75, σ = 10

a) A student scores 85. What percentile (assuming normal)?
b) Another scores 65. What percentile?
c) Score needed to be in top 16% (1σ above mean)?
d) Range containing middle 68% of students?

**Problem 3.10: Population vs Sample**

Full class (population): [70, 75, 80, 85, 90]

a) Calculate population σ (divide by n)
b) Sample: just first 3 values [70, 75, 80]
c) Calculate sample s (divide by n-1)
d) Why do we use n-1 for samples?

---

<a name="angle"></a>
## Chapter 3: Norm, Distance, Standard Deviation, and Angles

### A First-Principles Approach with Detailed Examples

---


1. [Norm (Vector Length/Magnitude)](#norm)
2. [Distance Between Vectors](#distance)
3. [Standard Deviation and Spread](#standard-deviation)
4. [Angle and Cosine Similarity](#angle)
5. [Complexity Analysis](#complexity)
6. [Chapter Summary](#summary)
7. [Comprehensive Practice Problems](#practice)

---

<a name="norm"></a>

## The Fundamental Question: How "Big" Is a Vector?

Imagine you're hiking and your GPS shows your displacement from the starting point:
- 3 km east
- 4 km north

Your displacement vector: **v** = (3, 4)

**Question:** How far did you travel from your starting point in a straight line?

You didn't walk 3 + 4 = 7 km (that's the Manhattan distance, walking along streets).

You want the **straight-line distance** - the length of the arrow from start to your current position.

## Building the Solution: Pythagorean Theorem

Remember the Pythagorean theorem from geometry?

For a right triangle with sides a and b, and hypotenuse c:
c² = a² + b²

**Your displacement forms a right triangle!**
- Horizontal side: 3 km
- Vertical side: 4 km
- Hypotenuse: your straight-line distance

Distance = √(3² + 4²) = √(9 + 16) = √25 = 5 km

**This distance is the NORM (or length, or magnitude) of the vector!**

## Formal Definition

The **norm** (or **length** or **magnitude**) of a vector **v** = (v₁, v₂, ..., vₙ) is:

||**v**|| = √(v₁² + v₂² + ... + vₙ²)

**Notation:** 
- ||**v**|| means "the norm of v"
- Also written as |**v**| or ‖**v**‖
- Sometimes called L2 norm or Euclidean norm

**Key insight:** The norm is always a **non-negative number** (scalar), not a vector!

## Why Square and Then Square Root?

**Question:** Why this specific formula? Why not just add components or add absolute values?

### Bad Idea 1: Just Add Components

||**v**|| = v₁ + v₂ + ... + vₙ

**Problem:** What if components are negative?

**v** = (3, -4)
||**v**|| = 3 + (-4) = -1 (Negative length?! Makes no sense!)

### Bad Idea 2: Add Absolute Values (L1 Norm)

||**v**||₁ = |v₁| + |v₂| + ... + |vₙ|

**v** = (3, 4)
||**v**||₁ = |3| + |4| = 7

This actually works and is called the **L1 norm** or **Manhattan distance**!

**But:** This gives you the "city block" distance (walking on a grid), not straight-line distance.

**Think about it:** To go from (0,0) to (3,4), you walk 3 blocks east + 4 blocks north = 7 blocks total.

### Why Squares Work Best (L2 Norm / Euclidean Norm)

||**v**||₂ = √(v₁² + v₂² + ... + vₙ²)

**Advantages:**
1. **Squaring makes everything positive:** Both 3² and (-3)² give 9
2. **Geometric meaning:** Matches Pythagorean theorem (straight-line distance)
3. **Smooth and differentiable:** Important for optimization (gradients exist everywhere)
4. **Penalizes large values more:** Errors of 2 count as 4, errors of 10 count as 100
5. **Nice mathematical properties:** Works beautifully with inner products

**This is the "standard" norm in ML** unless specified otherwise!

## Connection to Inner Product

**Beautiful relationship:** The norm is the square root of the inner product of a vector with itself!

||**v**||² = **v · v** = v₁² + v₂² + ... + vₙ²

**Therefore:**
||**v**|| = √(**v · v**)

**This connects everything we've learned!**

## Examples in Different Dimensions

### Example 1.1: 1D Vectors (Just Numbers)

**v** = (5)
||**v**|| = √(5²) = √25 = 5

**v** = (-3)
||**v**|| = √((-3)²) = √9 = 3

**Key point:** Length is always positive, even for negative numbers!

### Example 1.2: 2D Vectors

**v** = (3, 4)
||**v**|| = √(3² + 4²) = √(9 + 16) = √25 = 5

**v** = (-1, -1)
||**v**|| = √((-1)² + (-1)²) = √(1 + 1) = √2 ≈ 1.414

**v** = (0, 5)
||**v**|| = √(0² + 5²) = √25 = 5 (pointing straight up)

**v** = (5, 0)
||**v**|| = √(5² + 0²) = 5 (pointing straight right)

### Example 1.3: 3D Vectors

**v** = (1, 2, 2)
||**v**|| = √(1² + 2² + 2²) = √(1 + 4 + 4) = √9 = 3

**v** = (2, -3, 6)
||**v**|| = √(4 + 9 + 36) = √49 = 7

**Think about it:** This is the distance from the origin (0, 0, 0) to the point (2, -3, 6) in 3D space!

### Example 1.4: High-Dimensional Vectors

**v** = (1, 1, 1, 1, 1) ∈ ℝ⁵
||**v**|| = √(1 + 1 + 1 + 1 + 1) = √5 ≈ 2.236

**v** = (15, 3, 4, 1, 1) (our spam email features!)
||**v**|| = √(15² + 3² + 4² + 1² + 1²)
= √(225 + 9 + 16 + 1 + 1)
= √252 ≈ 15.87

**Even though we can't visualize 5D space, the math works the same!**

## Properties of the Norm

These properties define what makes something a "norm":

### Property 1: Non-negativity
||**v**|| ≥ 0 for all **v**

**And:** ||**v**|| = 0 **if and only if** **v** = **0**

**Interpretation:** Length is never negative, and only the zero vector has zero length.

### Property 2: Homogeneity (Scaling)
||α**v**|| = |α| · ||**v**|| for any scalar α

**Example:**
**v** = (3, 4), ||**v**|| = 5

**2v** = (6, 8)
||**2v**|| = √(36 + 64) = √100 = 10 = 2 · 5 ✓

**-v** = (-3, -4)
||-**v**|| = √(9 + 16) = 5 = |-1| · 5 ✓

**Interpretation:** If you double the vector, you double its length. If you flip direction (multiply by -1), length stays the same.

### Property 3: Triangle Inequality
||**u + v**|| ≤ ||**u**|| + ||**v**||

**Interpretation:** The direct path is never longer than taking a detour!

**Think about it:** 
- Walk from A to B directly: ||**u + v**||
- Walk from A to C to B: ||**u**|| + ||**v**||
- Direct is always shorter or equal!

**Example:**
**u** = (1, 0), **v** = (0, 1)

||**u**|| = 1
||**v**|| = 1
||**u + v**|| = ||(1, 1)|| = √2 ≈ 1.414

Check: 1.414 ≤ 1 + 1 = 2 ✓

### Property 4: Relationship with Inner Product (Cauchy-Schwarz)
|**u · v**| ≤ ||**u**|| · ||**v**||

**This will be crucial when we discuss angles!**

## Unit Vectors: Vectors with Length 1

A **unit vector** is a vector with norm exactly equal to 1.

**To create a unit vector from any non-zero vector:**
**v̂** = **v** / ||**v**||

(Read as "v-hat" - the hat notation means "unit vector")

**This process is called normalization.**

### Example 1.5: Normalizing Vectors

**v** = (3, 4)
||**v**|| = 5

**v̂** = (3, 4) / 5 = (3/5, 4/5) = (0.6, 0.8)

**Verify:**
||**v̂**|| = √(0.6² + 0.8²) = √(0.36 + 0.64) = √1 = 1 ✓

**Key insight:** **v̂** points in the **same direction** as **v**, but has **length 1**!

### Example 1.6: Standard Basis Vectors Are Unit Vectors

**e₁** = (1, 0, 0)
||**e₁**|| = √(1² + 0² + 0²) = 1 ✓

**e₂** = (0, 1, 0)
||**e₂**|| = 1 ✓

**e₃** = (0, 0, 1)
||**e₃**|| = 1 ✓

**All standard basis vectors are unit vectors!**

## Why Machine Learning Cares About Norms

Norms are **everywhere** in ML!

### 1. Regularization: Preventing Overfitting

**Problem:** Large weights can cause models to overfit (memorize training data instead of learning patterns).

**Solution:** Add penalty for large weights!

**Ridge Regression (L2 regularization):**
Loss = MSE + λ||**w**||²

Where:
- MSE = prediction error
- ||**w**||² = sum of squared weights
- λ = regularization strength

**Effect:** Model prefers smaller weights → simpler, more generalizable models

**Lasso Regression (L1 regularization):**
Loss = MSE + λ||**w**||₁

Where ||**w**||₁ = sum of absolute values of weights

**Effect:** Can force some weights to exactly zero → automatic feature selection!

### 2. Normalization: Standardizing Inputs

**Problem:** Features with different scales can dominate learning.

**Example:**
- Feature 1: Income ($20k - $200k)
- Feature 2: Age (20 - 65 years)

Income dominates just because numbers are bigger!

**Solution:** Normalize feature vectors to have same scale

**Unit normalization:**
**x_normalized** = **x** / ||**x**||

Now all feature vectors have length 1!

### 3. Gradient Clipping: Stabilizing Training

**Problem:** Sometimes gradients become very large (exploding gradients in deep learning).

**Solution:** If ||∇L|| > threshold, scale it down!

if ||∇L|| > max_norm:
    ∇L = ∇L · (max_norm / ||∇L||)

**Effect:** Gradient direction preserved, but magnitude limited.

### 4. Measuring Prediction Confidence

In neural networks, the norm of output vectors can indicate confidence.

**Example:** Image classification
- Output: **y** = (0.1, 0.1, 0.7, 0.1) (probabilities for 4 classes)
- ||**y**|| = √(0.01 + 0.01 + 0.49 + 0.01) = √0.52 ≈ 0.72

Large norm → confident prediction
Small norm → uncertain

### 5. Batch Normalization

Normalize activations in neural networks:
**a_normalized** = (**a** - mean) / std

Helps training converge faster and more stably!

## Norm Squared: A Useful Shortcut

Often in ML, we use **||v||²** instead of ||**v**| because:

**Advantages:**
1. **No square root needed** → computationally faster
2. **Easier to differentiate** → simpler gradients
3. **Still monotonic** → bigger ||**v**|| means bigger ||**v**||²

||**v**||² = **v · v** = v₁² + v₂² + ... + vₙ²

**When comparing lengths, ||v||² works just as well as ||v||!**

### Example 1.7: Comparing Distances

Which is closer to origin: **u** = (3, 4) or **v** = (2, 5)?

**Method 1: Using norm**
||**u**|| = √25 = 5
||**v**|| = √29 ≈ 5.39
**u** is closer

**Method 2: Using squared norm (faster!)**
||**u**||² = 25
||**v**||² = 29
**u** is closer (same answer, no square roots!)

## Different Types of Norms

### L0 "Norm" (Not Really a Norm)
||**v**||₀ = number of non-zero components

**v** = (0, 3, 0, 5, 0)
||**v**||₀ = 2 (two non-zero entries)

**Use:** Counting sparsity (how many features are active)

### L1 Norm (Manhattan Distance)
||**v**||₁ = |v₁| + |v₂| + ... + |vₙ|

**v** = (3, -4)
||**v**||₁ = |3| + |-4| = 3 + 4 = 7

**Use:** Lasso regularization, robust to outliers

### L2 Norm (Euclidean Distance) - The Standard!
||**v**||₂ = √(v₁² + v₂² + ... + vₙ²)

**This is what we mean by "norm" unless specified otherwise!**

### L∞ Norm (Maximum Norm)
||**v**||∞ = max(|v₁|, |v₂|, ..., |vₙ|)

**v** = (3, -7, 2)
||**v**||∞ = max(3, 7, 2) = 7

**Use:** Measuring worst-case deviation

### Comparison Example

**v** = (3, -4, 0)

||**v**||₀ = 2 (two non-zero)
||**v**||₁ = 3 + 4 + 0 = 7
||**v**||₂ = √(9 + 16 + 0) = √25 = 5
||**v**||∞ = max(3, 4, 0) = 4

**General relationship:** ||**v**||∞ ≤ ||**v**||₂ ≤ ||**v**||₁

## Detailed Worked Examples

### Example 1.8: GPS Navigation

Your hiking trail:
- Start: Origin (0, 0)
- Checkpoint 1: (3, 4) km
- Checkpoint 2: (8, 6) km
- End: (10, 2) km

**Calculate distances from origin:**

To Checkpoint 1:
**v₁** = (3, 4)
||**v₁**|| = √(9 + 16) = 5 km

To Checkpoint 2:
**v₂** = (8, 6)
||**v₂**|| = √(64 + 36) = √100 = 10 km

To End:
**v₃** = (10, 2)
||**v₃**|| = √(100 + 4) = √104 ≈ 10.2 km

**Which checkpoint is furthest?** Checkpoint 2 (10 km)

### Example 1.9: Feature Vector Magnitude

Customer behavior vector:
**c** = (purchases, avg_spend, days_active, reviews)
= (23, 150, 365, 12)

||**c**|| = √(23² + 150² + 365² + 12²)
= √(529 + 22500 + 133225 + 144)
= √156398 ≈ 395.47

**Interpretation:** This is the "magnitude" of customer engagement.

**Compare two customers:**
- Customer A: (23, 150, 365, 12), ||**cₐ**|| ≈ 395.47
- Customer B: (50, 200, 730, 25), ||**cᵦ**|| ≈ 762.76

Customer B has higher engagement magnitude!

### Example 1.10: Neural Network Weight Initialization

Initialize weights with small random values:

**w** = (0.01, -0.02, 0.015, -0.008, 0.012)

||**w**|| = √(0.0001 + 0.0004 + 0.000225 + 0.000064 + 0.000144)
= √0.001033 ≈ 0.032

**Check:** ||**w**|| < 0.1 ✓ (good initialization - small weights)

### Example 1.11: Normalizing Image Pixels

Image pixel vector (simplified, 3 pixels):
**img** = (128, 200, 64) (pixel brightness 0-255)

||**img**|| = √(16384 + 40000 + 4096) = √60480 ≈ 245.93

**Normalized:**
**img_normalized** = **img** / ||**img**||
= (128, 200, 64) / 245.93
= (0.520, 0.813, 0.260)

Now ||**img_normalized**|| = 1 ✓

### Example 1.12: Gradient Magnitude

Loss gradient: ∇L = (2.5, -1.8, 3.2, -0.9)

||∇L|| = √(6.25 + 3.24 + 10.24 + 0.81)
= √20.54 ≈ 4.53

**If gradient is too large (>10), clip it:**

Since 4.53 < 10, no clipping needed.

But if ||∇L|| = 15, then:
∇L_clipped = ∇L · (10 / 15) = 0.667 · ∇L

### Example 1.13: Portfolio Volatility

Stock returns vector (5 days):
**r** = (0.02, -0.01, 0.03, -0.02, 0.01) (daily returns as decimals)

||**r**|| = √(0.0004 + 0.0001 + 0.0009 + 0.0004 + 0.0001)
= √0.0019 ≈ 0.0436

**Interpretation:** This measures the magnitude of price movements (volatility indicator).

## Practice Problems - Norm

**Problem 1.1: Basic Norm Calculations**

Calculate ||**v**|| for:
a) **v** = (5, 12)
b) **v** = (-3, 4)
c) **v** = (1, 1, 1)
d) **v** = (2, -2, 1, -1)
e) **v** = (0, 0, 0)

**Problem 1.2: Pythagorean Triples**

Verify these are Pythagorean triples by calculating norms:
a) (3, 4) should have norm 5
b) (5, 12) should have norm 13
c) (8, 15) should have norm 17
d) (7, 24) should have norm 25

**Problem 1.3: Normalization**

Normalize these vectors (find unit vectors):
a) **v** = (3, 4)
b) **v** = (1, 1)
c) **v** = (0, 5)
d) **v** = (2, -2, 1)

Verify each normalized vector has norm 1.

**Problem 1.4: Comparing Magnitudes**

Which vector has larger norm?
a) **u** = (3, 4) vs **v** = (5, 2)
b) **u** = (1, 1, 1, 1) vs **v** = (2, 0, 0, 0)
c) **u** = (10, 1) vs **v** = (1, 10)

**Problem 1.5: Properties Verification**

For **u** = (3, 4) and scalar α = 2:
a) Calculate ||**u**||
b) Calculate ||α**u**||
c) Verify ||α**u**|| = |α| · ||**u**||
d) What if α = -3? Verify the property still holds.

**Problem 1.6: Triangle Inequality**

For **u** = (1, 2) and **v** = (3, 1):
a) Calculate ||**u**||, ||**v**||, and ||**u + v**||
b) Verify ||**u + v**|| ≤ ||**u**|| + ||**v**||
c) When does equality hold in triangle inequality?

**Problem 1.7: Different Norms**

For **v** = (3, -4, 5):
a) Calculate L1 norm: ||**v**||₁
b) Calculate L2 norm: ||**v**||₂
c) Calculate L∞ norm: ||**v**||∞
d) Which is largest? Why?

**Problem 1.8: Sparse Vectors**

Vector **v** = (0, 5, 0, 0, 3, 0, 0, 7, 0)
a) Calculate ||**v**||₀ (count non-zeros)
b) Calculate ||**v**||₁
c) Calculate ||**v**||₂
d) Why is this vector called "sparse"?

**Problem 1.9: Feature Scaling**

Two features with different scales:
- **f₁** = (1000, 2000, 1500) (income in $)
- **f₂** = (25, 35, 30) (age in years)

a) Calculate ||**f₁**|| and ||**f₂**||
b) Normalize both to unit vectors
c) Now calculate norms. What do you notice?
d) Why is this normalization useful in ML?

**Problem 1.10: Gradient Clipping**

Gradient: ∇L = (8, -6, 12, -4)
Max allowed norm: 10

a) Calculate ||∇L||
b) Is clipping needed?
c) If yes, calculate clipped gradient
d) Verify clipped gradient has norm ≤ 10

---

<a name="distance"></a>

## The Core Question: How Far Apart Are Two Things?

You have two movie preference vectors:
- **Alice:** (5, 2, 1, 4) = ratings for (action, comedy, drama, horror)
- **Bob:** (4, 3, 1, 3)

**Question:** How similar are their movie tastes?

To answer this, we need to measure how "far apart" their preference vectors are!

## Building the Solution: The Difference Vector

**First insight:** To measure distance between two points, find how they differ!

**Bob's preferences - Alice's preferences:**
**b - a** = (4, 3, 1, 3) - (5, 2, 1, 4) = (-1, 1, 0, -1)

**This difference vector tells us:**
- Action: Bob rates 1 point lower
- Comedy: Bob rates 1 point higher  
- Drama: Same!
- Horror: Bob rates 1 point lower

**Second insight:** The LENGTH of this difference vector is the distance!

distance(**a**, **b**) = ||**b - a**|| = ||(-1, 1, 0, -1)||
= √(1 + 1 + 0 + 1) = √3 ≈ 1.73

## Formal Definition

The **Euclidean distance** between vectors **u** and **v** is:

d(**u**, **v**) = ||**u - v**|| = √[(u₁-v₁)² + (u₂-v₂)² + ... + (uₙ-vₙ)²]

**Alternative formula using inner product:**
d(**u**, **v**) = √[(**u - v**) · (**u - v**)]

**Key properties:**
- Always non-negative: d(**u**, **v**) ≥ 0
- Zero iff identical: d(**u**, **v**) = 0 ⟺ **u** = **v**
- Symmetric: d(**u**, **v**) = d(**v**, **u**)
- Triangle inequality: d(**u**, **w**) ≤ d(**u**, **v**) + d(**v**, **w**)

## Geometric Interpretation in 2D

Two points in a plane:
- **p₁** = (1, 2)
- **p₂** = (4, 6)

**Visualize:** Plot these points. The distance is the length of the straight line connecting them!

d(**p₁**, **p₂**) = ||**p₂ - p₁**|| = ||(3, 4)|| = 5

**You can literally measure this with a ruler on graph paper!**

## Why This Formula Works

The distance formula comes directly from the Pythagorean theorem!

**Think about moving from **p₁** to **p₂**:**
- Horizontal change: Δx = 4 - 1 = 3
- Vertical change: Δy = 6 - 2 = 4
- These form a right triangle!
- Hypotenuse (distance): √(3² + 4²) = 5

**This extends to any dimension!**

## Properties of Distance

### Property 1: Non-negativity
d(**u**, **v**) ≥ 0

Distance is never negative!

### Property 2: Identity of Indiscernibles  
d(**u**, **v**) = 0 if and only if **u** = **v**

Zero distance means the vectors are identical.

### Property 3: Symmetry
d(**u**, **v**) = d(**v**, **u**)

Distance from A to B equals distance from B to A.

**Proof:**
d(**u**, **v**) = ||**u - v**||
d(**v**, **u**) = ||**v - u**|| = ||**-(u - v)**|| = |-1| · ||**u - v**|| = ||**u - v**|| ✓

### Property 4: Triangle Inequality
d(**u**, **w**) ≤ d(**u**, **v**) + d(**v**, **w**)

**Interpretation:** Going directly from **u** to **w** is never longer than going via **v**!

Think about driving:
- Direct route from home to store
- vs going home → friend's house → store

Direct is always shorter (or equal if friend is on the way)!

## Why Machine Learning Cares About Distance

Distance is **absolutely fundamental** to ML!

### 1. K-Nearest Neighbors (KNN)

**Algorithm:** To classify a new point:
1. Find the K closest training examples (smallest distances)
2. Use their labels to vote

**Example: Email Spam Detection**
- New email: **x** = (5, 20, 1, 50)
- Known spam: **s** = (12, 25, 1, 30)
- Known ham: **h** = (1, 3, 0, 200)

d(**x**, **s**) = ||(5, 20, 1, 50) - (12, 25, 1, 30)||
= ||(-7, -5, 0, 20)||
= √(49 + 25 + 0 + 400) = √474 ≈ 21.8

d(**x**, **h**) = ||(4, 17, 1, -150)||
= √(16 + 289 + 1 + 22500) = √22806 ≈ 151

**x** is MUCH closer to spam example → Classify as spam!

### 2. Clustering (K-Means)

**Goal:** Group similar data points together

**How:** Points are "similar" if distance is small!

**Algorithm:**
1. Assign each point to nearest cluster center
2. Update centers (mean of assigned points)
3. Repeat until convergence

**All based on distance calculations!**

### 3. Anomaly Detection

**Question:** Is this data point unusual?

**Answer:** If it has large distance from all normal examples → Anomaly!

**Example: Fraud Detection**
- Normal transaction: **t_normal** ≈ (50, 1, 10)
- New transaction: **t_new** = (10000, 5, 1000)

d(**t_new**, **t_normal**) = very large → Suspicious!

### 4. Recommendation Systems

**Find users with similar preferences:**

- Your ratings: **you** = (5, 1, 4, 2, 5)
- User A: **a** = (5, 2, 4, 1, 5)
- User B: **b** = (1, 5, 1, 5, 2)

d(**you**, **a**) = small → Similar tastes!
d(**you**, **b**) = large → Different tastes!

**Recommendation:** Show what similar users liked!

### 5. Loss Functions

**Mean Squared Error** is based on distance!

MSE = (1/n) Σᵢ ||**yᵢ** - **ŷᵢ**||²

Average squared distance between predictions and true values!

## Distance vs. Similarity

**Key insight:** Small distance = high similarity!

Often we convert distance to similarity:

**Similarity metrics:**
1. sim = 1 / (1 + distance)
2. sim = e^(-distance)
3. sim = 1 - (distance / max_distance)

**Example:**
- d = 0 → sim = 1 (identical)
- d = 1 → sim = 0.5 (somewhat similar)
- d = ∞ → sim = 0 (completely different)

## Different Distance Metrics

### Euclidean Distance (L2) - The Standard!
d₂(**u**, **v**) = ||**u - v**||₂ = √[Σ(uᵢ - vᵢ)²]

**Use:** General purpose, most common

### Manhattan Distance (L1)
d₁(**u**, **v**) = ||**u - v**||₁ = Σ|uᵢ - vᵢ|

**Use:** When you can only move along axes (like city blocks)

### Chebyshev Distance (L∞)
d∞(**u**, **v**) = ||**u - v**||∞ = max|uᵢ
