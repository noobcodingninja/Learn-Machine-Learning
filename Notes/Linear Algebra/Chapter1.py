"""
LINEAR ALGEBRA FOR MACHINE LEARNING - CHAPTER 1
Python Implementation: Vectors and Vector Operations

This module demonstrates:
1. Vector representation
2. Vector addition
3. Scalar-vector multiplication
4. Inner product (dot product)
5. Linear combinations
6. Computational complexity examples

Each concept is implemented in TWO ways:
- From scratch (using only basic Python)
- Using NumPy (standard ML library)
"""

# ============================================================================
# PART 1: BASIC PYTHON IMPLEMENTATION (No Libraries)
# ============================================================================

print("="*70)
print("PART 1: BASIC PYTHON IMPLEMENTATION (From Scratch)")
print("="*70)
print()

# ----------------------------------------------------------------------------
# 1.1 Vector Representation
# ----------------------------------------------------------------------------

print("1.1 VECTOR REPRESENTATION")
print("-" * 50)

# Vectors are represented as Python lists
email_vector = [15, 3, 4, 1, 1]  # (capitals, exclamations, words, has_FREE, has_MONEY)
print(f"Email vector: {email_vector}")
print(f"Dimension: {len(email_vector)}")
print(f"First component: {email_vector[0]}")
print(f"Last component: {email_vector[-1]}")
print()

# Creating vectors for different data types
song_vector = [120, 85, 180, 5]  # (tempo, loudness, duration, instruments)
house_vector = [2000, 3, 15, 8]  # (sqft, bedrooms, age, school_rating)

print(f"Song vector: {song_vector}")
print(f"House vector: {house_vector}")
print()

# Zero vector
zero_vector = [0, 0, 0, 0]
print(f"Zero vector: {zero_vector}")
print()

# ----------------------------------------------------------------------------
# 1.2 Vector Addition
# ----------------------------------------------------------------------------

print("1.2 VECTOR ADDITION")
print("-" * 50)

def vector_add(u, v):
    """
    Add two vectors component-wise.
    
    Args:
        u: First vector (list)
        v: Second vector (list)
    
    Returns:
        Sum vector (list)
    
    Raises:
        ValueError: If vectors have different dimensions
    """
    # Check dimensions match
    if len(u) != len(v):
        raise ValueError(f"Dimension mismatch: {len(u)} vs {len(v)}")
    
    # Add corresponding components
    result = []
    for i in range(len(u)):
        result.append(u[i] + v[i])
    
    return result

# Example: Daily activity tracking
monday = [8000, 5, 30]  # (steps, floors, minutes)
tuesday = [6000, 3, 45]

print(f"Monday activity: {monday}")
print(f"Tuesday activity: {tuesday}")

total = vector_add(monday, tuesday)
print(f"Total activity: {total}")
print()

# Example: Multiple additions
wednesday = [10000, 8, 60]
weekly_total = vector_add(vector_add(monday, tuesday), wednesday)
print(f"Three-day total: {weekly_total}")
print()

# Verify commutativity: u + v = v + u
u = [1, 2, 3]
v = [4, 5, 6]
print(f"u + v = {vector_add(u, v)}")
print(f"v + u = {vector_add(v, u)}")
print(f"Commutative? {vector_add(u, v) == vector_add(v, u)}")
print()

# ----------------------------------------------------------------------------
# 1.3 Scalar-Vector Multiplication
# ----------------------------------------------------------------------------

print("1.3 SCALAR-VECTOR MULTIPLICATION")
print("-" * 50)

def scalar_multiply(scalar, v):
    """
    Multiply a vector by a scalar.
    
    Args:
        scalar: Number to multiply by
        v: Vector (list)
    
    Returns:
        Scaled vector (list)
    """
    result = []
    for i in range(len(v)):
        result.append(scalar * v[i])
    
    return result

# Example: Doubling activity
print(f"Monday activity: {monday}")
double_monday = scalar_multiply(2, monday)
print(f"Double activity: {double_monday}")
print()

# Example: Recipe scaling
recipe = [4, 2, 1, 0.5]  # (eggs, cups_flour, cup_milk, tsp_salt) for 2 servings
print(f"Recipe (2 servings): {recipe}")
print(f"Recipe (6 servings): {scalar_multiply(3, recipe)}")
print(f"Recipe (1 serving): {scalar_multiply(0.5, recipe)}")
print()

# Example: Price discount
prices = [10, 20, 30, 15]
print(f"Original prices: {prices}")
print(f"20% discount (80% price): {scalar_multiply(0.8, prices)}")
print(f"50% markup (150% price): {scalar_multiply(1.5, prices)}")
print()

# Example: Direction reversal
velocity = [30, 40]
print(f"Velocity: {velocity}")
print(f"Opposite direction: {scalar_multiply(-1, velocity)}")
print()

# ----------------------------------------------------------------------------
# 1.4 Inner Product (Dot Product)
# ----------------------------------------------------------------------------

print("1.4 INNER PRODUCT (DOT PRODUCT)")
print("-" * 50)

def inner_product(u, v):
    """
    Compute inner product (dot product) of two vectors.
    
    Args:
        u: First vector (list)
        v: Second vector (list)
    
    Returns:
        Scalar (number)
    
    Raises:
        ValueError: If vectors have different dimensions
    """
    # Check dimensions match
    if len(u) != len(v):
        raise ValueError(f"Dimension mismatch: {len(u)} vs {len(v)}")
    
    # Multiply corresponding components and sum
    result = 0
    for i in range(len(u)):
        result += u[i] * v[i]
    
    return result

# Example: Dating app compatibility
preferences = [10, 7, 5, 2]  # (hiking, reading, cooking, gaming)
profile = [1, 0, 1, 1]       # (yes, no, yes, yes)

print(f"Your preferences: {preferences}")
print(f"Their profile: {profile}")
compatibility = inner_product(preferences, profile)
print(f"Compatibility score: {compatibility}")
print()

# Example: Spam detection
spam_pattern = [12, 2.5, 8, 0.9, 0.8]
email = [15, 3, 4, 1, 1]

print(f"Spam pattern: {spam_pattern}")
print(f"Email features: {email}")
spam_score = inner_product(spam_pattern, email)
print(f"Spam score: {spam_score}")
print(f"Classification: {'SPAM' if spam_score > 100 else 'HAM'}")
print()

# Example: Perpendicular vectors (inner product = 0)
u = [1, 0]
v = [0, 1]
print(f"u = {u}, v = {v}")
print(f"u · v = {inner_product(u, v)} (perpendicular!)")
print()

u = [3, 4]
v = [4, -3]
print(f"u = {u}, v = {v}")
print(f"u · v = {inner_product(u, v)} (perpendicular!)")
print()

# Example: Same direction (large positive)
u = [3, 4]
v = [6, 8]  # v = 2*u
print(f"u = {u}, v = {v}")
print(f"u · v = {inner_product(u, v)} (same direction!)")
print()

# Example: Opposite direction (negative)
u = [3, 4]
v = [-3, -4]  # v = -u
print(f"u = {u}, v = {v}")
print(f"u · v = {inner_product(u, v)} (opposite direction!)")
print()

# Verify commutativity: u · v = v · u
u = [2, 3, 4]
v = [5, 6, 7]
print(f"u · v = {inner_product(u, v)}")
print(f"v · u = {inner_product(v, u)}")
print(f"Commutative? {inner_product(u, v) == inner_product(v, u)}")
print()

# ----------------------------------------------------------------------------
# 1.5 Linear Combinations
# ----------------------------------------------------------------------------

print("1.5 LINEAR COMBINATIONS")
print("-" * 50)

def linear_combination(coefficients, vectors):
    """
    Compute linear combination: c1*v1 + c2*v2 + ... + cn*vn
    
    Args:
        coefficients: List of scalars [c1, c2, ..., cn]
        vectors: List of vectors [v1, v2, ..., vn]
    
    Returns:
        Result vector (list)
    """
    if len(coefficients) != len(vectors):
        raise ValueError("Number of coefficients must match number of vectors")
    
    # Start with zero vector of appropriate dimension
    dimension = len(vectors[0])
    result = [0] * dimension
    
    # Add each scaled vector
    for i in range(len(coefficients)):
        scaled = scalar_multiply(coefficients[i], vectors[i])
        result = vector_add(result, scaled)
    
    return result

# Example: Weighted average of test scores
scores = [
    [85, 90, 78],
    [92, 88, 95],
    [78, 85, 82]
]
weights = [0.3, 0.5, 0.2]

print(f"Test 1 scores: {scores[0]}")
print(f"Test 2 scores: {scores[1]}")
print(f"Test 3 scores: {scores[2]}")
print(f"Weights: {weights}")

weighted_avg = linear_combination(weights, scores)
print(f"Weighted average: {weighted_avg}")
print()

# Example: RGB color mixing
red = [255, 0, 0]
green = [0, 255, 0]
blue = [0, 0, 255]

print("Color mixing:")
print(f"Red: {red}, Green: {green}, Blue: {blue}")

# Yellow = red + green
yellow = linear_combination([0.5, 0.5, 0], [red, green, blue])
print(f"Yellow (R+G): {yellow}")

# Orange = 70% red + 30% green
orange = linear_combination([0.7, 0.3, 0], [red, green, blue])
print(f"Orange (0.7R+0.3G): {orange}")

# Purple = red + blue
purple = linear_combination([0.5, 0, 0.5], [red, green, blue])
print(f"Purple (R+B): {purple}")
print()

# ----------------------------------------------------------------------------
# 1.6 Helper Functions
# ----------------------------------------------------------------------------

print("1.6 HELPER FUNCTIONS")
print("-" * 50)

def vector_subtract(u, v):
    """Subtract v from u: u - v"""
    return vector_add(u, scalar_multiply(-1, v))

def vector_length_squared(v):
    """Compute squared length (norm squared): v · v"""
    return inner_product(v, v)

def vector_length(v):
    """Compute length (norm): sqrt(v · v)"""
    # Implement square root using Newton's method (no math library!)
    squared = vector_length_squared(v)
    
    if squared == 0:
        return 0
    
    # Newton's method for square root
    x = squared / 2  # Initial guess
    for _ in range(20):  # 20 iterations for good accuracy
        x = (x + squared / x) / 2
    
    return x

def distance(u, v):
    """Compute Euclidean distance: ||u - v||"""
    diff = vector_subtract(u, v)
    return vector_length(diff)

# Examples
u = [3, 4]
print(f"Vector u: {u}")
print(f"u · u = {vector_length_squared(u)}")
print(f"||u|| = {vector_length(u)}")
print()

v = [0, 0]
print(f"Distance from {u} to origin {v}: {distance(u, v)}")
print()

p1 = [1, 2]
p2 = [4, 6]
print(f"Distance from {p1} to {p2}: {distance(p1, p2)}")
print()

# ----------------------------------------------------------------------------
# 1.7 Complete Example: Simple Linear Classifier
# ----------------------------------------------------------------------------

print("1.7 COMPLETE EXAMPLE: SPAM CLASSIFIER")
print("-" * 50)

# Training data: (capitals, exclamations, short_words, has_FREE, has_MONEY)
spam_emails = [
    [20, 5, 3, 1, 1],
    [18, 4, 5, 1, 1],
    [22, 6, 4, 1, 1],
]

ham_emails = [
    [2, 0, 10, 0, 0],
    [3, 1, 8, 0, 0],
    [1, 0, 12, 0, 0],
]

print("Training data:")
print(f"Spam emails: {spam_emails}")
print(f"Ham emails: {ham_emails}")
print()

# Compute average spam vector
spam_avg = linear_combination([1/3, 1/3, 1/3], spam_emails)
print(f"Average spam: {spam_avg}")

# Compute average ham vector
ham_avg = linear_combination([1/3, 1/3, 1/3], ham_emails)
print(f"Average ham: {ham_avg}")

# Weight vector: spam - ham
weights = vector_subtract(spam_avg, ham_avg)
print(f"Weight vector: {weights}")
print()

# Test on new emails
test_emails = [
    [15, 3, 4, 1, 0],   # Suspicious
    [5, 0, 15, 0, 0],   # Normal
    [25, 8, 2, 1, 1],   # Very spammy
]

print("Testing new emails:")
for i, email in enumerate(test_emails, 1):
    score = inner_product(weights, email)
    classification = "SPAM" if score > 0 else "HAM"
    print(f"Email {i}: {email}")
    print(f"  Score: {score:.2f} → {classification}")
print()

# ============================================================================
# PART 2: NUMPY IMPLEMENTATION (Using Standard Library)
# ============================================================================

print("="*70)
print("PART 2: NUMPY IMPLEMENTATION (Industry Standard)")
print("="*70)
print()

try:
    import numpy as np
    
    print("NumPy imported successfully!")
    print(f"NumPy version: {np.__version__}")
    print()
    
    # ------------------------------------------------------------------------
    # 2.1 Vector Creation with NumPy
    # ------------------------------------------------------------------------
    
    print("2.1 VECTOR CREATION")
    print("-" * 50)
    
    # Create vectors (NumPy arrays)
    email_vec = np.array([15, 3, 4, 1, 1])
    print(f"Email vector: {email_vec}")
    print(f"Type: {type(email_vec)}")
    print(f"Shape: {email_vec.shape}")
    print(f"Dimension: {email_vec.shape[0]}")
    print()
    
    # Different ways to create vectors
    zeros = np.zeros(5)
    print(f"Zero vector: {zeros}")
    
    ones = np.ones(3)
    print(f"Ones vector: {ones}")
    
    range_vec = np.arange(0, 10, 2)  # Start, stop, step
    print(f"Range vector: {range_vec}")
    
    random_vec = np.random.rand(5)  # Random values [0, 1)
    print(f"Random vector: {random_vec}")
    print()
    
    # ------------------------------------------------------------------------
    # 2.2 Vector Addition with NumPy
    # ------------------------------------------------------------------------
    
    print("2.2 VECTOR ADDITION")
    print("-" * 50)
    
    monday_np = np.array([8000, 5, 30])
    tuesday_np = np.array([6000, 3, 45])
    
    print(f"Monday: {monday_np}")
    print(f"Tuesday: {tuesday_np}")
    
    # Simply use + operator!
    total_np = monday_np + tuesday_np
    print(f"Total: {total_np}")
    print()
    
    # Multiple additions
    wednesday_np = np.array([10000, 8, 60])
    weekly_np = monday_np + tuesday_np + wednesday_np
    print(f"Weekly total: {weekly_np}")
    print()
    
    # ------------------------------------------------------------------------
    # 2.3 Scalar Multiplication with NumPy
    # ------------------------------------------------------------------------
    
    print("2.3 SCALAR MULTIPLICATION")
    print("-" * 50)
    
    # Simply use * operator!
    double = 2 * monday_np
    print(f"Double Monday: {double}")
    
    half = 0.5 * monday_np
    print(f"Half Monday: {half}")
    
    opposite = -1 * monday_np
    print(f"Opposite direction: {opposite}")
    print()
    
    # Prices example
    prices_np = np.array([10, 20, 30, 15])
    print(f"Original prices: {prices_np}")
    print(f"20% discount: {0.8 * prices_np}")
    print(f"50% markup: {1.5 * prices_np}")
    print()
    
    # ------------------------------------------------------------------------
    # 2.4 Inner Product with NumPy
    # ------------------------------------------------------------------------
    
    print("2.4 INNER PRODUCT")
    print("-" * 50)
    
    u_np = np.array([2, 3, 4])
    v_np = np.array([5, 6, 7])
    
    # Method 1: np.dot()
    dot1 = np.dot(u_np, v_np)
    print(f"u · v (using np.dot): {dot1}")
    
    # Method 2: @ operator (Python 3.5+)
    dot2 = u_np @ v_np
    print(f"u · v (using @): {dot2}")
    
    # Method 3: .dot() method
    dot3 = u_np.dot(v_np)
    print(f"u · v (using .dot()): {dot3}")
    print()
    
    # Spam detection example
    spam_pattern_np = np.array([12, 2.5, 8, 0.9, 0.8])
    email_np = np.array([15, 3, 4, 1, 1])
    
    spam_score_np = spam_pattern_np @ email_np
    print(f"Spam score: {spam_score_np}")
    print(f"Classification: {'SPAM' if spam_score_np > 100 else 'HAM'}")
    print()
    
    # ------------------------------------------------------------------------
    # 2.5 Linear Combinations with NumPy
    # ------------------------------------------------------------------------
    
    print("2.5 LINEAR COMBINATIONS")
    print("-" * 50)
    
    # Weighted average
    scores_np = np.array([
        [85, 90, 78],
        [92, 88, 95],
        [78, 85, 82]
    ])
    weights_np = np.array([0.3, 0.5, 0.2])
    
    # Using matrix multiplication
    weighted_avg_np = weights_np @ scores_np
    print(f"Weighted average: {weighted_avg_np}")
    print()
    
    # Color mixing
    red_np = np.array([255, 0, 0])
    green_np = np.array([0, 255, 0])
    blue_np = np.array([0, 0, 255])
    
    yellow_np = 0.5 * red_np + 0.5 * green_np
    print(f"Yellow: {yellow_np}")
    
    orange_np = 0.7 * red_np + 0.3 * green_np
    print(f"Orange: {orange_np}")
    print()
    
    # ------------------------------------------------------------------------
    # 2.6 Useful NumPy Functions
    # ------------------------------------------------------------------------
    
    print("2.6 USEFUL NUMPY FUNCTIONS")
    print("-" * 50)
    
    v_np = np.array([3, 4])
    
    # Length (norm)
    length = np.linalg.norm(v_np)
    print(f"||v|| = {length}")
    
    # Squared length
    length_sq = np.dot(v_np, v_np)
    print(f"||v||² = {length_sq}")
    
    # Distance
    u_np = np.array([0, 0])
    dist = np.linalg.norm(v_np - u_np)
    print(f"Distance: {dist}")
    print()
    
    # Sum, mean, max, min
    data = np.array([10, 20, 30, 40, 50])
    print(f"Data: {data}")
    print(f"Sum: {np.sum(data)}")
    print(f"Mean: {np.mean(data)}")
    print(f"Max: {np.max(data)}")
    print(f"Min: {np.min(data)}")
    print(f"Std dev: {np.std(data)}")
    print()
    
    # ------------------------------------------------------------------------
    # 2.7 Complete Example: Spam Classifier with NumPy
    # ------------------------------------------------------------------------
    
    print("2.7 SPAM CLASSIFIER WITH NUMPY")
    print("-" * 50)
    
    # Training data
    spam_emails_np = np.array([
        [20, 5, 3, 1, 1],
        [18, 4, 5, 1, 1],
        [22, 6, 4, 1, 1],
    ])
    
    ham_emails_np = np.array([
        [2, 0, 10, 0, 0],
        [3, 1, 8, 0, 0],
        [1, 0, 12, 0, 0],
    ])
    
    # Compute averages (axis=0 means along rows)
    spam_avg_np = np.mean(spam_emails_np, axis=0)
    ham_avg_np = np.mean(ham_emails_np, axis=0)
    
    print(f"Spam average: {spam_avg_np}")
    print(f"Ham average: {ham_avg_np}")
    
    # Weight vector
    weights_np = spam_avg_np - ham_avg_np
    print(f"Weights: {weights_np}")
    print()
    
    # Test emails
    test_emails_np = np.array([
        [15, 3, 4, 1, 0],
        [5, 0, 15, 0, 0],
        [25, 8, 2, 1, 1],
    ])
    
    # Classify all at once using matrix-vector multiplication!
    scores_np = test_emails_np @ weights_np
    
    print("Test results:")
    for i, (email, score) in enumerate(zip(test_emails_np, scores_np), 1):
        classification = "SPAM" if score > 0 else "HAM"
        print(f"Email {i}: score={score:.2f} → {classification}")
    print()
    
    # ------------------------------------------------------------------------
    # 2.8 Performance Comparison
    # ------------------------------------------------------------------------
    
    print("2.8 PERFORMANCE COMPARISON")
    print("-" * 50)
    
    import time
    
    # Create large vectors
    n = 1000000
    u_large = [float(i) for i in range(n)]
    v_large = [float(i * 2) for i in range(n)]
    
    u_large_np = np.array(u_large)
    v_large_np = np.array(v_large)
    
    print(f"Vector dimension: {n:,}")
    print()
    
    # Time basic Python inner product
    start = time.time()
    result_basic = inner_product(u_large, v_large)
    time_basic = time.time() - start
    
    print(f"Basic Python inner product: {time_basic:.4f} seconds")
    
    # Time NumPy inner product
    start = time.time()
    result_numpy = u_large_np @ v_large_np
    time_numpy = time.time() - start
    
    print(f"NumPy inner product: {time_numpy:.4f} seconds")
    print(f"Speedup: {time_basic / time_numpy:.1f}x faster!")
    print()
    
    print("Why NumPy is faster:")
    print("- Written in C (compiled, not interpreted)")
    print("- Uses optimized BLAS/LAPACK libraries")
    print("- Vectorized operations (no Python loops)")
    print("- Better memory layout and caching")
    print()

except ImportError:
    print("NumPy not installed. To install:")
    print("  pip install numpy")
    print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*70)
print("SUMMARY")
print("="*70)
print()

print("Basic Python Implementation:")
print("✓ Full control and understanding")
print("✓ No external dependencies")
print("✓ Good for learning")
print("✓ Slower for large data")
print()

print("NumPy Implementation:")
print("✓ Industry standard")
print("✓ Much faster (10-100x)")
print("✓ Cleaner, more readable code")
print("✓ Interoperates with ML libraries")
print("✓ Required for real ML work")
print()

print("Key Operations Summary:")
print("-" * 50)
print("Operation          | Basic Python        | NumPy")
print("-" * 50)
print("Create vector      | [1, 2, 3]          | np.array([1, 2, 3])")
print("Addition           | vector_add(u, v)   | u + v")
print("Scalar multiply    | scalar_multiply    | α * v")
print("Inner product      | inner_product      | u @ v or np.dot(u, v)")
print("Length             | vector_length      | np.linalg.norm(v)")
print("Distance           | distance(u, v)     | np.linalg.norm(u - v)")
print()

print("Next Steps:")
print("1. Practice implementing these from scratch")
print("2. Learn NumPy for real applications")
print("3. Move on to Chapter 2: Linear Functions & Regression")
print()

print("="*70)
print("END OF CHAPTER 1 IMPLEMENTATION")
print("="*70)
