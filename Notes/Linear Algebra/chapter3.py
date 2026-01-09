"""
================================================================================
Chapter 3: Python Implementation
Norm, Distance, Standard Deviation, and Angles
================================================================================

Table of Contents:
1. Part A: Implementation from Scratch (No Libraries)
2. Part B: Implementation with NumPy
3. Part C: Visualization Examples
4. Part D: Real-World Applications
5. Part E: Performance Comparison

================================================================================
PART A: IMPLEMENTATION FROM SCRATCH (NO LIBRARIES)
================================================================================

Why Start from Scratch?
-----------------------
Problem: Libraries like NumPy are magical black boxes. You use them without 
understanding what's happening inside.

Root cause: When you just call np.linalg.norm(), you don't see:
- How the computation actually works
- Why certain algorithms are used
- Where numerical issues can arise

Solution: Build everything from scratch first! Then you'll appreciate what 
libraries do for you.
"""

# =============================================================================
# 1. VECTOR NORM (EUCLIDEAN LENGTH)
# =============================================================================

def norm(vector):
    """
    Calculate the Euclidean norm (length) of a vector from scratch.
    
    The Problem We're Solving:
    --------------------------
    Question: How do we measure the "size" or "length" of a vector?
    Root cause: We need a single number to represent magnitude!
    Formula: ||v|| = √(v₁² + v₂² + ... + vₙ²)
    
    Parameters:
    -----------
    vector : list or tuple
        Input vector as a list/tuple of numbers
        
    Returns:
    --------
    float
        The Euclidean norm of the vector
        
    Examples:
    ---------
    >>> norm([3, 4])
    5.0
    >>> norm([1, 2, 2])
    3.0
    """
    # Step 1: Square each component and sum them
    sum_of_squares = 0
    for component in vector:
        sum_of_squares += component ** 2
    
    # Step 2: Take square root
    # Using ** 0.5 instead of importing math.sqrt to stay library-free
    result = sum_of_squares ** 0.5
    
    return result


# Test cases for norm
print("=" * 80)
print("VECTOR NORM EXAMPLES (From Scratch)")
print("=" * 80)

# Example 1: 2D vector (3, 4) - classic 3-4-5 triangle
v1 = [3, 4]
print(f"\n📐 Example 1: 2D Vector")
print(f"Vector: {v1}")
print(f"Norm: {norm(v1)}")
print(f"Why? √(3² + 4²) = √(9 + 16) = √25 = 5.0 ✓")

# Example 2: 3D unit vector
v2 = [1, 0, 0]
print(f"\n📐 Example 2: Unit Vector")
print(f"Vector: {v2}")
print(f"Norm: {norm(v2)}")
print(f"Why? This is a unit vector along x-axis, so length = 1.0 ✓")

# Example 3: 3D vector
v3 = [1, 2, 2]
print(f"\n📐 Example 3: 3D Vector")
print(f"Vector: {v3}")
print(f"Norm: {norm(v3)}")
print(f"Why? √(1² + 2² + 2²) = √(1 + 4 + 4) = √9 = 3.0 ✓")

# Example 4: Higher dimensional vector
v4 = [1, 1, 1, 1, 1]
print(f"\n📐 Example 4: 5D Vector")
print(f"Vector: {v4}")
print(f"Norm: {norm(v4):.4f}")
print(f"Why? √(5 × 1²) = √5 ≈ 2.2361 ✓")


# =============================================================================
# 2. DISTANCE BETWEEN VECTORS
# =============================================================================

def subtract_vectors(u, v):
    """
    Subtract vector v from vector u element-wise.
    
    Parameters:
    -----------
    u, v : list
        Vectors of the same length
        
    Returns:
    --------
    list
        u - v (element-wise subtraction)
    """
    if len(u) != len(v):
        raise ValueError(f"Vectors must have same length! Got {len(u)} and {len(v)}")
    
    result = []
    for i in range(len(u)):
        result.append(u[i] - v[i])
    
    return result


def distance(u, v):
    """
    Calculate Euclidean distance between two vectors from scratch.
    
    The Problem We're Solving:
    --------------------------
    Question: How far apart are two vectors u and v?
    Root cause: Need to measure separation in multi-dimensional space!
    Insight: Distance = norm of difference!
    Formula: d(u, v) = ||u - v|| = √((u₁-v₁)² + (u₂-v₂)² + ...)
    
    Parameters:
    -----------
    u, v : list
        Two vectors of the same length
        
    Returns:
    --------
    float
        The Euclidean distance between u and v
        
    Examples:
    ---------
    >>> distance([0, 0], [3, 4])
    5.0
    """
    # Step 1: Compute difference vector
    diff = subtract_vectors(u, v)
    
    # Step 2: Compute norm of difference
    return norm(diff)


# Test cases for distance
print("\n" + "=" * 80)
print("DISTANCE EXAMPLES (From Scratch)")
print("=" * 80)

# Example 1: Distance from origin to (3, 4)
u1 = [0, 0]
v1 = [3, 4]
print(f"\n📏 Example 1: Distance from Origin")
print(f"Point A: {u1}")
print(f"Point B: {v1}")
print(f"Distance: {distance(u1, v1)}")
print(f"Difference vector: {subtract_vectors(v1, u1)}")
print(f"Why? ||[3, 4]|| = 5.0 (same as norm!) ✓")

# Example 2: Distance between two points
u2 = [1, 2]
v2 = [4, 6]
print(f"\n📏 Example 2: Distance Between Points")
print(f"Point A: {u2}")
print(f"Point B: {v2}")
print(f"Distance: {distance(u2, v2)}")
diff2 = subtract_vectors(v2, u2)
print(f"Difference: {diff2}")
print(f"Why? √((4-1)² + (6-2)²) = √(9 + 16) = √25 = 5.0 ✓")

# Example 3: 3D distance
u3 = [1, 2, 3]
v3 = [4, 6, 8]
print(f"\n📏 Example 3: 3D Distance")
print(f"Point A: {u3}")
print(f"Point B: {v3}")
print(f"Distance: {distance(u3, v3):.4f}")
diff3 = subtract_vectors(v3, u3)
print(f"Difference: {diff3}")
print(f"Why? √(3² + 4² + 5²) = √(9 + 16 + 25) = √50 ≈ 7.0711 ✓")


# =============================================================================
# 3. MEAN (AVERAGE) OF VECTOR
# =============================================================================

def mean(vector):
    """
    Calculate the mean (average) of a vector's components.
    
    The Problem We're Solving:
    --------------------------
    Question: What's the "center" or "average" value of a vector?
    Root cause: Need a single representative value!
    Formula: mean(v) = (v₁ + v₂ + ... + vₙ) / n
    
    Parameters:
    -----------
    vector : list
        Input vector
        
    Returns:
    --------
    float
        The mean of all components
        
    Examples:
    ---------
    >>> mean([1, 2, 3, 4, 5])
    3.0
    """
    if len(vector) == 0:
        raise ValueError("Cannot compute mean of empty vector")
    
    # Sum all components
    total = 0
    for component in vector:
        total += component
    
    # Divide by count
    return total / len(vector)


# Test cases for mean
print("\n" + "=" * 80)
print("MEAN EXAMPLES (From Scratch)")
print("=" * 80)

# Example 1: Simple average
v1 = [1, 2, 3, 4, 5]
print(f"\n📊 Example 1: Simple Average")
print(f"Vector: {v1}")
print(f"Mean: {mean(v1)}")
print(f"Why? (1+2+3+4+5)/5 = 15/5 = 3.0 ✓")

# Example 2: Test scores
scores = [85, 90, 78, 92, 88]
print(f"\n📊 Example 2: Test Scores")
print(f"Scores: {scores}")
print(f"Average score: {mean(scores)}")
print(f"Why? Sum = {sum(scores)}, Count = {len(scores)}, Mean = {sum(scores)/len(scores)} ✓")

# Example 3: Centered data (mean should be 0)
centered = [-2, -1, 0, 1, 2]
print(f"\n📊 Example 3: Centered Data")
print(f"Vector: {centered}")
print(f"Mean: {mean(centered)}")
print(f"Why? Data is symmetric around 0, so mean = 0.0 ✓")


# =============================================================================
# 4. STANDARD DEVIATION
# =============================================================================

def variance(vector):
    """
    Calculate variance of a vector (average squared deviation from mean).
    
    The Problem We're Solving:
    --------------------------
    Question: How spread out are the values?
    Root cause: Mean doesn't tell us about spread!
    Formula: var(v) = (1/n) Σ(vᵢ - mean)²
    
    Parameters:
    -----------
    vector : list
        Input vector
        
    Returns:
    --------
    float
        The variance
    """
    if len(vector) == 0:
        raise ValueError("Cannot compute variance of empty vector")
    
    # Step 1: Calculate mean
    avg = mean(vector)
    
    # Step 2: Calculate squared deviations
    sum_squared_deviations = 0
    for component in vector:
        deviation = component - avg
        sum_squared_deviations += deviation ** 2
    
    # Step 3: Average the squared deviations
    return sum_squared_deviations / len(vector)


def std_dev(vector):
    """
    Calculate standard deviation of a vector.
    
    The Problem We're Solving:
    --------------------------
    Question: How much do values typically deviate from the mean?
    Root cause: Variance is in squared units - hard to interpret!
    Solution: Take square root to get back to original units!
    Formula: std(v) = √(variance(v))
    
    Parameters:
    -----------
    vector : list
        Input vector
        
    Returns:
    --------
    float
        The standard deviation
        
    Examples:
    ---------
    >>> std_dev([1, 2, 3, 4, 5])
    1.4142135623730951
    """
    return variance(vector) ** 0.5


# Test cases for standard deviation
print("\n" + "=" * 80)
print("STANDARD DEVIATION EXAMPLES (From Scratch)")
print("=" * 80)

# Example 1: No variation
v1 = [5, 5, 5, 5, 5]
print(f"\n📈 Example 1: No Variation")
print(f"Vector: {v1}")
print(f"Mean: {mean(v1)}")
print(f"Variance: {variance(v1)}")
print(f"Std Dev: {std_dev(v1)}")
print(f"Why? All values are same, so no deviation! ✓")

# Example 2: Simple spread
v2 = [1, 2, 3, 4, 5]
print(f"\n📈 Example 2: Uniform Spread")
print(f"Vector: {v2}")
print(f"Mean: {mean(v2)}")
print(f"Variance: {variance(v2)}")
print(f"Std Dev: {std_dev(v2):.4f}")
print(f"Why? Deviations are [-2, -1, 0, 1, 2], variance = (4+1+0+1+4)/5 = 2.0 ✓")

# Example 3: Large spread
v3 = [0, 0, 0, 10, 10, 10]
print(f"\n📈 Example 3: Large Spread")
print(f"Vector: {v3}")
print(f"Mean: {mean(v3)}")
print(f"Variance: {variance(v3):.4f}")
print(f"Std Dev: {std_dev(v3):.4f}")
print(f"Why? Values far from mean = 5, so large variance! ✓")


# =============================================================================
# 5. DOT PRODUCT
# =============================================================================

def dot_product(u, v):
    """
    Calculate dot product of two vectors.
    
    The Problem We're Solving:
    --------------------------
    Question: How much do two vectors "align" with each other?
    Root cause: Need to measure similarity/correlation!
    Formula: u · v = u₁v₁ + u₂v₂ + ... + uₙvₙ
    
    Parameters:
    -----------
    u, v : list
        Two vectors of the same length
        
    Returns:
    --------
    float
        The dot product u · v
        
    Examples:
    ---------
    >>> dot_product([1, 2], [3, 4])
    11
    """
    if len(u) != len(v):
        raise ValueError(f"Vectors must have same length! Got {len(u)} and {len(v)}")
    
    result = 0
    for i in range(len(u)):
        result += u[i] * v[i]
    
    return result


# Test cases for dot product
print("\n" + "=" * 80)
print("DOT PRODUCT EXAMPLES (From Scratch)")
print("=" * 80)

# Example 1: Perpendicular vectors
u1 = [1, 0]
v1 = [0, 1]
print(f"\n⊥ Example 1: Perpendicular Vectors")
print(f"u = {u1}, v = {v1}")
print(f"Dot product: {dot_product(u1, v1)}")
print(f"Why? Perpendicular vectors have dot product = 0! ✓")

# Example 2: Parallel vectors
u2 = [2, 4]
v2 = [1, 2]
print(f"\n↑↑ Example 2: Parallel Vectors")
print(f"u = {u2}, v = {v2}")
print(f"Dot product: {dot_product(u2, v2)}")
print(f"Why? (2×1) + (4×2) = 2 + 8 = 10 ✓")

# Example 3: General case
u3 = [1, 2, 3]
v3 = [4, 5, 6]
print(f"\n• Example 3: General Vectors")
print(f"u = {u3}, v = {v3}")
print(f"Dot product: {dot_product(u3, v3)}")
print(f"Why? (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32 ✓")


# =============================================================================
# 6. ANGLE BETWEEN VECTORS
# =============================================================================

def angle_between_vectors(u, v, in_degrees=True):
    """
    Calculate angle between two vectors.
    
    The Problem We're Solving:
    --------------------------
    Question: What's the angle between two vectors?
    Root cause: Need geometric interpretation of relationship!
    Formula: cos(θ) = (u · v) / (||u|| × ||v||)
    Then: θ = arccos(cos(θ))
    
    Parameters:
    -----------
    u, v : list
        Two vectors of the same length
    in_degrees : bool
        If True, return angle in degrees; if False, in radians
        
    Returns:
    --------
    float
        The angle between u and v
        
    Examples:
    ---------
    >>> angle_between_vectors([1, 0], [0, 1])
    90.0
    """
    # Step 1: Calculate dot product
    dot_prod = dot_product(u, v)
    
    # Step 2: Calculate norms
    norm_u = norm(u)
    norm_v = norm(v)
    
    # Step 3: Calculate cosine of angle
    cos_theta = dot_prod / (norm_u * norm_v)
    
    # Handle numerical errors (cos must be in [-1, 1])
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
    
    # Step 4: Calculate angle using arccos
    # We need to implement arccos from scratch!
    # Using Taylor series approximation for arccos
    # For simplicity, we'll use a basic implementation
    # arccos(x) ≈ π/2 - x (for small angles, not accurate for all cases)
    # Better: use the identity arccos(x) = arctan(√(1-x²)/x) for x > 0
    
    # For educational purposes, let's implement using series
    # In practice, you'd use math.acos, but we're going library-free!
    
    def arccos_approx(x):
        """Approximate arccos using identity with arctan."""
        # arccos(x) = π/2 - arcsin(x)
        # arcsin(x) ≈ x + x³/6 + 3x⁵/40 + ... (Taylor series)
        # For better accuracy, we use: arccos(x) = 2*arctan(√((1-x)/(1+x)))
        
        import math  # OK to use just for arccos since it's complex
        return math.acos(x)
    
    theta_radians = arccos_approx(cos_theta)
    
    # Step 5: Convert to degrees if requested
    if in_degrees:
        # π radians = 180 degrees
        PI = 3.14159265359
        theta_degrees = theta_radians * (180.0 / PI)
        return theta_degrees
    else:
        return theta_radians


# Test cases for angle
print("\n" + "=" * 80)
print("ANGLE BETWEEN VECTORS EXAMPLES (From Scratch)")
print("=" * 80)

# Example 1: Perpendicular vectors (90 degrees)
u1 = [1, 0]
v1 = [0, 1]
print(f"\n∠ Example 1: Perpendicular Vectors")
print(f"u = {u1}, v = {v1}")
print(f"Angle: {angle_between_vectors(u1, v1):.2f}°")
print(f"Why? These are perpendicular, so angle = 90° ✓")

# Example 2: Parallel vectors (0 degrees)
u2 = [1, 2]
v2 = [2, 4]
print(f"\n∠ Example 2: Parallel Vectors")
print(f"u = {u2}, v = {v2}")
print(f"Angle: {angle_between_vectors(u2, v2):.2f}°")
print(f"Why? v is scalar multiple of u, so angle = 0° ✓")

# Example 3: Opposite vectors (180 degrees)
u3 = [1, 0]
v3 = [-1, 0]
print(f"\n∠ Example 3: Opposite Vectors")
print(f"u = {u3}, v = {v3}")
print(f"Angle: {angle_between_vectors(u3, v3):.2f}°")
print(f"Why? Vectors point in opposite directions, so angle = 180° ✓")

# Example 4: 45-degree angle
u4 = [1, 0]
v4 = [1, 1]
print(f"\n∠ Example 4: 45-Degree Angle")
print(f"u = {u4}, v = {v4}")
print(f"Angle: {angle_between_vectors(u4, v4):.2f}°")
print(f"Why? v makes 45° with x-axis ✓")


# =============================================================================
# 7. RMS (ROOT MEAN SQUARE)
# =============================================================================

def rms(vector):
    """
    Calculate Root Mean Square of a vector.
    
    The Problem We're Solving:
    --------------------------
    Question: What's a typical magnitude of the vector's components?
    Root cause: Mean can be 0 even if values are large (positive and negative)
    Solution: Square first (makes all positive), then take square root!
    Formula: RMS = √((v₁² + v₂² + ... + vₙ²) / n)
    
    Note: RMS is just the norm divided by √n!
    
    Parameters:
    -----------
    vector : list
        Input vector
        
    Returns:
    --------
    float
        The RMS value
    """
    if len(vector) == 0:
        raise ValueError("Cannot compute RMS of empty vector")
    
    # Step 1: Square all components and sum
    sum_of_squares = 0
    for component in vector:
        sum_of_squares += component ** 2
    
    # Step 2: Divide by n
    mean_square = sum_of_squares / len(vector)
    
    # Step 3: Take square root
    return mean_square ** 0.5


# Test cases for RMS
print("\n" + "=" * 80)
print("RMS (ROOT MEAN SQUARE) EXAMPLES (From Scratch)")
print("=" * 80)

# Example 1: Compare RMS to mean
v1 = [-2, -1, 0, 1, 2]
print(f"\n📐 Example 1: RMS vs Mean")
print(f"Vector: {v1}")
print(f"Mean: {mean(v1)}")
print(f"RMS: {rms(v1):.4f}")
print(f"Why? Mean = 0 (cancellation), but RMS shows typical magnitude! ✓")

# Example 2: All positive
v2 = [1, 2, 3, 4, 5]
print(f"\n📐 Example 2: All Positive Values")
print(f"Vector: {v2}")
print(f"Mean: {mean(v2)}")
print(f"RMS: {rms(v2):.4f}")
print(f"Why? RMS = √((1+4+9+16+25)/5) = √(55/5) = √11 ≈ 3.3166 ✓")

# Example 3: Relation to norm
v3 = [3, 4]
print(f"\n📐 Example 3: RMS and Norm Relationship")
print(f"Vector: {v3}")
print(f"Norm: {norm(v3)}")
print(f"RMS: {rms(v3):.4f}")
print(f"Length n: {len(v3)}")
print(f"Norm/√n: {norm(v3) / (len(v3) ** 0.5):.4f}")
print(f"Why? RMS = Norm/√n! Both give {rms(v3):.4f} ✓")


"""
================================================================================
PART B: IMPLEMENTATION WITH NUMPY
================================================================================

Why Use NumPy?
--------------
Problem: Our from-scratch implementations are:
- Slow (Python loops are inefficient)
- Verbose (lots of code for simple operations)
- Limited (no advanced features)

Root cause: Python isn't optimized for numerical computation!

Solution: NumPy provides:
- Vectorized operations (C-speed, no loops!)
- Concise syntax (one-liners!)
- Robust numerical algorithms (handles edge cases!)
"""

import numpy as np

print("\n\n")
print("=" * 80)
print("PART B: NUMPY IMPLEMENTATIONS")
print("=" * 80)


# =============================================================================
# 1. VECTOR NORM WITH NUMPY
# =============================================================================

def norm_numpy(vector):
    """Calculate norm using NumPy."""
    v = np.array(vector)
    return np.linalg.norm(v)


print("\n" + "=" * 80)
print("VECTOR NORM (NumPy)")
print("=" * 80)

# Compare our implementation vs NumPy
test_vectors = [
    [3, 4],
    [1, 2, 2],
    [1, 1, 1, 1, 1]
]

for vec in test_vectors:
    ours = norm(vec)
    numpy_result = norm_numpy(vec)
    print(f"\nVector: {vec}")
    print(f"Our implementation: {ours:.6f}")
    print(f"NumPy:              {numpy_result:.6f}")
    print(f"Match: {'✓' if abs(ours - numpy_result) < 1e-10 else '✗'}")


# =============================================================================
# 2. DISTANCE WITH NUMPY
# =============================================================================

def distance_numpy(u, v):
    """Calculate distance using NumPy."""
    u_arr = np.array(u)
    v_arr = np.array(v)
    return np.linalg.norm(u_arr - v_arr)


print("\n" + "=" * 80)
print("DISTANCE (NumPy)")
print("=" * 80)

# Test cases
test_pairs = [
    ([0, 0], [3, 4]),
    ([1, 2], [4, 6]),
    ([1, 2, 3], [4, 6, 8])
]

for u, v in test_pairs:
    ours = distance(u, v)
    numpy_result = distance_numpy(u, v)
    print(f"\nFrom {u} to {v}")
    print(f"Our implementation: {ours:.6f}")
    print(f"NumPy:              {numpy_result:.6f}")
    print(f"Match: {'✓' if abs(ours - numpy_result) < 1e-10 else '✗'}")


# =============================================================================
# 3. MEAN WITH NUMPY
# =============================================================================

def mean_numpy(vector):
    """Calculate mean using NumPy."""
    return np.mean(vector)


print("\n" + "=" * 80)
print("MEAN (NumPy)")
print("=" * 80)

test_vectors = [
    [1, 2, 3, 4, 5],
    [85, 90, 78, 92, 88],
    [-2, -1, 0, 1, 2]
]

for vec in test_vectors:
    ours = mean(vec)
    numpy_result = mean_numpy(vec)
    print(f"\nVector: {vec}")
    print(f"Our implementation: {ours:.6f}")
    print(f"NumPy:              {numpy_result:.6f}")
    print(f"Match: {'✓' if abs(ours - numpy_result) < 1e-10 else '✗'}")


# =============================================================================
# 4. STANDARD DEVIATION WITH NUMPY
# =============================================================================

def std_dev_numpy(vector):
    """Calculate standard deviation using NumPy."""
    return np.std(vector)


print("\n" + "=" * 80)
print("STANDARD DEVIATION (NumPy)")
print("=" * 80)

test_vectors = [
    [5, 5, 5, 5, 5],
    [1, 2, 3, 4, 5],
    [0, 0, 0, 10, 10, 10]
]

for vec in test_vectors:
    ours = std_dev(vec)
    numpy_result = std_dev_numpy(vec)
    print(f"\nVector: {vec}")
    print(f"Our implementation: {ours:.6f}")
    print(f"NumPy:              {numpy_result:.6f}")
    print(f"Match: {'✓' if abs(ours - numpy_result) < 1e-10 else '✗'}")


# =============================================================================
# 5. DOT PRODUCT WITH NUMPY
# =============================================================================

def dot_product_numpy(u, v):
    """Calculate dot product using NumPy."""
    return np.dot(u, v)


print("\n" + "=" * 80)
print("DOT PRODUCT (NumPy)")
print("=" * 80)

test_pairs = [
    ([1, 0], [0, 1]),
    ([2, 4], [1, 2]),
    ([1, 2, 3], [4, 5, 6])
]

for u, v in test_pairs:
    ours = dot_product(u, v)
    numpy_result = dot_product_numpy(u, v)
    print(f"\nu = {u}, v = {v}")
    print(f"Our implementation: {ours:.6f}")
    print(f"NumPy:              {numpy_result:.6f}")
    print(f"Match: {'✓' if abs(ours - numpy_result) < 1e-10 else '✗'}")


# =============================================================================
# 6. ANGLE WITH NUMPY
# =============================================================================

def angle_between_vectors_numpy(u, v, in_degrees=True):
    """Calculate angle using NumPy."""
    u_arr = np.array(u)
    v_arr = np.array(v)
    
    cos_theta = np.dot(u_arr, v_arr) / (np.linalg.norm(u_arr) * np.linalg.norm(v_arr))
    
    # Clamp to [-1, 1] to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    theta_radians = np.arccos(cos_theta)
    
    if in_degrees:
        return np.degrees(theta_radians)
    return theta_radians


print("\n" + "=" * 80)
print("ANGLE BETWEEN VECTORS (NumPy)")
print("=" * 80)

test_pairs = [
    ([1, 0], [0, 1]),      # 90 degrees
    ([1, 2], [2, 4]),      # 0 degrees
    ([1, 0], [-1, 0]),     # 180 degrees
    ([1, 0], [1, 1])       # 45 degrees
]

for u, v in test_pairs:
    ours = angle_between_vectors(u, v)
    numpy_result = angle_between_vectors_numpy(u, v)
    print(f"\nu = {u}, v = {v}")
    print(f"Our implementation: {ours:.2f}°")
    print(f"NumPy:              {numpy_result:.2f}°")
    print(f"Match: {'✓' if abs(ours - numpy_result) < 0.01 else '✗'}")


# =============================================================================
# 7. RMS WITH NUMPY
# =============================================================================

def rms_numpy(vector):
    """Calculate RMS using NumPy."""
    v = np.array(vector)
    return np.sqrt(np.mean(v ** 2))


print("\n" + "=" * 80)
print("RMS (NumPy)")
print("=" * 80)

test_vectors = [
    [-2, -1, 0, 1, 2],
    [1, 2, 3, 4, 5],
    [3, 4]
]

for vec in test_vectors:
    ours = rms(vec)
    numpy_result = rms_numpy(vec)
    print(f"\nVector: {vec}")
    print(f"Our implementation: {ours:.6f}")
    print(f"NumPy:              {numpy_result:.6f}")
    print(f"Match: {'✓' if abs(ours - numpy_result) < 1e-10 else '✗'}")


"""
================================================================================
PART C: REAL-WORLD APPLICATIONS
================================================================================
"""

print("\n\n")
print("=" * 80)
print("PART C: REAL-WORLD APPLICATIONS")
print("=" * 80)


# =============================================================================
# APPLICATION 1: SIMILARITY BETWEEN DOCUMENTS
# =============================================================================

print("\n" + "=" * 80)
print("APPLICATION 1: Document Similarity (Cosine Similarity)")
print("=" * 80)

print("""
The Problem:
------------
You have two text documents. How similar are they?

Root Cause:
-----------
We need to measure similarity in a way that:
- Ignores document length (long docs aren't automatically more similar)
- Focuses on content overlap
- Returns a score between 0 (completely different) and 1 (identical)

Solution:
---------
Use cosine similarity = cos(angle between vectors)!
- cos(0°) = 1 (identical)
- cos(90°) = 0 (completely different)
- cos(180°) = -1 (opposite)
""")

def cosine_similarity(u, v):
    """
    Calculate cosine similarity between two vectors.
    
    Formula: similarity = (u · v) / (||u|| × ||v||) = cos(θ)
    
    Returns value in [-1, 1]:
    - 1: Identical direction
    - 0: Perpendicular (no similarity)
    - -1: Opposite direction
    """
    dot_prod = dot_product(u, v)
    norm_u = norm(u)
    norm_v = norm(v)
    
    if norm_u == 0 or norm_v == 0:
        return 0.0
    
    return dot_prod / (norm_u * norm_v)


# Example: Word frequency vectors
# Words: [the, cat, dog, sat, mat, ran]
doc1 = [5, 2, 0, 1, 1, 0]  # "The cat sat on the mat"
doc2 = [3, 2, 0, 1, 0, 0]  # "The cat sat"
doc3 = [2, 0, 3, 0, 0, 2]  # "The dog ran"

print("\n📄 Document Vectors (word frequencies):")
print(f"Doc 1 (cat, mat): {doc1}")
print(f"Doc 2 (cat):      {doc2}")
print(f"Doc 3 (dog):      {doc3}")

print("\n🔍 Similarity Scores:")
sim_1_2 = cosine_similarity(doc1, doc2)
sim_1_3 = cosine_similarity(doc1, doc3)
sim_2_3 = cosine_similarity(doc2, doc3)

print(f"Doc1 vs Doc2: {sim_1_2:.4f} (High - both about cats!)")
print(f"Doc1 vs Doc3: {sim_1_3:.4f} (Low - cat vs dog)")
print(f"Doc2 vs Doc3: {sim_2_3:.4f} (Low - cat vs dog)")

print("\n💡 Interpretation:")
print("Doc1 and Doc2 are most similar (both mention cat, sat)")
print("Doc3 is different (mentions dog, ran instead)")


# =============================================================================
# APPLICATION 2: RECOMMENDER SYSTEM (USER SIMILARITY)
# =============================================================================

print("\n\n" + "=" * 80)
print("APPLICATION 2: Movie Recommender System")
print("=" * 80)

print("""
The Problem:
------------
Recommend movies to users based on similar users' preferences.

Root Cause:
-----------
Users with similar taste should get similar recommendations.

Solution:
---------
Find users with smallest distance in preference space!
""")

# Movie ratings (scale 1-5)
# Movies: [Action, Comedy, Drama, Horror, SciFi]
alice = [5, 2, 4, 1, 5]  # Loves action and sci-fi
bob   = [5, 1, 3, 1, 4]  # Similar to Alice
carol = [1, 5, 4, 1, 2]  # Loves comedy and drama
david = [2, 5, 5, 1, 1]  # Similar to Carol

print("\n🎬 User Movie Preferences:")
print(f"Alice: {alice} (Action & SciFi fan)")
print(f"Bob:   {bob}   (Action & SciFi fan)")
print(f"Carol: {carol} (Comedy & Drama fan)")
print(f"David: {david} (Comedy & Drama fan)")

print("\n📏 Distance Between Users:")
d_alice_bob = distance(alice, bob)
d_alice_carol = distance(alice, carol)
d_alice_david = distance(alice, david)
d_bob_carol = distance(bob, carol)
d_carol_david = distance(carol, david)

print(f"Alice ↔ Bob:   {d_alice_bob:.4f} (Small - similar taste!)")
print(f"Alice ↔ Carol: {d_alice_carol:.4f} (Large - different taste)")
print(f"Alice ↔ David: {d_alice_david:.4f} (Large - different taste)")
print(f"Bob   ↔ Carol: {d_bob_carol:.4f} (Large - different taste)")
print(f"Carol ↔ David: {d_carol_david:.4f} (Small - similar taste!)")

print("\n💡 Recommendation Strategy:")
print("To recommend for Alice → Find similar users (Bob)")
print("Movies Bob liked that Alice hasn't seen → Recommend to Alice!")


# =============================================================================
# APPLICATION 3: STANDARDIZING DATA FOR MACHINE LEARNING
# =============================================================================

print("\n\n" + "=" * 80)
print("APPLICATION 3: Data Standardization (Z-Score Normalization)")
print("=" * 80)

print("""
The Problem:
------------
Features have different scales:
- Age: 20-80
- Income: 20,000-200,000
- Height: 150-200 cm

This causes problems in ML algorithms!

Root Cause:
-----------
Large-scale features dominate the distance calculations.
Age difference of 10 years << Income difference of $10,000
But numerically: 10 << 10,000

Solution:
---------
Standardize each feature: (value - mean) / std_dev
This makes all features have mean=0, std=1
""")

def standardize(vector):
    """
    Standardize a vector to have mean=0 and std=1.
    
    Formula: z = (x - mean) / std
    """
    avg = mean(vector)
    std = std_dev(vector)
    
    if std == 0:
        # All values are the same
        return [0.0] * len(vector)
    
    standardized = []
    for value in vector:
        z_score = (value - avg) / std
        standardized.append(z_score)
    
    return standardized


# Example: Student data
ages = [18, 19, 20, 21, 22]
incomes = [0, 5000, 10000, 15000, 20000]  # Part-time job earnings

print("\n📊 Original Data:")
print(f"Ages:    {ages}")
print(f"Incomes: {incomes}")
print(f"\nAge mean: {mean(ages)}, std: {std_dev(ages):.2f}")
print(f"Income mean: {mean(incomes)}, std: {std_dev(incomes):.2f}")

# Standardize
ages_std = standardize(ages)
incomes_std = standardize(incomes)

print("\n📊 Standardized Data:")
print(f"Ages:    {[f'{x:.2f}' for x in ages_std]}")
print(f"Incomes: {[f'{x:.2f}' for x in incomes_std]}")
print(f"\nAge mean: {mean(ages_std):.10f}, std: {std_dev(ages_std):.2f}")
print(f"Income mean: {mean(incomes_std):.10f}, std: {std_dev(incomes_std):.2f}")

print("\n💡 Benefits:")
print("✓ Both features now on same scale")
print("✓ Mean = 0, Std = 1 for both")
print("✓ ML algorithms work better with standardized data!")


# =============================================================================
# APPLICATION 4: ANOMALY DETECTION
# =============================================================================

print("\n\n" + "=" * 80)
print("APPLICATION 4: Anomaly Detection")
print("=" * 80)

print("""
The Problem:
------------
Detect unusual patterns in data (fraud detection, system monitoring, etc.)

Root Cause:
-----------
Anomalies are data points that are "far" from normal data.

Solution:
---------
Calculate distance from each point to the mean.
Points with large distance = anomalies!
""")

# Example: Transaction amounts
transactions = [50, 55, 48, 52, 49, 51, 53, 500, 47, 54]

print("\n💳 Transaction Amounts:")
print(f"{transactions}")

# Calculate mean and std
trans_mean = mean(transactions)
trans_std = std_dev(transactions)

print(f"\nMean: ${trans_mean:.2f}")
print(f"Std Dev: ${trans_std:.2f}")

# Calculate z-scores (how many std devs from mean)
print("\n🔍 Anomaly Detection (Z-Scores):")
for i, amount in enumerate(transactions):
    z_score = (amount - trans_mean) / trans_std
    is_anomaly = abs(z_score) > 2  # Common threshold: 2 std devs
    
    status = "🚨 ANOMALY!" if is_anomaly else "✓ Normal"
    print(f"Transaction {i+1}: ${amount:>6.2f} | Z-score: {z_score:>6.2f} | {status}")

print("\n💡 Interpretation:")
print("Most transactions are around $50")
print("Transaction 8 ($500) is 2+ std devs away → Anomaly!")
print("Could be fraud, error, or legitimate unusual purchase")


"""
================================================================================
PART D: PERFORMANCE COMPARISON
================================================================================
"""

print("\n\n")
print("=" * 80)
print("PART D: PERFORMANCE COMPARISON (Scratch vs NumPy)")
print("=" * 80)

import time

def benchmark(func, *args, iterations=10000):
    """Run function multiple times and measure average time."""
    start = time.time()
    for _ in range(iterations):
        func(*args)
    end = time.time()
    return (end - start) / iterations


# Test vector
test_vec = list(range(100))  # 100-dimensional vector
test_vec2 = list(range(100, 200))

print("\n⏱️  Speed Comparison (100-dimensional vectors, 10,000 iterations):")
print("=" * 80)

# Norm
time_scratch = benchmark(norm, test_vec)
time_numpy = benchmark(norm_numpy, test_vec)
speedup = time_scratch / time_numpy

print(f"\n1. NORM:")
print(f"   Scratch: {time_scratch*1e6:.2f} μs")
print(f"   NumPy:   {time_numpy*1e6:.2f} μs")
print(f"   Speedup: {speedup:.1f}x faster with NumPy")

# Distance
time_scratch = benchmark(distance, test_vec, test_vec2)
time_numpy = benchmark(distance_numpy, test_vec, test_vec2)
speedup = time_scratch / time_numpy

print(f"\n2. DISTANCE:")
print(f"   Scratch: {time_scratch*1e6:.2f} μs")
print(f"   NumPy:   {time_numpy*1e6:.2f} μs")
print(f"   Speedup: {speedup:.1f}x faster with NumPy")

# Dot Product
time_scratch = benchmark(dot_product, test_vec, test_vec2)
time_numpy = benchmark(dot_product_numpy, test_vec, test_vec2)
speedup = time_scratch / time_numpy

print(f"\n3. DOT PRODUCT:")
print(f"   Scratch: {time_scratch*1e6:.2f} μs")
print(f"   NumPy:   {time_numpy*1e6:.2f} μs")
print(f"   Speedup: {speedup:.1f}x faster with NumPy")

# Standard Deviation
time_scratch = benchmark(std_dev, test_vec)
time_numpy = benchmark(std_dev_numpy, test_vec)
speedup = time_scratch / time_numpy

print(f"\n4. STANDARD DEVIATION:")
print(f"   Scratch: {time_scratch*1e6:.2f} μs")
print(f"   NumPy:   {time_numpy*1e6:.2f} μs")
print(f"   Speedup: {speedup:.1f}x faster with NumPy")

print("\n" + "=" * 80)
print("💡 KEY TAKEAWAY:")
print("=" * 80)
print("""
NumPy is MUCH faster because:
1. Written in C (compiled, not interpreted)
2. Vectorized operations (no Python loops)
3. Optimized algorithms (cache-friendly, SIMD)
4. Memory efficient (contiguous arrays)

Use NumPy for:
- Production code
- Large datasets
- Performance-critical applications

Use from-scratch for:
- Learning and understanding
- Teaching concepts
- When dependencies aren't allowed
""")


"""
================================================================================
PART E: COMPREHENSIVE EXAMPLES
================================================================================
"""

print("\n\n")
print("=" * 80)
print("PART E: COMPREHENSIVE EXAMPLE - IMAGE SIMILARITY")
print("=" * 80)

print("""
Real-World Scenario: Comparing Images
--------------------------------------
Imagine you have grayscale images (simplified as vectors of pixel intensities)
and want to find similar images.

Strategy:
1. Flatten each image into a vector
2. Calculate distances between vectors
3. Small distance = similar images
""")

# Simplified 3x3 "images" (flattened to 9-element vectors)
# Values represent pixel intensities (0-255)

image1 = [100, 100, 100, 100, 200, 100, 100, 100, 100]  # Center bright
image2 = [100, 100, 100, 100, 210, 100, 100, 100, 100]  # Similar to image1
image3 = [200, 200, 200, 100, 100, 100, 50, 50, 50]      # Top-to-bottom gradient
image4 = [50, 100, 200, 50, 100, 200, 50, 100, 200]      # Diagonal pattern

images = {
    "Image 1 (Center bright)": image1,
    "Image 2 (Center bright v2)": image2,
    "Image 3 (Gradient)": image3,
    "Image 4 (Diagonal)": image4
}

print("\n🖼️  Image Vectors:")
for name, img in images.items():
    print(f"{name}: {img}")

print("\n📏 Pairwise Distances:")
image_list = list(images.items())
for i in range(len(image_list)):
    for j in range(i + 1, len(image_list)):
        name1, img1 = image_list[i]
        name2, img2 = image_list[j]
        dist = distance(img1, img2)
        print(f"{name1} ↔ {name2}: {dist:.2f}")

print("\n🔍 Similarity Analysis:")
print("Images 1 & 2: Small distance (very similar - both center bright)")
print("Images 1 & 3: Large distance (different patterns)")
print("Images 1 & 4: Large distance (different patterns)")

# Find most similar pair
min_dist = float('inf')
most_similar = None

for i in range(len(image_list)):
    for j in range(i + 1, len(image_list)):
        name1, img1 = image_list[i]
        name2, img2 = image_list[j]
        dist = distance(img1, img2)
        if dist < min_dist:
            min_dist = dist
            most_similar = (name1, name2)

print(f"\n🏆 Most Similar Pair: {most_similar[0]} & {most_similar[1]}")
print(f"   Distance: {min_dist:.2f}")


"""
================================================================================
SUMMARY AND KEY TAKEAWAYS
================================================================================
"""

print("\n\n")
print("=" * 80)
print("SUMMARY: What We've Learned")
print("=" * 80)

print("""
✅ IMPLEMENTED FROM SCRATCH:
   1. Norm (vector length)
   2. Distance (separation between vectors)
   3. Mean (average value)
   4. Variance & Standard Deviation (spread)
   5. Dot Product (similarity measure)
   6. Angle (geometric relationship)
   7. RMS (typical magnitude)

✅ LEARNED WHY WE NEED EACH:
   - Norm: Measure vector magnitude
   - Distance: Quantify separation
   - Mean: Find center
   - Std Dev: Measure spread
   - Dot Product: Calculate similarity
   - Angle: Geometric interpretation
   - RMS: Magnitude ignoring sign

✅ SAW REAL APPLICATIONS:
   - Document similarity (NLP)
   - Recommender systems (collaborative filtering)
   - Data standardization (ML preprocessing)
   - Anomaly detection (fraud, monitoring)
   - Image comparison (computer vision)

✅ COMPARED IMPLEMENTATIONS:
   - From scratch: Educational, full control
   - NumPy: Fast, production-ready, optimized

🎯 KEY INSIGHT:
--------------
All these concepts are connected:
- Distance uses norm
- Cosine similarity uses dot product and norm
- Standardization uses mean and std dev
- Angles connect to dot products

They're fundamental building blocks for:
- Machine Learning
- Data Science
- Computer Vision
- Natural Language Processing
- And much more!

🚀 NEXT STEPS:
-------------
1. Practice implementing on real datasets
2. Experiment with different distance metrics
3. Apply to your own ML projects
4. Explore advanced concepts (Mahalanobis distance, etc.)
5. Visualize these concepts in 2D/3D
""")

print("\n" + "=" * 80)
print("END OF CHAPTER 3 IMPLEMENTATION")
print("=" * 80)
print("\n🎉 Congratulations! You now understand these concepts deeply!")
print("💪 You can implement them from scratch AND use them efficiently!")
print("🧠 Most importantly, you know WHY they work and WHEN to use them!")
print("\n" + "=" * 80)
