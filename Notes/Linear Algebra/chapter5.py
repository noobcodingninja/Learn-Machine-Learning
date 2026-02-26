"""
================================================================================
Chapter 5: Python Implementation
Linear Independence, Basis, and Orthogonality
================================================================================

Table of Contents:
1. Part A: Linear Independence (From Scratch)
2. Part B: Span and Subspaces (From Scratch)
3. Part C: Basis and Dimension (From Scratch)
4. Part D: Orthogonality and Dot Product (From Scratch)
5. Part E: Gram-Schmidt Process (From Scratch)
6. Part F: NumPy Implementations
7. Part G: Applications (PCA, QR Decomposition, Regression)
8. Part H: Visualizations

================================================================================
WHY DO WE NEED THESE CONCEPTS?
================================================================================

The Big Picture Problem:
------------------------
In Machine Learning, we work with HIGH-DIMENSIONAL data:
- Image: 1000×1000 pixels = 1,000,000 dimensions!
- Text: 50,000 words vocabulary = 50,000 dimensions!

Questions we must answer:
1. Are all these dimensions truly INDEPENDENT? (Linear Independence)
2. What space do they span? (Span & Subspaces)
3. What's the MINIMAL set we need? (Basis & Dimension)
4. Are features correlated? (Orthogonality)
5. How to decorrelate them? (Gram-Schmidt)

Root Cause:
-----------
Most high-dimensional data has REDUNDANCY!
- Many dimensions are just combinations of others
- True dimensionality is much lower
- We need to find the fundamental, independent directions

Solution:
---------
Master linear independence, basis, and orthogonality!
These concepts let us:
- Remove redundant features
- Find true dimensionality (PCA)
- Create better coordinate systems
- Make ML algorithms work better

================================================================================
PART A: LINEAR INDEPENDENCE
================================================================================

The Problem:
------------
Given vectors v₁, v₂, ..., vₖ, are they linearly independent?

Why this matters:
-----------------
Independent vectors = No redundancy = Efficient representation!
Dependent vectors = Redundancy = Wasted computation!

Test: c₁v₁ + c₂v₂ + ... + cₖvₖ = 0
If ONLY solution is all cᵢ = 0 → Independent
If non-zero solution exists → Dependent
"""

def are_linearly_independent(vectors):
    """
    Check if a set of vectors is linearly independent.
    
    Method: Form matrix with vectors as columns, compute rank.
    If rank = number of vectors → Independent
    If rank < number of vectors → Dependent
    
    Parameters:
    -----------
    vectors : list of lists
        Each inner list is a vector
        
    Returns:
    --------
    bool, int
        (is_independent, rank)
    """
    if not vectors:
        return True, 0
    
    # Convert to matrix (vectors as columns)
    n_vectors = len(vectors)
    n_dims = len(vectors[0])
    
    # Create matrix
    matrix = [[vectors[j][i] for j in range(n_vectors)] 
              for i in range(n_dims)]
    
    # Calculate rank using Gaussian elimination
    rank = calculate_rank(matrix)
    
    is_independent = (rank == n_vectors)
    
    return is_independent, rank


def calculate_rank(matrix):
    """
    Calculate rank of a matrix using Gaussian elimination.
    
    Rank = number of non-zero rows after row reduction
    
    Why we need rank:
    -----------------
    Rank tells us how many truly independent rows/columns exist!
    """
    if not matrix or not matrix[0]:
        return 0
    
    # Create a copy to avoid modifying original
    m = [row[:] for row in matrix]
    rows = len(m)
    cols = len(m[0])
    
    rank = 0
    tolerance = 1e-10
    
    for col in range(cols):
        # Find pivot (largest absolute value in column)
        pivot_row = None
        max_val = 0
        
        for row in range(rank, rows):
            if abs(m[row][col]) > max_val:
                max_val = abs(m[row][col])
                pivot_row = row
        
        # If no pivot found, this column is dependent
        if max_val < tolerance:
            continue
        
        # Swap rows to bring pivot to top
        if pivot_row != rank:
            m[rank], m[pivot_row] = m[pivot_row], m[rank]
        
        # Scale pivot row
        pivot = m[rank][col]
        for j in range(cols):
            m[rank][j] /= pivot
        
        # Eliminate column in other rows
        for row in range(rows):
            if row != rank:
                factor = m[row][col]
                for j in range(cols):
                    m[row][j] -= factor * m[rank][j]
        
        rank += 1
    
    return rank


# =============================================================================
# TEST LINEAR INDEPENDENCE
# =============================================================================

print("=" * 80)
print("PART A: LINEAR INDEPENDENCE")
print("=" * 80)
print()

print("📚 What is Linear Independence?")
print("-" * 80)
print("""
Vectors are linearly INDEPENDENT if:
- None can be written as a combination of others
- No redundancy in the set
- Each vector adds NEW information/direction

Vectors are linearly DEPENDENT if:
- At least one is a combination of others
- Redundancy exists
- Wasted dimensions
""")
print()

# Example 1: Independent vectors in 2D
print("Example 1: Testing Independence in 2D")
print("-" * 80)

v1 = [1, 0]
v2 = [0, 1]

print(f"v₁ = {v1}")
print(f"v₂ = {v2}")
print()

is_indep, rank = are_linearly_independent([v1, v2])
print(f"Are they independent? {is_indep}")
print(f"Rank: {rank}")
print(f"Number of vectors: {len([v1, v2])}")
print()
print("💡 Interpretation: These are standard basis vectors!")
print("   They point in completely different directions → Independent")
print()

# Example 2: Dependent vectors in 2D
print("Example 2: Dependent Vectors")
print("-" * 80)

v1 = [2, 4]
v2 = [1, 2]

print(f"v₁ = {v1}")
print(f"v₂ = {v2}")
print()

is_indep, rank = are_linearly_independent([v1, v2])
print(f"Are they independent? {is_indep}")
print(f"Rank: {rank}")
print(f"Number of vectors: {len([v1, v2])}")
print()
print("💡 Interpretation: v₁ = 2 × v₂")
print("   They point in the SAME direction → Dependent!")
print("   Rank = 1 means only ONE independent direction")
print()

# Example 3: Three vectors in 2D (impossible to all be independent!)
print("Example 3: Three Vectors in 2D")
print("-" * 80)

v1 = [1, 0]
v2 = [0, 1]
v3 = [1, 1]

print(f"v₁ = {v1}")
print(f"v₂ = {v2}")
print(f"v₃ = {v3}")
print()

is_indep, rank = are_linearly_independent([v1, v2, v3])
print(f"Are they independent? {is_indep}")
print(f"Rank: {rank}")
print(f"Number of vectors: {len([v1, v2, v3])}")
print()
print("💡 Interpretation: Can't have 3 independent vectors in 2D!")
print("   v₃ = v₁ + v₂ (it's a combination)")
print("   Maximum independent vectors in n-dimensional space = n")
print()

# Example 4: Feature redundancy (ML scenario)
print("Example 4: Feature Redundancy in ML")
print("-" * 80)

print("Dataset with 3 features for 4 houses:")
print("Features: [Area_sqft, Area_sqm, Bedrooms]")
print()

# Area in sqft, Area in sqm (= sqft × 0.0929), Bedrooms
house1 = [1000, 92.9, 2]
house2 = [1500, 139.4, 3]
house3 = [2000, 185.8, 4]
house4 = [2500, 232.3, 5]

features = [house1, house2, house3, house4]

# Transpose to get feature vectors
feature_vectors = [[features[i][j] for i in range(4)] for j in range(3)]

print("Feature vector 1 (Area_sqft):", feature_vectors[0])
print("Feature vector 2 (Area_sqm): ", feature_vectors[1])
print("Feature vector 3 (Bedrooms): ", feature_vectors[2])
print()

is_indep, rank = are_linearly_independent(feature_vectors)
print(f"Are features independent? {is_indep}")
print(f"Rank: {rank}")
print(f"Number of features: {len(feature_vectors)}")
print()
print("💡 Interpretation:")
print("   Features 1 & 2 are DEPENDENT (sqm = 0.0929 × sqft)")
print("   True dimensionality = 2 (not 3!)")
print("   Should remove one area feature for better ML!")
print()


"""
================================================================================
PART B: SPAN AND SUBSPACES
================================================================================

The Problem:
------------
Given vectors v₁, v₂, ..., vₖ, what points can we reach by combining them?

Why this matters:
-----------------
Span tells us the "space" our vectors can represent!
In ML: Can our features represent the data we care about?

Definition: span{v₁, v₂, ...} = all linear combinations c₁v₁ + c₂v₂ + ...
"""

def is_in_span(vector, basis_vectors, tolerance=1e-10):
    """
    Check if a vector is in the span of basis vectors.
    
    Method: Try to solve: c₁v₁ + c₂v₂ + ... = vector
    If solution exists → vector is in span
    If no solution → vector is NOT in span
    
    Parameters:
    -----------
    vector : list
        Vector to test
    basis_vectors : list of lists
        Vectors that define the span
    tolerance : float
        Numerical tolerance for checking
        
    Returns:
    --------
    bool, list or None
        (is_in_span, coefficients if in span else None)
    """
    # Set up system: [v₁ v₂ ...] * c = vector
    # This is Ac = b, solve using Gaussian elimination
    
    n_dims = len(vector)
    n_vectors = len(basis_vectors)
    
    # Create augmented matrix [A | b]
    matrix = []
    for i in range(n_dims):
        row = [basis_vectors[j][i] for j in range(n_vectors)]
        row.append(vector[i])
        matrix.append(row)
    
    # Solve using Gaussian elimination
    coefficients = solve_linear_system(matrix)
    
    if coefficients is None:
        return False, None
    
    # Verify solution
    result = [sum(coefficients[j] * basis_vectors[j][i] 
                  for j in range(n_vectors)) 
              for i in range(n_dims)]
    
    # Check if result matches vector
    error = sum((result[i] - vector[i])**2 for i in range(n_dims))**0.5
    
    if error < tolerance:
        return True, coefficients
    else:
        return False, None


def solve_linear_system(augmented_matrix):
    """
    Solve linear system Ax = b using Gaussian elimination.
    
    Input: Augmented matrix [A | b]
    Output: Solution vector x, or None if no solution
    """
    if not augmented_matrix or not augmented_matrix[0]:
        return None
    
    # Create copy
    m = [row[:] for row in augmented_matrix]
    rows = len(m)
    cols = len(m[0]) - 1  # Exclude augmented column
    
    tolerance = 1e-10
    
    # Forward elimination
    for col in range(min(rows, cols)):
        # Find pivot
        pivot_row = None
        max_val = 0
        
        for row in range(col, rows):
            if abs(m[row][col]) > max_val:
                max_val = abs(m[row][col])
                pivot_row = row
        
        if max_val < tolerance:
            continue
        
        # Swap rows
        if pivot_row != col:
            m[col], m[pivot_row] = m[pivot_row], m[col]
        
        # Scale pivot row
        pivot = m[col][col]
        for j in range(len(m[0])):
            m[col][j] /= pivot
        
        # Eliminate
        for row in range(rows):
            if row != col:
                factor = m[row][col]
                for j in range(len(m[0])):
                    m[row][j] -= factor * m[col][j]
    
    # Extract solution
    solution = []
    for col in range(cols):
        # Find the row with pivot in this column
        pivot_row = None
        for row in range(rows):
            if abs(m[row][col] - 1.0) < tolerance:
                all_zeros = all(abs(m[row][j]) < tolerance 
                               for j in range(cols) if j != col)
                if all_zeros:
                    pivot_row = row
                    break
        
        if pivot_row is not None:
            solution.append(m[pivot_row][-1])
        else:
            solution.append(0.0)
    
    return solution


# =============================================================================
# TEST SPAN AND SUBSPACES
# =============================================================================

print("\n" + "=" * 80)
print("PART B: SPAN AND SUBSPACES")
print("=" * 80)
print()

print("📚 What is Span?")
print("-" * 80)
print("""
The SPAN of vectors is all points you can reach by combining them!

span{v₁, v₂, ...} = {c₁v₁ + c₂v₂ + ... | c₁, c₂, ... are any scalars}

Geometric interpretation:
- Span of 1 vector in 2D/3D: A line through origin
- Span of 2 independent vectors in 3D: A plane through origin  
- Span of 3 independent vectors in 3D: All of 3D space
""")
print()

# Example 1: Point in span
print("Example 1: Is Point in Span?")
print("-" * 80)

v1 = [1, 0]
v2 = [0, 1]
test_point = [3, 4]

print(f"Basis vectors: v₁ = {v1}, v₂ = {v2}")
print(f"Test point: {test_point}")
print()

in_span, coeffs = is_in_span(test_point, [v1, v2])
print(f"Is point in span? {in_span}")
if in_span:
    print(f"Coefficients: {[f'{c:.2f}' for c in coeffs]}")
    print(f"Verification: {coeffs[0]:.2f}×{v1} + {coeffs[1]:.2f}×{v2} = {test_point}")
print()
print("💡 Interpretation: Standard basis spans all of 2D!")
print("   Any 2D point can be written as c₁(1,0) + c₂(0,1)")
print()

# Example 2: Point NOT in span
print("Example 2: Point NOT in Span")
print("-" * 80)

v1 = [1, 0, 0]
v2 = [0, 1, 0]
test_point = [1, 2, 3]

print(f"Basis vectors: v₁ = {v1}, v₂ = {v2}")
print(f"These span the xy-plane in 3D")
print(f"Test point: {test_point}")
print()

in_span, coeffs = is_in_span(test_point, [v1, v2])
print(f"Is point in span? {in_span}")
print()
print("💡 Interpretation: Point has z=3, but span only covers xy-plane!")
print("   Cannot reach any point with non-zero z-coordinate")
print("   Need a third vector pointing in z-direction")
print()

# Example 3: Dimension of span
print("Example 3: Dimension of Span")
print("-" * 80)

v1 = [1, 2, 3]
v2 = [2, 4, 6]
v3 = [1, 0, 0]

print(f"v₁ = {v1}")
print(f"v₂ = {v2}")
print(f"v₃ = {v3}")
print()

is_indep, rank = are_linearly_independent([v1, v2, v3])
print(f"Are all three independent? {is_indep}")
print(f"Dimension of span (rank): {rank}")
print()
print("💡 Interpretation:")
print("   v₂ = 2×v₁ (dependent!)")
print("   So span{v₁, v₂, v₃} = span{v₁, v₃}")
print("   Dimension = 2 (a plane in 3D space)")
print()


"""
================================================================================
PART C: BASIS AND DIMENSION
================================================================================

The Problem:
------------
What's the MINIMAL set of vectors needed to span a space?

Why this matters:
-----------------
Basis = Most efficient representation!
In ML: Use fewer features while keeping all information!

Definition: A BASIS is a set of vectors that:
1. Are linearly independent (no redundancy)
2. Span the space (can reach everything)
"""

def find_basis(vectors):
    """
    Find a basis from a set of vectors (remove dependent ones).
    
    Algorithm:
    ----------
    1. Start with empty basis
    2. For each vector:
       - If NOT in span of current basis → add it
       - If in span → skip it (dependent!)
    3. Result: Maximal independent subset = basis
    
    Parameters:
    -----------
    vectors : list of lists
        Input vectors
        
    Returns:
    --------
    list of lists
        Basis vectors (maximal independent subset)
    """
    if not vectors:
        return []
    
    basis = []
    
    for vec in vectors:
        if not basis:
            # First vector always added
            basis.append(vec[:])
        else:
            # Check if vec is in span of current basis
            in_span, _ = is_in_span(vec, basis)
            
            if not in_span:
                # New direction! Add to basis
                basis.append(vec[:])
    
    return basis


def get_dimension(vectors):
    """
    Get the dimension of the span of vectors.
    
    Dimension = size of any basis = rank
    """
    _, rank = are_linearly_independent(vectors)
    return rank


# =============================================================================
# TEST BASIS AND DIMENSION
# =============================================================================

print("\n" + "=" * 80)
print("PART C: BASIS AND DIMENSION")
print("=" * 80)
print()

print("📚 What is a Basis?")
print("-" * 80)
print("""
A BASIS is a minimal spanning set:
- Independent (no redundancy)
- Spans the space (reaches everything)
- DIMENSION = number of vectors in any basis

Think of it as:
- The fundamental building blocks
- Minimal "toolkit" to build any vector
- Most efficient coordinate system
""")
print()

# Example 1: Standard basis
print("Example 1: Standard Basis in 3D")
print("-" * 80)

e1 = [1, 0, 0]
e2 = [0, 1, 0]
e3 = [0, 0, 1]

vectors = [e1, e2, e3]
basis = find_basis(vectors)
dim = get_dimension(vectors)

print(f"Vectors: {vectors}")
print(f"Basis: {basis}")
print(f"Dimension: {dim}")
print()
print("💡 Interpretation: Standard basis for ℝ³")
print("   These 3 vectors are the foundation of 3D space!")
print()

# Example 2: Redundant vectors
print("Example 2: Removing Redundant Vectors")
print("-" * 80)

v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [2, 0, 0]  # = 2×v1 (redundant!)
v4 = [1, 1, 0]  # = v1 + v2 (redundant!)

vectors = [v1, v2, v3, v4]
basis = find_basis(vectors)
dim = get_dimension(vectors)

print(f"Original vectors: 4 vectors")
for i, v in enumerate(vectors):
    print(f"  v{i+1} = {v}")
print()
print(f"Basis found: {len(basis)} vectors")
for i, v in enumerate(basis):
    print(f"  b{i+1} = {v}")
print(f"Dimension: {dim}")
print()
print("💡 Interpretation:")
print("   Started with 4 vectors, but only 2 are independent!")
print("   v3 and v4 are combinations of v1 and v2")
print("   Basis = {v1, v2} spans a 2D plane (xy-plane)")
print()

# Example 3: Feature selection analogy
print("Example 3: Feature Selection in ML")
print("-" * 80)

print("Original features:")
print("  f1: Temperature (°C)")
print("  f2: Temperature (°F) = 1.8×f1 + 32")
print("  f3: Humidity (%)")
print("  f4: Humidity (decimal) = f3/100")
print("  f5: Pressure (kPa)")
print()

# Simulate feature vectors (5 samples)
f1 = [20, 25, 30, 15, 22]  # Temperature °C
f2 = [68, 77, 86, 59, 71.6]  # Temperature °F
f3 = [60, 70, 50, 80, 65]  # Humidity %
f4 = [0.6, 0.7, 0.5, 0.8, 0.65]  # Humidity decimal
f5 = [101, 102, 100, 103, 101.5]  # Pressure

features = [f1, f2, f3, f4, f5]
basis_features = find_basis(features)
dim = get_dimension(features)

print(f"Total features: {len(features)}")
print(f"Independent features: {dim}")
print(f"Redundant features: {len(features) - dim}")
print()
print("💡 Interpretation:")
print("   f2 depends on f1 (temperature conversion)")
print("   f4 depends on f3 (humidity conversion)")
print("   True dimensionality = 3 (temp, humidity, pressure)")
print("   Should use only 3 features, not 5!")
print()


"""
================================================================================
PART D: ORTHOGONALITY AND DOT PRODUCT
================================================================================

The Problem:
------------
How do we measure if vectors are "perpendicular" (independent geometrically)?

Why this matters:
-----------------
Orthogonal vectors = Zero correlation = True independence!
In ML: Orthogonal features have no redundant information!

Definition: Vectors u and v are ORTHOGONAL if u · v = 0
"""

def dot_product(v1, v2):
    """
    Calculate dot product of two vectors.
    
    Formula: v1 · v2 = v1[0]×v2[0] + v1[1]×v2[1] + ...
    
    Interpretation:
    ---------------
    - Positive: Vectors point in similar direction
    - Zero: Vectors are perpendicular (orthogonal!)
    - Negative: Vectors point in opposite directions
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have same length")
    
    return sum(v1[i] * v2[i] for i in range(len(v1)))


def norm(vector):
    """Calculate Euclidean norm (length) of vector."""
    return sum(x**2 for x in vector)**0.5


def are_orthogonal(v1, v2, tolerance=1e-10):
    """
    Check if two vectors are orthogonal (perpendicular).
    
    Test: v1 · v2 = 0?
    """
    dot = dot_product(v1, v2)
    return abs(dot) < tolerance


def angle_between(v1, v2):
    """
    Calculate angle between two vectors (in degrees).
    
    Formula: cos(θ) = (v1 · v2) / (||v1|| × ||v2||)
    """
    import math
    
    dot = dot_product(v1, v2)
    norm1 = norm(v1)
    norm2 = norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return None
    
    cos_theta = dot / (norm1 * norm2)
    
    # Clamp to [-1, 1] to avoid numerical errors
    cos_theta = max(-1.0, min(1.0, cos_theta))
    
    theta_rad = math.acos(cos_theta)
    theta_deg = math.degrees(theta_rad)
    
    return theta_deg


# =============================================================================
# TEST ORTHOGONALITY
# =============================================================================

print("\n" + "=" * 80)
print("PART D: ORTHOGONALITY")
print("=" * 80)
print()

print("📚 What is Orthogonality?")
print("-" * 80)
print("""
Vectors are ORTHOGONAL if they meet at a right angle (90°).

Mathematical test: u · v = 0

Why this matters in ML:
- Orthogonal features = zero correlation
- Each feature captures independent information
- Better for many ML algorithms
- Easier to interpret

Think of it as:
- Perpendicular directions
- No overlap in information
- Perfect independence
""")
print()

# Example 1: Standard basis (orthogonal)
print("Example 1: Standard Basis (Orthogonal)")
print("-" * 80)

e1 = [1, 0, 0]
e2 = [0, 1, 0]
e3 = [0, 0, 1]

print(f"e₁ = {e1}")
print(f"e₂ = {e2}")
print(f"e₃ = {e3}")
print()

print(f"e₁ · e₂ = {dot_product(e1, e2)} → Orthogonal: {are_orthogonal(e1, e2)}")
print(f"e₁ · e₃ = {dot_product(e1, e3)} → Orthogonal: {are_orthogonal(e1, e3)}")
print(f"e₂ · e₃ = {dot_product(e2, e3)} → Orthogonal: {are_orthogonal(e2, e3)}")
print()
print("💡 Interpretation: Standard basis vectors are mutually orthogonal!")
print("   They point in completely different directions (x, y, z)")
print()

# Example 2: Non-orthogonal vectors
print("Example 2: Non-Orthogonal Vectors")
print("-" * 80)

v1 = [1, 0]
v2 = [1, 1]

print(f"v₁ = {v1}")
print(f"v₂ = {v2}")
print()

dot = dot_product(v1, v2)
angle = angle_between(v1, v2)

print(f"v₁ · v₂ = {dot} → Orthogonal: {are_orthogonal(v1, v2)}")
print(f"Angle between them: {angle:.2f}°")
print()
print("💡 Interpretation:")
print("   Dot product = 1 (not zero) → Not orthogonal")
print(f"   Angle = {angle:.2f}° (not 90°)")
print("   They share some directional overlap")
print()

# Example 3: Correlated features
print("Example 3: Feature Correlation")
print("-" * 80)

# Simulated data: Height and Weight (correlated)
height = [160, 170, 180, 190, 200]  # cm
weight = [55, 65, 75, 85, 95]  # kg

# Mean-center the features
height_mean = sum(height) / len(height)
weight_mean = sum(weight) / len(weight)

height_centered = [h - height_mean for h in height]
weight_centered = [w - weight_mean for w in weight]

print("Features: Height and Weight (mean-centered)")
print(f"Height: {[f'{h:.1f}' for h in height_centered]}")
print(f"Weight: {[f'{w:.1f}' for w in weight_centered]}")
print()

dot = dot_product(height_centered, weight_centered)
print(f"Dot product: {dot:.2f}")
print(f"Orthogonal: {are_orthogonal(height_centered, weight_centered)}")
print()
print("💡 Interpretation:")
print("   Large positive dot product → Highly correlated!")
print("   Height and weight move together (not independent)")
print("   Would benefit from decorrelation (e.g., PCA)")
print()


"""
================================================================================
PART E: GRAM-SCHMIDT PROCESS
================================================================================

The Problem:
------------
We have independent vectors, but they're NOT orthogonal (correlated).
How do we make them orthogonal while keeping the same span?

Why this matters:
-----------------
Orthogonal basis = Easier computations + Better numerics!
In ML: QR decomposition, orthogonal initialization, PCA

Solution: Gram-Schmidt Process!
1. Keep first vector as is
2. Remove its "component" from second vector
3. Remove both components from third vector
4. Continue...
"""

def gram_schmidt(vectors):
    """
    Apply Gram-Schmidt process to create orthogonal vectors.
    
    Algorithm:
    ----------
    u₁ = v₁
    u₂ = v₂ - proj_u₁(v₂)
    u₃ = v₃ - proj_u₁(v₃) - proj_u₂(v₃)
    ...
    
    Where proj_u(v) = (v·u / u·u) × u
    
    Parameters:
    -----------
    vectors : list of lists
        Input vectors (should be linearly independent)
        
    Returns:
    --------
    list of lists
        Orthogonal vectors spanning the same space
    """
    if not vectors:
        return []
    
    orthogonal = []
    
    for i, v in enumerate(vectors):
        # Start with the original vector
        u = v[:]
        
        # Subtract projections onto all previous orthogonal vectors
        for j in range(i):
            u_j = orthogonal[j]
            # Calculate projection: proj_uj(v) = (v·uj / uj·uj) × uj
            dot_v_uj = dot_product(v, u_j)
            dot_uj_uj = dot_product(u_j, u_j)
            
            if dot_uj_uj != 0:
                projection = [dot_v_uj / dot_uj_uj * u_j[k] 
                             for k in range(len(u_j))]
                
                # Subtract projection
                u = [u[k] - projection[k] for k in range(len(u))]
        
        # Check if result is non-zero (vectors might be dependent)
        if norm(u) > 1e-10:
            orthogonal.append(u)
    
    return orthogonal


def normalize(vector):
    """
    Normalize a vector to unit length.
    
    Formula: v_normalized = v / ||v||
    """
    n = norm(vector)
    if n == 0:
        return vector
    return [x / n for x in vector]


def gram_schmidt_orthonormal(vectors):
    """
    Apply Gram-Schmidt and normalize to create orthonormal basis.
    
    Orthonormal = Orthogonal + Unit length
    """
    orthogonal = gram_schmidt(vectors)
    orthonormal = [normalize(v) for v in orthogonal]
    return orthonormal


# =============================================================================
# TEST GRAM-SCHMIDT PROCESS
# =============================================================================

print("\n" + "=" * 80)
print("PART E: GRAM-SCHMIDT PROCESS")
print("=" * 80)
print()

print("📚 What is Gram-Schmidt?")
print("-" * 80)
print("""
Gram-Schmidt transforms ANY independent set into an ORTHOGONAL set!

The Process:
1. Keep first vector
2. Make second orthogonal to first (remove overlap)
3. Make third orthogonal to both previous (remove all overlaps)
4. Continue...

Why it works:
- We systematically remove the "projection" onto previous vectors
- What remains is the perpendicular component
- Result: Same span, but orthogonal!
""")
print()

# Example 1: Simple 2D case
print("Example 1: Orthogonalizing 2D Vectors")
print("-" * 80)

v1 = [3, 1]
v2 = [2, 2]

print("Original vectors (NOT orthogonal):")
print(f"v₁ = {v1}")
print(f"v₂ = {v2}")
print(f"v₁ · v₂ = {dot_product(v1, v2)} (not zero!)")
print()

orthogonal = gram_schmidt([v1, v2])

print("After Gram-Schmidt (orthogonal):")
print(f"u₁ = {[f'{x:.3f}' for x in orthogonal[0]]}")
print(f"u₂ = {[f'{x:.3f}' for x in orthogonal[1]]}")
print(f"u₁ · u₂ = {dot_product(orthogonal[0], orthogonal[1]):.10f} ✓")
print()

# Create orthonormal version
orthonormal = gram_schmidt_orthonormal([v1, v2])

print("Orthonormal version (orthogonal + unit length):")
print(f"q₁ = {[f'{x:.3f}' for x in orthonormal[0]]}, ||q₁|| = {norm(orthonormal[0]):.3f}")
print(f"q₂ = {[f'{x:.3f}' for x in orthonormal[1]]}, ||q₂|| = {norm(orthonormal[1]):.3f}")
print()
print("💡 Interpretation:")
print("   Original vectors were independent but not perpendicular")
print("   Gram-Schmidt made them perpendicular while keeping same span!")
print("   Orthonormal version also has unit length (perfect for computations)")
print()

# Example 2: 3D case
print("Example 2: Orthogonalizing 3D Vectors")
print("-" * 80)

v1 = [1, 1, 0]
v2 = [1, 0, 1]
v3 = [0, 1, 1]

print("Original vectors:")
print(f"v₁ = {v1}")
print(f"v₂ = {v2}")
print(f"v₃ = {v3}")
print()

print("Pairwise dot products (before):")
print(f"v₁ · v₂ = {dot_product(v1, v2)}")
print(f"v₁ · v₃ = {dot_product(v1, v3)}")
print(f"v₂ · v₃ = {dot_product(v2, v3)}")
print("(Not all zero → not orthogonal)")
print()

orthonormal = gram_schmidt_orthonormal([v1, v2, v3])

print("After Gram-Schmidt (orthonormal):")
for i, q in enumerate(orthonormal):
    print(f"q{i+1} = {[f'{x:.3f}' for x in q]}, ||q{i+1}|| = {norm(q):.3f}")
print()

print("Pairwise dot products (after):")
print(f"q₁ · q₂ = {dot_product(orthonormal[0], orthonormal[1]):.10f} ✓")
print(f"q₁ · q₃ = {dot_product(orthonormal[0], orthonormal[2]):.10f} ✓")
print(f"q₂ · q₃ = {dot_product(orthonormal[1], orthonormal[2]):.10f} ✓")
print()
print("💡 Perfect orthonormal basis created!")
print()

# Example 3: Feature decorrelation
print("Example 3: Decorrelating Features for ML")
print("-" * 80)

print("Original features (correlated):")
# Feature 1: Overall size (sum of measurements)
# Feature 2: Aspect ratio-ish
# Both from same 5 data points
f1 = [10, 20, 30, 40, 50]
f2 = [12, 22, 32, 42, 52]

print(f"Feature 1: {f1}")
print(f"Feature 2: {f2}")
print(f"Correlation (dot product): {dot_product(f1, f2)}")
print()

orthogonal_features = gram_schmidt([f1, f2])

print("After Gram-Schmidt (decorrelated):")
print(f"New Feature 1: {[f'{x:.2f}' for x in orthogonal_features[0]]}")
print(f"New Feature 2: {[f'{x:.2f}' for x in orthogonal_features[1]]}")
print(f"Correlation: {dot_product(orthogonal_features[0], orthogonal_features[1]):.10f} ✓")
print()
print("💡 Interpretation:")
print("   New features are uncorrelated!")
print("   Each captures independent information")
print("   Better for regression and other ML algorithms")
print()


"""
================================================================================
PART F: NUMPY IMPLEMENTATIONS
================================================================================

Why NumPy?
----------
Our implementations are educational, but NumPy is:
- Much faster (C/Fortran backend)
- More numerically stable
- Industry standard
- Feature-rich
"""

import numpy as np

print("\n" + "=" * 80)
print("PART F: NUMPY IMPLEMENTATIONS")
print("=" * 80)
print()

# Linear Independence with NumPy
print("1. Linear Independence (NumPy)")
print("-" * 80)

vectors_np = np.array([[1, 0], [0, 1], [1, 1]]).T  # Transpose to get vectors as columns
rank_np = np.linalg.matrix_rank(vectors_np)

print(f"Vectors as columns:\n{vectors_np}")
print(f"Rank (NumPy): {rank_np}")
print(f"Number of vectors: {vectors_np.shape[1]}")
print(f"Independent: {rank_np == vectors_np.shape[1]}")
print()

# Gram-Schmidt with NumPy (QR decomposition)
print("2. Gram-Schmidt / QR Decomposition (NumPy)")
print("-" * 80)

A = np.array([[3, 2], [1, 2]], dtype=float).T  # Vectors as columns
print(f"Original vectors (as columns):\n{A}")
print()

Q, R = np.linalg.qr(A)
print("QR Decomposition:")
print(f"Q (orthonormal):\n{Q}")
print(f"R (upper triangular):\n{R}")
print()

# Verify orthonormality
print("Verification:")
print(f"Q^T Q (should be identity):\n{Q.T @ Q}")
print(f"Q @ R (should equal original A):\n{Q @ R}")
print()

# Dot product with NumPy
print("3. Dot Product (NumPy)")
print("-" * 80)

v1_np = np.array([1, 2, 3])
v2_np = np.array([4, 5, 6])

dot_np = np.dot(v1_np, v2_np)
print(f"v₁ = {v1_np}")
print(f"v₂ = {v2_np}")
print(f"v₁ · v₂ = {dot_np}")
print()


"""
================================================================================
PART G: APPLICATIONS
================================================================================
"""

print("\n" + "=" * 80)
print("PART G: APPLICATIONS TO MACHINE LEARNING")
print("=" * 80)
print()

# =============================================================================
# APPLICATION 1: PCA (PRINCIPAL COMPONENT ANALYSIS)
# =============================================================================

print("APPLICATION 1: Principal Component Analysis (PCA)")
print("-" * 80)
print()

print("""
Problem:
--------
High-dimensional data with correlated features.
Want to reduce dimensions while keeping most information.

Solution:
---------
1. Center the data (subtract mean)
2. Compute covariance matrix
3. Find eigenvectors (principal components)
4. These form an orthogonal basis!
5. Project data onto top k components
""")

# Create sample data (2D for visualization)
np.random.seed(42)

# Data with correlation
n_samples = 100
x1 = np.random.randn(n_samples)
x2 = 0.8 * x1 + 0.3 * np.random.randn(n_samples)  # Correlated with x1

data = np.column_stack([x1, x2])

print(f"Generated {n_samples} data points with 2 correlated features")
print(f"Sample data (first 5 points):")
print(data[:5])
print()

# Step 1: Center the data
mean = np.mean(data, axis=0)
data_centered = data - mean

print("Step 1: Data centered (mean subtracted)")
print(f"Original mean: {mean}")
print(f"Centered mean: {np.mean(data_centered, axis=0)}")
print()

# Step 2: Compute covariance matrix
cov_matrix = np.cov(data_centered.T)

print("Step 2: Covariance Matrix")
print(cov_matrix)
print()

# Step 3: Compute eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Step 3: Eigenvectors (Principal Components)")
print(f"PC1: {eigenvectors[:, 0]}")
print(f"PC2: {eigenvectors[:, 1]}")
print()
print(f"Eigenvalues (variance explained):")
print(f"PC1: {eigenvalues[0]:.3f} ({eigenvalues[0]/sum(eigenvalues)*100:.1f}%)")
print(f"PC2: {eigenvalues[1]:.3f} ({eigenvalues[1]/sum(eigenvalues)*100:.1f}%)")
print()

# Verify orthogonality
dot_pcs = np.dot(eigenvectors[:, 0], eigenvectors[:, 1])
print(f"PC1 · PC2 = {dot_pcs:.10f} (orthogonal! ✓)")
print()

# Step 4: Project onto first PC only (dimension reduction!)
pc1 = eigenvectors[:, 0:1]  # Keep as column vector
data_1d = data_centered @ pc1

print("Step 4: Reduced to 1D")
print(f"Original shape: {data_centered.shape}")
print(f"Reduced shape: {data_1d.shape}")
print(f"Kept {eigenvalues[0]/sum(eigenvalues)*100:.1f}% of variance")
print()

print("💡 PCA Application:")
print("   - Started with 2 correlated features")
print("   - Found 2 orthogonal principal components")
print("   - Can reduce to 1D and keep ~90% of information!")
print("   - PC1 captures the main direction of variation")
print()


# =============================================================================
# APPLICATION 2: LINEAR REGRESSION (ORTHOGONAL PROJECTION)
# =============================================================================

print("\n" + "=" * 80)
print("APPLICATION 2: Linear Regression as Projection")
print("=" * 80)
print()

print("""
Problem:
--------
Find best fit: y = Xw
Where y might not be in column space of X!

Solution:
---------
The best w makes Xw = orthogonal projection of y onto column space!

Key insight: Residuals (y - Xw) are ORTHOGONAL to column space!
This gives us: X^T(y - Xw) = 0
Therefore: w = (X^T X)^(-1) X^T y
""")

# Create simple regression problem
np.random.seed(42)
X = np.column_stack([np.ones(10), np.arange(10)])  # [1, x] for intercept + slope
y_true = 2 + 3 * X[:, 1]  # True: y = 2 + 3x
y = y_true + np.random.randn(10) * 0.5  # Add noise

print("Regression Problem: y = w₀ + w₁×x")
print(f"Data points: {len(y)}")
print(f"Features: {X.shape[1]} (intercept + x)")
print()

# Solve normal equation
w = np.linalg.solve(X.T @ X, X.T @ y)

print(f"Solution: y = {w[0]:.2f} + {w[1]:.2f}×x")
print(f"True values: y = 2.00 + 3.00×x")
print()

# Calculate predictions and residuals
y_pred = X @ w
residuals = y - y_pred

print("Verification: Residuals orthogonal to column space")
print(f"X^T × residuals = {X.T @ residuals}")
print("(Should be close to zero → orthogonal! ✓)")
print()

print("💡 Regression Application:")
print("   - Best fit is orthogonal projection of y onto column space of X")
print("   - Residuals perpendicular to feature space")
print("   - This minimizes squared error!")
print()


# =============================================================================
# APPLICATION 3: FEATURE ORTHOGONALIZATION
# =============================================================================

print("\n" + "=" * 80)
print("APPLICATION 3: Feature Orthogonalization")
print("=" * 80)
print()

print("""
Problem:
--------
Multicollinearity: Features are highly correlated
→ Unstable regression coefficients
→ Hard to interpret which feature matters

Solution:
---------
Apply Gram-Schmidt to create orthogonal features!
""")

# Create correlated features
np.random.seed(42)
n_samples = 50

# Original features
age = 20 + 20 * np.random.rand(n_samples)
experience = 0.8 * age + 2 * np.random.randn(n_samples)  # Highly correlated
education = 12 + 8 * np.random.rand(n_samples)  # Independent

X_original = np.column_stack([age, experience, education])

print("Original Features:")
print(f"Feature 1: Age")
print(f"Feature 2: Experience (correlated with age)")
print(f"Feature 3: Education (independent)")
print()

# Check correlation
print("Correlation (dot products of centered features):")
X_centered = X_original - np.mean(X_original, axis=0)
for i in range(3):
    for j in range(i+1, 3):
        corr = np.dot(X_centered[:, i], X_centered[:, j])
        print(f"Feature {i+1} · Feature {j+1}: {corr:.2f}")
print()

# Orthogonalize using QR decomposition
Q, R = np.linalg.qr(X_centered)

print("After Orthogonalization (QR):")
print("New features are orthogonal:")
for i in range(3):
    for j in range(i+1, 3):
        corr = np.dot(Q[:, i], Q[:, j])
        print(f"Feature {i+1} · Feature {j+1}: {corr:.10f} ✓")
print()

print("💡 Feature Orthogonalization:")
print("   - Removed correlation between age and experience")
print("   - Each new feature captures independent information")
print("   - Better for regression (stable coefficients)")
print("   - Easier to interpret (no confounding)")
print()


"""
================================================================================
PART H: VISUALIZATIONS
================================================================================
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("\n" + "=" * 80)
print("PART H: VISUALIZATIONS")
print("=" * 80)
print()

# Visualization 1: Gram-Schmidt Process
print("Creating Visualization 1: Gram-Schmidt Process...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before Gram-Schmidt
v1 = np.array([3, 1])
v2 = np.array([2, 2])

ax = axes[0]
ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
          color='blue', width=0.008, label='v₁')
ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, 
          color='red', width=0.008, label='v₂')

# Show projection
proj_v2_on_v1 = (np.dot(v2, v1) / np.dot(v1, v1)) * v1
ax.quiver(0, 0, proj_v2_on_v1[0], proj_v2_on_v1[1], 
          angles='xy', scale_units='xy', scale=1, 
          color='green', width=0.006, linestyle='--', label='proj(v₂ on v₁)')

ax.set_xlim(-0.5, 4)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Before Gram-Schmidt\n(Not Orthogonal)')
ax.set_xlabel('x')
ax.set_ylabel('y')

# After Gram-Schmidt
u1 = v1
u2 = v2 - proj_v2_on_v1

ax = axes[1]
ax.quiver(0, 0, u1[0], u1[1], angles='xy', scale_units='xy', scale=1, 
          color='blue', width=0.008, label='u₁ = v₁')
ax.quiver(0, 0, u2[0], u2[1], angles='xy', scale_units='xy', scale=1, 
          color='red', width=0.008, label='u₂ (orthogonal)')

# Draw right angle indicator
from matplotlib.patches import Rectangle
angle_size = 0.3
angle_square = Rectangle((0, 0), angle_size, angle_size, 
                          angle=np.degrees(np.arctan2(u1[1], u1[0])),
                          fill=False, edgecolor='green', linewidth=1.5)
ax.add_patch(angle_square)

ax.set_xlim(-0.5, 4)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('After Gram-Schmidt\n(Orthogonal! ✓)')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.tight_layout()
plt.show()
print("✓ Visualization 1 complete")
print()

# Visualization 2: Linear Independence
print("Creating Visualization 2: Linear Independence...")

fig = plt.figure(figsize=(12, 5))

# Subplot 1: Independent vectors
ax1 = fig.add_subplot(121)

v1 = np.array([1, 0])
v2 = np.array([0, 1])

ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.01, label='v₁')
ax1.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, 
           color='red', width=0.01, label='v₂')

# Show span (entire 2D plane)
x = np.linspace(-1.5, 1.5, 10)
y = np.linspace(-1.5, 1.5, 10)
X, Y = np.meshgrid(x, y)
ax1.contourf(X, Y, X*0, alpha=0.1, colors=['lightblue'])
ax1.text(0.7, 0.7, 'Span = all of ℝ²', fontsize=10, style='italic')

ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_title('Independent Vectors\n(Different directions)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Subplot 2: Dependent vectors
ax2 = fig.add_subplot(122)

v1 = np.array([1, 0.5])
v2 = np.array([2, 1])  # v2 = 2×v1

ax2.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.01, label='v₁')
ax2.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, 
           color='red', width=0.01, label='v₂ = 2v₁')

# Show span (just a line)
t = np.linspace(-2, 2, 100)
line_x = t * v1[0]
line_y = t * v1[1]
ax2.plot(line_x, line_y, 'g--', alpha=0.5, linewidth=2, label='Span (line)')
ax2.text(0.5, 1, 'Span = 1D line\n(not all of ℝ²)', fontsize=10, style='italic')

ax2.set_xlim(-1.5, 2.5)
ax2.set_ylim(-1, 1.5)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_title('Dependent Vectors\n(Same direction)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

plt.tight_layout()
plt.show()
print("✓ Visualization 2 complete")
print()

# Visualization 3: PCA
print("Creating Visualization 3: PCA Direction...")

fig, ax = plt.subplots(figsize=(8, 8))

# Plot correlated data
ax.scatter(data[:, 0], data[:, 1], alpha=0.6, s=30, label='Data points')

# Plot principal components
scale = 3
pc1_vec = eigenvectors[:, 0] * scale * np.sqrt(eigenvalues[0])
pc2_vec = eigenvectors[:, 1] * scale * np.sqrt(eigenvalues[1])

ax.quiver(mean[0], mean[1], pc1_vec[0], pc1_vec[1], 
          angles='xy', scale_units='xy', scale=1, 
          color='red', width=0.01, label=f'PC1 ({eigenvalues[0]/sum(eigenvalues)*100:.1f}% var)')
ax.quiver(mean[0], mean[1], pc2_vec[0], pc2_vec[1], 
          angles='xy', scale_units='xy', scale=1, 
          color='blue', width=0.01, label=f'PC2 ({eigenvalues[1]/sum(eigenvalues)*100:.1f}% var)')

ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('PCA: Finding Orthogonal Directions of Max Variance')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
print("✓ Visualization 3 complete")
print()


"""
================================================================================
SUMMARY AND KEY TAKEAWAYS
================================================================================
"""

print("\n" + "=" * 80)
print("SUMMARY: What We've Learned")
print("=" * 80)
print()

print("""
✅ IMPLEMENTED FROM SCRATCH:
   1. Linear Independence Test (via rank)
   2. Span Membership Test
   3. Basis Finding (remove redundant vectors)
   4. Orthogonality Test (dot product = 0)
   5. Gram-Schmidt Process (orthogonalization)
   6. Angle Between Vectors

✅ UNDERSTOOD WHY THEY MATTER:
   - Linear Independence: No redundancy, efficient representation
   - Span: What space can we reach/represent?
   - Basis: Minimal spanning set, fundamental directions
   - Dimension: True degrees of freedom
   - Orthogonality: Zero correlation, true independence
   - Gram-Schmidt: Create orthogonal basis

✅ SAW REAL ML APPLICATIONS:
   - PCA: Find orthogonal directions of max variance
   - Regression: Projection onto column space
   - Feature Orthogonalization: Remove multicollinearity
   - QR Decomposition: Numerical stability

✅ KEY INSIGHTS:
   - Most high-dimensional data has LOWER true dimension
   - Orthogonal features = uncorrelated = independent info
   - Basis provides coordinate system
   - Gram-Schmidt creates orthogonal from any independent set

🎯 CONNECTIONS TO ML:
-------------------
Linear Independence → Feature Selection
   - Remove redundant features
   - Detect multicollinearity

Span & Subspaces → Representational Capacity
   - What can our model represent?
   - Dimension of hypothesis space

Basis & Dimension → Dimensionality Reduction
   - PCA finds optimal basis
   - True dimension < apparent dimension

Orthogonality → Decorrelation
   - Independent features
   - Better numerics
   - Easier interpretation

Gram-Schmidt → QR, Initialization
   - QR decomposition for solving systems
   - Orthogonal initialization in neural networks
   - Creating orthonormal bases

⚠️  PRACTICAL TIPS:
------------------
1. Always check for linear dependence before regression
2. Use orthogonal features when possible
3. NumPy is much faster for production
4. Visualize in 2D/3D to build intuition
5. Remember: rank = true dimensionality

🚀 NEXT STEPS:
-------------
1. Apply to real datasets
2. Implement PCA from scratch
3. Compare with sklearn.decomposition.PCA
4. Explore SVD (generalization of eigendecomposition)
5. Study applications in deep learning
6. Practice on high-dimensional data

🎉 CONGRATULATIONS!
------------------
You now understand the LINEAR ALGEBRA FOUNDATIONS of ML:
- WHY features might be redundant (dependence)
- HOW to find minimal representation (basis)
- WHEN to use orthogonal features (always when possible!)
- How to CREATE orthogonal features (Gram-Schmidt)

These concepts underpin:
- PCA, SVD, Matrix Factorization
- Regression, Least Squares
- Neural Network Initialization
- And much more!

You're ready to tackle advanced ML mathematics! 🚀
""")

print("=" * 80)
print("END OF CHAPTER 5 IMPLEMENTATION")
print("=" * 80)
print()
print("💡 Remember: Linear algebra is the LANGUAGE of machine learning!")
print("   Master these foundations and everything else becomes easier.")
print()
print("=" * 80)
