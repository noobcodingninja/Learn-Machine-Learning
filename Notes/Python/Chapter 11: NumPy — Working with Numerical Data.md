# Chapter 11: NumPy — Working with Numerical Data

## Part 1: Why NumPy Exists

### The Problem: Pure Python Is Painfully Slow for Numbers

Imagine you're working at a fintech startup. You have daily stock prices for 5,000 companies over 20 years — that's **36.5 million numbers**. You need to:
- Calculate rolling 30-day averages for every stock
- Find correlations between every pair of stocks
- Run portfolio risk simulations 10,000 times

Let's see what happens with pure Python:

```python
import time

# Pure Python: multiply 1 million numbers by 2
data = list(range(1_000_000))

start = time.perf_counter()
result = [x * 2 for x in data]
python_time = time.perf_counter() - start
print(f"Python list:  {python_time:.4f}s")

# NumPy: same operation
import numpy as np
arr = np.arange(1_000_000)

start = time.perf_counter()
result = arr * 2
numpy_time = time.perf_counter() - start
print(f"NumPy array:  {numpy_time:.4f}s")
print(f"Speedup: {python_time / numpy_time:.0f}x faster")

# Typical output:
# Python list:  0.0821s
# NumPy array:  0.0008s
# Speedup: 102x faster
```

### Why Is NumPy So Much Faster?

```
Python list of floats:          NumPy array of float64:
┌───────────────────────┐       ┌─────────────────────────────┐
│ Pointer → object      │       │ Contiguous block of memory  │
│ Pointer → object      │       │ 8.0  8.0  8.0  8.0  8.0 .. │
│ Pointer → object      │       │ (packed float64 values)     │
│ Pointer → object      │       └─────────────────────────────┘
│ ...                   │
└───────────────────────┘
Each element: 28 bytes +        Each element: 8 bytes
pointer overhead                No pointer, no overhead

Operations: Python bytecode     Operations: compiled C/Fortran
            (interpreted)                  running on CPU SIMD
```

**Key reasons:**
1. **Fixed type**: Every element is the same type → no type checking per element
2. **Contiguous memory**: All values packed together → CPU cache friendly
3. **Vectorized operations**: Operations run on entire arrays at once using CPU's SIMD instructions
4. **Compiled code**: Core operations written in C/Fortran, not Python

---

## Part 2: Creating Arrays

### Installation and Import

```python
# Install (if needed)
# pip install numpy

import numpy as np   # Convention: always import as np
print(np.__version__)  # e.g., 1.24.0
```

### Ways to Create Arrays

```python
# ── FROM PYTHON LISTS ─────────────────────────────────────────────────────

# 1D array (vector)
a = np.array([1, 2, 3, 4, 5])
print(a)           # [1 2 3 4 5]
print(type(a))     # <class 'numpy.ndarray'>
print(a.dtype)     # int64  (NumPy inferred the type)

# 2D array (matrix)
m = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(m)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

# With explicit dtype
floats = np.array([1, 2, 3], dtype=np.float64)
print(floats)       # [1. 2. 3.]
print(floats.dtype) # float64

integers = np.array([1.7, 2.9, 3.1], dtype=np.int32)
print(integers)     # [1 2 3]  ← truncated, not rounded!


# ── BUILT-IN CONSTRUCTORS ─────────────────────────────────────────────────

# zeros — all zeros
print(np.zeros(5))            # [0. 0. 0. 0. 0.]
print(np.zeros((3, 4)))       # 3×4 matrix of zeros
print(np.zeros((2, 3, 4)))    # 3D array: 2 layers, 3 rows, 4 cols

# ones — all ones
print(np.ones(5))             # [1. 1. 1. 1. 1.]
print(np.ones((3, 3)))        # 3×3 matrix of ones

# full — fill with a value
print(np.full(5, 7))          # [7 7 7 7 7]
print(np.full((3, 3), 3.14))  # 3×3 matrix of π

# identity matrix
print(np.eye(4))
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]

# empty — uninitialized (whatever is in memory — FAST but garbage values)
e = np.empty((3, 3))  # Use when you'll fill it yourself immediately

# ── RANGE-LIKE CONSTRUCTORS ───────────────────────────────────────────────

# arange — like range() but returns array
print(np.arange(10))          # [0 1 2 3 4 5 6 7 8 9]
print(np.arange(2, 10, 2))    # [2 4 6 8]
print(np.arange(0, 1, 0.1))   # [0.  0.1 0.2 ... 0.9]

# linspace — evenly spaced, specify COUNT not step
print(np.linspace(0, 1, 5))   # [0.   0.25 0.5  0.75 1.  ]
print(np.linspace(0, 100, 11))# [0. 10. 20. 30. ... 100.]
# Useful for: plotting, sampling, creating test data

# logspace — evenly spaced on log scale
print(np.logspace(0, 3, 4))   # [1. 10. 100. 1000.]


# ── RANDOM ARRAYS ─────────────────────────────────────────────────────────

rng = np.random.default_rng(seed=42)  # Reproducible randomness

# Uniform [0, 1)
print(rng.random(5))           # [0.773 0.438 0.859 0.697 0.094]

# Uniform [low, high)
print(rng.uniform(10, 20, 5))  # [17.8 14.3 18.9 16.9 10.9]

# Normal (Gaussian) distribution
print(rng.normal(mean=0, scale=1, size=5))  # [-0.4  0.6  1.2 ...]

# Integers
print(rng.integers(1, 7, size=10))  # Dice rolls: [3 1 5 2 4 6 ...]

# Choice from array
arr = np.array([10, 20, 30, 40, 50])
print(rng.choice(arr, size=3, replace=False))  # [30, 10, 50]

# Shuffle (in-place)
arr = np.arange(10)
rng.shuffle(arr)
print(arr)  # [7 3 1 9 0 5 2 8 4 6]
```

---

## Part 3: Array Properties — Know Your Array

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# Core properties — check these first when debugging!
print(arr.shape)    # (3, 4)   — (rows, cols) or (dim1, dim2, dim3, ...)
print(arr.ndim)     # 2        — number of dimensions
print(arr.size)     # 12       — total number of elements
print(arr.dtype)    # int64    — data type of elements
print(arr.nbytes)   # 96       — total bytes in memory (12 × 8 bytes)

# Shape vocabulary:
# 1D: (5,)       — vector, 5 elements
# 2D: (3, 4)     — matrix, 3 rows, 4 columns
# 3D: (2, 3, 4)  — 2 "layers" of 3×4 matrices
# nD: (a, b, c, d) — a × b × c × d tensor

# Real-world shapes:
# (100,)           — 100 prices
# (1000, 20)       — 1000 samples, 20 features
# (64, 28, 28)     — 64 images, 28×28 pixels (grayscale)
# (64, 28, 28, 3)  — 64 images, 28×28 pixels, 3 color channels (RGB)
```

---

## Part 4: Indexing and Slicing

### 1D Arrays — Just Like Lists

```python
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
#               0   1   2   3   4   5   6   7   8
#              -9  -8  -7  -6  -5  -4  -3  -2  -1

print(arr[0])     # 10  — first element
print(arr[-1])    # 90  — last element
print(arr[2:5])   # [30 40 50]  — slice
print(arr[::2])   # [10 30 50 70 90]  — every other
print(arr[::-1])  # [90 80 70 60 50 40 30 20 10]  — reversed
```

### 2D Arrays — Row, Column

```python
m = np.array([[1,  2,  3,  4],
              [5,  6,  7,  8],
              [9, 10, 11, 12]])

# Single element — [row, col]
print(m[0, 0])   # 1   — top-left
print(m[1, 2])   # 7   — row 1, col 2
print(m[-1, -1]) # 12  — bottom-right

# Entire row
print(m[0])      # [1 2 3 4]    — first row
print(m[1, :])   # [5 6 7 8]    — second row (explicit)

# Entire column
print(m[:, 0])   # [1 5 9]      — first column
print(m[:, -1])  # [4 8 12]     — last column

# Submatrix (slice of rows AND columns)
print(m[0:2, 1:3])
# [[2 3]
#  [6 7]]

# ── IMPORTANT: NumPy slices are VIEWS, not copies! ─────────────────────
original = np.array([1, 2, 3, 4, 5])
view = original[1:4]   # This is a VIEW of original

view[0] = 99           # Modifying view CHANGES original!
print(original)        # [ 1 99  3  4  5]  ← Changed!

# To get a copy, use .copy()
copy = original[1:4].copy()
copy[0] = 0
print(original)        # [ 1 99  3  4  5]  ← Unchanged

# Why views? Memory efficiency — no data is copied!
```

### Fancy Indexing — Power Feature

```python
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80])

# ── INTEGER ARRAY INDEXING ────────────────────────────────────────────────
# Select multiple arbitrary elements
indices = [0, 2, 5, 7]
print(arr[indices])   # [10 30 60 80]

# Reorder or repeat elements
print(arr[[2, 2, 0, 5]])  # [30 30 10 60]

# ── BOOLEAN INDEXING (most powerful!) ────────────────────────────────────
# Create a boolean mask
mask = arr > 40
print(mask)          # [False False False False  True  True  True  True]

# Use mask to select elements
print(arr[mask])     # [50 60 70 80]

# Or inline:
print(arr[arr > 40])      # [50 60 70 80]
print(arr[arr % 20 == 0]) # [20 40 60 80]

# ── REAL-WORLD: Filter data by condition ─────────────────────────────────
prices = np.array([150.5, 203.2, 89.9, 445.0, 312.8, 67.3, 189.5])
tickers = np.array(["AAPL", "GOOGL", "AMZN", "MSFT", "META", "TWTR", "NFLX"])

# Stocks over $200
expensive_mask = prices > 200
print(tickers[expensive_mask])   # ['GOOGL' 'MSFT' 'META']
print(prices[expensive_mask])    # [203.2 445.  312.8]

# Combined conditions (use & | ~, NOT and or not!)
mid_range = (prices > 100) & (prices < 300)
print(tickers[mid_range])   # ['AAPL' 'GOOGL' 'META' 'NFLX']

# Invert a mask
not_expensive = ~(prices > 300)
print(tickers[not_expensive])

# np.where — like a vectorized ternary
categories = np.where(prices > 200, "expensive", "affordable")
print(categories)
# ['affordable' 'expensive' 'affordable' 'expensive' 'expensive' 'affordable' 'affordable']

# Three-arg np.where is super useful:
# np.where(condition, value_if_true, value_if_false)
safe_log = np.where(prices > 0, np.log(prices), 0)  # Log of positive only
```

---

## Part 5: Reshaping Arrays

```python
arr = np.arange(12)
print(arr)         # [ 0  1  2  3  4  5  6  7  8  9 10 11]
print(arr.shape)   # (12,)

# reshape — change shape WITHOUT changing data
m = arr.reshape(3, 4)   # 3 rows, 4 columns
print(m)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# -1 means "figure it out"
print(arr.reshape(4, -1))  # -1 → NumPy calculates: 12/4 = 3 cols
print(arr.reshape(-1, 3))  # -1 → 12/3 = 4 rows

# Common reshaping patterns
# Add a dimension (for broadcasting/model input)
v = np.array([1, 2, 3])       # shape (3,)
col = v.reshape(-1, 1)         # shape (3, 1) — column vector
row = v.reshape(1, -1)         # shape (1, 3) — row vector

# Or use newaxis (same result, more readable)
col = v[:, np.newaxis]         # shape (3, 1)
row = v[np.newaxis, :]         # shape (1, 3)

# flatten vs ravel
m = np.array([[1, 2], [3, 4]])
print(m.flatten())   # [1 2 3 4] — always returns a COPY
print(m.ravel())     # [1 2 3 4] — returns a VIEW if possible (faster)

# transpose — flip axes
print(m.T)   # Transpose (rows become columns)
# or np.transpose(m)

# Stack arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.vstack([a, b]))   # Stack vertically (row-wise)
# [[1 2 3]
#  [4 5 6]]

print(np.hstack([a, b]))   # Stack horizontally
# [1 2 3 4 5 6]

# Stack along new axis
print(np.stack([a, b], axis=0))    # [[1 2 3], [4 5 6]]  — same as vstack
print(np.stack([a, b], axis=1))    # [[1 4], [2 5], [3 6]]  — column-wise

# Concatenate along existing axis
c = np.array([[1, 2], [3, 4]])
d = np.array([[5, 6], [7, 8]])
print(np.concatenate([c, d], axis=0))  # Stack rows (4×2)
print(np.concatenate([c, d], axis=1))  # Stack cols (2×4)
```

---

## Part 6: Vectorized Operations

### The Core Idea — No More Loops

```python
# ── ELEMENT-WISE ARITHMETIC ───────────────────────────────────────────────
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

print(a + b)    # [11 22 33 44 55]  — element-wise add
print(a - b)    # [ -9 -18 -27 -36 -45]
print(a * b)    # [ 10  40  90 160 250]
print(a / b)    # [0.1 0.1 0.1 0.1 0.1]
print(a ** 2)   # [ 1  4  9 16 25]
print(b % 3)    # [ 1  2  0  1  2]

# Operations with scalars (broadcast to every element)
print(a + 10)   # [11 12 13 14 15]
print(a * 2)    # [ 2  4  6  8 10]
print(a > 3)    # [False False False  True  True]

# ── UNIVERSAL FUNCTIONS (ufuncs) ──────────────────────────────────────────
# Math functions applied element-wise — these are FAST (C-compiled)
x = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(np.sin(x))     # [0.    0.5   0.707 0.866 1.   ]
print(np.cos(x))     # [1.    0.866 0.707 0.5   0.   ]
print(np.exp(x))     # e^x
print(np.log(x + 1)) # log(x+1) — avoids log(0)
print(np.sqrt(a))    # [1.    1.414 1.732 2.    2.236]
print(np.abs(np.array([-3, -1, 0, 2, -5])))  # [3 1 0 2 5]

# ── COMPARISON — returns boolean arrays ───────────────────────────────────
print(a == b)   # [False False False False False]
print(a < b)    # [ True  True  True  True  True]

# ── REAL-WORLD: normalize prices between 0 and 1 ─────────────────────────
prices = np.array([150.5, 203.2, 89.9, 445.0, 312.8])

# Without NumPy — manual loop
min_p, max_p = min(prices), max(prices)
normalized_slow = [(p - min_p) / (max_p - min_p) for p in prices]

# With NumPy — vectorized
normalized = (prices - prices.min()) / (prices.max() - prices.min())
print(normalized)  # [0.169 0.318 0.    1.    0.624]

# ── Z-score normalization ──────────────────────────────────────────────────
z_scores = (prices - prices.mean()) / prices.std()
print(z_scores)  # [-0.67 -0.17 -1.13  1.65  0.32]
```

### Broadcasting — NumPy's Superpower

```python
# Broadcasting: operations between arrays of DIFFERENT shapes
# NumPy "stretches" the smaller array to match the larger one

# ── RULE: Two dimensions are compatible if:
# 1. They are equal, OR
# 2. One of them is 1

# ── EXAMPLE 1: scalar + array ────────────────────────────────────────────
a = np.array([1, 2, 3, 4])
result = a + 10    # 10 is broadcast to [10, 10, 10, 10]
print(result)      # [11 12 13 14]

# ── EXAMPLE 2: (3,1) + (1,4) → (3,4) ─────────────────────────────────────
col = np.array([[1], [2], [3]])   # shape (3, 1)
row = np.array([[10, 20, 30, 40]])  # shape (1, 4)

result = col + row   # NumPy broadcasts both to (3, 4)
print(result)
# [[11 21 31 41]   ← 1 + [10, 20, 30, 40]
#  [12 22 32 42]   ← 2 + [10, 20, 30, 40]
#  [13 23 33 43]]  ← 3 + [10, 20, 30, 40]

# ── EXAMPLE 3: distance matrix ───────────────────────────────────────────
# Calculate pairwise distances between points
points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # shape (4, 2)

# Reshape for broadcasting
# points[:, np.newaxis] → shape (4, 1, 2)
# points[np.newaxis, :] → shape (1, 4, 2)
# Difference: shape (4, 4, 2)
diff = points[:, np.newaxis] - points[np.newaxis, :]
distances = np.sqrt((diff ** 2).sum(axis=2))
print(distances)
# [[0.    1.    1.    1.414]
#  [1.    0.    1.414 1.   ]
#  [1.    1.414 0.    1.   ]
#  [1.414 1.    1.    0.   ]]

# ── EXAMPLE 4: mean-center columns of a dataset ──────────────────────────
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])  # shape (3, 3)

col_means = data.mean(axis=0)   # shape (3,) — mean of each column
print(col_means)  # [4. 5. 6.]

centered = data - col_means     # (3,3) - (3,) broadcasts row-wise
print(centered)
# [[-3. -3. -3.]
#  [ 0.  0.  0.]
#  [ 3.  3.  3.]]

# ── BROADCASTING RULES ────────────────────────────────────────────────────
# Arrays are compared dimension by dimension, RIGHT to LEFT
# (3, 4) + (4,)    → (4,) becomes (1,4) → (3,4) ✓
# (3, 4) + (3,)    → Error! (3,) becomes (1,3) → (3,3) ≠ (3,4)
# (3, 4) + (3, 1)  → (3,1) becomes (3,4) ✓
# (3, 4) + (1, 4)  → (1,4) becomes (3,4) ✓
```

---

## Part 7: Aggregation Functions

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# ── WITHOUT axis — operates on ENTIRE array ───────────────────────────────
print(arr.sum())     # 78   — sum of everything
print(arr.min())     # 1    — global minimum
print(arr.max())     # 12   — global maximum
print(arr.mean())    # 6.5  — overall mean
print(arr.std())     # 3.45 — overall std deviation
print(arr.var())     # 11.9 — overall variance

# ── WITH axis=0 — collapse ROWS (operate down columns) ────────────────────
print(arr.sum(axis=0))    # [15 18 21 24]  — column sums
print(arr.mean(axis=0))   # [5.  6.  7.  8.] — column means
print(arr.max(axis=0))    # [9 10 11 12] — column maxima

# ── WITH axis=1 — collapse COLUMNS (operate across rows) ──────────────────
print(arr.sum(axis=1))    # [10 26 42]  — row sums
print(arr.mean(axis=1))   # [2.5  6.5 10.5] — row means
print(arr.max(axis=1))    # [ 4  8 12] — row maxima

# Memory trick: axis=0 → result loses dim 0 (rows collapse)
#               axis=1 → result loses dim 1 (cols collapse)

# ── OTHER USEFUL AGGREGATIONS ─────────────────────────────────────────────
print(arr.cumsum())       # Running total: [1 3 6 10 15 21 28 36 45 55 66 78]
print(arr.cumprod(axis=1))  # Running product along rows
print(arr.prod())         # Product of all elements

# argmin / argmax — INDEX of min/max
print(arr.argmin())      # 0   — index in FLAT array
print(arr.argmax())      # 11  — index in FLAT array
print(arr.argmin(axis=0))   # [0 0 0 0] — row index of min per column
print(arr.argmax(axis=1))   # [3 3 3]   — col index of max per row

# ── BOOLEAN AGGREGATIONS ──────────────────────────────────────────────────
data = np.array([1, 5, 3, 8, 2, 9, 4, 7, 6])

print(np.any(data > 8))   # True  — is ANY value > 8?
print(np.all(data > 0))   # True  — are ALL values > 0?
print(np.count_nonzero(data > 5))  # 3  — how many > 5?
print((data > 5).sum())             # 3  — same thing

# np.where to get INDICES of matching elements
indices = np.where(data > 5)
print(indices)         # (array([1, 3, 5, 7, 8]),)  — tuple of arrays
print(data[indices])   # [5 8 9 7 6]
```

---

## Part 8: Linear Algebra

```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# ── MATRIX MULTIPLICATION ─────────────────────────────────────────────────
# NOT element-wise! This is true matrix multiplication
print(A @ B)      # Matrix multiply (Python 3.5+)
# [[19 22]
#  [43 50]]

print(np.dot(A, B))  # Same as A @ B

# Element-wise (NOT matrix multiply!)
print(A * B)
# [[ 5 12]
#  [21 32]]

# ── LINEAR ALGEBRA OPERATIONS ─────────────────────────────────────────────
# Transpose
print(A.T)        # [[1 3], [2 4]]

# Determinant
print(np.linalg.det(A))  # -2.0

# Inverse
A_inv = np.linalg.inv(A)
print(A_inv)
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# Verify: A @ A_inv ≈ Identity
print(np.round(A @ A_inv, 10))
# [[1. 0.]
#  [0. 1.]]

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)   # [-0.372  5.372]

# Solve linear system: Ax = b
A = np.array([[2, 1], [1, 3]])
b = np.array([8, 13])
x = np.linalg.solve(A, b)
print(x)          # [2.6 3.47...]
print(A @ x)      # Should be [8. 13.] (verify solution)

# Norm (magnitude)
v = np.array([3, 4])
print(np.linalg.norm(v))     # 5.0  — Euclidean distance

# SVD (Singular Value Decomposition)
U, S, Vt = np.linalg.svd(A)
```

---

## Part 9: Worked Examples

### Worked Example 1: Financial Portfolio Analysis

```python
import numpy as np

# Daily returns for 5 stocks over 252 trading days (1 year)
rng = np.random.default_rng(42)
n_stocks = 5
n_days = 252
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]

# Simulate daily returns (roughly realistic)
returns = rng.normal(loc=0.0005, scale=0.02, size=(n_days, n_stocks))
# Shape: (252, 5) — rows=days, cols=stocks

# ── 1. CUMULATIVE RETURNS ─────────────────────────────────────────────────
# $1 invested grows by compounding daily returns
cumulative = (1 + returns).cumprod(axis=0)   # Cumulative product along rows
final_values = cumulative[-1]                 # Last row = final value

print("Final portfolio values ($1 invested):")
for ticker, value in zip(tickers, final_values):
    print(f"  {ticker}: ${value:.4f}")

# ── 2. ANNUALIZED STATISTICS ──────────────────────────────────────────────
annual_return = returns.mean(axis=0) * 252   # Annualize (252 trading days)
annual_vol = returns.std(axis=0) * np.sqrt(252)  # Annualize volatility

print("\nAnnualized Statistics:")
print(f"{'Ticker':<8} {'Return':>10} {'Volatility':>12} {'Sharpe':>8}")
print("-" * 42)
for i, ticker in enumerate(tickers):
    sharpe = annual_return[i] / annual_vol[i]
    print(f"{ticker:<8} {annual_return[i]:>9.1%} {annual_vol[i]:>11.1%} {sharpe:>8.2f}")

# ── 3. CORRELATION MATRIX ─────────────────────────────────────────────────
corr_matrix = np.corrcoef(returns.T)  # Transpose: each row = one stock
print("\nCorrelation Matrix:")
print(f"{'':>6}", end="")
for t in tickers:
    print(f"{t:>8}", end="")
print()
for i, ticker in enumerate(tickers):
    print(f"{ticker:<6}", end="")
    for j in range(n_stocks):
        print(f"{corr_matrix[i, j]:>8.2f}", end="")
    print()

# ── 4. MAX DRAWDOWN ───────────────────────────────────────────────────────
# Largest peak-to-trough decline
def max_drawdown(cumulative_returns):
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = cumulative_returns / rolling_max - 1
    return drawdowns.min()

print("\nMax Drawdown:")
for i, ticker in enumerate(tickers):
    dd = max_drawdown(cumulative[:, i])
    print(f"  {ticker}: {dd:.1%}")

# ── 5. PORTFOLIO OPTIMIZATION (Equal Weight) ─────────────────────────────
weights = np.ones(n_stocks) / n_stocks   # Equal weights: [0.2, 0.2, 0.2, 0.2, 0.2]
portfolio_returns = returns @ weights    # Daily portfolio return

port_annual_return = portfolio_returns.mean() * 252
port_annual_vol = portfolio_returns.std() * np.sqrt(252)
port_sharpe = port_annual_return / port_annual_vol

print(f"\nEqual-Weight Portfolio:")
print(f"  Annual Return:     {port_annual_return:.1%}")
print(f"  Annual Volatility: {port_annual_vol:.1%}")
print(f"  Sharpe Ratio:      {port_sharpe:.2f}")
```

### Worked Example 2: Image Processing with Arrays

```python
import numpy as np

# ── IMAGES ARE JUST 3D ARRAYS ─────────────────────────────────────────────
# Shape: (height, width, channels)
# channels: 3 for RGB, 1 for grayscale

# Create a synthetic image
height, width = 100, 100
image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
# Shape: (100, 100, 3)

print(f"Image shape: {image.shape}")   # (100, 100, 3)
print(f"Dtype: {image.dtype}")          # uint8
print(f"Min pixel: {image.min()}")      # ~0
print(f"Max pixel: {image.max()}")      # ~255

# ── CHANNEL SEPARATION ────────────────────────────────────────────────────
red   = image[:, :, 0]   # All rows, all cols, channel 0
green = image[:, :, 1]   # Channel 1
blue  = image[:, :, 2]   # Channel 2

print(f"Red channel mean:   {red.mean():.1f}")
print(f"Green channel mean: {green.mean():.1f}")
print(f"Blue channel mean:  {blue.mean():.1f}")

# ── CONVERT TO GRAYSCALE ──────────────────────────────────────────────────
# Human-perceived luminance weights
grayscale = (0.2989 * red + 0.5870 * green + 0.1140 * blue).astype(np.uint8)
print(f"Grayscale shape: {grayscale.shape}")  # (100, 100)

# ── BRIGHTNESS ADJUSTMENT ─────────────────────────────────────────────────
# Increase brightness by 50, clip to valid range [0, 255]
brighter = np.clip(image.astype(np.int16) + 50, 0, 255).astype(np.uint8)

# ── CONTRAST ADJUSTMENT ───────────────────────────────────────────────────
# Stretch pixel values to fill [0, 255]
def adjust_contrast(img):
    min_val = img.min()
    max_val = img.max()
    stretched = (img.astype(float) - min_val) / (max_val - min_val) * 255
    return stretched.astype(np.uint8)

contrasted = adjust_contrast(image)

# ── CROP ──────────────────────────────────────────────────────────────────
cropped = image[25:75, 25:75]     # Center 50×50 crop
print(f"Cropped shape: {cropped.shape}")  # (50, 50, 3)

# ── FLIP ──────────────────────────────────────────────────────────────────
flipped_horizontal = image[:, ::-1]    # Reverse columns
flipped_vertical   = image[::-1, :]    # Reverse rows
rotated_180        = image[::-1, ::-1]

# ── SIMPLE CONVOLUTION (blur) ─────────────────────────────────────────────
def box_blur(channel, kernel_size=3):
    """Apply a box blur to a 2D channel."""
    h, w = channel.shape
    k = kernel_size // 2
    result = np.zeros_like(channel, dtype=float)
    
    for i in range(k, h - k):
        for j in range(k, w - k):
            result[i, j] = channel[i-k:i+k+1, j-k:j+k+1].mean()
    
    return result.astype(np.uint8)

blurred_red = box_blur(red, 5)
print(f"Blur applied. Original std: {red.std():.1f}, Blurred std: {blurred_red.std():.1f}")

# ── PIXEL STATISTICS ──────────────────────────────────────────────────────
# Histogram of pixel values
hist, bin_edges = np.histogram(grayscale, bins=256, range=(0, 256))
print(f"Most common pixel value: {np.argmax(hist)}")
print(f"Median pixel value: {np.median(grayscale):.0f}")
```

### Worked Example 3: Statistical Analysis

```python
import numpy as np

rng = np.random.default_rng(42)

# ── GENERATING TEST DATA ───────────────────────────────────────────────────
# Simulate A/B test: two groups of users, measure conversion
n_control = 1000
n_treatment = 1000

# Control group: 10% conversion rate
control = rng.binomial(n=1, p=0.10, size=n_control)

# Treatment group: 12% conversion rate (new feature works!)
treatment = rng.binomial(n=1, p=0.12, size=n_treatment)

# ── DESCRIPTIVE STATISTICS ────────────────────────────────────────────────
print("=== A/B Test Results ===")
print(f"Control:   {control.sum():>4} conversions / {n_control} users = {control.mean():.1%}")
print(f"Treatment: {treatment.sum():>4} conversions / {n_treatment} users = {treatment.mean():.1%}")
print(f"Lift: {(treatment.mean() - control.mean()) / control.mean():.1%}")

# ── BOOTSTRAP CONFIDENCE INTERVALS ───────────────────────────────────────
def bootstrap_ci(data, stat_func=np.mean, n_bootstrap=10000, confidence=0.95):
    """Calculate bootstrap confidence interval."""
    bootstrap_stats = np.array([
        stat_func(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_stats, alpha * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)
    return lower, upper

ctrl_ci = bootstrap_ci(control)
trt_ci  = bootstrap_ci(treatment)

print(f"\n95% Confidence Intervals:")
print(f"Control:   [{ctrl_ci[0]:.1%}, {ctrl_ci[1]:.1%}]")
print(f"Treatment: [{trt_ci[0]:.1%}, {trt_ci[1]:.1%}]")

# ── PERMUTATION TEST ──────────────────────────────────────────────────────
observed_diff = treatment.mean() - control.mean()
combined = np.concatenate([control, treatment])

perm_diffs = np.array([
    rng.permutation(combined)[:n_treatment].mean() -
    rng.permutation(combined)[n_treatment:].mean()
    for _ in range(10000)
])

p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
print(f"\nPermutation Test:")
print(f"Observed difference: {observed_diff:.3%}")
print(f"P-value: {p_value:.4f}")
print(f"Statistically significant (p<0.05): {p_value < 0.05}")

# ── PERCENTILE ANALYSIS ───────────────────────────────────────────────────
# Analyze response times for a web service
response_times = np.abs(rng.normal(200, 50, 10000)) + 50  # in ms

percentiles = [50, 75, 90, 95, 99, 99.9]
print(f"\nResponse Time Percentiles (ms):")
for p in percentiles:
    print(f"  P{p:<5}: {np.percentile(response_times, p):>7.1f}ms")

# SLA check: 95% of requests under 300ms?
sla_p95 = np.percentile(response_times, 95)
print(f"\nSLA Check (P95 < 300ms): {'✓ PASS' if sla_p95 < 300 else '✗ FAIL'}")
print(f"Actual P95: {sla_p95:.1f}ms")
```

---

## Part 10: Performance Comparison

### Pure Python vs NumPy

```python
import numpy as np
import time

def benchmark(name, func, *args, runs=5):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times)
    return avg, result

N = 1_000_000

# ── TEST 1: Sum of squares ─────────────────────────────────────────────────
python_list = list(range(N))
numpy_array = np.arange(N, dtype=np.float64)

def python_sum_squares(data):
    return sum(x**2 for x in data)

def numpy_sum_squares(arr):
    return (arr ** 2).sum()

py_time, _ = benchmark("Python", python_sum_squares, python_list)
np_time, _ = benchmark("NumPy",  numpy_sum_squares, numpy_array)

print(f"Sum of squares (N={N:,})")
print(f"  Python: {py_time:.4f}s")
print(f"  NumPy:  {np_time:.4f}s")
print(f"  Speedup: {py_time/np_time:.0f}x")

# ── TEST 2: Dot product ────────────────────────────────────────────────────
a_list = list(range(N))
b_list = list(range(N, 2*N))
a_arr = np.arange(N, dtype=np.float64)
b_arr = np.arange(N, 2*N, dtype=np.float64)

def python_dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def numpy_dot(a, b):
    return np.dot(a, b)

py_time, _ = benchmark("Python", python_dot, a_list, b_list)
np_time, _ = benchmark("NumPy",  numpy_dot, a_arr, b_arr)

print(f"\nDot product (N={N:,})")
print(f"  Python: {py_time:.4f}s")
print(f"  NumPy:  {np_time:.4f}s")
print(f"  Speedup: {py_time/np_time:.0f}x")

# ── TEST 3: Matrix multiplication ─────────────────────────────────────────
size = 500
A = np.random.random((size, size))
B = np.random.random((size, size))

def python_matmul(A, B):
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def numpy_matmul(A, B):
    return A @ B

# NOTE: Don't actually run python_matmul on 500x500 — it takes minutes!
# np_time, _ = benchmark("NumPy", numpy_matmul, A, B, runs=10)
# The NumPy version runs in ~0.002s; Python would take ~60s

print(f"\nMatrix multiply ({size}×{size})")
print(f"  NumPy:  ~0.002s")
print(f"  Python: ~60s (estimated)")
print(f"  Speedup: ~30,000x")

# ── MEMORY COMPARISON ─────────────────────────────────────────────────────
import sys

python_list = list(range(1000))
numpy_array = np.arange(1000, dtype=np.int64)

py_memory = sys.getsizeof(python_list) + sum(sys.getsizeof(x) for x in python_list)
np_memory = numpy_array.nbytes

print(f"\nMemory for 1000 integers:")
print(f"  Python list: {py_memory:,} bytes ({py_memory/1000:.0f} bytes/element)")
print(f"  NumPy array: {np_memory:,} bytes ({np_memory/1000:.0f} bytes/element)")
print(f"  Ratio: {py_memory/np_memory:.1f}x more memory for Python")
```

---

## Practice Problems

### Easy (1–5)

**Problem 1: Array Creation**
Create the following arrays without typing individual values.

```python
import numpy as np

# a) [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]  — 10 zeros (float)
a = ...

# b) [[1. 0. 0.], [0. 1. 0.], [0. 0. 1.]]  — 3x3 identity
b = ...

# c) [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]  — 11 evenly spaced 0→10
c = ...

# d) [[1 1 1 1 1],
#     [1 1 1 1 1],
#     [1 1 1 1 1]]  — 3x5 matrix of ones (integer!)
d = ...

# e) [1 4 9 16 25 36 49 64 81 100]  — squares of 1→10
e = ...
```

**Problem 2: Temperature Dataset**
Analyze a week of temperature readings.

```python
temps_celsius = np.array([
    [18, 21, 25, 24, 20],   # Monday: 5 readings
    [19, 22, 26, 25, 21],   # Tuesday
    [15, 18, 22, 20, 16],   # Wednesday (cold front)
    [12, 15, 19, 18, 14],   # Thursday
    [14, 17, 23, 22, 18],   # Friday
    [20, 24, 28, 27, 23],   # Saturday
    [21, 25, 29, 28, 24],   # Sunday
])

# Answer these:
# 1. What is the highest temperature of the week?
# 2. What day had the coldest average?
# 3. Convert all to Fahrenheit (F = C * 9/5 + 32)
# 4. Which readings were above 25°C?
# 5. Daily temperature range (max - min per day)
```

**Problem 3: Boolean Masking**
Use boolean indexing to answer questions about sales data.

```python
sales = np.array([
    [101, "North", 1200.50],
    [102, "South", 450.00],
    [103, "North", 3200.00],
    [104, "East",  890.75],
    [105, "North",  230.00],
    [106, "West",  4500.00],
    [107, "South", 1100.00],
], dtype=object)

amounts = sales[:, 2].astype(float)

# Find:
# 1. All sales over $1000
# 2. Average sale amount
# 3. Number of sales under $500
# 4. Total revenue from North region
```

**Problem 4: Statistical Summary**
Write a function that returns a full statistical summary of an array.

```python
def describe(arr):
    """Return dict with: count, mean, std, min, 25th, median, 75th, max"""
    pass

data = np.array([12, 45, 3, 67, 23, 89, 45, 23, 12, 56, 78, 34, 90, 1, 45])
stats = describe(data)
for k, v in stats.items():
    print(f"{k:>8}: {v:.2f}")
```

**Problem 5: Array Operations Without Loops**
Complete these vectorized operations.

```python
# Given:
prices = np.array([10.0, 25.0, 5.0, 100.0, 50.0, 75.0])
quantities = np.array([100, 50, 200, 10, 80, 30])

# Without any for loops:
# 1. Total revenue per product (price × quantity)
# 2. Total revenue across all products
# 3. Which products have revenue over $2000?
# 4. Normalize prices to [0, 1] range
# 5. Rank products by revenue (argsort!)
```

### Medium (6–12)

**Problem 6: Moving Average**
Calculate moving averages without using any loops.

```python
def moving_average(arr, window):
    """
    Calculate moving average of given window size.
    Use np.convolve for a pure NumPy solution.
    """
    pass

prices = np.array([100, 102, 98, 105, 103, 99, 107, 110, 108, 112])

ma3  = moving_average(prices, 3)   # 3-day moving average
ma5  = moving_average(prices, 5)   # 5-day moving average

print(ma3)   # [100.  101.67 101.67 102.  103.  105.  108.  110.]
```

**Problem 7: Euclidean Distance Matrix**
Calculate pairwise distances between all points.

```python
def distance_matrix(points):
    """
    Given array of shape (n, 2),
    return (n, n) matrix where [i, j] = distance between point i and j.
    Use broadcasting — NO loops!
    """
    pass

points = np.array([[0, 0], [3, 4], [6, 0], [3, -4]])
D = distance_matrix(points)
# D[0, 1] should be 5.0 (3-4-5 triangle)
```

**Problem 8: One-Hot Encoding**
Convert integer labels to one-hot encoded matrix.

```python
def one_hot_encode(labels, n_classes=None):
    """
    labels: 1D array of integer class labels
    Returns: 2D array of shape (len(labels), n_classes)
    Each row has one 1 and rest 0s.
    Use fancy indexing — NO loops!
    """
    pass

labels = np.array([0, 2, 1, 2, 0, 1])
one_hot = one_hot_encode(labels, n_classes=3)
print(one_hot)
# [[1 0 0]
#  [0 0 1]
#  [0 1 0]
#  [0 0 1]
#  [1 0 0]
#  [0 1 0]]
```

**Problem 9: Z-Score Outlier Detection**
Find outliers using Z-scores, vectorized.

```python
def find_outliers(data, threshold=3.0):
    """
    Return mask of outliers (|z_score| > threshold).
    Works for 1D or 2D arrays (per-column).
    """
    pass

data = np.array([12, 15, 14, 10, 1000, 13, 11, 14, 999, 12])
outliers = find_outliers(data)
print(data[outliers])   # [1000, 999]

# 2D version
data_2d = np.array([[1, 100, 2],
                    [2, 3, 3],
                    [3, 2, 500],
                    [2, 4, 2]])
```

**Problem 10: Random Walk Simulation**
Simulate multiple random walks simultaneously.

```python
def simulate_random_walks(n_walkers, n_steps, step_size=1.0):
    """
    Simulate n_walkers random walks of n_steps each.
    Each step is +step_size or -step_size.
    Return: (n_steps+1, n_walkers) array of positions.
    Use cumsum — NO loops over steps!
    """
    pass

positions = simulate_random_walks(1000, 252)  # 1000 stocks, 1 year
print(f"Shape: {positions.shape}")  # (253, 1000)

final = positions[-1]
print(f"Mean final position: {final.mean():.2f}")
print(f"Std of final positions: {final.std():.2f}")
print(f"Max gain: {final.max():.2f}")
print(f"Max loss: {final.min():.2f}")
```

**Problem 11: Image Convolution**
Apply a convolution filter to a 2D array using strides.

```python
def convolve2d(image, kernel):
    """
    Apply a kernel (filter) to a 2D image using sliding window.
    Return the filtered image (valid padding — smaller than input).
    """
    pass

# Sharpen filter
image = np.random.randint(0, 256, (100, 100), dtype=np.uint8).astype(float)
sharpen = np.array([[ 0, -1,  0],
                    [-1,  5, -1],
                    [ 0, -1,  0]])

result = convolve2d(image, sharpen)
print(f"Input shape: {image.shape}, Output shape: {result.shape}")
```

**Problem 12: Vectorized String Operations**
Use NumPy string operations on arrays of strings.

```python
# np.char module provides vectorized string ops
names = np.array(["alice johnson", "BOB SMITH", "  carol davis  ", "dave wilson"])

# 1. Capitalize all names properly
# 2. Get first names only
# 3. Get lengths of all names
# 4. Filter names containing "son"
# 5. Replace spaces with underscores
```

### Hard (13–20)

**Problem 13: Portfolio Optimization**
Find the optimal portfolio weights using Monte Carlo simulation.

```python
def monte_carlo_optimization(returns, n_simulations=10000):
    """
    Generate random portfolio weights.
    Calculate Sharpe ratio for each.
    Return weights with highest Sharpe ratio.
    
    returns: (n_days, n_stocks) array of daily returns
    """
    pass
```

**Problem 14: Vectorized K-Means**
Implement K-means clustering using only NumPy.

```python
def kmeans(X, k, max_iters=100, random_state=42):
    """
    K-means clustering.
    X: (n_samples, n_features)
    k: number of clusters
    Returns: (labels, centroids)
    """
    pass

X = np.vstack([
    np.random.normal([0, 0], 0.5, (100, 2)),
    np.random.normal([5, 5], 0.5, (100, 2)),
    np.random.normal([0, 5], 0.5, (100, 2))
])

labels, centroids = kmeans(X, k=3)
```

**Problem 15: Numerical Gradient**
Compute gradients numerically using finite differences.

```python
def numerical_gradient(f, x, eps=1e-5):
    """
    Compute gradient of f at point x using central differences.
    f: scalar function of array x
    Returns: gradient array (same shape as x)
    """
    pass

def rosenbrock(x):
    """Rosenbrock function — classic optimization test."""
    return sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

x = np.array([0.0, 0.0, 0.0])
grad = numerical_gradient(rosenbrock, x)
print(grad)  # [-2. -200. 0.]  (approximately)
```

**Problem 16: Sliding Window Statistics**
Compute rolling statistics using NumPy strides.

```python
def rolling_stats(arr, window):
    """
    Compute rolling mean and std without any Python loops.
    Use np.lib.stride_tricks.sliding_window_view
    """
    pass

prices = np.random.normal(100, 10, 1000)
mean_20, std_20 = rolling_stats(prices, 20)
# Bollinger Bands:
upper = mean_20 + 2 * std_20
lower = mean_20 - 2 * std_20
```

**Problem 17: Efficient Covariance Matrix**
Compute the covariance matrix from scratch using broadcasting.

```python
def covariance_matrix(X):
    """
    X: (n_samples, n_features)
    Return (n_features, n_features) covariance matrix.
    NO loops, use broadcasting and matrix multiplication.
    Should match np.cov(X.T) within floating point precision.
    """
    pass

X = np.random.randn(1000, 5)
custom_cov = covariance_matrix(X)
numpy_cov = np.cov(X.T)
print(np.allclose(custom_cov, numpy_cov))   # Should be True
```

**Problem 18: Vectorized Backpropagation**
Implement forward and backward pass of a simple neural layer.

```python
def relu(x): pass
def relu_grad(x): pass

def dense_forward(X, W, b):
    """
    X: (batch_size, input_dim)
    W: (input_dim, output_dim)
    b: (output_dim,)
    Returns: Z (pre-activation), A (post-activation)
    """
    pass

def dense_backward(dA, Z, X, W):
    """
    Backpropagate through a dense + ReLU layer.
    Returns: dX, dW, db
    """
    pass
```

**Problem 19: Spectral Analysis**
Use NumPy's FFT to analyze frequency content.

```python
def analyze_signal(t, signal):
    """
    Perform FFT and return dominant frequencies.
    t: time array
    signal: amplitude array
    Returns: (frequencies, amplitudes) sorted by amplitude
    """
    pass

# Create test signal: 50Hz + 120Hz + noise
t = np.linspace(0, 1, 1000)
signal = (3 * np.sin(2 * np.pi * 50 * t) +
          1.5 * np.sin(2 * np.pi * 120 * t) +
          np.random.normal(0, 0.1, 1000))

freqs, amps = analyze_signal(t, signal)
print("Top 3 frequencies:")
for f, a in zip(freqs[:3], amps[:3]):
    print(f"  {f:.0f} Hz (amplitude: {a:.2f})")
# ~50 Hz (3.0), ~120 Hz (1.5), ...
```

**Problem 20: NumPy Custom Universal Function**
Create a vectorized function using np.frompyfunc.

```python
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two GPS points in km.
    (The Haversine formula)
    """
    R = 6371  # Earth's radius in km
    # Your implementation here
    pass

# Vectorize it
vectorized_haversine = np.vectorize(haversine_distance)

# Test: distance from Mumbai to Delhi
lats1 = np.array([19.0760, 28.6139, 13.0827])   # Cities lat
lons1 = np.array([72.8777, 77.2090, 80.2707])   # Cities lon
lats2 = np.array([28.6139, 13.0827, 19.0760])   # Target lat
lons2 = np.array([77.2090, 80.2707, 72.8777])   # Target lon

distances = vectorized_haversine(lats1, lons1, lats2, lons2)
cities = ["Mumbai→Delhi", "Delhi→Chennai", "Chennai→Mumbai"]
for city, dist in zip(cities, distances):
    print(f"{city}: {dist:.0f} km")
```

---

## Answer Keys (Selected Problems)

### Problem 1 Solution:
```python
a = np.zeros(10)
b = np.eye(3)
c = np.linspace(0, 10, 11)
d = np.ones((3, 5), dtype=int)
e = np.arange(1, 11) ** 2
```

### Problem 6 Solution:
```python
def moving_average(arr, window):
    weights = np.ones(window) / window
    return np.convolve(arr, weights, mode="valid")
```

### Problem 8 Solution:
```python
def one_hot_encode(labels, n_classes=None):
    if n_classes is None:
        n_classes = labels.max() + 1
    one_hot = np.zeros((len(labels), n_classes), dtype=int)
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot
```

### Problem 10 Solution:
```python
def simulate_random_walks(n_walkers, n_steps, step_size=1.0):
    rng = np.random.default_rng()
    steps = rng.choice([-step_size, step_size], size=(n_steps, n_walkers))
    positions = np.vstack([np.zeros((1, n_walkers)), np.cumsum(steps, axis=0)])
    return positions
```

### Problem 15 Solution:
```python
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad
```

---

## Mini-Project: Quantitative Finance Toolkit

```python
"""
quant_finance.py — A NumPy-powered quantitative finance library
"""

import numpy as np

# ── DATA SIMULATION ────────────────────────────────────────────────────────

def simulate_gbm(S0, mu, sigma, T, dt, n_paths=1000, seed=42):
    """
    Simulate stock prices using Geometric Brownian Motion.
    
    GBM formula: dS = mu*S*dt + sigma*S*dW
    where dW is a Wiener process increment (random normal)
    
    Args:
        S0: Initial stock price
        mu: Drift (expected annual return, e.g. 0.10 for 10%)
        sigma: Volatility (annual, e.g. 0.20 for 20%)
        T: Time horizon in years
        dt: Time step (e.g. 1/252 for daily)
        n_paths: Number of Monte Carlo paths
        seed: Random seed for reproducibility
    
    Returns:
        prices: (n_steps+1, n_paths) array
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    
    # Random increments: shape (n_steps, n_paths)
    Z = rng.standard_normal((n_steps, n_paths))
    
    # Daily returns using exact GBM solution
    daily_returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    # Start at S0, then cumulatively multiply
    prices = np.zeros((n_steps + 1, n_paths))
    prices[0] = S0
    prices[1:] = S0 * np.cumprod(daily_returns, axis=0)
    
    return prices


# ── RISK METRICS ───────────────────────────────────────────────────────────

def value_at_risk(returns, confidence=0.95):
    """
    Calculate Value at Risk (VaR) — the loss not exceeded with confidence%.
    
    Returns: VaR as a positive number (e.g., 0.05 means 5% loss)
    """
    return -np.percentile(returns, (1 - confidence) * 100)

def conditional_var(returns, confidence=0.95):
    """
    CVaR (Expected Shortfall) — expected loss when loss > VaR.
    More conservative than VaR.
    """
    var = value_at_risk(returns, confidence)
    tail_losses = returns[returns < -var]
    return -tail_losses.mean() if len(tail_losses) > 0 else var

def max_drawdown(prices):
    """
    Maximum peak-to-trough decline.
    Returns: max drawdown as a negative decimal (e.g., -0.35 means 35% drop)
    """
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    return drawdown.min()

def sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Sharpe Ratio: excess return per unit of risk.
    Higher is better. > 1 is good, > 2 is great.
    """
    excess = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / returns.std()

def sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Like Sharpe but only penalizes DOWNSIDE volatility.
    """
    excess = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-8
    return np.sqrt(periods_per_year) * excess.mean() / downside_std


# ── PORTFOLIO ANALYTICS ────────────────────────────────────────────────────

def portfolio_returns(asset_returns, weights):
    """
    asset_returns: (n_days, n_assets)
    weights: (n_assets,)  — must sum to 1
    Returns: (n_days,) portfolio daily returns
    """
    return asset_returns @ weights   # Dot product = weighted sum

def efficient_frontier(asset_returns, n_portfolios=5000, seed=42):
    """
    Monte Carlo simulation of random portfolios to trace efficient frontier.
    Returns: (n_portfolios, n_assets+2) — [weights..., return, volatility]
    """
    rng = np.random.default_rng(seed)
    n_assets = asset_returns.shape[1]
    n_days = asset_returns.shape[0]
    
    # Random weights (Dirichlet distribution ensures they sum to 1)
    weights = rng.dirichlet(np.ones(n_assets), size=n_portfolios)
    
    # Portfolio metrics for all simulations at once
    # (n_portfolios, n_days) = (n_portfolios, n_assets) @ (n_assets, n_days)
    port_returns_all = (weights @ asset_returns.T)  # (n_portfolios, n_days)
    
    annual_returns = port_returns_all.mean(axis=1) * 252
    annual_vols = port_returns_all.std(axis=1) * np.sqrt(252)
    sharpes = annual_returns / annual_vols
    
    return weights, annual_returns, annual_vols, sharpes


# ── OPTION PRICING ─────────────────────────────────────────────────────────

def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes price for European call option.
    Fully vectorized — accepts arrays for any parameter.
    
    S: current stock price
    K: strike price
    T: time to expiry (years)
    r: risk-free rate (e.g., 0.05)
    sigma: volatility (e.g., 0.20)
    """
    from scipy.stats import norm   # pip install scipy
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def monte_carlo_option_price(S0, K, T, r, sigma, n_paths=100000, seed=42):
    """
    Price a European call option using Monte Carlo simulation.
    Compare with Black-Scholes to verify.
    """
    rng = np.random.default_rng(seed)
    
    # Simulate terminal stock prices
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Payoff: max(ST - K, 0)
    payoffs = np.maximum(ST - K, 0)
    
    # Discount to present value
    price = np.exp(-r * T) * payoffs.mean()
    std_error = payoffs.std() / np.sqrt(n_paths)
    
    return price, std_error


# ── FULL DEMO ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("QUANTITATIVE FINANCE TOOLKIT DEMO")
    print("=" * 60)
    
    # Simulate 3 assets for 1 year
    rng = np.random.default_rng(42)
    n_days, n_assets = 252, 3
    tickers = ["AAPL", "GOOGL", "MSFT"]
    
    asset_returns = np.column_stack([
        np.diff(simulate_gbm(100, mu=0.15, sigma=0.25, T=1, dt=1/252, n_paths=1)[:, 0])
        / simulate_gbm(100, mu=0.15, sigma=0.25, T=1, dt=1/252, n_paths=1)[:-1, 0],
        np.diff(simulate_gbm(150, mu=0.12, sigma=0.22, T=1, dt=1/252, n_paths=1)[:, 0])
        / simulate_gbm(150, mu=0.12, sigma=0.22, T=1, dt=1/252, n_paths=1)[:-1, 0],
        np.diff(simulate_gbm(80, mu=0.18, sigma=0.28, T=1, dt=1/252, n_paths=1)[:, 0])
        / simulate_gbm(80, mu=0.18, sigma=0.28, T=1, dt=1/252, n_paths=1)[:-1, 0],
    ])

    # Use simpler direct simulation
    asset_returns = rng.normal(
        loc=np.array([0.0006, 0.0005, 0.0007]),
        scale=np.array([0.016, 0.014, 0.018]),
        size=(n_days, n_assets)
    )
    
    print("\n── INDIVIDUAL ASSET METRICS ──────────────────────────")
    print(f"{'Asset':<8} {'Ann. Return':>12} {'Ann. Vol':>10} {'Sharpe':>8} {'Max DD':>8}")
    print("-" * 50)
    for i, ticker in enumerate(tickers):
        r = asset_returns[:, i]
        prices = (1 + r).cumprod()
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        sr = sharpe_ratio(r)
        mdd = max_drawdown(prices)
        print(f"{ticker:<8} {ann_ret:>11.1%} {ann_vol:>9.1%} {sr:>8.2f} {mdd:>8.1%}")
    
    print("\n── RISK METRICS (Equal-Weight Portfolio) ─────────────")
    weights = np.ones(n_assets) / n_assets
    port_ret = portfolio_returns(asset_returns, weights)
    
    print(f"Daily VaR (95%):       {value_at_risk(port_ret, 0.95):.2%}")
    print(f"Daily CVaR (95%):      {conditional_var(port_ret, 0.95):.2%}")
    print(f"Annual Sharpe:         {sharpe_ratio(port_ret):.2f}")
    print(f"Annual Sortino:        {sortino_ratio(port_ret):.2f}")
    prices_port = (1 + port_ret).cumprod()
    print(f"Max Drawdown:          {max_drawdown(prices_port):.1%}")
    
    print("\n── MONTE CARLO PORTFOLIO SIMULATION ──────────────────")
    weights_mc, returns_mc, vols_mc, sharpes_mc = efficient_frontier(
        asset_returns, n_portfolios=10000
    )
    best_idx = sharpes_mc.argmax()
    print(f"Best Sharpe portfolio found: {sharpes_mc.max():.2f}")
    print(f"  Weights: ", end="")
    for t, w in zip(tickers, weights_mc[best_idx]):
        print(f"{t}={w:.0%}", end=" ")
    print(f"\n  Expected Return: {returns_mc[best_idx]:.1%}")
    print(f"  Volatility:      {vols_mc[best_idx]:.1%}")
    
    print("\n── OPTION PRICING ────────────────────────────────────")
    S0, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.20
    mc_price, mc_se = monte_carlo_option_price(S0, K, T, r, sigma, n_paths=100000)
    print(f"Call option: S={S0}, K={K}, T={T}y, r={r:.0%}, σ={sigma:.0%}")
    print(f"Monte Carlo Price: ${mc_price:.4f} ± ${mc_se:.4f}")
    
    print("\n" + "=" * 60)

main()
```

---

## Chapter Summary

You've learned NumPy — the foundation of all scientific Python!

✅ **Why NumPy**: Fixed-type arrays, contiguous memory, C-compiled operations → 100x faster
✅ **Creating Arrays**: `zeros`, `ones`, `arange`, `linspace`, `random`, from lists
✅ **Properties**: `shape`, `ndim`, `size`, `dtype`, `nbytes`
✅ **Indexing**: Slicing (returns views!), fancy indexing, boolean masking
✅ **Reshaping**: `reshape`, `flatten`, `ravel`, `newaxis`, `stack`, `concatenate`
✅ **Vectorized Ops**: Element-wise arithmetic, ufuncs, no loops needed
✅ **Broadcasting**: Operations between different-shaped arrays
✅ **Aggregations**: `sum`, `mean`, `std`, `min`, `max` along axes
✅ **Linear Algebra**: Matrix multiply `@`, `inv`, `det`, `solve`, `eig`
✅ **Performance**: 10–1000x faster than Python loops, 10x less memory

**Key Takeaways:**
- If you find yourself writing a loop over array elements, ask: "Can NumPy do this vectorized?"
- Boolean masking is more powerful and readable than filtered loops
- Slices return **views**, not copies — modify with care, use `.copy()` when needed
- `axis=0` collapses rows (result is shape of one row); `axis=1` collapses columns
- Broadcasting is automatic when shapes are compatible — learn the rules

**Next Chapter Preview:**
Chapter 12 covers **Pandas** — the workhorse of data analysis. DataFrames, Series, reading CSVs and Excel, merging datasets, groupby aggregations, and the full data cleaning pipeline!
