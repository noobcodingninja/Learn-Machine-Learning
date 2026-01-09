# Linear Algebra for Machine Learning
## Chapter 5: Linear Independence, Basis, and Orthogonality

### A First-Principles Approach with Detailed Examples

---

# Table of Contents

1. [Linear Independence](#independence)
2. [Span and Subspaces](#span)
3. [Basis and Dimension](#basis)
4. [Orthogonality](#orthogonality)
5. [Gram-Schmidt Process](#gram-schmidt)
6. [Applications to Machine Learning](#applications)
7. [Comprehensive Practice Problems](#practice)

---

<a name="independence"></a>
# 1. Linear Independence

## The Core Question: When Do Vectors Give Us New Information?

Imagine you're a data scientist collecting features for a machine learning model.

**Scenario:** Predicting house prices

You have these features:
- Feature 1: House area in square feet
- Feature 2: House area in square meters
- Feature 3: Number of bedrooms

**Question:** Are all three features useful?

**Problem:** Features 1 and 2 are the same information!
- Square meters = Square feet × 0.0929
- They're redundant - one is just a scaled version of the other

**This is the problem of linear dependence!**

## What Does "Independent" Mean?

**Intuitive definition:** Vectors are **linearly independent** if none of them can be written as a combination of the others.

**Question:** But what does "combination of others" mean exactly?

**Answer:** A vector is a **linear combination** of others if you can write it as:

**v** = c₁**v₁** + c₂**v₂** + ... + cₙ**vₙ**

for some scalars c₁, c₂, ..., cₙ.

**Back to house example:**
- Square meters = 0.0929 × Square feet
- Feature 2 = 0.0929 × Feature 1
- **Feature 2 is a linear combination of Feature 1!**
- Therefore, they are **linearly dependent**

## Why Does This Matter?

**Question:** Why should we care about linear independence?

**Problems with dependent features:**

1. **Redundant information:** Not adding new knowledge
2. **Computational waste:** Processing duplicate information
3. **Unstable models:** Some algorithms break (matrix inversion fails!)
4. **Overfitting risk:** More parameters, same information
5. **Hard to interpret:** Can't tell which feature matters

**Example in machine learning:**

```python
# Bad model (dependent features)
price = w₁ × sqft + w₂ × sqm + w₃ × bedrooms

Problem: w₁ and w₂ are not uniquely determined!
Could have: w₁=100, w₂=0 (same prediction!)
Or:         w₁=0, w₂=1076 (same prediction!)
Or:         w₁=50, w₂=538 (same prediction!)

# Good model (independent features)
price = w₁ × sqft + w₂ × bedrooms

Now: w₁ and w₂ are uniquely determined!
```

## Formal Definition

**Definition:** Vectors **v₁, v₂, ..., vₙ** are **linearly independent** if:

The only solution to:
c₁**v₁** + c₂**v₂** + ... + cₙ**vₙ** = **0**

is:
c₁ = c₂ = ... = cₙ = 0

**In plain English:** The only way to combine them to get zero is to use all zero coefficients.

**If there's another way (non-zero coefficients) → linearly dependent!**

### Understanding the Definition

**Question:** Why this specific definition?

**Let's think through it:**

Suppose c₁**v₁** + c₂**v₂** + c₃**v₃** = **0** with c₁ ≠ 0.

Then we can solve for **v₁**:
- c₁**v₁** = -c₂**v₂** - c₃**v₃**
- **v₁** = (-c₂/c₁)**v₂** + (-c₃/c₁)**v₃**

**So v₁ is a combination of v₂ and v₃!**

**Therefore:** If there's a non-trivial combination giving zero, then one vector can be written as a combination of others = dependent!

## Simple Examples

### Example 1.1: Two Vectors in 2D

**Vectors:**
- **v₁** = (2, 1)
- **v₂** = (4, 2)

**Question:** Are they independent?

**Test:** Can we write c₁**v₁** + c₂**v₂** = **0** with non-zero c's?

c₁(2, 1) + c₂(4, 2) = (0, 0)

This gives:
- 2c₁ + 4c₂ = 0
- c₁ + 2c₂ = 0

From second equation: c₁ = -2c₂

**Choose c₂ = 1:** Then c₁ = -2

**Check:** -2(2, 1) + 1(4, 2) = (-4, -2) + (4, 2) = (0, 0) ✓

**Non-zero coefficients give zero!**

**Therefore: DEPENDENT**

**Why geometrically?** **v₂** = 2**v₁** (same direction, just longer!)

### Example 1.2: Two Vectors in 2D (Different Directions)

**Vectors:**
- **v₁** = (1, 0)
- **v₂** = (0, 1)

**Test:** c₁(1, 0) + c₂(0, 1) = (0, 0)

This gives:
- c₁ = 0
- c₂ = 0

**Only trivial solution!**

**Therefore: INDEPENDENT**

**Why geometrically?** Point in different directions - neither is a multiple of the other!

### Example 1.3: Three Vectors in 2D

**Vectors:**
- **v₁** = (1, 0)
- **v₂** = (0, 1)
- **v₃** = (1, 1)

**Question:** Can three vectors in 2D be independent?

**Intuition:** In 2D space, you only need 2 directions. A third must be redundant!

**Test:** c₁(1, 0) + c₂(0, 1) + c₃(1, 1) = (0, 0)

This gives:
- c₁ + c₃ = 0
- c₂ + c₃ = 0

**Solution:** c₁ = -c₃, c₂ = -c₃

**Choose c₃ = 1:** Then c₁ = -1, c₂ = -1

**Check:** -1(1, 0) + -1(0, 1) + 1(1, 1) = (-1, 0) + (0, -1) + (1, 1) = (0, 0) ✓

**Therefore: DEPENDENT**

**In fact:** **v₃** = **v₁** + **v₂**

**Key insight:** You can't have more than 2 independent vectors in 2D!

## The Geometric Intuition

**In 2D:**
- 1 vector: Just a line through origin
- 2 independent vectors: Span the entire plane
- 3+ vectors: At least one must be dependent!

**Why?** Once you have 2 vectors pointing in different directions, any third vector in the plane can be written as a combination of the first two!

**In 3D:**
- 1 vector: A line
- 2 independent vectors: A plane
- 3 independent vectors: All of 3D space
- 4+ vectors: At least one must be dependent!

**General rule:** In n-dimensional space, you can have at most n independent vectors!

## Testing for Independence

### Method 1: Set Up the Equation

For vectors **v₁, v₂, ..., vₖ**:

1. Write: c₁**v₁** + c₂**v₂** + ... + cₖ**vₖ** = **0**
2. This gives a system of equations
3. Solve for c₁, c₂, ..., cₖ
4. If only solution is all zeros → Independent
5. If non-zero solution exists → Dependent

### Method 2: Matrix Method

Put vectors as columns of a matrix:

A = [**v₁** **v₂** ... **vₖ**]

Then:
- **Independent** if matrix has full column rank
- **Dependent** if columns are linearly dependent

(We'll learn more about rank in later chapters!)

## Detailed Example: Testing Independence

**Vectors in 3D:**
- **v₁** = (1, 2, 3)
- **v₂** = (4, 5, 6)
- **v₃** = (7, 8, 9)

**Test:** Are they independent?

**Set up equation:**
c₁(1, 2, 3) + c₂(4, 5, 6) + c₃(7, 8, 9) = (0, 0, 0)

**This gives three equations:**
- c₁ + 4c₂ + 7c₃ = 0 ... (1)
- 2c₁ + 5c₂ + 8c₃ = 0 ... (2)
- 3c₁ + 6c₂ + 9c₃ = 0 ... (3)

**Solve the system:**

From equation (1): c₁ = -4c₂ - 7c₃

Substitute into equation (2):
2(-4c₂ - 7c₃) + 5c₂ + 8c₃ = 0
-8c₂ - 14c₃ + 5c₂ + 8c₃ = 0
-3c₂ - 6c₃ = 0
c₂ = -2c₃

**Choose c₃ = 1:** Then c₂ = -2, c₁ = -4(-2) - 7(1) = 8 - 7 = 1

**Solution:** c₁ = 1, c₂ = -2, c₃ = 1

**Verify:**
1(1, 2, 3) + (-2)(4, 5, 6) + 1(7, 8, 9)
= (1, 2, 3) + (-8, -10, -12) + (7, 8, 9)
= (1 - 8 + 7, 2 - 10 + 8, 3 - 12 + 9)
= (0, 0, 0) ✓

**Non-zero coefficients work!**

**Therefore: DEPENDENT**

**Relationship:** **v₃** = 2**v₂** - **v₁**

Or equivalently: **v₁** - 2**v₂** + **v₃** = **0**

## Why These Vectors Are Dependent

**Look at the pattern:**
- **v₁** = (1, 2, 3)
- **v₂** = (4, 5, 6)
- **v₃** = (7, 8, 9)

**Notice:** Each component increases by 3!

First component: 1, 4, 7 (arithmetic sequence, difference = 3)
Second component: 2, 5, 8 (arithmetic sequence, difference = 3)
Third component: 3, 6, 9 (arithmetic sequence, difference = 3)

**This pattern means:** **v₂** - **v₁** = (3, 3, 3) and **v₃** - **v₂** = (3, 3, 3)

So: **v₃** - **v₂** = **v₂** - **v₁**

Therefore: **v₃** = 2**v₂** - **v₁**

**Geometric meaning:** The three vectors are coplanar (lie in the same plane)!

## Maximum Number of Independent Vectors

**Theorem:** In n-dimensional space ℝⁿ, you can have at most n linearly independent vectors.

**Examples:**
- In ℝ² (2D): At most 2 independent vectors
- In ℝ³ (3D): At most 3 independent vectors
- In ℝ¹⁰⁰: At most 100 independent vectors

**Question:** Why this limit?

**Intuitive answer:** Each independent vector adds a new "dimension" or "direction." Once you have n independent vectors in ℝⁿ, you've filled up all possible directions!

**Formal answer:** This is related to the rank of a matrix. We'll see more in later chapters!

## Connection to Machine Learning Features

**Back to the original question:** Which features should we include in our model?

**Goal:** Choose **linearly independent** features!

**Example: House price prediction**

**Bad feature set (dependent):**
- Area in sqft
- Area in sqm (= 0.0929 × sqft)
- Area in acres (= sqft / 43560)
- Price per sqft × sqft

**Why bad?** Features 2, 3, 4 are linear combinations of Feature 1!

**Good feature set (independent):**
- Area in sqft
- Number of bedrooms
- Age of house
- School district rating
- Distance to downtown

**Why good?** No feature can be written as a combination of others!

## Checking Independence in Practice

**For k features in n-dimensional space:**

**Quick checks:**

1. **If k > n:** Definitely dependent!
   - Example: 5 features in 3D space → dependent

2. **If one feature is a multiple of another:** Dependent!
   - Example: (x, 2x, 3x) → all multiples of first component

3. **If one feature is sum/difference of others:** Dependent!
   - Example: feature₃ = feature₁ + feature₂

**Formal test:**
1. Create matrix with features as columns
2. Calculate rank (number of independent columns)
3. If rank < k → dependent
4. If rank = k → independent

## Practice Problems - Linear Independence

**Problem 1.1: Testing Independence**

Are these vectors independent?

a) **v₁** = (1, 2), **v₂** = (3, 6)
b) **v₁** = (1, 0), **v₂** = (0, 1)
c) **v₁** = (1, 1, 0), **v₂** = (0, 1, 1), **v₃** = (1, 0, 1)
d) **v₁** = (1, 0, 0), **v₂** = (0, 1, 0), **v₃** = (0, 0, 1), **v₄** = (1, 1, 1)

**Problem 1.2: Finding Relationships**

These vectors are dependent. Find the linear relationship:

a) **v₁** = (2, 4), **v₂** = (1, 2)
b) **v₁** = (1, 2, 3), **v₂** = (2, 4, 6)
c) **v₁** = (1, 1, 0), **v₂** = (1, 0, 1), **v₃** = (2, 1, 1)

**Problem 1.3: Maximum Independent Set**

Given: **v₁** = (1, 0, 0), **v₂** = (0, 1, 0), **v₃** = (0, 0, 1), **v₄** = (1, 1, 0), **v₅** = (1, 1, 1)

a) Are all 5 vectors independent?
b) What's the maximum number of independent vectors you can choose?
c) Give an example of a maximum independent subset

**Problem 1.4: Feature Selection**

You're predicting car prices with these features:
- Price in USD
- Price in EUR (= USD × 0.92)
- Engine size in liters
- Engine size in cubic inches (= liters × 61.024)
- Horsepower
- Torque (highly correlated with horsepower, but not exact multiple)

a) Which features are linearly dependent?
b) Create a set of independent features
c) Would you include both horsepower and torque? Why or why not?

**Problem 1.5: Geometric Interpretation**

Three vectors in 3D: **v₁** = (1, 0, 0), **v₂** = (0, 1, 0), **v₃** = (a, b, 0)

a) For what values of a and b are the vectors independent?
b) For what values are they dependent?
c) Geometric meaning: What surface do dependent vectors lie on?

---

**End of Section 5.1: Linear Independence**

Key takeaways:
- ✅ Vectors are independent if none can be written as combination of others
- ✅ Test: Only trivial solution to c₁v₁ + c₂v₂ + ... = 0
- ✅ At most n independent vectors in ℝⁿ
- ✅ Critical for machine learning feature selection
- ✅ Dependent features cause redundancy and instability

# Chapter 5: Section 2 - Span and Subspaces

<a name="span"></a>
# 2. Span and Subspaces

## The Core Question: What Can We Reach?

Imagine you're a robot on a factory floor. You can move in two directions:
- Direction 1: 3 steps forward, 1 step right
- Direction 2: 1 step forward, 2 steps right

**Question:** By combining these movements, what positions can you reach?

**This is the concept of SPAN!**

## What is Span?

**Definition:** The **span** of a set of vectors is all possible linear combinations of those vectors.

**Formal notation:** 

span{**v₁, v₂, ..., vₖ**} = {c₁**v₁** + c₂**v₂** + ... + cₖ**vₖ** : c₁, c₂, ..., cₖ ∈ ℝ}

**In plain English:** The span is the set of all points you can reach by taking any combination (with any coefficients) of the given vectors.

**Robot example:**
- **v₁** = (3, 1) [3 forward, 1 right]
- **v₂** = (1, 2) [1 forward, 2 right]

span{**v₁, v₂**} = all points of form c₁(3, 1) + c₂(1, 2)

**Can we reach (5, 4)?**
- Try: c₁(3, 1) + c₂(1, 2) = (5, 4)
- This gives: 3c₁ + c₂ = 5 and c₁ + 2c₂ = 4
- Solving: c₁ = 2/5, c₂ = 17/5... wait, let me recalculate

From first equation: c₂ = 5 - 3c₁
Substitute into second: c₁ + 2(5 - 3c₁) = 4
c₁ + 10 - 6c₁ = 4
-5c₁ = -6
c₁ = 6/5

Then c₂ = 5 - 3(6/5) = 5 - 18/5 = 7/5

**Check:** (6/5)(3, 1) + (7/5)(1, 2) = (18/5 + 7/5, 6/5 + 14/5) = (25/5, 20/5) = (5, 4) ✓

**Yes! We can reach (5, 4)!**

## Geometric Understanding

### Span of One Vector

**Given:** **v** = (2, 1)

span{**v**} = all points c(2, 1) for any scalar c

**Question:** What does this look like?

**Answer:** A line through the origin!

```
        ↑
      (4,2) = 2v
        |
      (2,1) = v
        |
        * ← origin
        |
     (-2,-1) = -v
        |
```

**All multiples of **v** form a line!**

**In 3D:** span of one vector is still a line through origin.

### Span of Two Vectors (Independent)

**Given:** **v₁** = (1, 0), **v₂** = (0, 1)

span{**v₁, v₂**} = all points c₁(1, 0) + c₂(0, 1) = (c₁, c₂)

**This is every point in the plane!**

**Key insight:** Two independent vectors in 2D span the entire 2D space!

**In 3D:** Two independent vectors span a plane through the origin.

```
In 3D, span{v₁, v₂} looks like:
        
        ↑ z
        |
        |  / (plane)
        | /
        |/___→ y
       /
      /
     → x
```

### Span of Two Vectors (Dependent)

**Given:** **v₁** = (2, 1), **v₂** = (4, 2) = 2**v₁**

span{**v₁, v₂**} = all points c₁(2, 1) + c₂(4, 2)

But since **v₂** = 2**v₁**:
= c₁(2, 1) + c₂·2(2, 1)
= (c₁ + 2c₂)(2, 1)

**This is still just a line!** (Same as span{**v₁**})

**Key insight:** Dependent vectors don't add new directions! Span stays the same.

### Span of Three Vectors in 3D

**Case 1: All independent**

**v₁** = (1, 0, 0), **v₂** = (0, 1, 0), **v₃** = (0, 0, 1)

span{**v₁, v₂, v₃**} = all points (c₁, c₂, c₃)

**This is all of 3D space!**

**Case 2: Two independent, one dependent**

**v₁** = (1, 0, 0), **v₂** = (0, 1, 0), **v₃** = (1, 1, 0) = **v₁** + **v₂**

span{**v₁, v₂, v₃**} = span{**v₁, v₂**} = xy-plane

**Only a plane!** (Third vector doesn't add new direction)

**Case 3: All dependent (on one line)**

**v₁** = (1, 2, 3), **v₂** = (2, 4, 6), **v₃** = (3, 6, 9)

span{**v₁, v₂, v₃**} = span{**v₁**} = a line

**Only a line!** (All point in same direction)

## The Pattern

**Number of independent vectors = dimension of span**

| Independent Vectors | Span in 3D |
|---------------------|------------|
| 1 | Line through origin |
| 2 | Plane through origin |
| 3 | All of 3D space |

**General rule in ℝⁿ:**
- k independent vectors span a k-dimensional subspace
- Maximum: n independent vectors span all of ℝⁿ

## What is a Subspace?

**Definition:** A **subspace** is a subset of ℝⁿ that:
1. Contains the zero vector **0**
2. Closed under addition: If **u**, **v** in subspace, then **u** + **v** in subspace
3. Closed under scalar multiplication: If **v** in subspace, then c**v** in subspace

**In plain English:** A subspace is a space "within" the larger space that behaves like a vector space itself.

**Examples of subspaces in ℝ³:**
- {**0**} (just the origin) - 0-dimensional
- Any line through origin - 1-dimensional
- Any plane through origin - 2-dimensional
- All of ℝ³ - 3-dimensional

**Important:** Subspaces MUST go through the origin!

**Non-examples (not subspaces):**
- A line NOT through origin
- A plane NOT through origin
- A sphere
- A single point (unless it's the origin)

**Why must it contain origin?** Because 0·**v** = **0** (scalar multiplication must stay in the subspace!)

## Span Creates Subspaces

**Theorem:** The span of any set of vectors is a subspace.

**Why?**

Let S = span{**v₁, v₂, ..., vₖ**}

**Check property 1:** Does S contain **0**?
- **0** = 0·**v₁** + 0·**v₂** + ... + 0·**vₖ** ✓

**Check property 2:** Closed under addition?
- Take **u** = a₁**v₁** + a₂**v₂** + ... + aₖ**vₖ** (in S)
- Take **w** = b₁**v₁** + b₂**v₂** + ... + bₖ**vₖ** (in S)
- **u** + **w** = (a₁+b₁)**v₁** + (a₂+b₂)**v₂** + ... + (aₖ+bₖ)**vₖ**
- This is also a linear combination of **v₁, ..., vₖ** → in S ✓

**Check property 3:** Closed under scalar multiplication?
- Take **u** = a₁**v₁** + a₂**v₂** + ... + aₖ**vₖ** (in S)
- c**u** = ca₁**v₁** + ca₂**v₂** + ... + caₖ**vₖ**
- This is also a linear combination → in S ✓

**Therefore, span is always a subspace!**

## Testing if a Point is in a Span

**Question:** Is **b** in span{**v₁, v₂, ..., vₖ**}?

**Method:** Try to find coefficients c₁, c₂, ..., cₖ such that:

c₁**v₁** + c₂**v₂** + ... + cₖ**vₖ** = **b**

**If solution exists → Yes, **b** is in span**
**If no solution → No, **b** is not in span**

### Example 2.1: Is (7, 5) in span{(1, 2), (3, 1)}?

**Setup:** Does c₁(1, 2) + c₂(3, 1) = (7, 5)?

**Equations:**
- c₁ + 3c₂ = 7
- 2c₁ + c₂ = 5

**Solve:**
From first equation: c₁ = 7 - 3c₂
Substitute into second: 2(7 - 3c₂) + c₂ = 5
14 - 6c₂ + c₂ = 5
-5c₂ = -9
c₂ = 9/5

Then: c₁ = 7 - 3(9/5) = 7 - 27/5 = 8/5

**Check:** (8/5)(1, 2) + (9/5)(3, 1) = (8/5 + 27/5, 16/5 + 9/5) = (35/5, 25/5) = (7, 5) ✓

**Answer: YES!** (7, 5) is in the span.

**Geometric meaning:** (7, 5) lies in the plane spanned by (1, 2) and (3, 1).

### Example 2.2: Is (1, 0, 0) in span{(1, 1, 0), (0, 1, 1)}?

**Setup:** Does c₁(1, 1, 0) + c₂(0, 1, 1) = (1, 0, 0)?

**Equations:**
- c₁ = 1
- c₁ + c₂ = 0
- c₂ = 0

From first: c₁ = 1
From third: c₂ = 0
Check second: 1 + 0 = 1 ≠ 0 ✗

**Contradiction!**

**Answer: NO!** (1, 0, 0) is NOT in the span.

**Geometric meaning:** The two vectors span a plane, but (1, 0, 0) is outside that plane.

## Column Space of a Matrix

**Connection to matrices:**

Given vectors **v₁, v₂, ..., vₖ**, form matrix A = [**v₁** **v₂** ... **vₖ**] (vectors as columns).

Then:
span{**v₁, v₂, ..., vₖ**} = column space of A

**Column space:** All possible outputs A**x** for any **x**

**Question:** Is **b** in column space of A?

**Answer:** Can we solve A**x** = **b**?

**If yes → **b** in column space**
**If no → **b** not in column space**

This connects linear algebra to solving systems of equations!

## Dimension of a Span

**Definition:** The **dimension** of span{**v₁, ..., vₖ**} is the maximum number of linearly independent vectors in the set.

**Examples:**

**1 vector:**
- span{(2, 1)} has dimension 1 (a line)

**2 independent vectors in 2D:**
- span{(1, 0), (0, 1)} has dimension 2 (entire plane)

**2 dependent vectors:**
- span{(2, 1), (4, 2)} has dimension 1 (just a line, since one is redundant)

**3 independent vectors in 3D:**
- span{(1,0,0), (0,1,0), (0,0,1)} has dimension 3 (all of 3D)

**3 vectors, 2 independent:**
- span{(1,0,0), (0,1,0), (1,1,0)} has dimension 2 (a plane, third is redundant)

**Key insight:** Dimension = number of truly independent directions!

## Detailed Example: Finding Dimension of Span

**Given vectors in ℝ⁴:**
- **v₁** = (1, 2, 0, 1)
- **v₂** = (2, 4, 0, 2)
- **v₃** = (0, 1, 1, 0)
- **v₄** = (1, 3, 1, 1)

**Question:** What is the dimension of span{**v₁, v₂, v₃, v₄**}?

**Step 1: Check which vectors are independent**

Notice: **v₂** = 2**v₁** (second is double of first)

So **v₂** is redundant! Remove it.

Now check: **v₁, v₃, v₄**

**Step 2: Check if **v₄** is combination of **v₁** and **v₃****

Does c₁**v₁** + c₂**v₃** = **v₄**?

c₁(1, 2, 0, 1) + c₂(0, 1, 1, 0) = (1, 3, 1, 1)

**Equations:**
- c₁ = 1
- 2c₁ + c₂ = 3 → 2(1) + c₂ = 3 → c₂ = 1
- c₂ = 1 ✓ (consistent!)
- c₁ = 1 ✓ (consistent!)

**Yes!** **v₄** = **v₁** + **v₃**

**Step 3: Conclusion**

Independent vectors: **v₁** and **v₃** (only 2!)

**Dimension of span = 2**

Even though we had 4 vectors, they only span a 2-dimensional subspace of ℝ⁴!

## Applications to Machine Learning

### Feature Space

**In ML, each feature is a dimension.**

**Example:** House prices with features
- Feature 1: Square footage
- Feature 2: Number of bedrooms  
- Feature 3: Age
- Feature 4: Square footage in meters (= Feature 1 × 0.0929)

**Question:** What's the dimension of the feature space?

**Answer:** Only 3! (Feature 4 is in span of Feature 1)

**Real dimensionality:** 3, not 4

**Implication:** Model actually lives in 3D space, not 4D!

### Principal Component Analysis (PCA)

**Goal:** Find lower-dimensional subspace that captures most variance

**Process:**
1. Data lives in high-dimensional space (many features)
2. Find directions of maximum variance
3. Project onto span of these directions
4. This span is a lower-dimensional subspace!

**Example:**
- Original: 100 features (100-dimensional)
- PCA: Find that 95% of variance in span of 10 directions
- New representation: 10-dimensional subspace
- Dimension reduced: 100 → 10!

### Linear Regression

**Setup:** Predict y from features **x**

**Model:** ŷ = **w**ᵀ**x** + b

**Question:** What values can ŷ take?

**Answer:** All values in span{**x₁**, **x₂**, ..., **xₙ**} (plus bias b)

**If features are dependent:** 
- Span is lower-dimensional than expected
- Multiple **w** give same predictions
- Model not uniquely determined!

**This is why we want independent features!**

## Practice Problems - Span and Subspaces

**Problem 2.1: Computing Span**

Find the span of:
a) {(1, 2)}
b) {(1, 0), (0, 1)}
c) {(1, 1), (2, 2)}
d) {(1, 0, 0), (0, 1, 0), (0, 0, 1)}

Describe geometrically (line, plane, space, etc.)

**Problem 2.2: Testing Membership**

Is **b** in span{**v₁, v₂**}?

a) **b** = (5, 7), **v₁** = (1, 2), **v₂** = (2, 1)
b) **b** = (1, 1, 1), **v₁** = (1, 0, 0), **v₂** = (0, 1, 0)
c) **b** = (6, 3), **v₁** = (2, 1), **v₂** = (4, 2)

**Problem 2.3: Subspace Verification**

Which of these are subspaces of ℝ²?

a) All vectors of form (a, 2a)
b) All vectors of form (a, a+1)
c) All vectors (a, b) where a ≥ 0
d) All vectors (a, 0)

**Problem 2.4: Dimension of Span**

Find dimension of span:

a) {(1, 0, 0), (0, 1, 0), (1, 1, 0)}
b) {(1, 2, 3), (2, 4, 6), (3, 6, 9)}
c) {(1, 0), (0, 1), (1, 1), (2, 3)}
d) {(1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1), (1, 0, 0, 1)}

**Problem 2.5: Application to Features**

Dataset has features:
- Temperature in Celsius
- Temperature in Fahrenheit (= C × 9/5 + 32)
- Humidity (%)
- Pressure (kPa)

a) What's the span of these feature vectors?
b) What's the true dimension of feature space?
c) Which features should you keep?

**Problem 2.6: Geometric Understanding**

In ℝ³, you have **v₁** = (1, 0, 0) and **v₂** = (0, 1, 0)

a) Describe span{**v₁, v₂**} geometrically
b) Give 3 examples of vectors IN the span
c) Give 3 examples of vectors NOT in the span
d) What vector could you add to span all of ℝ³?

**Problem 2.7: Span Equality**

Prove or disprove:
a) span{**v₁, v₂**} = span{**v₁, v₂, **v₁** + **v₂**}
b) span{**v₁, v₂**} = span{2**v₁, 2**v₂**}
c) span{**v₁**} = span{3**v₁**}

---

**End of Section 5.2: Span and Subspaces**

Key takeaways:
- ✅ Span = all linear combinations of vectors
- ✅ Span creates a subspace (line, plane, or higher dimension)
- ✅ Dimension of span = number of independent vectors
- ✅ Dependent vectors don't add new dimensions
- ✅ Critical for understanding feature spaces in ML
- ✅ Connection to column space of matrices

Next: Section 5.3 - Basis and Dimension

# Chapter 5: Section 3 - Basis and Dimension

<a name="basis"></a>
# 3. Basis and Dimension

## The Core Question: What's the Minimal Set We Need?

Imagine you're packing for a trip and want to create any outfit from a minimal wardrobe.

**Bad strategy:** Pack 50 shirts that are all slight variations of the same color
- Redundant! Many similar items
- Heavy luggage
- Still limited variety

**Good strategy:** Pack 5 versatile, distinct pieces
- Each piece adds something new
- Minimal weight
- Can create many combinations

**This is the idea of a BASIS - a minimal spanning set!**

## What is a Basis?

**Definition:** A **basis** for a subspace V is a set of vectors that:
1. **Spans V** (can reach everything in V)
2. **Is linearly independent** (no redundancy)

**In plain English:** A basis is the smallest set of vectors you need to reach everything in the space.

**Think of it as:**
- A minimal "toolkit" for building any vector
- The fundamental "directions" in the space
- Efficient coordinates without redundancy

## Why Do We Need Bases?

**Question:** Why not just use any spanning set?

**Answer:** Bases give us unique representations!

**Example without basis (redundant):**

Vectors: **v₁** = (1, 0), **v₂** = (0, 1), **v₃** = (1, 1)

These span ℝ², but (2, 2) can be written as:
- 2**v₁** + 2**v₂** = (2, 2)
- 2**v₃** = (2, 2)
- 1**v₁** + 1**v₂** + 0**v₃** = (2, 2)

**Multiple representations! Ambiguous!**

**Example with basis (minimal):**

Vectors: **v₁** = (1, 0), **v₂** = (0, 1)

(2, 2) can ONLY be written as: 2**v₁** + 2**v₂**

**Unique representation! Clear!**

## Standard Basis

**In ℝ²:**
- **e₁** = (1, 0)
- **e₂** = (0, 1)

**In ℝ³:**
- **e₁** = (1, 0, 0)
- **e₂** = (0, 1, 0)
- **e₃** = (0, 0, 1)

**In ℝⁿ:**
- **e₁** = (1, 0, 0, ..., 0)
- **e₂** = (0, 1, 0, ..., 0)
- ...
- **eₙ** = (0, 0, 0, ..., 1)

**These are called standard basis vectors.**

**Why?**
- Linearly independent ✓
- Span all of ℝⁿ ✓
- Form a basis!

**Any vector (a, b, c) = a**e₁** + b**e₂** + c**e₃****

The coefficients (a, b, c) are just the components!

## Verifying a Basis

**To check if {**v₁, v₂, ..., vₖ**} is a basis for V:**

**Step 1:** Verify linear independence
- Only trivial solution to c₁**v₁** + ... + cₖ**vₖ** = **0**

**Step 2:** Verify spanning
- Every vector in V can be written as combination of **v₁, ..., vₖ**

**If both hold → It's a basis!**

### Example 3.1: Verifying a Basis in ℝ²

**Claim:** {(1, 2), (3, 1)} is a basis for ℝ²

**Step 1: Check independence**

Does c₁(1, 2) + c₂(3, 1) = (0, 0) only when c₁ = c₂ = 0?

Equations:
- c₁ + 3c₂ = 0
- 2c₁ + c₂ = 0

From second: c₁ = -c₂/2
Substitute into first: -c₂/2 + 3c₂ = 0 → 5c₂/2 = 0 → c₂ = 0

Then c₁ = 0.

**Only trivial solution! Independent ✓**

**Step 2: Check spanning**

Can we write any (a, b) as c₁(1, 2) + c₂(3, 1)?

Equations:
- c₁ + 3c₂ = a
- 2c₁ + c₂ = b

Solve for c₁, c₂:
From first: c₁ = a - 3c₂
Substitute: 2(a - 3c₂) + c₂ = b
2a - 6c₂ + c₂ = b
-5c₂ = b - 2a
c₂ = (2a - b)/5

Then: c₁ = a - 3(2a - b)/5 = (5a - 6a + 3b)/5 = (-a + 3b)/5

**For any (a, b), we can solve for c₁, c₂! Spans ℝ² ✓**

**Therefore: {(1, 2), (3, 1)} is a basis for ℝ²**

### Example 3.2: NOT a Basis

**Claim:** {(1, 2), (2, 4)} is NOT a basis for ℝ²

**Check independence:**

Does c₁(1, 2) + c₂(2, 4) = (0, 0)?

Notice: (2, 4) = 2(1, 2)

So: c₁(1, 2) + c₂ · 2(1, 2) = (c₁ + 2c₂)(1, 2) = (0, 0)

**Non-trivial solution:** c₁ = -2, c₂ = 1

**Not independent! ✗**

**Therefore: NOT a basis**

**Also fails spanning:** Can only reach multiples of (1, 2) - just a line, not entire ℝ²!

## Dimension

**Definition:** The **dimension** of a subspace V is the number of vectors in any basis for V.

**Remarkable fact:** All bases for the same subspace have the SAME number of vectors!

**Examples:**

**ℝ²:**
- Standard basis: {(1, 0), (0, 1)} - 2 vectors
- Another basis: {(1, 2), (3, 1)} - 2 vectors
- Any basis for ℝ²: 2 vectors
- **Dimension of ℝ² = 2**

**ℝ³:**
- Standard basis: {(1,0,0), (0,1,0), (0,0,1)} - 3 vectors
- **Dimension of ℝ³ = 3**

**Line through origin in ℝ²:**
- Basis: {(2, 1)} - 1 vector
- **Dimension = 1**

**Plane through origin in ℝ³:**
- Basis: {(1, 0, 0), (0, 1, 0)} - 2 vectors
- **Dimension = 2**

**Just the origin {**0**}:**
- Basis: empty set (no vectors needed!)
- **Dimension = 0**

## Why All Bases Have Same Size

**Theorem:** If {**v₁, ..., vₖ**} and {**w₁, ..., wₘ**} are both bases for V, then k = m.

**Intuitive proof:**

**Suppose k < m** (fewer v's than w's).

Since {**v₁, ..., vₖ**} is a basis, it spans V.

So each **wᵢ** can be written as combination of **v**'s.

But we have MORE **w**'s than **v**'s!

This means the **w**'s must be linearly dependent (more vectors than dimensions).

But **w**'s are supposed to be a basis (independent)!

**Contradiction!**

Similarly, m < k leads to contradiction.

**Therefore: k = m**

**The dimension is well-defined!**

## Finding a Basis

**Problem:** Given vectors **v₁, v₂, ..., vₖ**, find a basis for their span.

**Method:** Remove redundant (dependent) vectors!

**Algorithm:**

1. Start with empty set S = {}
2. For each vector **vᵢ**:
   - If **vᵢ** is NOT in span(S), add **vᵢ** to S
   - If **vᵢ** IS in span(S), skip it (redundant!)
3. Result: S is a basis for span{**v₁, ..., vₖ**}

### Example 3.3: Finding a Basis

**Given vectors in ℝ³:**
- **v₁** = (1, 0, 0)
- **v₂** = (1, 1, 0)
- **v₃** = (0, 1, 0)
- **v₄** = (2, 1, 0)

**Find basis for span{**v₁, v₂, v₃, v₄**}**

**Step 1:** S = {}, consider **v₁** = (1, 0, 0)
- S is empty, so add **v₁**
- S = {(1, 0, 0)}

**Step 2:** Consider **v₂** = (1, 1, 0)
- Is **v₂** in span{(1, 0, 0)}?
- Need: c(1, 0, 0) = (1, 1, 0)
- This gives c = 1 (first component) but c = ∞ (second component)
- Impossible! **v₂** NOT in span(S)
- Add **v₂**
- S = {(1, 0, 0), (1, 1, 0)}

**Step 3:** Consider **v₃** = (0, 1, 0)
- Is **v₃** in span{(1, 0, 0), (1, 1, 0)}?
- Need: c₁(1, 0, 0) + c₂(1, 1, 0) = (0, 1, 0)
- Equations: c₁ + c₂ = 0, c₂ = 1
- Solution: c₂ = 1, c₁ = -1
- Check: -1(1, 0, 0) + 1(1, 1, 0) = (0, 1, 0) ✓
- **v₃** IS in span(S) - redundant!
- Don't add, S = {(1, 0, 0), (1, 1, 0)}

**Step 4:** Consider **v₄** = (2, 1, 0)
- Is **v₄** in span{(1, 0, 0), (1, 1, 0)}?
- Need: c₁(1, 0, 0) + c₂(1, 1, 0) = (2, 1, 0)
- Equations: c₁ + c₂ = 2, c₂ = 1
- Solution: c₂ = 1, c₁ = 1
- Check: 1(1, 0, 0) + 1(1, 1, 0) = (2, 1, 0) ✓
- **v₄** IS in span(S) - redundant!
- Don't add

**Result:** Basis = {(1, 0, 0), (1, 1, 0)}

**Dimension of span = 2** (a plane in ℝ³)

**The plane is:** All vectors of form (a, b, 0) - the xy-plane!

## Coordinates with Respect to a Basis

**Once we have a basis, we can represent any vector uniquely!**

**Given:** Basis B = {**b₁, b₂, ..., bₙ**} for V

**For any **v** in V:** There exist UNIQUE scalars c₁, c₂, ..., cₙ such that:

**v** = c₁**b₁** + c₂**b₂** + ... + cₙ**bₙ**

**The coordinates of **v** with respect to basis B** are: [c₁, c₂, ..., cₙ]ᵦ

### Example 3.4: Coordinates in Different Bases

**Vector:** **v** = (3, 5) in ℝ²

**Standard basis:** E = {(1, 0), (0, 1)}

**v** = 3(1, 0) + 5(0, 1)

**Coordinates with respect to E:** [3, 5]ₑ

**Different basis:** B = {(1, 2), (2, 1)}

Find coordinates [c₁, c₂]ᵦ such that **v** = c₁(1, 2) + c₂(2, 1)

c₁(1, 2) + c₂(2, 1) = (3, 5)

Equations:
- c₁ + 2c₂ = 3
- 2c₁ + c₂ = 5

From first: c₁ = 3 - 2c₂
Substitute: 2(3 - 2c₂) + c₂ = 5
6 - 4c₂ + c₂ = 5
-3c₂ = -1
c₂ = 1/3

Then: c₁ = 3 - 2(1/3) = 3 - 2/3 = 7/3

**Coordinates with respect to B:** [7/3, 1/3]ᵦ

**Verification:** (7/3)(1, 2) + (1/3)(2, 1) = (7/3 + 2/3, 14/3 + 1/3) = (9/3, 15/3) = (3, 5) ✓

**Same vector, different coordinates!** It depends on the basis!

## Change of Basis

**Problem:** Given coordinates in basis B, find coordinates in basis C.

**Applications:**
- Computer graphics (different coordinate systems)
- Physics (different reference frames)
- Signal processing (frequency domain ↔ time domain)

**Method:** Use change of basis matrix (covered in matrix chapters)

**For now, key insight:** Same vector, different representations depending on basis!

## Dimension and Degrees of Freedom

**The dimension tells us "how many numbers do we need to specify a vector"**

**In ℝ³:**
- Dimension = 3
- Need 3 numbers (x, y, z) to specify any point
- "3 degrees of freedom"

**On a plane in ℝ³:**
- Dimension = 2
- Need only 2 numbers (constrained to plane)
- "2 degrees of freedom"

**On a line in ℝ³:**
- Dimension = 1
- Need only 1 number (position along line)
- "1 degree of freedom"

**Machine learning connection:**

**High-dimensional data:** 1000 features
**Actual dimension:** Maybe only 20 (via PCA)

**Meaning:** Data actually lives on 20-dimensional subspace!
- 980 dimensions are redundant
- Only 20 "degrees of freedom" matter
- Can represent with 20 numbers instead of 1000!

## Applications to Machine Learning

### Dimensionality Reduction

**Problem:** Data in ℝ¹⁰⁰⁰ (1000 features)

**Reality:** Data lies near a low-dimensional subspace (say, dimension 50)

**PCA process:**
1. Find basis for this 50-dimensional subspace
2. Represent each data point using coordinates in this basis
3. Now data is effectively in ℝ⁵⁰!

**Benefits:**
- Faster computation
- Less storage
- Better visualization
- Reduced overfitting

### Feature Selection vs Feature Extraction

**Feature Selection:** Choose subset of original features
- Keep 50 out of 1000 original features
- Interpretable (still original features)

**Feature Extraction (PCA):** Create new basis
- Find 50 new "features" (linear combinations of originals)
- New features are basis vectors!
- Less interpretable, but more efficient

**Both reduce dimension, different approaches!**

### Rank of Data Matrix

**Data matrix X:** n samples × d features

**Rank = dimension of column space**

**If rank < d:**
- Features are linearly dependent
- True dimensionality < d
- Redundancy in features

**If rank = d:**
- Features are independent
- Using full dimensionality

**Finding rank → Finding a basis for column space!**

## Practice Problems - Basis and Dimension

**Problem 3.1: Verifying Bases**

Determine if these are bases for ℝ²:

a) {(1, 0), (0, 1)}
b) {(1, 1), (1, -1)}
c) {(2, 4), (1, 2)}
d) {(1, 0)}
e) {(1, 0), (0, 1), (1, 1)}

**Problem 3.2: Finding Bases**

Find a basis for the span of:

a) {(1, 2, 3), (2, 4, 6), (1, 1, 1)}
b) {(1, 0, 0), (1, 1, 0), (1, 1, 1)}
c) {(1, 2), (2, 4), (3, 6), (1, 1)}
d) {(1, 0, 1), (0, 1, 0), (2, 0, 2), (0, 2, 0)}

**Problem 3.3: Dimension**

Find the dimension of:

a) span{(1, 2, 3), (4, 5, 6)}
b) span{(1, 0, 0), (0, 1, 0), (1, 1, 0)}
c) The set of all vectors (a, b, c) where a + b + c = 0
d) The set of all vectors (a, 2a, 3a)

**Problem 3.4: Coordinates**

Vector **v** = (5, 7) in ℝ²

Basis B = {(1, 2), (3, 1)}

a) Find coordinates [c₁, c₂]ᵦ
b) Verify: c₁(1, 2) + c₂(3, 1) = (5, 7)
c) Find coordinates with respect to standard basis
d) Why are the coordinates different?

**Problem 3.5: Subspace Dimension**

In ℝ⁴, consider subspace of all vectors (a, b, c, d) where:
- a + b = 0
- c - d = 0

a) Find a basis for this subspace
b) What is the dimension?
c) Interpret: How many "free" variables?

**Problem 3.6: Application - Feature Reduction**

Dataset: 100 samples, 20 features

Analysis shows: rank of data matrix = 12

a) What does this mean?
b) How many features are truly independent?
c) Can you represent data with fewer features? How many?
d) What technique would you use?

**Problem 3.7: Extending to a Basis**

Given: {(1, 0, 0), (0, 1, 0)} (two vectors in ℝ³)

a) Do these form a basis for ℝ³? Why not?
b) What do they span?
c) Find a third vector to complete a basis for ℝ³
d) Is your choice unique?

**Problem 3.8: Basis Properties**

Prove or disprove:

a) If {**v₁, v₂**} is a basis for ℝ², then {2**v₁**, 2**v₂**} is also a basis
b) If {**v₁, v₂**} is a basis for ℝ², then {**v₁ + v₂**, **v₁ - v₂**} is also a basis
c) Every subspace of ℝⁿ has a unique basis
d) Every basis for ℝⁿ has exactly n vectors

---

**End of Section 5.3: Basis and Dimension**

Key takeaways:
- ✅ Basis = minimal spanning set (independent + spans)
- ✅ Provides unique representation for every vector
- ✅ All bases for same space have same size (dimension)
- ✅ Dimension = degrees of freedom in the space
- ✅ Coordinates depend on choice of basis
- ✅ Critical for dimensionality reduction in ML
- ✅ Basis vectors are the "fundamental directions"

Next: Section 5.4 - Orthogonality

# Chapter 5: Section 4 - Orthogonality

<a name="orthogonality"></a>
# 4. Orthogonality

## The Core Question: When Are Vectors Perpendicular?

Imagine you're building a house. You want to place walls at right angles (90°) to each other.

**Question:** How do you check if two walls are perpendicular?

**In 2D:** Use a carpenter's square
**In 3D:** Use geometry
**In higher dimensions?** We need mathematics!

**This is the concept of ORTHOGONALITY - the mathematical notion of "perpendicular."**

## What Does Orthogonal Mean?

**Definition:** Two vectors **u** and **v** are **orthogonal** if their inner product (dot product) is zero:

**u** · **v** = 0

**In plain English:** Vectors are orthogonal if they meet at a right angle (90°).

**Why this definition?**

Recall the dot product formula:
**u** · **v** = ||**u**|| ||**v**|| cos(θ)

where θ is the angle between them.

**If **u** · **v** = 0:**
- Then: ||**u**|| ||**v**|| cos(θ) = 0
- Since lengths are non-zero: cos(θ) = 0
- Therefore: θ = 90°

**Perpendicular!**

## Simple Examples

### Example 4.1: Orthogonal Vectors in 2D

**Vectors:**
- **u** = (1, 0)
- **v** = (0, 1)

**Check:** **u** · **v** = 1(0) + 0(1) = 0 ✓

**Orthogonal!** (x-axis perpendicular to y-axis)

### Example 4.2: Non-Orthogonal Vectors

**Vectors:**
- **u** = (1, 1)
- **v** = (1, 0)

**Check:** **u** · **v** = 1(1) + 1(0) = 1 ≠ 0 ✗

**Not orthogonal!** (angle is 45°, not 90°)

### Example 4.3: Orthogonal Vectors in 3D

**Vectors:**
- **u** = (1, 0, 0)
- **v** = (0, 1, 0)
- **w** = (0, 0, 1)

**Check all pairs:**
- **u** · **v** = 0 ✓
- **u** · **w** = 0 ✓
- **v** · **w** = 0 ✓

**All mutually orthogonal!** (Standard basis vectors)

### Example 4.4: Finding an Orthogonal Vector

**Given:** **u** = (3, 4)

**Find:** **v** orthogonal to **u**

**Method:** Need **u** · **v** = 0

Let **v** = (a, b)

3a + 4b = 0

**Solutions:** b = -3a/4

**Choose a = 4:** Then b = -3

**One solution:** **v** = (4, -3)

**Check:** (3, 4) · (4, -3) = 12 - 12 = 0 ✓

**Another solution:** **v** = (-4, 3) (or any multiple!)

**Key insight:** Many vectors orthogonal to a given vector!

## Geometric Intuition

**In 2D:**
- Given a vector **u**
- All vectors orthogonal to **u** form a line through origin
- This line is perpendicular to **u**

**In 3D:**
- Given a vector **u**
- All vectors orthogonal to **u** form a plane through origin
- This plane is perpendicular to **u**

**In ℝⁿ:**
- Given a vector **u**
- All vectors orthogonal to **u** form an (n-1)-dimensional subspace
- This is called the **orthogonal complement**

## Orthogonal Complement

**Definition:** Given a subspace W, the **orthogonal complement** W⊥ (read "W perp") is:

W⊥ = {**v** : **v** · **w** = 0 for all **w** in W}

**In plain English:** All vectors perpendicular to everything in W.

### Example 4.5: Orthogonal Complement in ℝ³

**Subspace W:** The xy-plane = span{(1, 0, 0), (0, 1, 0)}

**W consists of:** All vectors (a, b, 0)

**Question:** What is W⊥?

**Answer:** All vectors perpendicular to every vector in xy-plane.

**Test:** For **v** = (x, y, z) to be in W⊥:
- **v** · (1, 0, 0) = x = 0
- **v** · (0, 1, 0) = y = 0
- Therefore: **v** = (0, 0, z)

**W⊥ = z-axis!**

**Beautiful fact:** xy-plane ⊥ z-axis

**General rule:** 
- dim(W) + dim(W⊥) = n (in ℝⁿ)
- dim(xy-plane) = 2, dim(z-axis) = 1, total = 3 ✓

## Orthogonal Sets

**Definition:** A set of vectors {**v₁, v₂, ..., vₖ**} is **orthogonal** if every pair is orthogonal:

**vᵢ** · **vⱼ** = 0 for all i ≠ j

**Example:** Standard basis {(1,0,0), (0,1,0), (0,0,1)} is orthogonal!

**Why orthogonal sets are special:**

1. **Automatically linearly independent!**
   - If vectors are orthogonal (and non-zero), they're independent
   - No redundancy

2. **Easy to find coordinates!**
   - Simple formulas (coming up!)

3. **Numerical stability**
   - Better for computation

### Theorem: Orthogonal Sets are Independent

**Claim:** If {**v₁, v₂, ..., vₖ**} is orthogonal (all non-zero), then it's linearly independent.

**Proof:**

Suppose c₁**v₁** + c₂**v₂** + ... + cₖ**vₖ** = **0**

Take dot product with **v₁**:
**v₁** · (c₁**v₁** + c₂**v₂** + ... + cₖ**vₖ**) = **v₁** · **0** = 0

c₁(**v₁** · **v₁**) + c₂(**v₁** · **v₂**) + ... + cₖ(**v₁** · **vₖ**) = 0

Since orthogonal: **v₁** · **vⱼ** = 0 for j ≠ 1

c₁||**v₁**||² = 0

Since **v₁** ≠ **0**: ||**v₁**|| ≠ 0, so c₁ = 0

Similarly for all cᵢ: c₁ = c₂ = ... = cₖ = 0

**Therefore: independent!**

## Orthonormal Sets

**Definition:** A set is **orthonormal** if:
1. Orthogonal (all pairs perpendicular)
2. Each vector has length 1 (unit vectors)

**Formal:** {**v₁, v₂, ..., vₖ**} is orthonormal if:

**vᵢ** · **vⱼ** = {1 if i = j, 0 if i ≠ j}

**Example:** Standard basis is orthonormal!
- (1,0,0) · (1,0,0) = 1 ✓
- (1,0,0) · (0,1,0) = 0 ✓
- All have length 1 ✓

### Creating Orthonormal from Orthogonal

**Given:** Orthogonal set {**v₁, v₂, ..., vₖ**}

**Create orthonormal:** Normalize each vector!

**uᵢ** = **vᵢ** / ||**vᵢ**||

**Example:**

Orthogonal set: {(3, 0), (0, 4)}

Normalize:
- **u₁** = (3, 0) / ||(3, 0)|| = (3, 0) / 3 = (1, 0)
- **u₂** = (0, 4) / ||(0, 4)|| = (0, 4) / 4 = (0, 1)

Orthonormal set: {(1, 0), (0, 1)}

## Orthonormal Bases

**Definition:** An **orthonormal basis** is a basis that is also orthonormal.

**Properties:**
- Spans the space ✓
- Linearly independent ✓ (automatic from orthogonal!)
- All vectors perpendicular to each other ✓
- All vectors have length 1 ✓

**Why we love orthonormal bases:**

### Benefit 1: Easy Coordinate Calculation

**Given:** Orthonormal basis B = {**u₁, u₂, ..., uₙ**}

**For any vector **v**:** The coordinates are:

cᵢ = **v** · **uᵢ**

**That's it! Just dot products!**

**Example:**

Orthonormal basis: {(1, 0), (0, 1)}
Vector: **v** = (3, 5)

Coordinates:
- c₁ = (3, 5) · (1, 0) = 3
- c₂ = (3, 5) · (0, 1) = 5

**v** = 3(1, 0) + 5(0, 1) ✓

**Compare to non-orthonormal basis:**
- Need to solve system of equations
- Much harder!

### Benefit 2: Pythagorean Theorem

**For orthonormal basis:**

||**v**||² = c₁² + c₂² + ... + cₙ²

**Just like Pythagorean theorem in higher dimensions!**

### Benefit 3: Projections

**Project **v** onto **u** (where **u** is unit vector):**

proj_**u**(**v**) = (**v** · **u**)**u**

**Simple formula!**

## Orthogonal Projection

**Problem:** Given vector **v** and subspace W, find the point in W closest to **v**.

**Solution:** Orthogonal projection of **v** onto W.

**Geometric picture in 2D:**

```
    **v**
    /|
   / |
  /  | (perpendicular)
 /   |
/_____| **proj_W(v)**
   W (a line)
```

**The projection is the "shadow" of **v** onto W!**

### Formula for Projection onto Line

**Given:** Line spanned by unit vector **u**

**Project **v** onto this line:**

proj_**u**(**v**) = (**v** · **u**)**u**

**If **u** not unit:** 

proj_**u**(**v**) = (**v** · **u** / ||**u**||²)**u**

### Example 4.6: Projection onto a Line

**Vector:** **v** = (3, 4)
**Line:** Spanned by **u** = (1, 0)

**Project:**

proj_**u**(**v**) = (**v** · **u**)**u** = [(3, 4) · (1, 0)](1, 0) = 3(1, 0) = (3, 0)

**Geometric meaning:** Drop perpendicular from (3, 4) to x-axis, land at (3, 0)!

**Component perpendicular to **u**:**

**v** - proj_**u**(**v**) = (3, 4) - (3, 0) = (0, 4)

**Verify orthogonal:** (0, 4) · (1, 0) = 0 ✓

### Formula for Projection onto Subspace

**Given:** Orthonormal basis {**u₁, u₂, ..., uₖ**} for subspace W

**Project **v** onto W:**

proj_W(**v**) = (**v** · **u₁**)**u₁** + (**v** · **u₂**)**u₂** + ... + (**v** · **uₖ**)**uₖ**

**Just sum of projections onto each basis vector!**

### Example 4.7: Projection onto Plane

**Subspace W:** xy-plane in ℝ³

**Orthonormal basis:** {(1, 0, 0), (0, 1, 0)}

**Vector:** **v** = (3, 4, 5)

**Project onto W:**

proj_W(**v**) = [**v** · (1,0,0)](1,0,0) + [**v** · (0,1,0)](0,1,0)
           = 3(1,0,0) + 4(0,1,0)
           = (3, 4, 0)

**Geometric meaning:** Drop perpendicular from (3, 4, 5) to xy-plane, land at (3, 4, 0)!

**Component perpendicular to W:**

**v** - proj_W(**v**) = (3, 4, 5) - (3, 4, 0) = (0, 0, 5)

**This is in W⊥ (the z-axis)!**

## Why Orthogonality Matters in ML

### 1. Principal Component Analysis (PCA)

**Goal:** Find orthogonal directions of maximum variance

**Why orthogonal?**
- Each direction captures independent information
- No redundancy
- Easy to interpret components

**Process:**
1. Find first direction of max variance → **u₁**
2. Find second direction orthogonal to **u₁** with max variance → **u₂**
3. Continue...

**Result:** Orthonormal basis aligned with data structure!

### 2. Feature Engineering

**Problem:** Features are correlated (not orthogonal)

**Solution:** Transform to orthogonal features
- PCA creates orthogonal features
- Whitening decorrelates features
- Better for many algorithms

**Example:**
- Original: Height and Weight (correlated)
- Transform: Size and Shape (orthogonal)

### 3. Least Squares Regression

**Problem:** Find best fit line/plane

**Solution involves:** Orthogonal projection!
- Project data onto model space
- Residuals are orthogonal to model space
- Minimizes squared error

**The "least squares" solution is the orthogonal projection of y onto column space of X!**

### 4. QR Decomposition

**Any matrix A can be factored:**

A = QR

where:
- Q has orthonormal columns
- R is upper triangular

**Uses:**
- Solving linear systems (numerically stable)
- Least squares regression
- Computing eigenvalues

### 5. Orthogonal Matrices

**Definition:** Matrix Q is **orthogonal** if its columns are orthonormal.

**Properties:**
- Q^T Q = I (preserves dot products!)
- ||Q**x**|| = ||**x**|| (preserves lengths!)
- Q represents rotation/reflection

**Applications:**
- Rotations in computer graphics
- Change of orthonormal basis
- Preserving geometry

## Computing with Orthonormal Bases

### Example 4.8: Complete Workflow

**Task:** Given vectors, create orthonormal basis

**Vectors in ℝ³:**
- **v₁** = (1, 1, 0)
- **v₂** = (1, 0, 1)

**Step 1:** Normalize **v₁**

||**v₁**|| = √(1² + 1² + 0²) = √2

**u₁** = **v₁**/||**v₁**|| = (1, 1, 0)/√2 = (1/√2, 1/√2, 0)

**Step 2:** Make **v₂** orthogonal to **u₁**

**w₂** = **v₂** - proj_**u₁**(**v₂**)

proj_**u₁**(**v₂**) = [**v₂** · **u₁**]**u₁**

**v₂** · **u₁** = (1, 0, 1) · (1/√2, 1/√2, 0) = 1/√2

proj_**u₁**(**v₂**) = (1/√2)(1/√2, 1/√2, 0) = (1/2, 1/2, 0)

**w₂** = (1, 0, 1) - (1/2, 1/2, 0) = (1/2, -1/2, 1)

**Step 3:** Normalize **w₂**

||**w₂**|| = √[(1/2)² + (-1/2)² + 1²] = √[1/4 + 1/4 + 1] = √(3/2)

**u₂** = **w₂**/||**w₂**|| = (1/2, -1/2, 1)/√(3/2) = (1/√6, -1/√6, 2/√6)

**Result:** Orthonormal set {**u₁, u₂**}

**Verify orthonormal:**
- **u₁** · **u₂** = (1/√2)(1/√6) + (1/√2)(-1/√6) + 0(2/√6) = 0 ✓
- ||**u₁**|| = 1 ✓
- ||**u₂**|| = 1 ✓

**This is the Gram-Schmidt process!** (Next section)

## Practice Problems - Orthogonality

**Problem 4.1: Testing Orthogonality**

Are these vectors orthogonal?

a) (1, 2) and (2, -1)
b) (1, 0, 0) and (0, 0, 1)
c) (1, 1, 1) and (1, -1, 0)
d) (3, 4) and (4, -3)

**Problem 4.2: Finding Orthogonal Vectors**

Find a vector orthogonal to:

a) (2, 3)
b) (1, 2, 3)
c) (1, 1, 1, 1)

Give at least two different solutions for each.

**Problem 4.3: Orthogonal Complement**

Find W⊥ for:

a) W = span{(1, 0, 0)} in ℝ³
b) W = span{(1, 1, 0), (0, 1, 1)} in ℝ³
c) W = span{(1, 2)} in ℝ²

**Problem 4.4: Projection Calculations**

Project **v** onto **u**:

a) **v** = (4, 3), **u** = (1, 0)
b) **v** = (1, 2, 3), **u** = (1, 0, 0)
c) **v** = (3, 4), **u** = (3, 4) (normalized first!)

**Problem 4.5: Orthonormal Sets**

Which are orthonormal?

a) {(1, 0), (0, 1)}
b) {(1/√2, 1/√2), (1/√2, -1/√2)}
c) {(2, 0), (0, 2)}
d) {(1, 0, 0), (0, 1, 0), (0, 0, 1)}

**Problem 4.6: Creating Orthonormal Basis**

Given orthogonal set {(3, 0), (0, 4)}, create orthonormal set.

**Problem 4.7: Coordinate Calculation**

Orthonormal basis: B = {(1/√2, 1/√2), (1/√2, -1/√2)}
Vector: **v** = (3, 1)

a) Find coordinates [c₁, c₂]_B
b) Verify: c₁**u₁** + c₂**u₂** = **v**
c) Calculate ||**v**||² using coordinates

**Problem 4.8: Projection onto Subspace**

**v** = (1, 2, 3)
W = span{(1, 0, 0), (0, 1, 0)} (xy-plane)

a) Find proj_W(**v**)
b) Find component orthogonal to W
c) Verify orthogonality

**Problem 4.9: Application - Decorrelating Features**

Features: **x₁** = (1, 2, 3), **x₂** = (2, 3, 4)

These are correlated: **x₁** · **x₂** = 20 ≠ 0

a) Create orthogonal features using **x₁** and **x₂** - proj_**x₁**(**x₂**)
b) Verify new features are orthogonal
c) Why is this useful in ML?

**Problem 4.10: QR-like Decomposition**

Given: **v₁** = (1, 1), **v₂** = (1, -1)

a) Are they orthogonal?
b) Create orthonormal basis from them
c) Express (5, 3) in this new basis

---

**End of Section 5.4: Orthogonality**

Key takeaways:
- ✅ Orthogonal = perpendicular (dot product = 0)
- ✅ Orthogonal sets are automatically linearly independent
- ✅ Orthonormal = orthogonal + unit length
- ✅ Orthonormal bases make calculations easy
- ✅ Projections onto subspaces via orthogonal decomposition
- ✅ Critical for PCA, regression, and numerical stability
- ✅ Foundation for Gram-Schmidt process

Next: Section 5.5 - Gram-Schmidt Process
