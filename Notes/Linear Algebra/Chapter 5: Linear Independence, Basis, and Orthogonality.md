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

# Chapter 5: Section 5 - Gram-Schmidt Process

<a name="gram-schmidt"></a>
# 5. Gram-Schmidt Process

## The Core Question: How Do We Build Perpendicular Directions?

Imagine you're in a dark room trying to understand its shape. You have a flashlight.

**Your strategy:**
- First, shine the light straight ahead → that's one direction
- Next, you want to explore a NEW direction, but it should be completely different from the first
- Then another direction that's different from both previous ones
- And so on...

**Question:** How do you ensure each new direction is truly "different" (perpendicular) from all previous ones?

**Problem without perpendicular directions:**
If your second direction partially overlaps with the first, you're wasting effort exploring areas you've already seen!

**Solution:** Remove the "overlap" before exploring the new direction!

**This is exactly what the Gram-Schmidt process does!**

## What Problem Does Gram-Schmidt Solve?

**The Problem We Face:**

You have a set of linearly independent vectors: **{v₁, v₂, v₃, ..., vₖ}**

But they're NOT orthogonal - they point in "messy" directions with overlap.

**What we want:**
- Keep the same span (reach the same space)
- But use orthogonal vectors instead
- Even better: orthonormal vectors!

**Why do we want this?**
- Easier calculations (as we saw in Section 4)
- Numerical stability in computers
- Clearer geometric understanding
- Foundation for many ML algorithms

**Real-world analogy:**
- **Before:** You have k crooked, overlapping rulers
- **After:** You have k perfectly perpendicular rulers measuring the same space
- Same space covered, but much cleaner system!

## The Root Cause: Why Are Vectors Not Orthogonal?

**Let's think about what "not orthogonal" means:**

Take two vectors **v₁** and **v₂** where **v₁ · v₂ ≠ 0**

**Question:** Why is their dot product non-zero?

**Answer:** Because **v₂** has a component in the direction of **v₁**!

**Geometric picture:**
```
        v₂
       /
      /
     /______ projection of v₂ onto v₁
    /
   v₁
```

**The projection of v₂ onto v₁** is the "overlap" - the part of **v₂** that goes in the same direction as **v₁**.

**Root cause:** This overlap makes them non-orthogonal!

## So How Can We Remove This Overlap?

**Natural question:** If the overlap is the problem, can we just... remove it?

**Yes! That's the brilliant insight!**

**If we have:**
- **v₂** = (part parallel to **v₁**) + (part perpendicular to **v₁**)

**Then we can isolate the perpendicular part:**
- Perpendicular part = **v₂** - (part parallel to **v₁**)
- Perpendicular part = **v₂** - proj**ᵥ₁**(**v₂**)

**This perpendicular part is orthogonal to v₁!**

**Let's verify:**
- Let **w₂** = **v₂** - proj**ᵥ₁**(**v₂**)
- **w₂ · v₁** = (**v₂** - proj**ᵥ₁**(**v₂**)) · **v₁**
- = **v₂ · v₁** - proj**ᵥ₁**(**v₂**) · **v₁**

Now, proj**ᵥ₁**(**v₂**) = ((**v₂ · v₁**) / ||**v₁**||²)**v₁**

So: proj**ᵥ₁**(**v₂**) · **v₁** = ((**v₂ · v₁**) / ||**v₁**||²) **v₁ · v₁**
                                  = ((**v₂ · v₁**) / ||**v₁**||²) ||**v₁**||²
                                  = **v₂ · v₁**

Therefore: **w₂ · v₁** = **v₂ · v₁** - **v₂ · v₁** = 0 ✓

**Beautiful! The overlap is removed!**

## The Gram-Schmidt Process: Step by Step

**Goal:** Convert linearly independent set {**v₁, v₂, ..., vₖ**} into orthogonal set {**u₁, u₂, ..., uₖ**}

**The algorithm:**

**Step 1:** Keep the first vector as is
- **u₁** = **v₁**
- (Nothing to make it orthogonal to yet!)

**Step 2:** Make **v₂** orthogonal to **u₁**
- **u₂** = **v₂** - proj**ᵤ₁**(**v₂**)
- Remove the component of **v₂** in the direction of **u₁**

**Step 3:** Make **v₃** orthogonal to BOTH **u₁** AND **u₂**
- **u₃** = **v₃** - proj**ᵤ₁**(**v₃**) - proj**ᵤ₂**(**v₃**)
- Remove components in both previous directions

**Step 4:** Continue this pattern...
- **u₄** = **v₄** - proj**ᵤ₁**(**v₄**) - proj**ᵤ₂**(**v₄**) - proj**ᵤ₃**(**v₄**)

**General step i:**
- **uᵢ** = **vᵢ** - Σⱼ₌₁^(i-1) proj**ᵤⱼ**(**vᵢ**)
- Remove ALL overlaps with previous orthogonal vectors

**Optional final step:** Normalize to get orthonormal
- **q₁** = **u₁** / ||**u₁**||
- **q₂** = **u₂** / ||**u₂**||
- etc.

## Why Does This Work?

**Let's think through why each new vector is orthogonal to all previous ones:**

**After Step 2:** Is **u₂ ⊥ u₁**?
- Yes! We specifically removed the **u₁** component from **v₂**

**After Step 3:** Is **u₃ ⊥ u₁** and **u₃ ⊥ u₂**?
- We removed BOTH the **u₁** component AND the **u₂** component from **v₃**
- So **u₃** is perpendicular to both!

**The pattern continues:** Each new vector has ALL previous components removed, so it's perpendicular to ALL previous vectors.

**Key insight:** By systematically removing overlaps, we build perpendicular directions one at a time!

## Detailed Example 5.1: Gram-Schmidt in ℝ²

**Given vectors:**
- **v₁** = (3, 1)
- **v₂** = (2, 2)

**Goal:** Create orthogonal set {**u₁, u₂**}

### Step 1: First vector

**u₁** = **v₁** = (3, 1)

### Step 2: Make v₂ orthogonal to u₁

**Calculate projection of v₂ onto u₁:**

proj**ᵤ₁**(**v₂**) = ((**v₂ · u₁**) / ||**u₁**||²) **u₁**

**v₂ · u₁** = (2)(3) + (2)(1) = 6 + 2 = 8

||**u₁**||² = 3² + 1² = 9 + 1 = 10

proj**ᵤ₁**(**v₂**) = (8/10)(3, 1) = (4/5)(3, 1) = (12/5, 4/5)

**Remove the overlap:**

**u₂** = **v₂** - proj**ᵤ₁**(**v₂**)
      = (2, 2) - (12/5, 4/5)
      = (10/5 - 12/5, 10/5 - 4/5)
      = (-2/5, 6/5)

### Verify orthogonality:

**u₁ · u₂** = (3)(-2/5) + (1)(6/5) = -6/5 + 6/5 = 0 ✓

**Perfect! They're orthogonal!**

### Geometric interpretation:

**Before:** **v₁** and **v₂** pointed in somewhat similar directions (not perpendicular)

**After:** **u₁** and **u₂** are exactly perpendicular

**Same span:** span{**v₁, v₂**} = span{**u₁, u₂**} = all of ℝ²

### Optional: Create orthonormal basis

**Normalize u₁:**

||**u₁**|| = √(9 + 1) = √10

**q₁** = (3, 1) / √10 = (3/√10, 1/√10)

**Normalize u₂:**

||**u₂**|| = √(4/25 + 36/25) = √(40/25) = √(8/5) = 2√(2/5) = 2/√5

**q₂** = (-2/5, 6/5) / (2/√5) = (-2/5, 6/5) × (√5/2) = (-√5/5, 3√5/5)

**Verify orthonormal:**
- **q₁ · q₂** = (3/√10)(-√5/5) + (1/√10)(3√5/5) = -3√5/(5√10) + 3√5/(5√10) = 0 ✓
- ||**q₁**|| = 1 ✓
- ||**q₂**|| = 1 ✓

## Detailed Example 5.2: Gram-Schmidt in ℝ³

**Given vectors:**
- **v₁** = (1, 1, 0)
- **v₂** = (1, 0, 1)  
- **v₃** = (0, 1, 1)

**Goal:** Create orthogonal set {**u₁, u₂, u₃**}

### Step 1: First vector

**u₁** = **v₁** = (1, 1, 0)

### Step 2: Make v₂ orthogonal to u₁

**Calculate projection:**

**v₂ · u₁** = (1)(1) + (0)(1) + (1)(0) = 1

||**u₁**||² = 1² + 1² + 0² = 2

proj**ᵤ₁**(**v₂**) = (1/2)(1, 1, 0) = (1/2, 1/2, 0)

**Remove overlap:**

**u₂** = **v₂** - proj**ᵤ₁**(**v₂**)
      = (1, 0, 1) - (1/2, 1/2, 0)
      = (1/2, -1/2, 1)

**Verify:** **u₁ · u₂** = (1)(1/2) + (1)(-1/2) + (0)(1) = 1/2 - 1/2 = 0 ✓

### Step 3: Make v₃ orthogonal to BOTH u₁ and u₂

**Calculate projection onto u₁:**

**v₃ · u₁** = (0)(1) + (1)(1) + (1)(0) = 1

proj**ᵤ₁**(**v₃**) = (1/2)(1, 1, 0) = (1/2, 1/2, 0)

**Calculate projection onto u₂:**

**v₃ · u₂** = (0)(1/2) + (1)(-1/2) + (1)(1) = 0 - 1/2 + 1 = 1/2

||**u₂**||² = (1/2)² + (-1/2)² + 1² = 1/4 + 1/4 + 1 = 3/2

proj**ᵤ₂**(**v₃**) = (1/2)/(3/2) × (1/2, -1/2, 1) = (1/3)(1/2, -1/2, 1) = (1/6, -1/6, 1/3)

**Remove BOTH overlaps:**

**u₃** = **v₃** - proj**ᵤ₁**(**v₃**) - proj**ᵤ₂**(**v₃**)
      = (0, 1, 1) - (1/2, 1/2, 0) - (1/6, -1/6, 1/3)
      = (0 - 1/2 - 1/6, 1 - 1/2 + 1/6, 1 - 0 - 1/3)
      = (-3/6 - 1/6, 3/6 + 1/6, 3/3 - 1/3)
      = (-4/6, 4/6, 2/3)
      = (-2/3, 2/3, 2/3)

### Verify orthogonality:

**u₁ · u₃** = (1)(-2/3) + (1)(2/3) + (0)(2/3) = -2/3 + 2/3 = 0 ✓

**u₂ · u₃** = (1/2)(-2/3) + (-1/2)(2/3) + (1)(2/3) = -1/3 - 1/3 + 2/3 = 0 ✓

**Perfect! All three are mutually orthogonal!**

### Result:

**Orthogonal basis:** {(1, 1, 0), (1/2, -1/2, 1), (-2/3, 2/3, 2/3)}

**These span the same space as the original vectors, but are perpendicular!**

## The Pattern Behind Gram-Schmidt

**Let's step back and see the beautiful pattern:**

**Question to yourself:** "I have a new vector **vᵢ**. How do I make it orthogonal to all my previous orthogonal vectors **u₁, u₂, ..., uᵢ₋₁**?"

**Answer:** "Remove the parts that overlap with each previous vector!"

**How do we find these overlapping parts?** 
- Use projections! proj**ᵤⱼ**(**vᵢ**) gives us the component of **vᵢ** in the direction of **uⱼ**

**What's the root cause of non-orthogonality?**
- The new vector **vᵢ** has components pointing in the same directions as previous vectors

**How does Gram-Schmidt fix it?**
- By systematically subtracting out ALL these overlapping components, leaving only the "new" direction

**Why does this give us orthogonal vectors?**
- After removing all components in previous directions, what's left MUST be perpendicular to all previous vectors!

## Common Mistakes and How to Avoid Them

### Mistake 1: Projecting onto original vectors instead of orthogonalized ones

**Wrong:**
```
u₃ = v₃ - proj_v₁(v₃) - proj_v₂(v₃)  ❌
```

**Right:**
```
u₃ = v₃ - proj_u₁(v₃) - proj_u₂(v₃)  ✓
```

**Why it matters:** The **u** vectors are already orthogonal, but the **v** vectors are not! We must project onto the orthogonal vectors we've already created.

### Mistake 2: Forgetting to include all previous projections

**Wrong:**
```
u₃ = v₃ - proj_u₂(v₃)  ❌ (forgot u₁!)
```

**Right:**
```
u₃ = v₃ - proj_u₁(v₃) - proj_u₂(v₃)  ✓
```

**Why it matters:** We need to remove overlaps with ALL previous vectors, not just the most recent one!

### Mistake 3: Using incorrect projection formula

**Remember:** proj**ᵤ**(**v**) = ((**v · u**) / ||**u**||²) **u**

**If u is already normalized (unit vector):**
proj**ᵤ**(**v**) = (**v · u**) **u**  (simpler!)

### Mistake 4: Not verifying orthogonality

**Always check:** **uᵢ · uⱼ** = 0 for all i ≠ j

If not zero, you made a calculation error!

## Modified Gram-Schmidt (More Numerically Stable)

**Classical Gram-Schmidt** (what we've shown):
- Calculate all projections at once
- **uᵢ** = **vᵢ** - Σⱼ₌₁^(i-1) proj**ᵤⱼ**(**vᵢ**)

**Modified Gram-Schmidt** (better for computers):
- Update vector after each projection
- More stable when dealing with nearly-dependent vectors

**Algorithm:**
```
u₁ = v₁

For i = 2 to k:
    uᵢ = vᵢ
    For j = 1 to i-1:
        uᵢ = uᵢ - proj_uⱼ(uᵢ)  // Update uᵢ immediately
```

**Why is this better?**
- Reduces accumulation of rounding errors
- Each subtraction uses the most up-to-date vector
- More accurate when vectors are almost dependent

**In practice:** Computers use Modified Gram-Schmidt for numerical stability

## Applications in Machine Learning

### 1. QR Decomposition

**Any matrix A can be factored as:**

A = QR

where:
- **Q** has orthonormal columns (from Gram-Schmidt!)
- **R** is upper triangular

**How Gram-Schmidt creates QR:**
- Columns of A are our original vectors **v₁, v₂, ...**
- Apply Gram-Schmidt → get orthogonal vectors **u₁, u₂, ...**
- Normalize → get orthonormal vectors (columns of Q)
- The coefficients used form R

**Uses of QR:**
- Solving least squares problems
- Computing eigenvalues
- Numerically stable matrix computations

### 2. Orthogonalizing Features

**Problem:** You have correlated features in your dataset

**Example:**
- Feature 1: House area
- Feature 2: Number of rooms (highly correlated with area)

**Solution:** Apply Gram-Schmidt!
- Keep Feature 1 as is
- Transform Feature 2 to be orthogonal to Feature 1
- Now they capture independent information!

**Benefits:**
- Removes multicollinearity
- Each feature adds unique information
- Better for regression models

### 3. Principal Component Analysis (PCA)

**PCA finds orthogonal directions of maximum variance**

**Connection to Gram-Schmidt:**
- PCA computes eigenvectors of covariance matrix
- These eigenvectors are orthogonal
- If they weren't, we'd use Gram-Schmidt to make them so!
- Gram-Schmidt ensures orthogonality in numerical implementations

### 4. Conjugate Gradient Method

**For solving large systems Ax = b:**
- Creates sequence of search directions
- These directions must be orthogonal (actually "conjugate")
- Uses Gram-Schmidt-like process to ensure orthogonality
- Much faster than standard methods for huge systems

### 5. Signal Processing

**Fourier transforms decompose signals into orthogonal components**

**Why orthogonal basis matters:**
- Each frequency component is independent
- No interference between components
- Easy to filter specific frequencies
- Gram-Schmidt ensures clean separation

## Example 5.3: Application to Data

**Dataset:** 3 measurements from 4 experiments

```
v₁ = (1, 2, 1, 0) - Measurement 1
v₂ = (1, 3, 1, 0) - Measurement 2  
v₃ = (1, 2, 0, 1) - Measurement 3
```

**These are correlated! Let's orthogonalize them.**

### Step 1: Keep v₁

**u₁** = (1, 2, 1, 0)

### Step 2: Orthogonalize v₂

**v₂ · u₁** = 1 + 6 + 1 + 0 = 8
||**u₁**||² = 1 + 4 + 1 + 0 = 6

proj**ᵤ₁**(**v₂**) = (8/6)(1, 2, 1, 0) = (4/3, 8/3, 4/3, 0)

**u₂** = (1, 3, 1, 0) - (4/3, 8/3, 4/3, 0) = (-1/3, 1/3, -1/3, 0)

### Step 3: Orthogonalize v₃

**v₃ · u₁** = 1 + 4 + 0 + 0 = 5
proj**ᵤ₁**(**v₃**) = (5/6)(1, 2, 1, 0) = (5/6, 10/6, 5/6, 0)

**v₃ · u₂** = -1/3 + 2/3 + 0 + 0 = 1/3
||**u₂**||² = 1/9 + 1/9 + 1/9 + 0 = 3/9 = 1/3

proj**ᵤ₂**(**v₃**) = (1/3)/(1/3) × (-1/3, 1/3, -1/3, 0) = (-1/3, 1/3, -1/3, 0)

**u₃** = (1, 2, 0, 1) - (5/6, 10/6, 5/6, 0) - (-1/3, 1/3, -1/3, 0)
      = (1 - 5/6 + 1/3, 2 - 10/6 - 1/3, 0 - 5/6 + 1/3, 1)
      = (3/6, 1/6, -3/6, 1)
      = (1/2, 1/6, -1/2, 1)

### Result: Orthogonal features

**New features:**
- **u₁** = (1, 2, 1, 0) - Base measurement
- **u₂** = (-1/3, 1/3, -1/3, 0) - Variation uncorrelated with u₁
- **u₃** = (1/2, 1/6, -1/2, 1) - Variation uncorrelated with both

**Each new feature captures independent information!**

## Visualizing the Process

**Think of Gram-Schmidt as building a coordinate system step by step:**

**Step 1:** Place first axis along **v₁**
```
     v₁
    ──────→ u₁
```

**Step 2:** Place second axis perpendicular to first
```
        ↑ u₂ (perpendicular)
        |
        |
    ────┼───→ u₁
        |
```

**Step 3:** Place third axis perpendicular to both
```
        ⊗ u₃ (coming out of page)
        
        ↑ u₂
        |
    ────┼───→ u₁
```

**At each step, we're building a perpendicular coordinate system!**

## When Can Gram-Schmidt Fail?

**Gram-Schmidt requires linearly independent input vectors.**

**What happens if vectors are dependent?**

**Example:**
- **v₁** = (1, 0)
- **v₂** = (2, 0) = 2**v₁**

**Apply Gram-Schmidt:**
- **u₁** = (1, 0)
- **u₂** = **v₂** - proj**ᵤ₁**(**v₂**)
        = (2, 0) - 2(1, 0)
        = (0, 0)

**We get the zero vector!**

**This tells us:** **v₂** is completely in the direction of **v₁** - it adds no new information!

**In practice:**
- Check that ||**uᵢ**|| ≠ 0 at each step
- If you get zero vector, original vectors were dependent
- Remove the dependent vector and continue

**Numerical issue:**
- On computers, might get very small ||**uᵢ**|| instead of exactly zero
- This indicates near-dependence
- Modified Gram-Schmidt handles this better

## Practice Problems - Gram-Schmidt

**Problem 5.1: Basic Application**

Apply Gram-Schmidt to create an orthogonal set:

a) **v₁** = (1, 1), **v₂** = (1, -1)
b) **v₁** = (1, 0, 0), **v₂** = (1, 1, 0)
c) **v₁** = (1, 2), **v₂** = (2, 1)

**Problem 5.2: Three Vectors in ℝ³**

Apply Gram-Schmidt:

**v₁** = (1, 0, 0)
**v₂** = (1, 1, 0)  
**v₃** = (1, 1, 1)

a) Find orthogonal set {**u₁, u₂, u₃**}
b) Verify all pairs are orthogonal
c) Normalize to create orthonormal basis

**Problem 5.3: Identifying Dependence**

Apply Gram-Schmidt to:

**v₁** = (1, 2, 3)
**v₂** = (2, 4, 6)
**v₃** = (1, 0, 0)

a) What happens when you process **v₂**?
b) Why does this happen?
c) Which vectors should you keep?

**Problem 5.4: QR Decomposition**

Matrix A has columns:
**v₁** = (1, 1, 0)
**v₂** = (0, 1, 1)

a) Apply Gram-Schmidt to get orthonormal columns (matrix Q)
b) Express original vectors in terms of orthonormal ones
c) What is R in the QR factorization?

**Problem 5.5: Orthogonalizing Functions**

Consider functions on [0, 1] with inner product ⟨f, g⟩ = ∫₀¹ f(x)g(x)dx

Apply Gram-Schmidt to:
- f₁(x) = 1
- f₂(x) = x
- f₃(x) = x²

(These become Legendre polynomials!)

**Problem 5.6: Application to Data**

You have features:
- **x₁** = (1, 2, 3, 4) - Age
- **x₂** = (2, 3, 4, 5) - Experience ≈ Age + 1
- **x₃** = (1, 1, 2, 2) - Education level

a) Apply Gram-Schmidt to decorrelate features
b) Interpret the new orthogonal features
c) Which original feature is most "unique"?

**Problem 5.7: Geometric Understanding**

In ℝ², you have **v₁** = (3, 0) and **v₂** = (1, 1)

a) Sketch both vectors
b) Apply Gram-Schmidt - sketch resulting **u₁** and **u₂**
c) Verify geometrically that they're perpendicular
d) Do they span the same space as original?

**Problem 5.8: Computational Challenge**

Apply Gram-Schmidt to nearly-dependent vectors:

**v₁** = (1, 0, 0)
**v₂** = (1, 0.0001, 0)
**v₃** = (0, 0, 1)

a) What happens to ||**u₂**||?
b) What does this indicate?
c) Why is this problematic for computers?

**Problem 5.9: Building from Scratch**

You want an orthonormal basis for ℝ³ where first vector is (1, 1, 1).

a) Normalize (1, 1, 1) → **q₁**
b) Choose ANY vector not parallel to **q₁** → **v₂**
c) Apply Gram-Schmidt to get **q₂**
d) Choose ANY third vector → **v₃**
e) Apply Gram-Schmidt to get **q₃**
f) Verify you have orthonormal basis

**Problem 5.10: Modified vs Classical**

For **v₁** = (1, 1, 1), **v₂** = (1, 1, 0), **v₃** = (1, 0, 0):

a) Apply classical Gram-Schmidt (all projections at once)
b) Apply modified Gram-Schmidt (update after each projection)
c) Are results identical?
d) Which would be better if vectors were nearly dependent?

---

**End of Section 5.5: Gram-Schmidt Process**

**Key takeaways:**
- ✅ Gram-Schmidt converts any independent set to orthogonal set
- ✅ Core idea: Remove overlaps (projections) systematically  
- ✅ Build perpendicular directions one at a time
- ✅ Results in same span, but orthogonal basis
- ✅ Foundation for QR decomposition
- ✅ Critical for numerical stability in computations
- ✅ Modified version better for computers
- ✅ Fails only when input vectors are dependent

**Next: Section 5.6 - Applications to Machine Learning**

---

<a name="applications"></a>
# 6. Applications to Machine Learning

## The Core Question: Why Does All This Math Matter for AI?

Imagine you're teaching a computer to recognize cats in photos.

**You have:** 1 million photos, each 1000×1000 pixels = 1 million dimensions per photo!

**Problem:** That's way too much data!
- Takes forever to process
- Needs massive memory
- Most information is redundant (neighboring pixels are similar)

**Question we should ask ourselves:** "Do we really need all 1 million dimensions? Or is there a smaller set of directions that captures most of the important information?"

**This is where linear independence, basis, orthogonality, and Gram-Schmidt become crucial!**

**Root cause of the problem:** High-dimensional data has lots of redundancy - many dimensions are not truly independent.

**How can we solve this?** Find the fundamental, independent directions that capture the essence of the data!

## Application 1: Principal Component Analysis (PCA)

### What Problem Does PCA Solve?

**Scenario:** You have a dataset with 100 features

**Problems you face:**
1. **Computational cost:** Processing 100 dimensions is slow
2. **Visualization:** Can't plot 100-dimensional data
3. **Overfitting:** Too many features, not enough data
4. **Redundancy:** Many features are correlated (not independent!)

**Question:** Can we represent the data using fewer dimensions without losing much information?

**Answer:** Yes! Use PCA to find the most important directions!

### The Root Cause: Redundancy in High Dimensions

**Let's think about why data has redundancy:**

**Example:** Predicting house prices with features:
- Living area (sq ft)
- Number of rooms
- Lot size
- Number of bathrooms
- Total area (living + lot)

**Notice:** Total area = Living area + Lot size (redundant!)

**More subtle:** Number of rooms and bathrooms are highly correlated
- More rooms usually means more bathrooms
- They're not independent directions!

**Root cause:** Features contain overlapping information - they're not orthogonal!

### So How Does PCA Fix This?

**PCA's brilliant insight:**

**Question to yourself:** "What if I could find NEW features that are:
1. Orthogonal (no overlap/redundancy)
2. Ordered by importance (first captures most variation)
3. Fewer in number (only keep important ones)"

**That's exactly what PCA does!**

**Process:**
1. Find the direction where data varies the MOST → Principal Component 1 (PC1)
2. Find the direction (orthogonal to PC1) where data varies second-most → PC2
3. Continue finding orthogonal directions...
4. Keep only the top k components that capture 95% of variance
5. Throw away the rest!

### Connecting to Our Linear Algebra Concepts

**PCA uses EVERYTHING we've learned:**

1. **Linear Independence:** PCA finds independent directions
2. **Basis:** PCs form a new basis for the data
3. **Orthogonality:** All PCs are orthogonal to each other
4. **Gram-Schmidt:** Used to ensure PCs are orthogonal
5. **Projection:** Project data onto the PC subspace

**Beautiful! All concepts come together!**

### Step-by-Step: How PCA Works

**Given:** Data matrix X (n samples × d features)

**Step 1: Center the data**
- Subtract mean from each feature
- Now data is centered at origin
- **Why?** We want directions through origin (subspaces!)

**Step 2: Compute covariance matrix**
- C = (1/n)X^T X
- Measures how features vary together
- **Size:** d × d

**Step 3: Find eigenvectors of C**
- These are the principal components!
- Each eigenvector is a direction
- Eigenvalue = variance in that direction

**Step 4: Sort by eigenvalue**
- Largest eigenvalue → most important direction (PC1)
- Second largest → PC2
- And so on...

**Step 5: Create transformation matrix**
- W = [PC1 | PC2 | ... | PCk]
- Columns are top k eigenvectors
- **These form an orthonormal basis!**

**Step 6: Transform data**
- Z = XW
- Projects data onto new k-dimensional subspace
- **Dimensionality reduced: d → k!**

### Detailed Example 6.1: PCA on Simple Data

**Dataset:** Student performance (4 students, 3 test scores)

```
        Math  Physics  Chemistry
Student 1:  90     85       88
Student 2:  70     68       72
Student 3:  80     78       81
Student 4:  60     58       62
```

**Matrix form:**
```
X = [90  85  88]
    [70  68  72]
    [80  78  81]
    [60  58  62]
```

**Observation:** Scores are highly correlated!
- Good at Math → probably good at Physics too
- Math and Physics scores move together

**Question:** Can we capture this with fewer dimensions?

**Step 1: Center the data**

Mean: (75, 72.25, 75.75)

```
X_centered = [15   12.75  12.25]
             [-5   -4.25  -3.75]
             [5    5.75   5.25]
             [-15  -14.25 -13.75]
```

**Step 2: Covariance matrix (simplified calculation)**

After computing C = (1/n)X_centered^T X_centered:

```
C ≈ [133.3  128.4  127.5]
    [128.4  124.2  123.1]
    [127.5  123.1  122.2]
```

**Notice:** All values are large and similar!
- High covariance → strong correlation
- Features are not independent!

**Step 3: Find eigenvectors (principal components)**

After eigendecomposition:

**PC1** ≈ [0.577, 0.578, 0.576] (nearly equal weights)
**Eigenvalue₁** ≈ 379.7

**PC2** ≈ [0.707, -0.707, 0.000] (contrast Math vs Physics)
**Eigenvalue₂** ≈ 0.5

**PC3** ≈ [0.408, 0.408, -0.816] (contrast Math+Physics vs Chemistry)  
**Eigenvalue₃** ≈ 0.3

**Interpretation:**

**PC1 (explains 99.7% of variance!):**
- All three subjects contribute equally
- **Represents "overall ability"**
- This ONE direction captures almost everything!

**PC2 (explains 0.1% of variance):**
- Contrast between Math and Physics
- Almost no variation here
- Students good at Math → good at Physics

**PC3 (explains 0.08% of variance):**
- Tiny variation
- Can ignore!

**Step 4: Dimensionality reduction**

**Keep only PC1!** It captures 99.7% of variance.

**Transform to 1D:**
```
Student 1: 15×0.577 + 12.75×0.578 + 12.25×0.576 ≈ 23.4
Student 2: -5×0.577 + (-4.25)×0.578 + (-3.75)×0.576 ≈ -7.7
Student 3: 5×0.577 + 5.75×0.578 + 5.25×0.576 ≈ 9.2
Student 4: -15×0.577 + (-14.25)×0.578 + (-13.75)×0.576 ≈ -24.9
```

**Result:** Reduced from 3 dimensions to 1 dimension!

**Lost only 0.3% of information!**

**New feature (PC1 score):** Represents overall academic ability

### Why PCA Uses Orthogonal Components

**Question to yourself:** "Why not just use ANY directions that explain variance?"

**Problem without orthogonality:**
- Directions might overlap (redundancy again!)
- Can't tell which direction contributes what
- Lose interpretability

**With orthogonal components:**
- Each PC captures INDEPENDENT variation
- No overlap between components
- Can analyze contribution of each separately
- Clean, interpretable decomposition

**This is why Gram-Schmidt matters!** Ensures orthogonality in numerical implementations.

## Application 2: Least Squares Regression

### What Problem Does Regression Solve?

**Scenario:** Predicting house prices

**You have:**
- Features: area, bedrooms, age → vector **x**
- Target: price → value y

**Goal:** Find relationship y = **w**^T**x** + b

**Problem:** Given n data points, find best **w**!

**This is regression!**

### The Root Cause: Data Doesn't Fit Perfectly

**Reality:** No perfect line fits all points

```
Price
  ^
  |    x     x
  |  x    x
  | x  x      (scattered points)
  |x    
  +-----------> Area
```

**Question:** What's the "best" line?

**Answer:** The line that minimizes errors!

### Geometric Interpretation: Orthogonal Projection!

**Here's where linear algebra becomes beautiful:**

**Setup:**
- Data matrix X (n × d): each row is a data point
- Target vector **y** (n × 1): actual prices
- Predicted: **ŷ** = X**w**

**Question:** What values can **ŷ** take?

**Answer:** All vectors in column space of X!
- **ŷ** = X**w** for some **w**
- This is span of columns of X
- It's a subspace!

**But:** Actual **y** might NOT be in this subspace!

```
        y (actual)
       /|
      / |
     /  | ← error (residual)
    /   |
   ŷ____| ← closest point in column space
   
   (column space of X - a subspace)
```

**Question:** How do we find the closest point **ŷ** in the subspace to **y**?

**Answer:** ORTHOGONAL PROJECTION!

**The best prediction is the orthogonal projection of y onto the column space of X!**

### Why Orthogonal Projection Minimizes Error

**Residual (error):** **e** = **y** - **ŷ**

**For orthogonal projection:**
- **e** is perpendicular to column space
- **e** is perpendicular to ALL columns of X
- This means: X^T**e** = 0

**This gives us the famous normal equation:**

X^T(**y** - X**w**) = 0
X^TX**w** = X^T**y**
**w** = (X^TX)^(-1)X^T**y**

**Beautiful! The best fit is where residuals are orthogonal to the feature space!**

### Example 6.2: Linear Regression

**Data:** Predicting test score from hours studied

```
Hours (x): [1, 2, 3, 4]
Score (y): [2, 3, 5, 6]
```

**Model:** y = wx + b

**In matrix form:**
```
X = [1  1]    y = [2]
    [1  2]        [3]
    [1  3]        [5]
    [1  4]        [6]
```
(First column for bias term)

**Normal equation:** (X^TX)**w** = X^T**y**

**Calculate X^TX:**
```
X^TX = [1 1 1 1] [1  1]   = [4  10]
       [1 2 3 4] [1  2]     [10 30]
                 [1  3]
                 [1  4]
```

**Calculate X^Ty:**
```
X^Ty = [1 1 1 1] [2]   = [16]
       [1 2 3 4] [3]     [44]
                 [5]
                 [6]
```

**Solve:**
```
[4  10] [b]   = [16]
[10 30] [w]     [44]
```

From first equation: 4b + 10w = 16 → b = 4 - 2.5w
Substitute into second: 10(4 - 2.5w) + 30w = 44
40 - 25w + 30w = 44
5w = 4
w = 0.8

Then: b = 4 - 2.5(0.8) = 4 - 2 = 2

**Solution:** y = 0.8x + 2

**Predictions:**
- x=1: ŷ=2.8 (actual: 2)
- x=2: ŷ=3.6 (actual: 3)  
- x=3: ŷ=4.4 (actual: 5)
- x=4: ŷ=5.2 (actual: 6)

**Residuals:** [-0.8, -0.6, 0.6, 0.8]

**Verify orthogonality:** X^T**e** should be near zero ✓

### When Does X^TX Fail to be Invertible?

**Question:** What if (X^TX) is not invertible?

**This happens when:** Columns of X are linearly dependent!

**Example:** Features are [area in sq ft, area in sq meters]
- Second is just 0.0929 × first
- Dependent features!
- X^TX is singular
- Can't solve uniquely

**Solution:** Remove dependent features!

**This is why linear independence matters in ML!**

## Application 3: Feature Orthogonalization

### The Multicollinearity Problem

**Scenario:** Predicting salary with:
- Years of experience
- Age  
- Years since degree

**Problem:** These are highly correlated!
- More experience → older
- More experience → longer since degree
- Not independent features!

**Consequences:**
1. **Unstable coefficients:** Small data changes → huge coefficient changes
2. **Hard to interpret:** Can't tell which feature matters
3. **Numerical issues:** Matrix inversion breaks down
4. **Inflated variance:** Coefficient estimates unreliable

**Root cause:** Features share information - they're not orthogonal!

### How Orthogonalization Helps

**Idea:** Transform features to be orthogonal!

**Method:** Apply Gram-Schmidt to feature vectors!

**Process:**
1. Keep first feature as is: **u₁** = **x₁**
2. Remove **x₁** component from **x₂**: **u₂** = **x₂** - proj**ᵤ₁**(**x₂**)
3. Remove **u₁**, **u₂** components from **x₃**: **u₃** = **x₃** - proj**ᵤ₁**(**x₃**) - proj**ᵤ₂**(**x₃**)

**Result:** New features {**u₁, u₂, u₃**} are orthogonal!

**Benefits:**
- Each feature adds unique information
- Stable coefficients
- Clear interpretation
- Better numerical properties

### Example 6.3: Orthogonalizing Correlated Features

**Dataset:** 4 houses

```
Feature 1 (Area):      [1000, 1500, 2000, 2500]
Feature 2 (Rooms):     [2, 3, 4, 5]
```

**Check correlation:**

Mean-centered:
```
x₁ = [-750, -250, 250, 750]
x₂ = [-1.5, -0.5, 0.5, 1.5]
```

**x₁ · x₂** = (-750)(-1.5) + (-250)(-0.5) + (250)(0.5) + (750)(1.5)
           = 1125 + 125 + 125 + 1125 = 2500

**Highly correlated!** (Large dot product)

**Orthogonalize:**

**u₁** = **x₁** = [-750, -250, 250, 750]

proj**ᵤ₁**(**x₂**) = (2500 / 1,562,500) × **x₁** = 0.0016 × **x₁**

**u₂** = **x₂** - proj**ᵤ₁**(**x₂**) 
      ≈ **x₂** - [−1.2, -0.4, 0.4, 1.2]
      ≈ [-0.3, -0.1, 0.1, 0.3]

**Verify:** **u₁ · u₂** ≈ 0 ✓

**Interpretation:**
- **u₁:** Overall size (area)
- **u₂:** Room density (rooms per area), independent of size

**Now:** Each feature captures unique information!

## Application 4: Singular Value Decomposition (SVD)

### What is SVD?

**Any matrix A (m × n) can be decomposed as:**

A = UΣV^T

where:
- **U** (m × m): Orthonormal basis for column space
- **Σ** (m × n): Diagonal with singular values
- **V** (n × n): Orthonormal basis for row space

**This is like PCA on steroids!**

### How SVD Uses Our Concepts

**SVD combines everything:**

1. **Orthonormal bases:** U and V have orthonormal columns
2. **Gram-Schmidt:** Used to compute U and V
3. **Eigenvalues/vectors:** Related to singular values
4. **Dimensionality reduction:** Keep only large singular values
5. **Linear independence:** SVD reveals rank (number of independent columns)

### Applications of SVD

**1. Image Compression**

**Original image:** 1000 × 1000 matrix = 1,000,000 values

**Apply SVD:** A = UΣV^T

**Key insight:** Only first k singular values are large!

**Keep only top k:** A ≈ U_k Σ_k V_k^T

**Storage:** k(1000 + 1 + 1000) instead of 1,000,000
- If k=100: Need 200,100 instead of 1,000,000
- 80% compression with minimal quality loss!

**Why it works:** Most singular values are tiny - image has redundancy!

**2. Recommender Systems**

**Matrix:** Users × Movies

**Problem:** Sparse! (Most entries missing)

**SVD gives:**
- User features (from U)
- Movie features (from V)  
- Predict missing ratings!

**Example:** Netflix prize winner used SVD-based methods

**3. Natural Language Processing**

**Term-document matrix:** Words × Documents

**SVD finds:**
- Latent topics (from Σ)
- Word-topic relationships (from U)
- Document-topic relationships (from V)

**This is Latent Semantic Analysis (LSA)!**

### Example 6.4: SVD for Image Compression

**Small image (4×4):**
```
A = [8  7  6  5]
    [7  6  5  4]
    [6  5  4  3]
    [5  4  3  2]
```

**Apply SVD:** A = UΣV^T

**Singular values (diagonal of Σ):**
σ₁ ≈ 18.37 (very large!)
σ₂ ≈ 0.54 (tiny)
σ₃ ≈ 0.00 (negligible)
σ₄ ≈ 0.00 (negligible)

**Observation:** First singular value dominates!

**Rank-1 approximation:** Keep only σ₁

```
A₁ = σ₁ u₁ v₁^T ≈ [7.9  6.9  5.9  4.9]
                   [6.9  6.0  5.1  4.1]
                   [5.9  5.1  4.3  3.4]
                   [4.9  4.1  3.4  2.6]
```

**Error:** Very small! (Most values within 0.1 of original)

**Compression:** 
- Original: 16 values
- Compressed: 4 (u₁) + 1 (σ₁) + 4 (v₁) = 9 values
- 44% reduction!

## Application 5: Neural Network Weight Initialization

### Why Orthogonal Initialization?

**Problem:** Training deep neural networks

**Challenge:** Gradients vanish or explode
- Too small → no learning
- Too large → instability

**Question:** How should we initialize weights?

**Answer:** Use orthogonal matrices!

### Why Orthogonality Helps

**Orthogonal matrices preserve norms:**

If Q is orthogonal: ||Q**x**|| = ||**x**||

**This means:**
- No explosion (can't make signals bigger)
- No vanishing (can't make signals smaller)
- Stable gradient flow!

**How to create:** Use Gram-Schmidt on random vectors!

### Example: Initializing a Layer

**Layer:** 100 neurons → 100 neurons

**Weight matrix:** 100 × 100

**Random initialization:**
```python
W = random_matrix(100, 100)
# Apply Gram-Schmidt to columns
Q = gram_schmidt(W)
# Q is now orthogonal!
```

**Benefits:**
- Signals preserve magnitude through layer
- Gradients flow smoothly
- Faster convergence
- Better final performance

**This is called "orthogonal initialization"!**

## Application 6: Independent Component Analysis (ICA)

### The Cocktail Party Problem

**Scenario:** Recording a party

**You have:** 3 microphones
**Recording:** Mix of 3 people talking

```
Mic 1: 0.7×person1 + 0.2×person2 + 0.1×person3
Mic 2: 0.1×person1 + 0.8×person2 + 0.1×person3  
Mic 3: 0.2×person1 + 0.1×person2 + 0.7×person3
```

**Goal:** Separate individual voices!

**This is ICA's job!**

### How ICA Relates to Linear Independence

**Key assumption:** Original sources are independent
- Person 1's speech independent of Person 2's
- Statistically independent signals

**ICA finds:** Transformation that makes signals independent
- Like finding independent basis vectors
- But using statistical independence, not just orthogonality

**Difference from PCA:**
- PCA: Finds orthogonal directions (geometric independence)
- ICA: Finds statistically independent sources (probabilistic independence)

**Both use:** Linear algebra concepts we've learned!

## Application 7: Dimensionality of Embeddings

### Word Embeddings in NLP

**Problem:** Representing words for machine learning

**Naive approach:** One-hot encoding
- Vocabulary: 50,000 words
- Each word: 50,000-dimensional vector (49,999 zeros, one 1)
- Huge! Sparse! No relationships captured!

**Better approach:** Word embeddings
- Each word: dense vector in 300 dimensions
- Similar words have similar vectors
- Captures semantic relationships

**Question:** Why 300 dimensions? Why not 50,000?

**Answer:** The true "dimensionality" of word meanings is much lower!
- Actual semantic space has much lower dimension
- Words lie in lower-dimensional manifold
- 300 dimensions capture essence

**This uses concepts of:**
- Basis and dimension
- Dimensionality reduction
- Linear independence of semantic directions

### Example: Word Relationships

**Classic example:**

king - man + woman ≈ queen

**In vector space:**
```
v(king) - v(man) + v(woman) ≈ v(queen)
```

**This works because:**
- "Royalty" dimension
- "Gender" dimension  
- These are roughly orthogonal directions!

**The embedding captures independent semantic dimensions!**

## Summary: Why Linear Algebra Powers ML

**Let's step back and see the big picture:**

**Question:** Why does all this abstract math appear everywhere in machine learning?

**Answer:** Data is fundamentally about relationships and structure!

**Linear algebra provides:**

1. **Language for relationships**
   - Dot products measure similarity
   - Projections find closest representations
   - Orthogonality means independence

2. **Tools for dimensionality**
   - Find true dimension of data (PCA)
   - Reduce computational cost
   - Reveal hidden structure

3. **Geometric intuition**
   - Data lives in high-dimensional spaces
   - We visualize using subspaces
   - Projections, distances, angles all matter

4. **Computational efficiency**
   - Matrix operations are fast
   - Orthogonal transformations stable
   - Vectorization accelerates processing

5. **Theoretical foundations**
   - Uniqueness of solutions (linear independence)
   - Optimal approximations (projections)
   - Guaranteed convergence (orthogonality)

**Every concept we learned has direct applications:**

| Concept | ML Application |
|---------|---------------|
| Linear Independence | Feature selection, detecting redundancy |
| Span | Representing data, understanding capacity |
| Basis | Coordinate systems, embeddings |
| Dimension | Complexity, capacity, compression |
| Orthogonality | Decorrelation, independence |
| Gram-Schmidt | QR decomposition, orthogonal initialization |
| Projection | Regression, PCA, nearest neighbors |
| Orthonormal bases | Fast computations, stability |

**The beauty:** Simple geometric ideas scale to millions of dimensions!

## Practice Problems - Applications

**Problem 6.1: PCA by Hand**

Dataset (3 students, 2 tests):
```
Math: [90, 60, 75]
Physics: [85, 58, 72]
```

a) Center the data
b) Compute covariance matrix
c) Find principal components (eigenvectors)
d) What % variance does PC1 explain?
e) Reduce to 1D using PC1

**Problem 6.2: Regression Geometry**

Data: x = [1, 2, 3], y = [2, 3, 5]

a) Form X matrix (include bias column)
b) Solve normal equation
c) Compute residuals
d) Verify X^T e = 0 (orthogonality)
e) Geometric interpretation: where does ŷ live?

**Problem 6.3: Feature Orthogonalization**

Features (4 samples):
```
x₁ = [1, 2, 3, 4]
x₂ = [2, 4, 6, 8] 
```

a) Why is this problematic?
b) Apply Gram-Schmidt
c) Interpret new features
d) Verify orthogonality

**Problem 6.4: SVD Compression**

Matrix:
```
A = [4  4]
    [4  4]
    [2  2]
```

a) What's the rank?
b) Apply SVD (or describe expected result)
c) How many singular values are non-zero?
d) Compress using rank-1 approximation
e) What's the error?

**Problem 6.5: Multicollinearity Detection**

You have features: [age, experience, years_since_degree]

a) Why might these be dependent?
b) How would you check mathematically?
c) What problems does this cause in regression?
d) Propose solution using concepts from this chapter

**Problem 6.6: PCA Interpretation**

After PCA on image dataset:
- PC1: [0.3, 0.3, 0.4, ...] (300 pixel weights)
- Explains 60% variance
- All weights positive

a) What might PC1 represent? (Hint: brightness?)
b) If PC2 has positive and negative weights?
c) Why do we need fewer PCs than original dimensions?

**Problem 6.7: Orthogonal Initialization**

Neural network layer: 50 input → 50 output neurons

a) Size of weight matrix?
b) Why initialize with orthogonal matrix?
c) How does this help gradient flow?
d) How to create orthogonal matrix? (Use Gram-Schmidt!)

**Problem 6.8: ICA vs PCA**

Audio signals: Mix of 2 independent sources

a) Would PCA separate the sources? Why/why not?
b) What's different about ICA?
c) Both find basis vectors - what's different about the bases?

**Problem 6.9: Embeddings**

Word embeddings: 50,000 words, 300 dimensions

a) Why not use 50,000 dimensions?
b) What determines actual dimensionality needed?
c) If "king - man + woman = queen" works, what does this say about the basis?

**Problem 6.10: Bringing It All Together**

You're building a recommender system:
- User-item matrix: 10,000 users × 5,000 movies
- Mostly empty (sparse)
- Want to predict missing ratings

a) What's wrong with using the data directly? (Dimensionality!)
b) How could PCA/SVD help?
c) What do the singular vectors represent?
d) How many dimensions do you really need? (Much less!)
e) Connect to concepts: basis, dimension, orthogonality

---

**End of Section 5.6: Applications to Machine Learning**

**Key takeaways:**
- ✅ PCA finds orthogonal directions of maximum variance
- ✅ Regression is orthogonal projection onto feature space
- ✅ Feature orthogonalization removes multicollinearity
- ✅ SVD provides low-rank approximations for compression
- ✅ Orthogonal initialization stabilizes neural network training
- ✅ ICA finds statistically independent components
- ✅ Embeddings capture true dimensionality of semantic spaces
- ✅ All concepts (independence, basis, orthogonality) are fundamental to ML

**Next: Section 5.7 - Comprehensive Practice Problems**

# Chapter 5: Section 5 - Gram-Schmidt Process

<a name="gram-schmidt"></a>
# 5. Gram-Schmidt Process

## The Core Question: How Do We Build Perpendicular Directions?

Imagine you're in a dark room trying to understand its shape. You have a flashlight.

**Your strategy:**
- First, shine the light straight ahead → that's one direction
- Next, you want to explore a NEW direction, but it should be completely different from the first
- Then another direction that's different from both previous ones
- And so on...

**Question:** How do you ensure each new direction is truly "different" (perpendicular) from all previous ones?

**Problem without perpendicular directions:**
If your second direction partially overlaps with the first, you're wasting effort exploring areas you've already seen!

**Solution:** Remove the "overlap" before exploring the new direction!

**This is exactly what the Gram-Schmidt process does!**

## What Problem Does Gram-Schmidt Solve?

**The Problem We Face:**

You have a set of linearly independent vectors: **{v₁, v₂, v₃, ..., vₖ}**

But they're NOT orthogonal - they point in "messy" directions with overlap.

**What we want:**
- Keep the same span (reach the same space)
- But use orthogonal vectors instead
- Even better: orthonormal vectors!

**Why do we want this?**
- Easier calculations (as we saw in Section 4)
- Numerical stability in computers
- Clearer geometric understanding
- Foundation for many ML algorithms

**Real-world analogy:**
- **Before:** You have k crooked, overlapping rulers
- **After:** You have k perfectly perpendicular rulers measuring the same space
- Same space covered, but much cleaner system!

## The Root Cause: Why Are Vectors Not Orthogonal?

**Let's think about what "not orthogonal" means:**

Take two vectors **v₁** and **v₂** where **v₁ · v₂ ≠ 0**

**Question:** Why is their dot product non-zero?

**Answer:** Because **v₂** has a component in the direction of **v₁**!

**Geometric picture:**
```
        v₂
       /
      /
     /______ projection of v₂ onto v₁
    /
   v₁
```

**The projection of v₂ onto v₁** is the "overlap" - the part of **v₂** that goes in the same direction as **v₁**.

**Root cause:** This overlap makes them non-orthogonal!

## So How Can We Remove This Overlap?

**Natural question:** If the overlap is the problem, can we just... remove it?

**Yes! That's the brilliant insight!**

**If we have:**
- **v₂** = (part parallel to **v₁**) + (part perpendicular to **v₁**)

**Then we can isolate the perpendicular part:**
- Perpendicular part = **v₂** - (part parallel to **v₁**)
- Perpendicular part = **v₂** - proj**ᵥ₁**(**v₂**)

**This perpendicular part is orthogonal to v₁!**

**Let's verify:**
- Let **w₂** = **v₂** - proj**ᵥ₁**(**v₂**)
- **w₂ · v₁** = (**v₂** - proj**ᵥ₁**(**v₂**)) · **v₁**
- = **v₂ · v₁** - proj**ᵥ₁**(**v₂**) · **v₁**

Now, proj**ᵥ₁**(**v₂**) = ((**v₂ · v₁**) / ||**v₁**||²)**v₁**

So: proj**ᵥ₁**(**v₂**) · **v₁** = ((**v₂ · v₁**) / ||**v₁**||²) **v₁ · v₁**
                                  = ((**v₂ · v₁**) / ||**v₁**||²) ||**v₁**||²
                                  = **v₂ · v₁**

Therefore: **w₂ · v₁** = **v₂ · v₁** - **v₂ · v₁** = 0 ✓

**Beautiful! The overlap is removed!**

## The Gram-Schmidt Process: Step by Step

**Goal:** Convert linearly independent set {**v₁, v₂, ..., vₖ**} into orthogonal set {**u₁, u₂, ..., uₖ**}

**The algorithm:**

**Step 1:** Keep the first vector as is
- **u₁** = **v₁**
- (Nothing to make it orthogonal to yet!)

**Step 2:** Make **v₂** orthogonal to **u₁**
- **u₂** = **v₂** - proj**ᵤ₁**(**v₂**)
- Remove the component of **v₂** in the direction of **u₁**

**Step 3:** Make **v₃** orthogonal to BOTH **u₁** AND **u₂**
- **u₃** = **v₃** - proj**ᵤ₁**(**v₃**) - proj**ᵤ₂**(**v₃**)
- Remove components in both previous directions

**Step 4:** Continue this pattern...
- **u₄** = **v₄** - proj**ᵤ₁**(**v₄**) - proj**ᵤ₂**(**v₄**) - proj**ᵤ₃**(**v₄**)

**General step i:**
- **uᵢ** = **vᵢ** - Σⱼ₌₁^(i-1) proj**ᵤⱼ**(**vᵢ**)
- Remove ALL overlaps with previous orthogonal vectors

**Optional final step:** Normalize to get orthonormal
- **q₁** = **u₁** / ||**u₁**||
- **q₂** = **u₂** / ||**u₂**||
- etc.

## Why Does This Work?

**Let's think through why each new vector is orthogonal to all previous ones:**

**After Step 2:** Is **u₂ ⊥ u₁**?
- Yes! We specifically removed the **u₁** component from **v₂**

**After Step 3:** Is **u₃ ⊥ u₁** and **u₃ ⊥ u₂**?
- We removed BOTH the **u₁** component AND the **u₂** component from **v₃**
- So **u₃** is perpendicular to both!

**The pattern continues:** Each new vector has ALL previous components removed, so it's perpendicular to ALL previous vectors.

**Key insight:** By systematically removing overlaps, we build perpendicular directions one at a time!

## Detailed Example 5.1: Gram-Schmidt in ℝ²

**Given vectors:**
- **v₁** = (3, 1)
- **v₂** = (2, 2)

**Goal:** Create orthogonal set {**u₁, u₂**}

### Step 1: First vector

**u₁** = **v₁** = (3, 1)

### Step 2: Make v₂ orthogonal to u₁

**Calculate projection of v₂ onto u₁:**

proj**ᵤ₁**(**v₂**) = ((**v₂ · u₁**) / ||**u₁**||²) **u₁**

**v₂ · u₁** = (2)(3) + (2)(1) = 6 + 2 = 8

||**u₁**||² = 3² + 1² = 9 + 1 = 10

proj**ᵤ₁**(**v₂**) = (8/10)(3, 1) = (4/5)(3, 1) = (12/5, 4/5)

**Remove the overlap:**

**u₂** = **v₂** - proj**ᵤ₁**(**v₂**)
      = (2, 2) - (12/5, 4/5)
      = (10/5 - 12/5, 10/5 - 4/5)
      = (-2/5, 6/5)

### Verify orthogonality:

**u₁ · u₂** = (3)(-2/5) + (1)(6/5) = -6/5 + 6/5 = 0 ✓

**Perfect! They're orthogonal!**

### Geometric interpretation:

**Before:** **v₁** and **v₂** pointed in somewhat similar directions (not perpendicular)

**After:** **u₁** and **u₂** are exactly perpendicular

**Same span:** span{**v₁, v₂**} = span{**u₁, u₂**} = all of ℝ²

### Optional: Create orthonormal basis

**Normalize u₁:**

||**u₁**|| = √(9 + 1) = √10

**q₁** = (3, 1) / √10 = (3/√10, 1/√10)

**Normalize u₂:**

||**u₂**|| = √(4/25 + 36/25) = √(40/25) = √(8/5) = 2√(2/5) = 2/√5

**q₂** = (-2/5, 6/5) / (2/√5) = (-2/5, 6/5) × (√5/2) = (-√5/5, 3√5/5)

**Verify orthonormal:**
- **q₁ · q₂** = (3/√10)(-√5/5) + (1/√10)(3√5/5) = -3√5/(5√10) + 3√5/(5√10) = 0 ✓
- ||**q₁**|| = 1 ✓
- ||**q₂**|| = 1 ✓

## Detailed Example 5.2: Gram-Schmidt in ℝ³

**Given vectors:**
- **v₁** = (1, 1, 0)
- **v₂** = (1, 0, 1)  
- **v₃** = (0, 1, 1)

**Goal:** Create orthogonal set {**u₁, u₂, u₃**}

### Step 1: First vector

**u₁** = **v₁** = (1, 1, 0)

### Step 2: Make v₂ orthogonal to u₁

**Calculate projection:**

**v₂ · u₁** = (1)(1) + (0)(1) + (1)(0) = 1

||**u₁**||² = 1² + 1² + 0² = 2

proj**ᵤ₁**(**v₂**) = (1/2)(1, 1, 0) = (1/2, 1/2, 0)

**Remove overlap:**

**u₂** = **v₂** - proj**ᵤ₁**(**v₂**)
      = (1, 0, 1) - (1/2, 1/2, 0)
      = (1/2, -1/2, 1)

**Verify:** **u₁ · u₂** = (1)(1/2) + (1)(-1/2) + (0)(1) = 1/2 - 1/2 = 0 ✓

### Step 3: Make v₃ orthogonal to BOTH u₁ and u₂

**Calculate projection onto u₁:**

**v₃ · u₁** = (0)(1) + (1)(1) + (1)(0) = 1

proj**ᵤ₁**(**v₃**) = (1/2)(1, 1, 0) = (1/2, 1/2, 0)

**Calculate projection onto u₂:**

**v₃ · u₂** = (0)(1/2) + (1)(-1/2) + (1)(1) = 0 - 1/2 + 1 = 1/2

||**u₂**||² = (1/2)² + (-1/2)² + 1² = 1/4 + 1/4 + 1 = 3/2

proj**ᵤ₂**(**v₃**) = (1/2)/(3/2) × (1/2, -1/2, 1) = (1/3)(1/2, -1/2, 1) = (1/6, -1/6, 1/3)

**Remove BOTH overlaps:**

**u₃** = **v₃** - proj**ᵤ₁**(**v₃**) - proj**ᵤ₂**(**v₃**)
      = (0, 1, 1) - (1/2, 1/2, 0) - (1/6, -1/6, 1/3)
      = (0 - 1/2 - 1/6, 1 - 1/2 + 1/6, 1 - 0 - 1/3)
      = (-3/6 - 1/6, 3/6 + 1/6, 3/3 - 1/3)
      = (-4/6, 4/6, 2/3)
      = (-2/3, 2/3, 2/3)

### Verify orthogonality:

**u₁ · u₃** = (1)(-2/3) + (1)(2/3) + (0)(2/3) = -2/3 + 2/3 = 0 ✓

**u₂ · u₃** = (1/2)(-2/3) + (-1/2)(2/3) + (1)(2/3) = -1/3 - 1/3 + 2/3 = 0 ✓

**Perfect! All three are mutually orthogonal!**

### Result:

**Orthogonal basis:** {(1, 1, 0), (1/2, -1/2, 1), (-2/3, 2/3, 2/3)}

**These span the same space as the original vectors, but are perpendicular!**

## The Pattern Behind Gram-Schmidt

**Let's step back and see the beautiful pattern:**

**Question to yourself:** "I have a new vector **vᵢ**. How do I make it orthogonal to all my previous orthogonal vectors **u₁, u₂, ..., uᵢ₋₁**?"

**Answer:** "Remove the parts that overlap with each previous vector!"

**How do we find these overlapping parts?** 
- Use projections! proj**ᵤⱼ**(**vᵢ**) gives us the component of **vᵢ** in the direction of **uⱼ**

**What's the root cause of non-orthogonality?**
- The new vector **vᵢ** has components pointing in the same directions as previous vectors

**How does Gram-Schmidt fix it?**
- By systematically subtracting out ALL these overlapping components, leaving only the "new" direction

**Why does this give us orthogonal vectors?**
- After removing all components in previous directions, what's left MUST be perpendicular to all previous vectors!

## Common Mistakes and How to Avoid Them

### Mistake 1: Projecting onto original vectors instead of orthogonalized ones

**Wrong:**
```
u₃ = v₃ - proj_v₁(v₃) - proj_v₂(v₃)  ❌
```

**Right:**
```
u₃ = v₃ - proj_u₁(v₃) - proj_u₂(v₃)  ✓
```

**Why it matters:** The **u** vectors are already orthogonal, but the **v** vectors are not! We must project onto the orthogonal vectors we've already created.

### Mistake 2: Forgetting to include all previous projections

**Wrong:**
```
u₃ = v₃ - proj_u₂(v₃)  ❌ (forgot u₁!)
```

**Right:**
```
u₃ = v₃ - proj_u₁(v₃) - proj_u₂(v₃)  ✓
```

**Why it matters:** We need to remove overlaps with ALL previous vectors, not just the most recent one!

### Mistake 3: Using incorrect projection formula

**Remember:** proj**ᵤ**(**v**) = ((**v · u**) / ||**u**||²) **u**

**If u is already normalized (unit vector):**
proj**ᵤ**(**v**) = (**v · u**) **u**  (simpler!)

### Mistake 4: Not verifying orthogonality

**Always check:** **uᵢ · uⱼ** = 0 for all i ≠ j

If not zero, you made a calculation error!

## Modified Gram-Schmidt (More Numerically Stable)

**Classical Gram-Schmidt** (what we've shown):
- Calculate all projections at once
- **uᵢ** = **vᵢ** - Σⱼ₌₁^(i-1) proj**ᵤⱼ**(**vᵢ**)

**Modified Gram-Schmidt** (better for computers):
- Update vector after each projection
- More stable when dealing with nearly-dependent vectors

**Algorithm:**
```
u₁ = v₁

For i = 2 to k:
    uᵢ = vᵢ
    For j = 1 to i-1:
        uᵢ = uᵢ - proj_uⱼ(uᵢ)  // Update uᵢ immediately
```

**Why is this better?**
- Reduces accumulation of rounding errors
- Each subtraction uses the most up-to-date vector
- More accurate when vectors are almost dependent

**In practice:** Computers use Modified Gram-Schmidt for numerical stability

## Applications in Machine Learning

### 1. QR Decomposition

**Any matrix A can be factored as:**

A = QR

where:
- **Q** has orthonormal columns (from Gram-Schmidt!)
- **R** is upper triangular

**How Gram-Schmidt creates QR:**
- Columns of A are our original vectors **v₁, v₂, ...**
- Apply Gram-Schmidt → get orthogonal vectors **u₁, u₂, ...**
- Normalize → get orthonormal vectors (columns of Q)
- The coefficients used form R

**Uses of QR:**
- Solving least squares problems
- Computing eigenvalues
- Numerically stable matrix computations

### 2. Orthogonalizing Features

**Problem:** You have correlated features in your dataset

**Example:**
- Feature 1: House area
- Feature 2: Number of rooms (highly correlated with area)

**Solution:** Apply Gram-Schmidt!
- Keep Feature 1 as is
- Transform Feature 2 to be orthogonal to Feature 1
- Now they capture independent information!

**Benefits:**
- Removes multicollinearity
- Each feature adds unique information
- Better for regression models

### 3. Principal Component Analysis (PCA)

**PCA finds orthogonal directions of maximum variance**

**Connection to Gram-Schmidt:**
- PCA computes eigenvectors of covariance matrix
- These eigenvectors are orthogonal
- If they weren't, we'd use Gram-Schmidt to make them so!
- Gram-Schmidt ensures orthogonality in numerical implementations

### 4. Conjugate Gradient Method

**For solving large systems Ax = b:**
- Creates sequence of search directions
- These directions must be orthogonal (actually "conjugate")
- Uses Gram-Schmidt-like process to ensure orthogonality
- Much faster than standard methods for huge systems

### 5. Signal Processing

**Fourier transforms decompose signals into orthogonal components**

**Why orthogonal basis matters:**
- Each frequency component is independent
- No interference between components
- Easy to filter specific frequencies
- Gram-Schmidt ensures clean separation

## Example 5.3: Application to Data

**Dataset:** 3 measurements from 4 experiments

```
v₁ = (1, 2, 1, 0) - Measurement 1
v₂ = (1, 3, 1, 0) - Measurement 2  
v₃ = (1, 2, 0, 1) - Measurement 3
```

**These are correlated! Let's orthogonalize them.**

### Step 1: Keep v₁

**u₁** = (1, 2, 1, 0)

### Step 2: Orthogonalize v₂

**v₂ · u₁** = 1 + 6 + 1 + 0 = 8
||**u₁**||² = 1 + 4 + 1 + 0 = 6

proj**ᵤ₁**(**v₂**) = (8/6)(1, 2, 1, 0) = (4/3, 8/3, 4/3, 0)

**u₂** = (1, 3, 1, 0) - (4/3, 8/3, 4/3, 0) = (-1/3, 1/3, -1/3, 0)

### Step 3: Orthogonalize v₃

**v₃ · u₁** = 1 + 4 + 0 + 0 = 5
proj**ᵤ₁**(**v₃**) = (5/6)(1, 2, 1, 0) = (5/6, 10/6, 5/6, 0)

**v₃ · u₂** = -1/3 + 2/3 + 0 + 0 = 1/3
||**u₂**||² = 1/9 + 1/9 + 1/9 + 0 = 3/9 = 1/3

proj**ᵤ₂**(**v₃**) = (1/3)/(1/3) × (-1/3, 1/3, -1/3, 0) = (-1/3, 1/3, -1/3, 0)

**u₃** = (1, 2, 0, 1) - (5/6, 10/6, 5/6, 0) - (-1/3, 1/3, -1/3, 0)
      = (1 - 5/6 + 1/3, 2 - 10/6 - 1/3, 0 - 5/6 + 1/3, 1)
      = (3/6, 1/6, -3/6, 1)
      = (1/2, 1/6, -1/2, 1)

### Result: Orthogonal features

**New features:**
- **u₁** = (1, 2, 1, 0) - Base measurement
- **u₂** = (-1/3, 1/3, -1/3, 0) - Variation uncorrelated with u₁
- **u₃** = (1/2, 1/6, -1/2, 1) - Variation uncorrelated with both

**Each new feature captures independent information!**

## Visualizing the Process

**Think of Gram-Schmidt as building a coordinate system step by step:**

**Step 1:** Place first axis along **v₁**
```
     v₁
    ──────→ u₁
```

**Step 2:** Place second axis perpendicular to first
```
        ↑ u₂ (perpendicular)
        |
        |
    ────┼───→ u₁
        |
```

**Step 3:** Place third axis perpendicular to both
```
        ⊗ u₃ (coming out of page)
        
        ↑ u₂
        |
    ────┼───→ u₁
```

**At each step, we're building a perpendicular coordinate system!**

## When Can Gram-Schmidt Fail?

**Gram-Schmidt requires linearly independent input vectors.**

**What happens if vectors are dependent?**

**Example:**
- **v₁** = (1, 0)
- **v₂** = (2, 0) = 2**v₁**

**Apply Gram-Schmidt:**
- **u₁** = (1, 0)
- **u₂** = **v₂** - proj**ᵤ₁**(**v₂**)
        = (2, 0) - 2(1, 0)
        = (0, 0)

**We get the zero vector!**

**This tells us:** **v₂** is completely in the direction of **v₁** - it adds no new information!

**In practice:**
- Check that ||**uᵢ**|| ≠ 0 at each step
- If you get zero vector, original vectors were dependent
- Remove the dependent vector and continue

**Numerical issue:**
- On computers, might get very small ||**uᵢ**|| instead of exactly zero
- This indicates near-dependence
- Modified Gram-Schmidt handles this better

## Practice Problems - Gram-Schmidt

**Problem 5.1: Basic Application**

Apply Gram-Schmidt to create an orthogonal set:

a) **v₁** = (1, 1), **v₂** = (1, -1)
b) **v₁** = (1, 0, 0), **v₂** = (1, 1, 0)
c) **v₁** = (1, 2), **v₂** = (2, 1)

**Problem 5.2: Three Vectors in ℝ³**

Apply Gram-Schmidt:

**v₁** = (1, 0, 0)
**v₂** = (1, 1, 0)  
**v₃** = (1, 1, 1)

a) Find orthogonal set {**u₁, u₂, u₃**}
b) Verify all pairs are orthogonal
c) Normalize to create orthonormal basis

**Problem 5.3: Identifying Dependence**

Apply Gram-Schmidt to:

**v₁** = (1, 2, 3)
**v₂** = (2, 4, 6)
**v₃** = (1, 0, 0)

a) What happens when you process **v₂**?
b) Why does this happen?
c) Which vectors should you keep?

**Problem 5.4: QR Decomposition**

Matrix A has columns:
**v₁** = (1, 1, 0)
**v₂** = (0, 1, 1)

a) Apply Gram-Schmidt to get orthonormal columns (matrix Q)
b) Express original vectors in terms of orthonormal ones
c) What is R in the QR factorization?

**Problem 5.5: Orthogonalizing Functions**

Consider functions on [0, 1] with inner product ⟨f, g⟩ = ∫₀¹ f(x)g(x)dx

Apply Gram-Schmidt to:
- f₁(x) = 1
- f₂(x) = x
- f₃(x) = x²

(These become Legendre polynomials!)

**Problem 5.6: Application to Data**

You have features:
- **x₁** = (1, 2, 3, 4) - Age
- **x₂** = (2, 3, 4, 5) - Experience ≈ Age + 1
- **x₃** = (1, 1, 2, 2) - Education level

a) Apply Gram-Schmidt to decorrelate features
b) Interpret the new orthogonal features
c) Which original feature is most "unique"?

**Problem 5.7: Geometric Understanding**

In ℝ², you have **v₁** = (3, 0) and **v₂** = (1, 1)

a) Sketch both vectors
b) Apply Gram-Schmidt - sketch resulting **u₁** and **u₂**
c) Verify geometrically that they're perpendicular
d) Do they span the same space as original?

**Problem 5.8: Computational Challenge**

Apply Gram-Schmidt to nearly-dependent vectors:

**v₁** = (1, 0, 0)
**v₂** = (1, 0.0001, 0)
**v₃** = (0, 0, 1)

a) What happens to ||**u₂**||?
b) What does this indicate?
c) Why is this problematic for computers?

**Problem 5.9: Building from Scratch**

You want an orthonormal basis for ℝ³ where first vector is (1, 1, 1).

a) Normalize (1, 1, 1) → **q₁**
b) Choose ANY vector not parallel to **q₁** → **v₂**
c) Apply Gram-Schmidt to get **q₂**
d) Choose ANY third vector → **v₃**
e) Apply Gram-Schmidt to get **q₃**
f) Verify you have orthonormal basis

**Problem 5.10: Modified vs Classical**

For **v₁** = (1, 1, 1), **v₂** = (1, 1, 0), **v₃** = (1, 0, 0):

a) Apply classical Gram-Schmidt (all projections at once)
b) Apply modified Gram-Schmidt (update after each projection)
c) Are results identical?
d) Which would be better if vectors were nearly dependent?

---

**End of Section 5.5: Gram-Schmidt Process**

**Key takeaways:**
- ✅ Gram-Schmidt converts any independent set to orthogonal set
- ✅ Core idea: Remove overlaps (projections) systematically  
- ✅ Build perpendicular directions one at a time
- ✅ Results in same span, but orthogonal basis
- ✅ Foundation for QR decomposition
- ✅ Critical for numerical stability in computations
- ✅ Modified version better for computers
- ✅ Fails only when input vectors are dependent

**Next: Section 5.6 - Applications to Machine Learning**

---

<a name="applications"></a>
# 6. Applications to Machine Learning

## The Core Question: Why Does All This Math Matter for AI?

Imagine you're teaching a computer to recognize cats in photos.

**You have:** 1 million photos, each 1000×1000 pixels = 1 million dimensions per photo!

**Problem:** That's way too much data!
- Takes forever to process
- Needs massive memory
- Most information is redundant (neighboring pixels are similar)

**Question we should ask ourselves:** "Do we really need all 1 million dimensions? Or is there a smaller set of directions that captures most of the important information?"

**This is where linear independence, basis, orthogonality, and Gram-Schmidt become crucial!**

**Root cause of the problem:** High-dimensional data has lots of redundancy - many dimensions are not truly independent.

**How can we solve this?** Find the fundamental, independent directions that capture the essence of the data!

## Application 1: Principal Component Analysis (PCA)

### What Problem Does PCA Solve?

**Scenario:** You have a dataset with 100 features

**Problems you face:**
1. **Computational cost:** Processing 100 dimensions is slow
2. **Visualization:** Can't plot 100-dimensional data
3. **Overfitting:** Too many features, not enough data
4. **Redundancy:** Many features are correlated (not independent!)

**Question:** Can we represent the data using fewer dimensions without losing much information?

**Answer:** Yes! Use PCA to find the most important directions!

### The Root Cause: Redundancy in High Dimensions

**Let's think about why data has redundancy:**

**Example:** Predicting house prices with features:
- Living area (sq ft)
- Number of rooms
- Lot size
- Number of bathrooms
- Total area (living + lot)

**Notice:** Total area = Living area + Lot size (redundant!)

**More subtle:** Number of rooms and bathrooms are highly correlated
- More rooms usually means more bathrooms
- They're not independent directions!

**Root cause:** Features contain overlapping information - they're not orthogonal!

### So How Does PCA Fix This?

**PCA's brilliant insight:**

**Question to yourself:** "What if I could find NEW features that are:
1. Orthogonal (no overlap/redundancy)
2. Ordered by importance (first captures most variation)
3. Fewer in number (only keep important ones)"

**That's exactly what PCA does!**

**Process:**
1. Find the direction where data varies the MOST → Principal Component 1 (PC1)
2. Find the direction (orthogonal to PC1) where data varies second-most → PC2
3. Continue finding orthogonal directions...
4. Keep only the top k components that capture 95% of variance
5. Throw away the rest!

### Connecting to Our Linear Algebra Concepts

**PCA uses EVERYTHING we've learned:**

1. **Linear Independence:** PCA finds independent directions
2. **Basis:** PCs form a new basis for the data
3. **Orthogonality:** All PCs are orthogonal to each other
4. **Gram-Schmidt:** Used to ensure PCs are orthogonal
5. **Projection:** Project data onto the PC subspace

**Beautiful! All concepts come together!**

### Step-by-Step: How PCA Works

**Given:** Data matrix X (n samples × d features)

**Step 1: Center the data**
- Subtract mean from each feature
- Now data is centered at origin
- **Why?** We want directions through origin (subspaces!)

**Step 2: Compute covariance matrix**
- C = (1/n)X^T X
- Measures how features vary together
- **Size:** d × d

**Step 3: Find eigenvectors of C**
- These are the principal components!
- Each eigenvector is a direction
- Eigenvalue = variance in that direction

**Step 4: Sort by eigenvalue**
- Largest eigenvalue → most important direction (PC1)
- Second largest → PC2
- And so on...

**Step 5: Create transformation matrix**
- W = [PC1 | PC2 | ... | PCk]
- Columns are top k eigenvectors
- **These form an orthonormal basis!**

**Step 6: Transform data**
- Z = XW
- Projects data onto new k-dimensional subspace
- **Dimensionality reduced: d → k!**

### Detailed Example 6.1: PCA on Simple Data

**Dataset:** Student performance (4 students, 3 test scores)

```
        Math  Physics  Chemistry
Student 1:  90     85       88
Student 2:  70     68       72
Student 3:  80     78       81
Student 4:  60     58       62
```

**Matrix form:**
```
X = [90  85  88]
    [70  68  72]
    [80  78  81]
    [60  58  62]
```

**Observation:** Scores are highly correlated!
- Good at Math → probably good at Physics too
- Math and Physics scores move together

**Question:** Can we capture this with fewer dimensions?

**Step 1: Center the data**

Mean: (75, 72.25, 75.75)

```
X_centered = [15   12.75  12.25]
             [-5   -4.25  -3.75]
             [5    5.75   5.25]
             [-15  -14.25 -13.75]
```

**Step 2: Covariance matrix (simplified calculation)**

After computing C = (1/n)X_centered^T X_centered:

```
C ≈ [133.3  128.4  127.5]
    [128.4  124.2  123.1]
    [127.5  123.1  122.2]
```

**Notice:** All values are large and similar!
- High covariance → strong correlation
- Features are not independent!

**Step 3: Find eigenvectors (principal components)**

After eigendecomposition:

**PC1** ≈ [0.577, 0.578, 0.576] (nearly equal weights)
**Eigenvalue₁** ≈ 379.7

**PC2** ≈ [0.707, -0.707, 0.000] (contrast Math vs Physics)
**Eigenvalue₂** ≈ 0.5

**PC3** ≈ [0.408, 0.408, -0.816] (contrast Math+Physics vs Chemistry)  
**Eigenvalue₃** ≈ 0.3

**Interpretation:**

**PC1 (explains 99.7% of variance!):**
- All three subjects contribute equally
- **Represents "overall ability"**
- This ONE direction captures almost everything!

**PC2 (explains 0.1% of variance):**
- Contrast between Math and Physics
- Almost no variation here
- Students good at Math → good at Physics

**PC3 (explains 0.08% of variance):**
- Tiny variation
- Can ignore!

**Step 4: Dimensionality reduction**

**Keep only PC1!** It captures 99.7% of variance.

**Transform to 1D:**
```
Student 1: 15×0.577 + 12.75×0.578 + 12.25×0.576 ≈ 23.4
Student 2: -5×0.577 + (-4.25)×0.578 + (-3.75)×0.576 ≈ -7.7
Student 3: 5×0.577 + 5.75×0.578 + 5.25×0.576 ≈ 9.2
Student 4: -15×0.577 + (-14.25)×0.578 + (-13.75)×0.576 ≈ -24.9
```

**Result:** Reduced from 3 dimensions to 1 dimension!

**Lost only 0.3% of information!**

**New feature (PC1 score):** Represents overall academic ability

### Why PCA Uses Orthogonal Components

**Question to yourself:** "Why not just use ANY directions that explain variance?"

**Problem without orthogonality:**
- Directions might overlap (redundancy again!)
- Can't tell which direction contributes what
- Lose interpretability

**With orthogonal components:**
- Each PC captures INDEPENDENT variation
- No overlap between components
- Can analyze contribution of each separately
- Clean, interpretable decomposition

**This is why Gram-Schmidt matters!** Ensures orthogonality in numerical implementations.

## Application 2: Least Squares Regression

### What Problem Does Regression Solve?

**Scenario:** Predicting house prices

**You have:**
- Features: area, bedrooms, age → vector **x**
- Target: price → value y

**Goal:** Find relationship y = **w**^T**x** + b

**Problem:** Given n data points, find best **w**!

**This is regression!**

### The Root Cause: Data Doesn't Fit Perfectly

**Reality:** No perfect line fits all points

```
Price
  ^
  |    x     x
  |  x    x
  | x  x      (scattered points)
  |x    
  +-----------> Area
```

**Question:** What's the "best" line?

**Answer:** The line that minimizes errors!

### Geometric Interpretation: Orthogonal Projection!

**Here's where linear algebra becomes beautiful:**

**Setup:**
- Data matrix X (n × d): each row is a data point
- Target vector **y** (n × 1): actual prices
- Predicted: **ŷ** = X**w**

**Question:** What values can **ŷ** take?

**Answer:** All vectors in column space of X!
- **ŷ** = X**w** for some **w**
- This is span of columns of X
- It's a subspace!

**But:** Actual **y** might NOT be in this subspace!

```
        y (actual)
       /|
      / |
     /  | ← error (residual)
    /   |
   ŷ____| ← closest point in column space
   
   (column space of X - a subspace)
```

**Question:** How do we find the closest point **ŷ** in the subspace to **y**?

**Answer:** ORTHOGONAL PROJECTION!

**The best prediction is the orthogonal projection of y onto the column space of X!**

### Why Orthogonal Projection Minimizes Error

**Residual (error):** **e** = **y** - **ŷ**

**For orthogonal projection:**
- **e** is perpendicular to column space
- **e** is perpendicular to ALL columns of X
- This means: X^T**e** = 0

**This gives us the famous normal equation:**

X^T(**y** - X**w**) = 0
X^TX**w** = X^T**y**
**w** = (X^TX)^(-1)X^T**y**

**Beautiful! The best fit is where residuals are orthogonal to the feature space!**

### Example 6.2: Linear Regression

**Data:** Predicting test score from hours studied

```
Hours (x): [1, 2, 3, 4]
Score (y): [2, 3, 5, 6]
```

**Model:** y = wx + b

**In matrix form:**
```
X = [1  1]    y = [2]
    [1  2]        [3]
    [1  3]        [5]
    [1  4]        [6]
```
(First column for bias term)

**Normal equation:** (X^TX)**w** = X^T**y**

**Calculate X^TX:**
```
X^TX = [1 1 1 1] [1  1]   = [4  10]
       [1 2 3 4] [1  2]     [10 30]
                 [1  3]
                 [1  4]
```

**Calculate X^Ty:**
```
X^Ty = [1 1 1 1] [2]   = [16]
       [1 2 3 4] [3]     [44]
                 [5]
                 [6]
```

**Solve:**
```
[4  10] [b]   = [16]
[10 30] [w]     [44]
```

From first equation: 4b + 10w = 16 → b = 4 - 2.5w
Substitute into second: 10(4 - 2.5w) + 30w = 44
40 - 25w + 30w = 44
5w = 4
w = 0.8

Then: b = 4 - 2.5(0.8) = 4 - 2 = 2

**Solution:** y = 0.8x + 2

**Predictions:**
- x=1: ŷ=2.8 (actual: 2)
- x=2: ŷ=3.6 (actual: 3)  
- x=3: ŷ=4.4 (actual: 5)
- x=4: ŷ=5.2 (actual: 6)

**Residuals:** [-0.8, -0.6, 0.6, 0.8]

**Verify orthogonality:** X^T**e** should be near zero ✓

### When Does X^TX Fail to be Invertible?

**Question:** What if (X^TX) is not invertible?

**This happens when:** Columns of X are linearly dependent!

**Example:** Features are [area in sq ft, area in sq meters]
- Second is just 0.0929 × first
- Dependent features!
- X^TX is singular
- Can't solve uniquely

**Solution:** Remove dependent features!

**This is why linear independence matters in ML!**

## Application 3: Feature Orthogonalization

### The Multicollinearity Problem

**Scenario:** Predicting salary with:
- Years of experience
- Age  
- Years since degree

**Problem:** These are highly correlated!
- More experience → older
- More experience → longer since degree
- Not independent features!

**Consequences:**
1. **Unstable coefficients:** Small data changes → huge coefficient changes
2. **Hard to interpret:** Can't tell which feature matters
3. **Numerical issues:** Matrix inversion breaks down
4. **Inflated variance:** Coefficient estimates unreliable

**Root cause:** Features share information - they're not orthogonal!

### How Orthogonalization Helps

**Idea:** Transform features to be orthogonal!

**Method:** Apply Gram-Schmidt to feature vectors!

**Process:**
1. Keep first feature as is: **u₁** = **x₁**
2. Remove **x₁** component from **x₂**: **u₂** = **x₂** - proj**ᵤ₁**(**x₂**)
3. Remove **u₁**, **u₂** components from **x₃**: **u₃** = **x₃** - proj**ᵤ₁**(**x₃**) - proj**ᵤ₂**(**x₃**)

**Result:** New features {**u₁, u₂, u₃**} are orthogonal!

**Benefits:**
- Each feature adds unique information
- Stable coefficients
- Clear interpretation
- Better numerical properties

### Example 6.3: Orthogonalizing Correlated Features

**Dataset:** 4 houses

```
Feature 1 (Area):      [1000, 1500, 2000, 2500]
Feature 2 (Rooms):     [2, 3, 4, 5]
```

**Check correlation:**

Mean-centered:
```
x₁ = [-750, -250, 250, 750]
x₂ = [-1.5, -0.5, 0.5, 1.5]
```

**x₁ · x₂** = (-750)(-1.5) + (-250)(-0.5) + (250)(0.5) + (750)(1.5)
           = 1125 + 125 + 125 + 1125 = 2500

**Highly correlated!** (Large dot product)

**Orthogonalize:**

**u₁** = **x₁** = [-750, -250, 250, 750]

proj**ᵤ₁**(**x₂**) = (2500 / 1,562,500) × **x₁** = 0.0016 × **x₁**

**u₂** = **x₂** - proj**ᵤ₁**(**x₂**) 
      ≈ **x₂** - [−1.2, -0.4, 0.4, 1.2]
      ≈ [-0.3, -0.1, 0.1, 0.3]

**Verify:** **u₁ · u₂** ≈ 0 ✓

**Interpretation:**
- **u₁:** Overall size (area)
- **u₂:** Room density (rooms per area), independent of size

**Now:** Each feature captures unique information!

## Application 4: Singular Value Decomposition (SVD)

### What is SVD?

**Any matrix A (m × n) can be decomposed as:**

A = UΣV^T

where:
- **U** (m × m): Orthonormal basis for column space
- **Σ** (m × n): Diagonal with singular values
- **V** (n × n): Orthonormal basis for row space

**This is like PCA on steroids!**

### How SVD Uses Our Concepts

**SVD combines everything:**

1. **Orthonormal bases:** U and V have orthonormal columns
2. **Gram-Schmidt:** Used to compute U and V
3. **Eigenvalues/vectors:** Related to singular values
4. **Dimensionality reduction:** Keep only large singular values
5. **Linear independence:** SVD reveals rank (number of independent columns)

### Applications of SVD

**1. Image Compression**

**Original image:** 1000 × 1000 matrix = 1,000,000 values

**Apply SVD:** A = UΣV^T

**Key insight:** Only first k singular values are large!

**Keep only top k:** A ≈ U_k Σ_k V_k^T

**Storage:** k(1000 + 1 + 1000) instead of 1,000,000
- If k=100: Need 200,100 instead of 1,000,000
- 80% compression with minimal quality loss!

**Why it works:** Most singular values are tiny - image has redundancy!

**2. Recommender Systems**

**Matrix:** Users × Movies

**Problem:** Sparse! (Most entries missing)

**SVD gives:**
- User features (from U)
- Movie features (from V)  
- Predict missing ratings!

**Example:** Netflix prize winner used SVD-based methods

**3. Natural Language Processing**

**Term-document matrix:** Words × Documents

**SVD finds:**
- Latent topics (from Σ)
- Word-topic relationships (from U)
- Document-topic relationships (from V)

**This is Latent Semantic Analysis (LSA)!**

### Example 6.4: SVD for Image Compression

**Small image (4×4):**
```
A = [8  7  6  5]
    [7  6  5  4]
    [6  5  4  3]
    [5  4  3  2]
```

**Apply SVD:** A = UΣV^T

**Singular values (diagonal of Σ):**
σ₁ ≈ 18.37 (very large!)
σ₂ ≈ 0.54 (tiny)
σ₃ ≈ 0.00 (negligible)
σ₄ ≈ 0.00 (negligible)

**Observation:** First singular value dominates!

**Rank-1 approximation:** Keep only σ₁

```
A₁ = σ₁ u₁ v₁^T ≈ [7.9  6.9  5.9  4.9]
                   [6.9  6.0  5.1  4.1]
                   [5.9  5.1  4.3  3.4]
                   [4.9  4.1  3.4  2.6]
```

**Error:** Very small! (Most values within 0.1 of original)

**Compression:** 
- Original: 16 values
- Compressed: 4 (u₁) + 1 (σ₁) + 4 (v₁) = 9 values
- 44% reduction!

## Application 5: Neural Network Weight Initialization

### Why Orthogonal Initialization?

**Problem:** Training deep neural networks

**Challenge:** Gradients vanish or explode
- Too small → no learning
- Too large → instability

**Question:** How should we initialize weights?

**Answer:** Use orthogonal matrices!

### Why Orthogonality Helps

**Orthogonal matrices preserve norms:**

If Q is orthogonal: ||Q**x**|| = ||**x**||

**This means:**
- No explosion (can't make signals bigger)
- No vanishing (can't make signals smaller)
- Stable gradient flow!

**How to create:** Use Gram-Schmidt on random vectors!

### Example: Initializing a Layer

**Layer:** 100 neurons → 100 neurons

**Weight matrix:** 100 × 100

**Random initialization:**
```python
W = random_matrix(100, 100)
# Apply Gram-Schmidt to columns
Q = gram_schmidt(W)
# Q is now orthogonal!
```

**Benefits:**
- Signals preserve magnitude through layer
- Gradients flow smoothly
- Faster convergence
- Better final performance

**This is called "orthogonal initialization"!**

## Application 6: Independent Component Analysis (ICA)

### The Cocktail Party Problem

**Scenario:** Recording a party

**You have:** 3 microphones
**Recording:** Mix of 3 people talking

```
Mic 1: 0.7×person1 + 0.2×person2 + 0.1×person3
Mic 2: 0.1×person1 + 0.8×person2 + 0.1×person3  
Mic 3: 0.2×person1 + 0.1×person2 + 0.7×person3
```

**Goal:** Separate individual voices!

**This is ICA's job!**

### How ICA Relates to Linear Independence

**Key assumption:** Original sources are independent
- Person 1's speech independent of Person 2's
- Statistically independent signals

**ICA finds:** Transformation that makes signals independent
- Like finding independent basis vectors
- But using statistical independence, not just orthogonality

**Difference from PCA:**
- PCA: Finds orthogonal directions (geometric independence)
- ICA: Finds statistically independent sources (probabilistic independence)

**Both use:** Linear algebra concepts we've learned!

## Application 7: Dimensionality of Embeddings

### Word Embeddings in NLP

**Problem:** Representing words for machine learning

**Naive approach:** One-hot encoding
- Vocabulary: 50,000 words
- Each word: 50,000-dimensional vector (49,999 zeros, one 1)
- Huge! Sparse! No relationships captured!

**Better approach:** Word embeddings
- Each word: dense vector in 300 dimensions
- Similar words have similar vectors
- Captures semantic relationships

**Question:** Why 300 dimensions? Why not 50,000?

**Answer:** The true "dimensionality" of word meanings is much lower!
- Actual semantic space has much lower dimension
- Words lie in lower-dimensional manifold
- 300 dimensions capture essence

**This uses concepts of:**
- Basis and dimension
- Dimensionality reduction
- Linear independence of semantic directions

### Example: Word Relationships

**Classic example:**

king - man + woman ≈ queen

**In vector space:**
```
v(king) - v(man) + v(woman) ≈ v(queen)
```

**This works because:**
- "Royalty" dimension
- "Gender" dimension  
- These are roughly orthogonal directions!

**The embedding captures independent semantic dimensions!**

## Summary: Why Linear Algebra Powers ML

**Let's step back and see the big picture:**

**Question:** Why does all this abstract math appear everywhere in machine learning?

**Answer:** Data is fundamentally about relationships and structure!

**Linear algebra provides:**

1. **Language for relationships**
   - Dot products measure similarity
   - Projections find closest representations
   - Orthogonality means independence

2. **Tools for dimensionality**
   - Find true dimension of data (PCA)
   - Reduce computational cost
   - Reveal hidden structure

3. **Geometric intuition**
   - Data lives in high-dimensional spaces
   - We visualize using subspaces
   - Projections, distances, angles all matter

4. **Computational efficiency**
   - Matrix operations are fast
   - Orthogonal transformations stable
   - Vectorization accelerates processing

5. **Theoretical foundations**
   - Uniqueness of solutions (linear independence)
   - Optimal approximations (projections)
   - Guaranteed convergence (orthogonality)

**Every concept we learned has direct applications:**

| Concept | ML Application |
|---------|---------------|
| Linear Independence | Feature selection, detecting redundancy |
| Span | Representing data, understanding capacity |
| Basis | Coordinate systems, embeddings |
| Dimension | Complexity, capacity, compression |
| Orthogonality | Decorrelation, independence |
| Gram-Schmidt | QR decomposition, orthogonal initialization |
| Projection | Regression, PCA, nearest neighbors |
| Orthonormal bases | Fast computations, stability |

**The beauty:** Simple geometric ideas scale to millions of dimensions!

## Practice Problems - Applications

**Problem 6.1: PCA by Hand**

Dataset (3 students, 2 tests):
```
Math: [90, 60, 75]
Physics: [85, 58, 72]
```

a) Center the data
b) Compute covariance matrix
c) Find principal components (eigenvectors)
d) What % variance does PC1 explain?
e) Reduce to 1D using PC1

**Problem 6.2: Regression Geometry**

Data: x = [1, 2, 3], y = [2, 3, 5]

a) Form X matrix (include bias column)
b) Solve normal equation
c) Compute residuals
d) Verify X^T e = 0 (orthogonality)
e) Geometric interpretation: where does ŷ live?

**Problem 6.3: Feature Orthogonalization**

Features (4 samples):
```
x₁ = [1, 2, 3, 4]
x₂ = [2, 4, 6, 8] 
```

a) Why is this problematic?
b) Apply Gram-Schmidt
c) Interpret new features
d) Verify orthogonality

**Problem 6.4: SVD Compression**

Matrix:
```
A = [4  4]
    [4  4]
    [2  2]
```

a) What's the rank?
b) Apply SVD (or describe expected result)
c) How many singular values are non-zero?
d) Compress using rank-1 approximation
e) What's the error?

**Problem 6.5: Multicollinearity Detection**

You have features: [age, experience, years_since_degree]

a) Why might these be dependent?
b) How would you check mathematically?
c) What problems does this cause in regression?
d) Propose solution using concepts from this chapter

**Problem 6.6: PCA Interpretation**

After PCA on image dataset:
- PC1: [0.3, 0.3, 0.4, ...] (300 pixel weights)
- Explains 60% variance
- All weights positive

a) What might PC1 represent? (Hint: brightness?)
b) If PC2 has positive and negative weights?
c) Why do we need fewer PCs than original dimensions?

**Problem 6.7: Orthogonal Initialization**

Neural network layer: 50 input → 50 output neurons

a) Size of weight matrix?
b) Why initialize with orthogonal matrix?
c) How does this help gradient flow?
d) How to create orthogonal matrix? (Use Gram-Schmidt!)

**Problem 6.8: ICA vs PCA**

Audio signals: Mix of 2 independent sources

a) Would PCA separate the sources? Why/why not?
b) What's different about ICA?
c) Both find basis vectors - what's different about the bases?

**Problem 6.9: Embeddings**

Word embeddings: 50,000 words, 300 dimensions

a) Why not use 50,000 dimensions?
b) What determines actual dimensionality needed?
c) If "king - man + woman = queen" works, what does this say about the basis?

**Problem 6.10: Bringing It All Together**

You're building a recommender system:
- User-item matrix: 10,000 users × 5,000 movies
- Mostly empty (sparse)
- Want to predict missing ratings

a) What's wrong with using the data directly? (Dimensionality!)
b) How could PCA/SVD help?
c) What do the singular vectors represent?
d) How many dimensions do you really need? (Much less!)
e) Connect to concepts: basis, dimension, orthogonality

---

**End of Section 5.6: Applications to Machine Learning**

**Key takeaways:**
- ✅ PCA finds orthogonal directions of maximum variance
- ✅ Regression is orthogonal projection onto feature space
- ✅ Feature orthogonalization removes multicollinearity
- ✅ SVD provides low-rank approximations for compression
- ✅ Orthogonal initialization stabilizes neural network training
- ✅ ICA finds statistically independent components
- ✅ Embeddings capture true dimensionality of semantic spaces
- ✅ All concepts (independence, basis, orthogonality) are fundamental to ML

**Next: Section 5.7 - Comprehensive Practice Problems**

---

<a name="practice"></a>
# 7. Comprehensive Practice Problems

## Introduction: Testing Your Understanding

These problems integrate ALL concepts from this chapter:
- Linear Independence
- Span and Subspaces
- Basis and Dimension
- Orthogonality
- Gram-Schmidt Process
- Machine Learning Applications

**Approach each problem by asking:**
1. What's the underlying problem?
2. What's the root cause?
3. Which concept helps solve it?
4. How do the pieces connect?

---

## Section A: Conceptual Understanding

### Problem 7.1: The Big Picture

**Question:** You're explaining linear algebra to a friend learning ML. They ask:

*"I have 1000 features in my dataset. My professor said only 50 are 'truly independent' and I can reduce to 50 dimensions without losing much. What does this mean? How is this even possible?"*

Using concepts from this chapter, explain:

a) What does "truly independent" mean in terms of linear independence?

b) Why might 1000 features only have 50 independent directions?

c) How would you find these 50 directions? (Name the technique and explain the process)

d) What role does orthogonality play in this reduction?

e) Connect this to the concepts of basis and dimension

**Hint:** Walk through the reasoning: redundancy → dependence → finding basis → orthogonal directions → PCA

---

### Problem 7.2: Connecting the Dots

**Fill in the reasoning chain:**

*"I have vectors that are NOT orthogonal..."*

a) **Problem:** What issues does this create?

b) **Root cause:** Why aren't they orthogonal?

c) **Solution:** What process makes them orthogonal?

d) **Benefit:** Why is the orthogonal version better?

e) **Application:** Name 3 ML applications where orthogonal vectors matter

---

### Problem 7.3: Independence vs Orthogonality

**Question:** A student says: *"Independent vectors and orthogonal vectors are the same thing, right?"*

a) Are they correct? Why or why not?

b) Give an example of vectors that are:
   - Independent but NOT orthogonal
   - Orthogonal (and therefore independent)

c) If vectors are orthogonal, are they always independent? Prove it.

d) If vectors are independent, are they always orthogonal? Show counterexample.

e) Which is a stronger condition? Why?

---

## Section B: Computational Problems

### Problem 7.4: Complete Workflow - From Dependent to Orthonormal

**Given vectors in ℝ³:**
```
v₁ = (1, 2, 0)
v₂ = (2, 5, 0)
v₃ = (0, 0, 3)
```

**Part A: Analysis**

a) Are these vectors linearly independent? Test it.

b) What's the span of these vectors? Describe geometrically.

c) What's the dimension of the span?

d) Do these form a basis for ℝ³? Why or why not?

**Part B: Check Orthogonality**

e) Calculate all pairwise dot products

f) Which pairs are orthogonal?

g) Which pairs are NOT orthogonal?

**Part C: Apply Gram-Schmidt**

h) Apply Gram-Schmidt to create orthogonal set {u₁, u₂, u₃}

i) Verify orthogonality of your result

**Part D: Create Orthonormal Basis**

j) Normalize each orthogonal vector

k) Verify you have an orthonormal basis

l) Express the vector (3, 7, 6) in this new basis

---

### Problem 7.5: PCA Step-by-Step

**Dataset: 4 students, 2 test scores**

```
        Test 1   Test 2
Student A:  80      78
Student B:  90      88
Student C:  70      68
Student D:  60      58
```

**Part A: Understand the Data**

a) Plot the data points. Do scores seem correlated?

b) Calculate correlation coefficient between Test 1 and Test 2

c) What does this correlation tell you about independence?

**Part B: Apply PCA**

d) Center the data (subtract means)

e) Compute covariance matrix C = (1/n)X^T X

f) Find eigenvalues and eigenvectors of C

g) Which eigenvector is PC1? What's its eigenvalue?

h) What percentage of variance does PC1 explain?

**Part C: Interpretation**

i) What does PC1 represent? (Look at eigenvector components)

j) If you keep only PC1, what information do you lose?

k) Transform students to 1D using PC1

l) Verify: Can you recover approximate original scores?

**Part D: Connection to Concepts**

m) Is the PC basis orthogonal? Verify.

n) What's the dimension of the original space? The reduced space?

o) Why does this work? (Explain using span and basis)

---

### Problem 7.6: Regression as Projection

**Data: Predicting y from x**

```
x: [1, 2, 4, 5]
y: [2, 3, 4, 6]
```

**Part A: Set Up**

a) Form the design matrix X (include column of 1s for bias)

b) Visualize: plot the points

c) What space do predictions ŷ = Xw live in? (Describe geometrically)

**Part B: Solve**

d) Compute X^T X

e) Compute X^T y

f) Solve normal equation: (X^T X)w = X^T y

g) What's your prediction equation?

**Part C: Geometric Understanding**

h) Compute predictions ŷ = Xw

i) Compute residuals e = y - ŷ

j) Verify orthogonality: X^T e = 0 (or very close)

k) What does this orthogonality mean geometrically?

**Part D: Column Space**

l) What's the column space of X? Describe it.

m) Is y in the column space? How do you know?

n) Where is ŷ in relation to the column space?

o) Why is ŷ the "best" approximation? (Use projection concept)

---

### Problem 7.7: Feature Orthogonalization in Action

**Features from 5 houses:**

```
Area (sq ft):     [1000, 1500, 2000, 2500, 3000]
Bedrooms:         [2, 3, 4, 5, 6]
Age (years):      [5, 10, 15, 20, 25]
```

**Part A: Detect Dependence**

a) Center each feature (subtract mean)

b) Compute all pairwise dot products

c) Which features are most correlated?

d) Would you expect multicollinearity problems? Why?

**Part B: Orthogonalize**

e) Apply Gram-Schmidt: u₁ = centered Area, orthogonalize others

f) Compute the new orthogonal features {u₁, u₂, u₃}

g) Verify pairwise orthogonality

**Part C: Interpretation**

h) What does u₁ represent?

i) What does u₂ represent? (Independent of area)

j) What does u₃ represent? (Independent of both)

k) Why is this better for regression?

**Part D: Compare**

l) If you ran regression on original features, what problems might occur?

m) If you ran regression on orthogonal features, what improves?

n) Do both give same predictions? (They should!)

o) Which coefficient estimates are more stable?

---

## Section C: Theoretical Problems

### Problem 7.8: Proving Properties

**Part A: Orthogonal Sets**

Prove: If {v₁, v₂, ..., vₖ} is an orthogonal set of non-zero vectors, then it's linearly independent.

**Hint:** Assume c₁v₁ + c₂v₂ + ... + cₖvₖ = 0, then dot both sides with vᵢ

**Part B: Gram-Schmidt**

Prove: The Gram-Schmidt process preserves span.

That is, show: span{v₁, v₂, ..., vₖ} = span{u₁, u₂, ..., uₖ}

**Hint:** Show each vᵢ is in span{u₁, ..., uᵢ} and vice versa

**Part C: Projection**

Prove: If u is a unit vector, then ||v - proj_u(v)||² + ||proj_u(v)||² = ||v||²

**Hint:** This is Pythagorean theorem! Show (v - proj_u(v)) ⊥ proj_u(v)

---

### Problem 7.9: Dimension Counting

**Part A: General Principle**

In ℝⁿ, prove you cannot have more than n linearly independent vectors.

**Hint:** Think about what happens with n+1 vectors in n dimensions

**Part B: Subspaces**

a) Prove: If W is a k-dimensional subspace of ℝⁿ, then W^⊥ has dimension n-k

b) Give geometric interpretation in ℝ³

**Part C: Rank**

a) If A is m×n with rank r, what's dim(column space)?

b) What's dim(null space)?

c) Prove: rank + nullity = n

---

### Problem 7.10: Optimality of Projections

**Theorem:** proj_W(v) minimizes ||v - w|| over all w in W

**Part A:** Understand the claim
- What does this mean geometrically?
- Why would projection be closest?

**Part B:** Prove it
- Let w be any vector in W
- Show ||v - proj_W(v)||² ≤ ||v - w||²

**Hint:** Write w = proj_W(v) + (w - proj_W(v)) and use orthogonality

**Part C:** Connect to regression
- How does this relate to least squares?
- Why is the normal equation solution optimal?

---

## Section D: Real-World Applications

### Problem 7.11: Image Compression with SVD

**Scenario:** You have a 1000×1000 grayscale image (1 million pixels)

**After SVD:** You find:
- First 50 singular values are large
- Remaining 950 are negligible (< 1% of largest)

**Part A: Analysis**

a) What does this tell you about the image?

b) Why do most singular values being small matter?

c) What's the true "dimensionality" of the image data?

**Part B: Compression**

d) If you keep only 50 singular values, how much storage do you need?
   - Original: 1 million values
   - Compressed: ? values

e) Calculate compression ratio

f) Why doesn't this lose much image quality?

**Part C: Connection to Concepts**

g) The 50 singular vectors form what? (Think basis)

h) Are these vectors orthogonal? Why does that matter?

i) How is this similar to PCA?

j) Could you use Gram-Schmidt here? Where?

---

### Problem 7.12: Building a Recommender System

**Scenario:** Movie recommendations

**Data:** 1000 users × 500 movies matrix (mostly empty)

**Goal:** Predict missing ratings

**Part A: The Problem**

a) What's wrong with using the raw matrix?
   - Dimensionality issues?
   - Sparsity problems?

b) What do you suspect about the true dimensionality?
   - Are all 500 movie dimensions independent?
   - Might preferences lie in lower-dimensional space?

**Part B: Apply SVD**

c) After SVD, you keep 20 singular values. What do these represent?

d) U gives user features (1000×20). What does each column mean?

e) V gives movie features (500×20). What does each column mean?

f) How do you predict rating for user i, movie j?

**Part C: Interpretation**

g) The 20 dimensions might represent what? (Genre preferences? Actor preferences?)

h) Why is this better than original 500 dimensions?

i) How does orthogonality of singular vectors help?

j) Connect to concepts: basis, dimension reduction, independence

---

### Problem 7.13: Neural Network Initialization

**Scenario:** Training a neural network

**Layer:** 128 input neurons → 128 output neurons

**Part A: The Problem**

a) Weight matrix size?

b) If initialized randomly (say, Gaussian), what problems occur?
   - Gradient vanishing?
   - Gradient explosion?

c) What's the root cause? (Think about repeated multiplications)

**Part B: Orthogonal Initialization**

d) Generate random 128×128 matrix. Apply Gram-Schmidt. Now weights are orthogonal.

e) Why does orthogonal initialization help?
   - What property do orthogonal matrices have?
   - How does this affect gradient flow?

f) Prove: If Q is orthogonal, ||Qx|| = ||x||

**Part C: Practical Considerations**

g) Do you need EXACT orthogonality? (Computational cost?)

h) Modified Gram-Schmidt vs Classical - which for large matrices?

i) Alternative: Random orthogonal matrix (cheaper). How?

j) Why does this matter more for deep networks?

---

### Problem 7.14: Text Analysis with LSA

**Scenario:** Analyzing documents

**Term-document matrix:** 10,000 words × 1,000 documents

**Part A: Curse of Dimensionality**

a) Working in 10,000 dimensions - what problems?

b) Most word combinations never occur - what does this mean?

c) Synonym problem: "car" and "automobile" treated as different dimensions. Issue?

**Part B: Apply SVD (Latent Semantic Analysis)**

d) Keep top 100 singular values. What do these represent?

e) How does this solve synonym problem?
   - "car" and "automobile" now close in 100D space!

f) Why 100 dimensions better than 10,000?

**Part C: Geometric Interpretation**

g) The 100-dimensional space represents what? (Latent topics?)

h) Documents are projected onto this space. What does closeness mean?

i) How are the 100 dimensions chosen? (Maximum variance?)

j) Why must the basis vectors be orthogonal?

---

### Problem 7.15: Putting It All Together - ML Pipeline

**Complete scenario:** Predicting house prices

**Data:** 1,000 houses, 50 features

**Step 1: Exploration**

a) Check for linear dependence among features
   - Method?
   - What if you find dependent features?

b) Check for multicollinearity
   - Compute correlation matrix?
   - Threshold for concern?

**Step 2: Preprocessing**

c) Some features highly correlated. Options:
   - Remove redundant features?
   - Orthogonalize using Gram-Schmidt?
   - Apply PCA?

d) You choose PCA. Steps:
   - Center data
   - Compute covariance matrix
   - Find eigenvectors
   - How many components to keep?

**Step 3: Dimension Reduction**

e) You keep 15 principal components (explain 95% variance)

f) Transform data to 15D

g) Why is this better than original 50D?
   - Computational efficiency?
   - Overfitting prevention?
   - Feature independence?

**Step 4: Regression**

h) Fit regression in 15D space

i) Geometry: What is ŷ? (Projection!)

j) Residuals orthogonal to what?

**Step 5: Interpretation**

k) Coefficients now in PC space. Can you interpret?
   - PC1 might be "overall size"
   - PC2 might be "luxury vs budget"

l) How to transform back to original features if needed?

**Step 6: Reflection**

m) List every concept from this chapter you used:
   - Linear independence?
   - Basis?
   - Orthogonality?
   - Gram-Schmidt?
   - Projection?

n) How did they all connect?

o) Why does linear algebra make this possible?

---

## Section E: Challenge Problems

### Problem 7.16: Designing Your Own Basis

**Task:** Create an orthonormal basis for ℝ³ where:
- First vector points toward (1, 1, 1)
- Second vector lies in xy-plane
- Third vector completes the right-handed system

**Part A: Plan**

a) What's your strategy?

b) Which theorem/algorithm will you use?

c) Are there multiple solutions?

**Part B: Execute**

d) Normalize (1, 1, 1) → q₁

e) Choose a vector in xy-plane, orthogonalize to q₁ → q₂

f) Find q₃ (hint: cross product or Gram-Schmidt with any third vector)

g) Verify orthonormality

**Part C: Use It**

h) Express (5, 3, 2) in your new basis

i) Verify by converting back

j) Why might this basis be useful? (Applications?)

---

### Problem 7.17: Rank-Deficient Regression

**Problem:** You have perfect multicollinearity

**Features:**
```
x₁ = [1, 2, 3, 4]
x₂ = [2, 4, 6, 8]  (= 2x₁)
```

**Target:** y = [3, 5, 7, 9]

**Part A: The Problem**

a) Form design matrix X. What's its rank?

b) Try to compute (X^T X)^(-1). What happens?

c) Why can't you solve uniquely?

d) Geometric interpretation: Column space?

**Part B: Solutions**

e) Remove x₂. Solve with just x₁. What coefficient?

f) Remove x₁. Solve with just x₂. What coefficient?

g) Use x₁ + x₂ as single feature. Solve. What coefficient?

h) Do all give same predictions? Verify!

**Part C: General Lesson**

i) What's the root cause of the problem?

j) How to detect this before fitting?

k) General strategy for handling?

l) Why does linear independence matter here?

---

### Problem 7.18: Optimal Subspace for Data

**Challenge:** Given data points, find the best k-dimensional subspace

**Data (5 points in ℝ³):**
```
p₁ = (1, 2, 0)
p₂ = (2, 3, 1)
p₃ = (3, 5, 1)
p₄ = (4, 6, 2)
p₅ = (5, 8, 2)
```

**Goal:** Find best 2D subspace (plane through origin) that approximates data

**Part A: What is "Best"?**

a) Define "best" - minimize what?

b) This is PCA! Why?

c) Connection to projection?

**Part B: Solve**

d) Center the data

e) Find covariance matrix

f) Find top 2 eigenvectors (PC1 and PC2)

g) These span the best 2D subspace!

**Part C: Analysis**

h) Project each point onto the 2D subspace

i) Compute total error (sum of squared distances)

j) What percentage of variance is explained?

k) Why is this plane "optimal"?

**Part D: Theory**

l) Prove: The PC subspace minimizes total squared error

m) Why must PCs be orthogonal?

n) Could you find the plane without eigenvalues? (Yes, via Gram-Schmidt on certain directions)

---

### Problem 7.19: Condition Number and Orthogonality

**Advanced:** Numerical stability

**Part A: Condition Number**

Given matrix A, condition number κ(A) = ||A|| ||A^(-1)||
- Large κ → ill-conditioned (numerical problems)
- Small κ → well-conditioned (stable)

a) For orthogonal matrix Q, prove κ(Q) = 1 (optimal!)

b) Why does this make orthogonal matrices numerically stable?

**Part B: Gram-Schmidt vs QR**

c) Classical Gram-Schmidt can be numerically unstable. Why?

d) Modified Gram-Schmidt is better. Why?

e) Householder QR is even better. Research and explain.

**Part C: Real Impact**

f) When does numerical instability matter?
   - Small eigenvalues?
   - Nearly-dependent vectors?

g) How to detect ill-conditioning?

h) Practical strategies?

---

### Problem 7.20: Creating Synthetic Data

**Design challenge:** Create a dataset for teaching PCA

**Requirements:**
- 100 data points
- Originally 10 features
- True dimension is 3 (lies in 3D subspace of ℝ¹⁰)
- Add small noise

**Part A: Design**

a) How would you generate this?
   - Start with 3D data?
   - Create 10D via linear combinations?
   - Add noise?

b) Write the procedure step-by-step

c) Implement it (pseudocode or actual code)

**Part B: Verify**

d) Apply PCA to your synthetic data

e) Do you recover 3 principal components?

f) How much variance do they explain?

g) Can you recover the original 3D structure?

**Part C: Teaching Tool**

h) Why is synthetic data useful for teaching?

i) What concepts does this illustrate?

j) How could you modify it to teach other concepts?
   - Make features dependent?
   - Make features orthogonal from the start?
   - Vary noise levels?

---

## Section F: Reflection and Synthesis

### Problem 7.21: The Grand Connection

**Essay question:** Explain how ALL major concepts in this chapter connect to each other.

Your explanation should:

a) Start with linear independence - why it matters

b) Build to span and subspaces - what they represent

c) Introduce basis and dimension - minimal spanning sets

d) Add orthogonality - perpendicular directions

e) Show how Gram-Schmidt connects independence to orthogonality

f) Demonstrate why orthonormal bases are special

g) Connect to projections and least squares

h) Apply to PCA and SVD

i) Show relevance to machine learning

j) End with why this foundation matters for AI

**Format:** Write as if explaining to a fellow student. Use examples, analogies, and build intuition step by step.

---

### Problem 7.22: Your Own Application

**Creative challenge:** Find a new application of these concepts

**Task:** Identify a problem (in ML or elsewhere) where linear independence, orthogonality, or related concepts play a key role.

**Structure your answer:**

a) **Describe the problem**
   - What are you trying to do?
   - What data do you have?

b) **Identify the challenge**
   - What makes it hard?
   - Where does dimensionality/dependence/correlation appear?

c) **Apply concepts**
   - Which concepts from this chapter help?
   - How would you use them?

d) **Solve step-by-step**
   - Give concrete algorithm/procedure
   - Use actual linear algebra

e) **Analyze the solution**
   - Why does it work?
   - What role does each concept play?

**Examples to inspire you:**
- Signal processing (audio/video)
- Genomics (gene expression data)
- Finance (portfolio optimization)
- Computer graphics (transformations)
- Robotics (motion planning)
- Climate science (pattern detection)

---

## Answer Guide for Selected Problems

### Hints for Problem 7.4 (Complete Workflow)

**Part A:**
- Test independence: Solve c₁v₁ + c₂v₂ + c₃v₃ = 0
- Notice v₁ and v₂ both have zero z-component
- v₃ is clearly independent of the plane

**Part C:**
- u₁ = v₁ (start with first)
- For u₂: Remove projection of v₂ onto u₁
- For u₃: Remove projections onto both u₁ and u₂

**Part D:**
- Divide each uᵢ by ||uᵢ||
- Verify: qᵢ · qⱼ = δᵢⱼ (Kronecker delta)

### Hints for Problem 7.5 (PCA)

**Part B:**
- Mean of Test 1: 75
- Mean of Test 2: 73
- Center: Subtract these means
- Covariance matrix is 2×2 - easy to work with!

**Part C:**
- Eigenvector weights tell you how features combine
- Nearly equal weights → "overall ability"
- Opposite signs → "contrast"

### Hints for Problem 7.6 (Regression)

**Part A:**
```
X = [1  1]
    [1  2]
    [1  4]
    [1  5]
```

**Part C:**
- Column space of X is a 2D plane in ℝ⁴
- y might not be in this plane
- ŷ is closest point in plane to y

### Hints for Problem 7.15 (ML Pipeline)

This problem ties everything together!

**Key insights:**
- Linear dependence → Remove redundant features
- PCA → Find orthogonal directions of max variance
- Dimension reduction → Keep only important PCs
- Regression → Project onto column space
- Orthogonality → Clean, independent features

**The full pipeline uses EVERY concept from the chapter!**

---

## Final Thoughts

These problems are designed to:

1. **Test understanding** - Not just memorization
2. **Build intuition** - Geometric and algebraic
3. **Connect concepts** - See how ideas relate
4. **Apply to ML** - Real-world relevance
5. **Challenge thinking** - Go beyond basics

**Remember the core questions:**
- What's the problem?
- What's the root cause?
- How do concepts connect?
- Why does the solution work?

**Most importantly:** Linear algebra isn't just math - it's the language of data, geometry, and relationships. Master these foundations, and you master the mathematics of machine learning!

---

**End of Chapter 5: Linear Independence, Basis, and Orthogonality**

**Congratulations on completing this comprehensive chapter!**

**You now understand:**
- ✅ When vectors are truly independent
- ✅ How to find minimal spanning sets (bases)
- ✅ Why orthogonality makes computations easier
- ✅ How to create orthogonal bases (Gram-Schmidt)
- ✅ Why these concepts power modern machine learning
- ✅ How to apply theory to real problems

**Next steps:**
- Practice these problems
- Implement algorithms in code
- Apply to your own datasets
- Explore advanced topics (eigendecomposition, SVD details)
- Build intuition through visualization

**Keep asking "why?" and building from first principles!**
