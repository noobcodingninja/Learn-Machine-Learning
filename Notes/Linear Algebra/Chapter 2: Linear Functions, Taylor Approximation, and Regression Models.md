# Linear Algebra for Machine Learning
## Chapter 2: Linear Functions, Taylor Approximation, and Regression Models

### A First-Principles Approach with Detailed Examples

---

# Table of Contents

1. [Linear Functions](#linear-functions)
2. [Taylor Approximation](#taylor-approximation)
3. [Regression Models](#regression-models)
4. [Chapter Summary](#summary)
5. [Comprehensive Practice Problems](#practice)

---

<a name="linear-functions"></a>
# 1. Linear Functions

## What Problem Are We Solving?

Imagine you're running a coffee shop and trying to predict your daily revenue.

You notice patterns:
- More customers → more revenue (obviously!)
- Larger average purchase → more revenue
- Rainy days → less foot traffic → less revenue

**Question:** Can we create a simple mathematical function that predicts revenue from these factors?

**Idea:** What if revenue is just a weighted combination of these features?

Revenue = a₁(customers) + a₂(avg_purchase) + a₃(weather_factor) + base_amount

This is a **linear function**!

## What Does "Linear" Actually Mean?

Let's start with the simplest case: one input, one output.

### Example: Apple Pricing

You're at a fruit stand:
- 1 apple costs $2
- 2 apples cost $4
- 3 apples cost $6
- x apples cost 2x dollars

**Function:** f(x) = 2x

This is linear! But what makes it "linear"?

### The Two Defining Properties

A function f is **linear** if it satisfies BOTH:

**Property 1: Homogeneity (Scaling)**
f(αx) = αf(x) for any scalar α

**Translation:** If you scale the input by α, the output scales by α.

**Example with apples:**
- f(10) = 2(10) = 20 dollars
- f(2·5) = 2(10) = 20
- 2·f(5) = 2(10) = 20 ✓

Buy twice as much → pay twice as much!

**Property 2: Additivity**
f(x + y) = f(x) + f(y)

**Translation:** The function of a sum equals the sum of the functions.

**Example with apples:**
- Buy 3 apples: f(3) = 6
- Buy 5 apples: f(5) = 10
- Buy 3+5=8 apples: f(8) = 16
- f(3) + f(5) = 6 + 10 = 16 ✓

Two separate purchases cost the same as one combined purchase!

### What If There's a Constant Term?

**New scenario:** The store charges a $5 bag fee plus $2 per apple.

g(x) = 2x + 5

**Is this linear?**

Test scaling:
- g(2·1) = g(2) = 2(2) + 5 = 9
- 2·g(1) = 2(2·1 + 5) = 2(7) = 14
- 9 ≠ 14 ✗

**No! This is NOT linear because of the +5 constant.**

This is called **affine** (linear plus a constant).

### The Critical Distinction

**Linear:** f(x) = mx (passes through origin)
- f(0) = 0 always!
- No constant term

**Affine:** f(x) = mx + b (can have y-intercept)
- f(0) = b (doesn't have to be zero)
- Has constant term

**In everyday language:** People often say "linear" when they mean "affine"

**In mathematics:** We're precise about the distinction

**In machine learning:** We often use "linear model" to mean affine (with bias term)

## Extending to Vectors: Multivariate Linear Functions

Now let's move from single numbers to vectors!

### Example: Smoothie Shop Pricing

Your smoothie shop prices:
- Bananas: $1 each
- Strawberries: $2 per cup
- Protein powder: $3 per scoop

Customer orders: **x** = (2, 1, 1)
- 2 bananas
- 1 cup strawberries
- 1 scoop protein

**How much do they pay?**

Price = 1(2) + 2(1) + 3(1) = 2 + 2 + 3 = $7

**This is an inner product!**

Prices vector: **a** = (1, 2, 3)
Order vector: **x** = (2, 1, 1)
Total: f(**x**) = **a · x** = 7

### General Form

A **linear function** from ℝⁿ to ℝ has the form:

f(**x**) = **a · x** = a₁x₁ + a₂x₂ + ... + aₙxₙ

where **a** = (a₁, a₂, ..., aₙ) is the coefficient vector.

**Properties:**
- Input: vector **x** ∈ ℝⁿ
- Output: scalar (single number)
- Defined by coefficient vector **a**

### Verifying Linear Properties with Vectors

**Property 1: Scaling**
f(α**x**) = **a** · (α**x**) = α(**a · x**) = αf(**x**) ✓

**Example:**
- Order: **x** = (2, 1, 1), Cost: f(**x**) = 7
- Double order: **2x** = (4, 2, 2)
- f(**2x**) = 1(4) + 2(2) + 3(2) = 4 + 4 + 6 = 14
- 2·f(**x**) = 2(7) = 14 ✓

**Property 2: Additivity**
f(**x + y**) = **a** · (**x + y**) = **a · x** + **a · y** = f(**x**) + f(**y**) ✓

**Example:**
- Order 1: **x** = (2, 1, 1), Cost: 7
- Order 2: **y** = (1, 0, 1), Cost: 1(1) + 2(0) + 3(1) = 4
- Combined: **x + y** = (3, 1, 2)
- f(**x + y**) = 1(3) + 2(1) + 3(2) = 3 + 2 + 6 = 11
- f(**x**) + f(**y**) = 7 + 4 = 11 ✓

## The Geometric Interpretation

### In 2D: Lines Through the Origin

f(x₁, x₂) = a₁x₁ + a₂x₂

Example: f(x₁, x₂) = 2x₁ + 3x₂

**Level sets** (where f equals constant c):
- f(**x**) = 0: The line 2x₁ + 3x₂ = 0
- f(**x**) = 6: The line 2x₁ + 3x₂ = 6
- f(**x**) = -6: The line 2x₁ + 3x₂ = -6

These are parallel lines! The coefficient vector **a** = (2, 3) is perpendicular to these lines.

**Key insight:** The coefficient vector points in the direction of maximum increase!

### In 3D: Planes Through the Origin

f(x₁, x₂, x₃) = a₁x₁ + a₂x₂ + a₃x₃

Example: f(x₁, x₂, x₃) = x₁ + 2x₂ + 3x₃

Level sets are planes:
- f(**x**) = 0: plane through origin
- f(**x**) = 6: parallel plane
- Normal vector: **a** = (1, 2, 3)

### In Higher Dimensions: Hyperplanes

The concept extends to any dimension! Level sets are hyperplanes.

## Why Machine Learning Cares About Linear Functions

**Linear functions are the foundation of machine learning!**

### 1. Linear Regression: The Simplest ML Model

**Goal:** Predict a number from features

**Model:** prediction = **w · x**

Example - House price prediction:
- Features: **x** = (sqft, bedrooms, age, school_rating)
- Weights: **w** = (100, 50000, -1000, 20000)
- Prediction: price = **w · x**

For house **x** = (2000, 3, 10, 8):
price = 100(2000) + 50000(3) + (-1000)(10) + 20000(8)
= 200,000 + 150,000 - 10,000 + 160,000
= $500,000

**Training = finding the best weights **w****!

### 2. Linear Classification: Decision Boundaries

**Goal:** Classify into categories

**Model:** 
- Calculate score = **w · x**
- If score > 0 → Class 1
- If score < 0 → Class 2

The boundary (score = 0) is a hyperplane!

Example - Spam detection:
- Features: **x** = (capitals, exclamations, ...)
- Weights: **w** = learned from data
- If **w · x** > threshold → Spam, else → Ham

### 3. Neural Networks: Stacks of Linear Functions

Every neuron computes:
1. Linear combination: z = **w · x** + b
2. Nonlinear activation: a = σ(z)

**Deep learning = many layers of linear functions plus nonlinearities!**

Without the linear functions, neural networks couldn't work!

### 4. Feature Importance

The coefficient vector **w** tells you feature importance!

Example weights: **w** = (100, 50000, -1000, 20000)

**Interpretation:**
- sqft coefficient = 100 → $100 per square foot
- bedrooms = 50000 → $50k per bedroom
- age = -1000 → -$1k per year (negative = bad!)
- school = 20000 → $20k per rating point

Large magnitude = important feature!

### 5. Dimensionality Reduction

**Principal Component Analysis (PCA):**
Projects data onto linear combinations of features that capture most variance.

New features are linear functions of original features!

## Affine Functions: Adding the Bias Term

In practice, ML models include a constant term:

f(**x**) = **w · x** + b

where b is called the **bias** or **intercept**.

**Why do we need it?**

### Example: Baseline Prediction

Predicting exam scores:
- Features: **x** = (study_hours, attendance, sleep_hours)
- Linear: f(**x**) = 5(study) + 30(attendance) + 2(sleep)

**Problem:** If all features are zero (no study, no attendance, no sleep):
f(**0**) = 0 score

**But that doesn't make sense!** Even doing nothing might get 25% by guessing!

**Solution:** Add bias:
f(**x**) = 5(study) + 30(attendance) + 2(sleep) + 25

Now f(**0**) = 25 (baseline score)

### The Bias Term Shifts the Function

**Geometric interpretation:**
- Without bias: hyperplane passes through origin
- With bias: hyperplane can be anywhere

**For house prices:**
- **w · x** = value from features
- b = base price (land value, market baseline, etc.)

### Notation in Machine Learning

You'll often see:
- ŷ = **w · x** + b (prediction)
- y = actual value
- Goal: minimize difference between ŷ and y

Or in matrix form:
- ŷ = **Xw** + b (for multiple examples at once)

## Detailed Examples

### Example 1.1: Student Grade Prediction

Features:
- x₁ = hours studied
- x₂ = hours slept  
- x₃ = attendance rate (0-1)

Linear function: f(**x**) = 5x₁ + 2x₂ + 30x₃ + 40

**Student A:** (10 hours, 8 hours, 0.9)
grade = 5(10) + 2(8) + 30(0.9) + 40
= 50 + 16 + 27 + 40
= 133 (cap at 100, so → 100%)

**Student B:** (5 hours, 6 hours, 0.5)
grade = 5(5) + 2(6) + 30(0.5) + 40
= 25 + 12 + 15 + 40
= 92%

**Student C:** (0 hours, 0 hours, 0)
grade = 40 (the bias term = baseline)

**Interpretation:**
- Each study hour worth 5 points
- Each sleep hour worth 2 points
- Attendance very important (30 points for perfect)
- Baseline 40 points (from random guessing, partial credit)

### Example 1.2: Credit Score Model

Features:
- x₁ = payment history (0-100)
- x₂ = credit utilization (0-1, lower is better)
- x₃ = length of history (years)
- x₄ = number of recent inquiries

Linear function: f(**x**) = 3x₁ - 200x₂ + 5x₃ - 10x₄ + 300

**Person A:** (80, 0.3, 10, 2)
score = 3(80) - 200(0.3) + 5(10) - 10(2) + 300
= 240 - 60 + 50 - 20 + 300
= 510

**Person B:** (95, 0.1, 15, 0)
score = 3(95) - 200(0.1) + 5(15) - 10(0) + 300
= 285 - 20 + 75 + 0 + 300
= 640 (excellent credit!)

**Interpretation:**
- Payment history: very important (coefficient = 3)
- Credit utilization: penalized heavily (-200)
- History length: modest positive (5)
- Recent inquiries: small penalty (-10)
- Baseline: 300 (starting point)

### Example 1.3: Restaurant Revenue Prediction

Features:
- x₁ = number of customers
- x₂ = average check size ($)
- x₃ = day of week (1-7)
- x₄ = weather quality (0-1, 0=bad, 1=perfect)

Linear function: f(**x**) = 45x₁ + 30x₂ + 50x₃ + 200x₄ + 500

**Friday, great weather, 100 customers, $25 average:**
**x** = (100, 25, 5, 0.9)
revenue = 45(100) + 30(25) + 50(5) + 200(0.9) + 500
= 4500 + 750 + 250 + 180 + 500
= $6,180

**Monday, rain, 40 customers, $20 average:**
**x** = (40, 20, 1, 0.3)
revenue = 45(40) + 30(20) + 50(1) + 200(0.3) + 500
= 1800 + 600 + 50 + 60 + 500
= $3,010

**Interpretation:**
- Customers most important (45 per person)
- Check size matters (30 per dollar of check)
- Weekend bonus (50 per day number)
- Weather impact (200 for perfect weather)
- Base revenue: $500 (minimal day)

### Example 1.4: Verifying Linearity

Function: f(**x**) = 2x₁ + 3x₂

**Test 1: Scaling**
**x** = (1, 1), f(**x**) = 2(1) + 3(1) = 5

**2x** = (2, 2)
f(**2x**) = 2(2) + 3(2) = 4 + 6 = 10
2·f(**x**) = 2(5) = 10 ✓

**Test 2: Additivity**
**x** = (1, 2), f(**x**) = 2(1) + 3(2) = 2 + 6 = 8
**y** = (3, 1), f(**y**) = 2(3) + 3(1) = 6 + 3 = 9

**x + y** = (4, 3)
f(**x + y**) = 2(4) + 3(3) = 8 + 9 = 17
f(**x**) + f(**y**) = 8 + 9 = 17 ✓

**Confirmed linear!**

### Example 1.5: Not Linear Due to Bias

Function: g(**x**) = 2x₁ + 3x₂ + 5

**Test scaling:**
**x** = (1, 1), g(**x**) = 2 + 3 + 5 = 10

**2x** = (2, 2)
g(**2x**) = 2(2) + 3(2) + 5 = 4 + 6 + 5 = 15
2·g(**x**) = 2(10) = 20
15 ≠ 20 ✗

**Not linear! (But affine)**

## Practice Problems - Linear Functions

**Problem 1.1: Identifying Linear Functions**

Which of these are linear functions (not affine)?
a) f(x) = 3x
b) f(x) = 3x + 2
c) f(x) = x²
d) f(**x**) = 2x₁ + 5x₂
e) f(**x**) = 2x₁ + 5x₂ + 1
f) f(**x**) = x₁x₂

**Problem 1.2: Coffee Shop Pricing**

Your coffee shop prices:
- Coffee: $3 per cup
- Pastry: $4 each
- Sandwich: $7 each

a) Write the pricing function f(**x**) where **x** = (coffees, pastries, sandwiches)
b) Customer orders (2, 1, 1). What's the total?
c) Is this function linear? Verify the two properties.
d) If you add a $2 service fee, write the new function. Is it still linear?

**Problem 1.3: House Price Model**

Linear model: price = 150(sqft) + 40000(bedrooms) + (-500)(age) + 15000(school)

Predict prices for:
a) House 1: (1500 sqft, 2 bed, 20 years, 7 school)
b) House 2: (2500 sqft, 4 bed, 5 years, 9 school)
c) House 3: (1800 sqft, 3 bed, 15 years, 6 school)

d) Which feature has the biggest impact per unit?
e) If a house gets one year older, how does price change?

**Problem 1.4: Verifying Properties**

For f(**x**) = 3x₁ + 2x₂ + x₃:

a) Calculate f((1, 2, 3))
b) Calculate f((2, 4, 6)) and verify f(2**x**) = 2f(**x**)
c) Calculate f((1, 0, 1)) and f((0, 1, 1))
d) Verify f(**x + y**) = f(**x**) + f(**y**) where **x** = (1, 0, 1) and **y** = (0, 1, 1)

**Problem 1.5: Feature Importance**

You train a spam classifier with weights:
**w** = (10, 5, 15, 2, 20)

for features: (capitals, exclamations, urgent_words, links, misspellings)

a) Which feature is most important for detecting spam?
b) Which is least important?
c) Classify email (5, 2, 3, 1, 4). If threshold is 100, is it spam?
d) What's the minimum number of misspellings needed to classify as spam (other features zero)?

**Problem 1.6: Temperature Conversion**

Fahrenheit to Celsius: C = (5/9)(F - 32)

a) Is this a linear function of F? Check both properties.
b) Rewrite as C = aF + b. What are a and b?
c) Why does the -32 make it affine rather than linear?
d) What would a truly linear temperature conversion look like?

**Problem 1.7: Combining Linear Functions**

f(**x**) = 2x₁ + 3x₂
g(**x**) = x₁ - x₂

a) Calculate (f + g)(**x**) = f(**x**) + g(**x**). Is it linear?
b) Calculate (3f)(**x**) = 3·f(**x**). Is it linear?
c) Calculate (f · g)(**x**) = f(**x**) · g(**x**). Is it linear?
d) What can you conclude about sums and scalar multiples of linear functions?

---

<a name="taylor-approximation"></a>
# 2. Taylor Approximation

## The Big Problem: Real-World Relationships Aren't Linear

Linear functions are simple and beautiful, but the real world is messy!

### Example: Ice Cream Sales vs Temperature

Relationship probably ISN'T linear:
- 0°C: Nobody buys ice cream (too cold!)
- 20°C: Some sales
- 35°C: Peak sales!
- 50°C: Sales drop (too hot to go outside!)

The true relationship might be quadratic or more complex:
sales(t) = -0.1t² + 8t - 50

**Problem:** This isn't linear! It's harder to:
- Optimize
- Understand
- Compute with

**Question:** Can we approximate complicated functions with simpler linear ones?

**Answer:** Yes! Using Taylor approximation!

## The Core Idea: Local Linearization

**Key insight:** Even if a function is globally nonlinear, it looks approximately linear if you zoom in close enough!

Think about Earth:
- Globally: It's a sphere (curved!)
- Locally: Looks flat when standing on it

**Same with functions:**
- Globally: Might be complex curve
- Locally (near a point): Approximately a line!

## The Simplest Case: Approximating √x

### The Problem

You need to calculate √10 but you only know √9 = 3.

**Question:** Can you estimate √10 using what you know?

### Building the Approximation

**Step 1: What do we know?**
- At x = 9: f(9) = √9 = 3
- We want: f(10) = √10 = ?

**Step 2: How fast is the function changing?**

The derivative tells us the rate of change:
f(x) = √x = x^(1/2)
f'(x) = (1/2)x^(-1/2) = 1/(2√x)

At x = 9: f'(9) = 1/(2√9) = 1/6

**Interpretation:** Near x = 9, for each unit increase in x, f increases by about 1/6.

**Step 3: Approximate**

From x = 9 to x = 10 is a change of Δx = 1.

Expected change in f: f'(9) · Δx = (1/6)(1) = 1/6

**Approximation:**
√10 ≈ √9 + (change)
√10 ≈ 3 + 1/6
√10 ≈ 3.167

**Actual value:** √10 = 3.162...

**Error:** 0.005 (very close!)

### The General Formula

**First-order Taylor approximation** around point x = a:

f(x) ≈ f(a) + f'(a)(x - a)

**Components:**
- f(a) = value at the point
- f'(a) = slope (rate of change) at the point
- (x - a) = how far we've moved from a

**Geometric meaning:** This is the equation of the **tangent line** at x = a!

## Visualizing Taylor Approximation

Imagine a curve f(x):
- At point a, draw the tangent line
- This tangent line is your linear approximation
- Close to a: line ≈ curve (good approximation)
- Far from a: line diverges from curve (poor approximation)

**The approximation is best near the expansion point!**

## Multivariate Taylor Approximation

Now extend to functions of vectors: f(**x**) where **x** ∈ ℝⁿ

### Example: Predicting Sales from Multiple Factors

sales(price, advertising, season) = some complex function

At current state **a** = (10, 5000, 3):
- Price: $10
- Advertising: $5000
- Season: 3 (summer)
- Sales: f(**a**) = 15,000 units

**Question:** If we change to **x** = (11, 5200, 3), what happens to sales?

### The Gradient: Multivariate Derivative

In multiple dimensions, the derivative is the **gradient**:

∇f(**a**) = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ) evaluated at **a**

**Components:**
- ∂f/∂x₁ = rate of change with respect to x₁ (holding others fixed)
- ∂f/∂x₂ = rate of change with respect to x₂ (holding others fixed)
- etc.

**Geometric meaning:** The gradient points in the direction of steepest increase!

### Multivariate Taylor Formula

**First-order approximation** around point **a**:

f(**x**) ≈ f(**a**) + ∇f(**a**) · (**x** - **a**)

**This is a linear function of (x - a)!**

**Components:**
- f(**a**) = value at point **a** (scalar)
- ∇f(**a**) = gradient at **a** (vector)
- (**x** - **a**) = displacement from **a** (vector)
- ∇f(**a**) · (**x** - **a**) = inner product (scalar)

### Detailed Example

sales(p, a, s) = complex function

At **current** = (10, 5000, 3):
- sales = 15,000 units

Gradient at current:
∇sales = (-500, 2, 1000)

**Interpretation:**
- ∂sales/∂price = -500 (increase price $1 → lose 500 units)
- ∂sales/∂advertising = 2 (increase ad $1 → gain 2 units)
- ∂sales/∂season = 1000 (each season unit → gain 1000 units)

**New state:** **x** = (11, 5200, 3)
Change: **x** - **current** = (1, 200, 0)

**Approximate new sales:**
sales(**x**) ≈ 15000 + (-500, 2, 1000) · (1, 200, 0)
= 15000 + (-500)(1) + (2)(200) + (1000)(0)
= 15000 - 500 + 400 + 0
= 14,900 units

**Interpretation:**
- Price increase hurts: -500 units
- More advertising helps: +400 units
- Net effect: -100 units

## Why Machine Learning Cares About Taylor Approximation

Taylor approximation is EVERYWHERE in ML!

### 1. Gradient Descent: The Core of Training

**Problem:** Minimize loss function L(**w**)

At current weights **w**, we want to find better weights.

**Taylor approximation:**
L(**w** + Δ**w**) ≈ L(**w**) + ∇L(**w**) · Δ**w**

**Interpretation:**
- If we move by Δ**w**, loss changes by approximately ∇L(**w**) · Δ**w**
- To decrease loss, move opposite to gradient: Δ**w** = -α∇L(**w**)
- This is gradient descent!

**The entire training process uses local linear approximations!**

### 2. Newton's Method: Second-Order Approximation

Include curvature information (second derivatives):

L(**w** + Δ**w**) ≈ L(**w**) + ∇L(**w**) · Δ**w** + (1/2)Δ**w**ᵀ H Δ**w**

where H is the Hessian (matrix of second derivatives).

**Faster convergence than gradient descent!** (But more expensive per step)

### 3. Understanding Neural Networks Locally

A deep neural network f(**x**; **θ**) is highly nonlinear globally.

But **locally** around a point **x₀**:

f(**x**) ≈ f(**x₀**) + ∇f(**x₀**) · (**x** - **x₀**)

**This helps us:**
- Understand sensitivity to inputs
- Compute gradients for backpropagation
- Analyze model behavior

### 4. Sensitivity Analysis

**Question:** "If feature x₁ increases by 1, how does prediction change?"

**Answer:** Look at ∂f/∂x₁ (the gradient component)!

Example - House prices:
∇price = (100, 50000, -1000, 20000)

**Interpretation:**
- +1 sqft → +$100
- +1 bedroom → +$50,000
- +1 year age → -$1,000
- +1 school rating → +$20,000

**The gradient tells you feature importance!**

### 5. Linearization for Optimization

Many optimization problems are hard when nonlinear.

**Strategy:**
1. Linearize around current point
2. Solve the linear problem (easy!)
3. Move to new point
4. Repeat

**This is the basis of:**
- Sequential quadratic programming
- Trust region methods
- Many other optimization algorithms

## Detailed Examples

### Example 2.1: Square Root Approximation

Approximate √17 using √16 = 4.

**Function:** f(x) = √x
**Derivative:** f'(x) = 1/(2√x)
**Expansion point:** a = 16

**At a = 16:**
- f(16) = 4
- f'(16) = 1/(2·4) = 1/8

**Taylor approximation:**
f(17) ≈ f(16) + f'(16)(17 - 16)
√17 ≈ 4 + (1/8)(1)
√17 ≈ 4.125

**Actual:** √17 = 4.123...
**Error:** 0.002 (very good!)

### Example 2.2: Exponential Approximation

Approximate e^0.1 using e^0 = 1.

**Function:** f(x) = e^x
**Derivative:** f'(x) = e^x
**Expansion point:** a = 0

**At a = 0:**
- f(0) = 1
- f'(0) = 1

**Taylor approximation:**
e^0.1 ≈ 1 + (1)(0.1)
e^0.1 ≈ 1.1

**Actual:** e^0.1 = 1.1052...
**Error:** 0.0052 (good for small values!)

### Example 2.3: Logarithm Approximation

Approximate ln(1.1) using ln(1) = 0.

**Function:** f(x) = ln(x)
**Derivative:** f'(x) = 1/x
**Expansion point:** a = 1

**At a = 1:**
- f(1) = 0
- f'(1) = 1

**Taylor approximation:**
ln(1.1) ≈ 0 + (1)(1.1 - 1)
ln(1.1) ≈ 0.1

**Actual:** ln(1.1) = 0.0953...
**Error:** 0.0047 (pretty close!)

### Example 2.4: Multivariate Example - Production Function

Production function: Q(L, K) = L^0.7 · K^0.3 (Cobb-Douglas)

Current state: L = 100 workers, K = 50 capital units
Q(100, 50) = 100^0.7 · 50^0.3 ≈ 73.68 units

**Gradients:**
∂Q/∂L = 0.7 · L^(-0.3) · K^0.3
∂Q/∂K = 0.3 · L^0.7 · K^(-0.7)

At (100, 50):
∂Q/∂L = 0.7 · 100^(-0.3) · 50^0.3 ≈ 0.516
∂Q/∂K = 0.3 · 100^0.7 · 50^(-0.7) ≈ 0.442

**Gradient:** ∇Q = (0.516, 0.442)

**Question:** What if we hire 2 more workers and add 1 capital unit?

**New state:** (102, 51)
**Change:** Δ = (2, 1)

**Approximation:**
Q(102, 51) ≈ 73.68 + (0.516, 0.442) · (2, 1)
= 73.68 + 0.516(2) + 0.442(1)
= 73.68 + 1.032 + 0.442
= 75.15 units

**Actual:** Q(102, 51) = 75.13 units
**Error:** 0.02 (excellent!)

**Interpretation:**
- Each additional worker adds ~0.516 units
- Each capital unit adds ~0.442 units
- Workers slightly more productive at margin

### Example 2.5: Gradient Descent Step

Loss function: L(**w**) at current **w** = (2, -1, 3)
- L(2, -1, 3) = 5.2

Gradient: ∇L = (4, -2, 6)

**Interpretation:**
- Increasing w₁ increases loss (∂L/∂w₁ = 4 > 0)
- Increasing w₂ decreases loss (∂L/∂w₂ = -2 < 0)
- Increasing w₃ increases loss (∂L/∂w₃ = 6 > 0)

**To decrease loss, move opposite to gradient:**
Learning rate: α = 0.1
Update: Δ**w** = -α∇L = -0.1(4, -2, 6) = (-0.4, 0.2, -0.6)

**New weights:**
**w_new** = (2, -1, 3) + (-0.4, 0.2, -0.6)
= (1.6, -0.8, 2.4)

**Approximate new loss:**
L(**w_new**) ≈ L(**w**) + ∇L · Δ**w**
= 5.2 + (4, -2, 6) · (-0.4, 0.2, -0.6)
= 5.2 + 4(-0.4) + (-2)(0.2) + 6(-0.6)
= 5.2 - 1.6 - 0.4 - 3.6
= -0.4

**The loss decreased!** (from 5.2 to approximately -0.4)

This is how gradient descent works!

## When Does Taylor Approximation Work Well?

**Good approximation when:**
1. **Close to expansion point:** |x - a| is small
2. **Function is smooth:** Derivatives exist and behave nicely
3. **Function nearly linear locally:** Not too much curvature

**Poor approximation when:**
1. **Far from expansion point:** Large |x - a|
2. **Function has discontinuities:** Jumps, kinks
3. **High curvature:** Function curves sharply

### Example: Good vs Bad Approximation

Function: f(x) = x²

Expansion point: a = 2
f(2) = 4
f'(2) = 2(2) = 4

**Taylor approximation:** f(x) ≈ 4 + 4(x - 2)

**Close to a = 2:**
- x = 2.1: True = 4.41, Approx = 4.4 (error = 0.01) ✓
- x = 1.9: True = 3.61, Approx = 3.6 (error = 0.01) ✓

**Far from a = 2:**
- x = 5: True = 25, Approx = 16 (error = 9) ✗
- x = 10: True = 100, Approx = 36 (error = 64) ✗✗

**The approximation degrades as we move away!**

## Practice Problems - Taylor Approximation

**Problem 2.1: Basic Approximations**

Use Taylor approximation around the given point:

a) Approximate √26 using √25 = 5
b) Approximate e^(-0.05) using e^0 = 1
c) Approximate ln(0.95) using ln(1) = 0
d) Approximate cos(0.1) using cos(0) = 1 and sin(0) = 0

**Problem 2.2: Error Analysis**

For f(x) = x², approximate f(3) using expansion at a = 2.

a) Calculate the Taylor approximation
b) Calculate the true value
c) Calculate the error
d) Now try approximating f(5) from a = 2. How's the error?
e) What do you notice about error as distance increases?

**Problem 2.3: Multivariate Approximation**

Function: f(x, y) = x² + 2xy + y²

At point (1, 1): f(1, 1) = 4

Gradients: ∂f/∂x = 2x + 2y, ∂f/∂y = 2x + 2y

a) Calculate ∇f(1, 1)
b) Approximate f(1.1, 0.9) using Taylor expansion
c) Calculate true f(1.1, 0.9)
d) What's the error?

**Problem 2.4: Gradient Descent**

Loss function at **w** = (1, 2): L = 10
Gradient: ∇L = (6, -4)

a) What direction should you move to decrease loss?
b) With learning rate α = 0.1, calculate the update
c) What are the new weights?
d) Approximate the new loss value
e) Did the loss decrease?

**Problem 2.5: Sensitivity Analysis**

Revenue function: R(p, a) where p = price, a = advertising

Current: p = 50, a = 1000, R = 100,000

Gradient: ∇R = (-200, 50)

a) If price increases by $1, how does revenue change?
b) If advertising increases by $100, how does revenue change?
c) Should you increase price or advertising to boost revenue?
d) Approximate revenue if (p, a) = (48, 1200)

**Problem 2.6: Production Optimization**

Production: Q(L, K) = 20√L + 10√K

Current: L = 100, K = 25, Q = 250

a) Calculate ∂Q/∂L and ∂Q/∂K at current point
b) If you can hire 5 workers or buy 5 capital units (same cost), which is better?
c) Approximate Q(105, 25) and Q(100, 30)
d) Which matches your answer from (b)?

**Problem 2.7: Understanding Limits**

f(x) = e^x, expand around a = 0

a) What is f(0) and f'(0)?
b) Write Taylor approximation
c) Test approximation for x = 0.01, 0.1, 0.5, 1.0
d) For which values is approximation good (error < 5%)?
e) Why does approximation degrade for larger x?

---

<a name="regression-models"></a>
# 3. Regression Models

## The Ultimate Problem: Learning from Data

You have data about houses and their prices:
- House 1: 1500 sqft → sold for $300k
- House 2: 2000 sqft → sold for $400k
- House 3: 1200 sqft → sold for $250k
- House 4: 1800 sqft → sold for $350k
- House 5: 2500 sqft → sold for $480k

**Question:** A new house is 2200 sqft. What should it sell for?

**This is regression:** Finding a function that fits the data and can predict new values!

## Why Use Linear Functions for Regression?

We could fit ANY function to this data:
- Polynomial: price = a₀ + a₁·sqft + a₂·sqft² + ...
- Exponential: price = a·e^(b·sqft)
- Neural network: price = complex_function(sqft)

**Why start with linear?**

**Reason 1: Simplicity**
- Easy to understand
- Easy to compute
- Easy to interpret

**Reason 2: Often "good enough"**
- Many relationships are approximately linear
- Even if not globally linear, locally linear (Taylor!)

**Reason 3: Foundation**
- Master linear first
- Then extend to nonlinear

**Reason 4: Computational**
- Closed-form solution exists!
- Don't need iterative optimization

## The Linear Regression Model

**Goal:** Find the best linear function that fits the data.

**Model:** ŷ = **w**ᵀ**x** + b

Where:
- **x** = input features (vector)
- **w** = weights (vector) - to be learned!
- b = bias (scalar) - to be learned!
- ŷ = prediction (scalar)

**For simple case (one feature):** ŷ = wx + b

This is just a line! The classic y = mx + b from algebra.

## Setting Up the Problem

**Given:** n data points
- (**x**₁, y₁), (**x**₂, y₂), ..., (**xₙ**, yₙ)

Where:
- **xᵢ** = feature vector for example i
- yᵢ = true value for example i

**Goal:** Find **w** and b such that predictions are close to true values.

**For each point i:**
- Prediction: ŷᵢ = **w**ᵀ**xᵢ** + b
- True value: yᵢ
- Error: eᵢ = yᵢ - ŷᵢ

## What Makes a "Good" Fit?

We need to measure how good our predictions are.

### Bad Idea 1: Total Error

Minimize: Σ eᵢ = Σ (yᵢ - ŷᵢ)

**Problem:** Positive and negative errors cancel!

Example:
- Point 1: error = +10 (predicted too low)
- Point 2: error = -10 (predicted too high)
- Total error = 0 (looks perfect, but it's not!)

### Bad Idea 2: Total Absolute Error

Minimize: Σ |eᵢ|

**Better!** All errors count positively.

**Problem:** Absolute value is not differentiable at zero. Makes optimization harder.

### Good Idea: Sum of Squared Errors (Least Squares)

Minimize: Σ eᵢ² = Σ (yᵢ - ŷᵢ)²

**Why this works:**
1. Squaring makes all errors positive
2. Differentiable everywhere (smooth optimization)
3. Penalizes large errors more (quadratic growth)
4. Has nice statistical properties
5. Leads to closed-form solution!

## The Loss Function

**Mean Squared Error (MSE):**

L(**w**, b) = (1/n) Σᵢ₌₁ⁿ (yᵢ - (**w**ᵀ**xᵢ** + b))²

Or equivalently:

L(**w**, b) = (1/n) Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²

**Interpretation:**
- Measures average squared error across all points
- Lower is better (predictions closer to truth)
- Zero means perfect fit (ŷᵢ = yᵢ for all i)

**Our goal:** Find **w** and b that minimize L.

## Solving Simple Linear Regression (One Feature)

Let's start with the simplest case: one input feature.

**Model:** ŷ = wx + b

**Data:** (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)

**Loss:** L(w, b) = (1/n) Σ (yᵢ - (wxᵢ + b))²

**To minimize, take derivatives and set to zero:**

∂L/∂w = 0
∂L/∂b = 0

### Deriving the Solution

**Step 1: Expand the loss**

L = (1/n) Σ (yᵢ - wxᵢ - b)²

**Step 2: Derivative with respect to b**

∂L/∂b = (2/n) Σ (yᵢ - wxᵢ - b)(-1)
= -(2/n) Σ (yᵢ - wxᵢ - b)

Set to zero:
Σ (yᵢ - wxᵢ - b) = 0
Σ yᵢ - w Σ xᵢ - nb = 0

**Solving for b:**
b = (Σ yᵢ - w Σ xᵢ) / n
b = ȳ - wx̄

Where ȳ = mean of y values, x̄ = mean of x values

**Key insight:** The line passes through the point (x̄, ȳ)!

**Step 3: Derivative with respect to w**

∂L/∂w = (2/n) Σ (yᵢ - wxᵢ - b)(-xᵢ)
= -(2/n) Σ xᵢ(yᵢ - wxᵢ - b)

Set to zero:
Σ xᵢ(yᵢ - wxᵢ - b) = 0
Σ xᵢyᵢ - w Σ xᵢ² - b Σ xᵢ = 0

Substitute b = ȳ - wx̄:
Σ xᵢyᵢ - w Σ xᵢ² - (ȳ - wx̄) Σ xᵢ = 0

After algebra:
**w = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²**

This is the **closed-form solution**!

## Detailed Example: House Prices (One Feature)

**Data:** (sqft, price)
- (1500, 300)
- (2000, 400)
- (1200, 250)
- (1800, 350)
- (2500, 480)

n = 5 points

**Step 1: Calculate means**
x̄ = (1500 + 2000 + 1200 + 1800 + 2500) / 5 = 9000 / 5 = 1800
ȳ = (300 + 400 + 250 + 350 + 480) / 5 = 1780 / 5 = 356

**Step 2: Calculate w**

Numerator: Σ(xᵢ - x̄)(yᵢ - ȳ)
- (1500-1800)(300-356) = (-300)(-56) = 16800
- (2000-1800)(400-356) = (200)(44) = 8800
- (1200-1800)(250-356) = (-600)(-106) = 63600
- (1800-1800)(350-356) = (0)(-6) = 0
- (2500-1800)(480-356) = (700)(124) = 86800

Sum = 16800 + 8800 + 63600 + 0 + 86800 = 176000

Denominator: Σ(xᵢ - x̄)²
- (1500-1800)² = 90000
- (2000-1800)² = 40000
- (1200-1800)² = 360000
- (1800-1800)² = 0
- (2500-1800)² = 490000

Sum = 90000 + 40000 + 360000 + 0 + 490000 = 980000

**w = 176000 / 980000 ≈ 0.1796**

**Step 3: Calculate b**
b = ȳ - wx̄ = 356 - 0.1796(1800) ≈ 356 - 323.3 ≈ 32.7

**Our model:** price = 0.18·sqft + 33 (approximately)

**Interpretation:**
- Each square foot adds about $180 (since w ≈ 0.18 in thousands)
- Base price (intercept) is about $33k

**Step 4: Make predictions**

For 2200 sqft house:
price = 0.18(2200) + 33 = 396 + 33 = $429k

**Step 5: Evaluate fit**

Let's check predictions on training data:

| sqft | True Price | Predicted | Error | Squared Error |
|------|-----------|-----------|-------|---------------|
| 1500 | 300 | 302.4 | -2.4 | 5.76 |
| 2000 | 400 | 392.2 | 7.8 | 60.84 |
| 1200 | 250 | 248.5 | 1.5 | 2.25 |
| 1800 | 350 | 356.7 | -6.7 | 44.89 |
| 2500 | 480 | 482.0 | -2.0 | 4.00 |

MSE = (5.76 + 60.84 + 2.25 + 44.89 + 4.00) / 5 = 117.74 / 5 ≈ 23.5

RMSE = √23.5 ≈ 4.8k (average error about $4,800)

**Pretty good fit!**

## Multiple Linear Regression

Now extend to multiple features!

**Model:** ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

Or in vector form: ŷ = **w**ᵀ**x** + b

**Example features:**
- x₁ = square footage
- x₂ = number of bedrooms
- x₃ = age (years)
- x₄ = school rating

**Goal:** Find **w** = (w₁, w₂, w₃, w₄) and b

### Matrix Formulation

With N training examples and d features:

**Design matrix X:** N × d matrix where row i is **xᵢ**ᵀ
**Output vector y:** N × 1 vector of true values
**Weight vector w:** d × 1 vector of weights

**Predictions:** ŷ = X**w** + b**1**

Where **1** is N × 1 vector of ones.

**Loss:** L(**w**, b) = (1/N)||**y** - X**w** - b**1**||²

### The Normal Equations

The optimal solution satisfies:

**Xᵀ(y - Xw - b1) = 0**

This leads to the **normal equations**.

**For simplicity, if we augment X with a column of ones (to absorb b into w):**

**w** = (XᵀX)⁻¹Xᵀ**y**

This is the **closed-form solution** for linear regression!

**Requirements:**
- XᵀX must be invertible
- This happens when columns of X are linearly independent

## Detailed Example: Multiple Features

**Data:** (sqft, bedrooms, age) → price

| sqft | beds | age | price |
|------|------|-----|-------|
| 1500 | 2 | 20 | 300 |
| 2000 | 3 | 15 | 400 |
| 1200 | 2 | 30 | 250 |
| 1800 | 3 | 10 | 350 |
| 2500 | 4 | 5 | 480 |

**Using the normal equations** (computation details omitted for brevity):

**Result:** **w** ≈ (0.12, 25, -2) and b ≈ 150

**Model:** price = 0.12·sqft + 25·bedrooms - 2·age + 150

**Interpretation:**
- Each sqft: +$120
- Each bedroom: +$25k
- Each year of age: -$2k
- Base price: $150k

**Test predictions:**

For house (2200 sqft, 3 bed, 12 years):
price = 0.12(2200) + 25(3) - 2(12) + 150
= 264 + 75 - 24 + 150
= $465k

## Evaluating Regression Models

### Metrics

**1. Mean Squared Error (MSE)**
MSE = (1/n) Σ (yᵢ - ŷᵢ)²

**2. Root Mean Squared Error (RMSE)**
RMSE = √MSE

**Advantage:** Same units as y (easier to interpret)

**3. Mean Absolute Error (MAE)**
MAE = (1/n) Σ |yᵢ - ŷᵢ|

**Advantage:** Less sensitive to outliers than MSE

**4. R² Score (Coefficient of Determination)**
R² = 1 - (SS_res / SS_tot)

Where:
- SS_res = Σ(yᵢ - ŷᵢ)² (residual sum of squares)
- SS_tot = Σ(yᵢ - ȳ)² (total sum of squares)

**Interpretation:**
- R² = 1: Perfect fit
- R² = 0: Model no better than predicting the mean
- R² < 0: Model worse than predicting the mean (bad!)

**Typical values:** 0.7-0.9 is often good in practice

### Visualizing Fit

**Residual plot:** Plot (ŷᵢ, yᵢ - ŷᵢ)

**Good fit:** Residuals randomly scattered around zero
**Bad fit:** Residuals show patterns (systematic errors)

## Why Linear Regression Works

### Connection to Taylor Approximation

Remember Taylor said: Complex functions ≈ linear locally

**Linear regression finds the best linear approximation globally!**

Even if true relationship is nonlinear:
true_price = complex_function(features)

Linear regression finds:
predicted_price = **w**ᵀ**x** + b

That minimizes average squared error!

**It's like finding the best "average" linear approximation across all the data.**

### When Does It Work Well?

**Good scenarios:**
1. True relationship is approximately linear
2. Features are relevant and informative
3. Enough data relative to number of features
4. Not too many outliers
5. Features not too highly correlated

**Poor scenarios:**
1. Highly nonlinear relationships (use nonlinear models)
2. Irrelevant features (add noise)
3. Too few data points (overfitting)
4. Many outliers (robust regression needed)
5. Multicollinearity (correlated features cause instability)

## Extensions and Variations

### 1. Polynomial Regression

**Idea:** Add polynomial features!

Original: price = w₁·sqft + b

Add squared term: price = w₁·sqft + w₂·sqft² + b

**This is still linear regression!** (Linear in the parameters **w**)

We just created a new feature: x₂ = x₁²

### 2. Ridge Regression (L2 Regularization)

**Problem:** Large weights can cause overfitting

**Solution:** Penalize large weights!

Loss = MSE + λ||**w**||²

Where λ controls regularization strength.

**Effect:** Prefers smaller weights (simpler models)

### 3. Lasso Regression (L1 Regularization)

Loss = MSE + λΣ|wᵢ|

**Effect:** Can force some weights to exactly zero (feature selection!)

### 4. Gradient Descent Solution

Instead of closed-form (XᵀX)⁻¹Xᵀ**y**, use iterative optimization:

**Repeat:**
1. Compute gradient: ∇L = -(2/N)Xᵀ(**y** - X**w**)
2. Update: **w** ← **w** - α∇L

**Advantage:** Works for huge datasets where (XᵀX)⁻¹ is too expensive

## Connecting Everything

**Linear Functions** → basis of our model
**Taylor Approximation** → why linear works (local approximation)
**Regression** → finding best linear function for data

**The full story:**
1. Real relationships might be nonlinear
2. But locally, they look linear (Taylor)
3. Regression finds best linear fit to data
4. Uses least squares (minimize squared errors)
5. Has closed-form solution (normal equations)
6. Or iterative solution (gradient descent)
7. Foundation for all supervised learning!

## Practice Problems - Regression

**Problem 3.1: Simple Linear Regression by Hand**

Data: (x, y) = (1, 3), (2, 5), (3, 7), (4, 9)

a) Calculate x̄ and ȳ
b) Calculate w using the formula
c) Calculate b
d) Write the regression equation
e) This data is perfectly linear. What should MSE be?

**Problem 3.2: Prediction and Evaluation**

You fit model: ŷ = 2x + 1

Test data: (x, y) = (5, 12), (6, 14), (7, 15)

a) Calculate predictions for each point
b) Calculate errors (yᵢ - ŷᵢ)
c) Calculate MSE
d) Calculate RMSE
e) Calculate MAE

**Problem 3.3: Multiple Regression Interpretation**

Model: salary = 5·years_experience + 10·education_level - 2·age + 30

Units: salary in $1000s, education (1-5), age in years

a) What's the base salary (all features zero)?
b) Each year of experience adds how much?
c) PhD (education=5) vs high school (education=1) difference?
d) Why might age have a negative coefficient?
e) Predict salary for: 10 years exp, education=4, age=35

**Problem 3.4: Designing Features**

You're predicting apartment rent with features: (sqft, floor_number, distance_to_subway_km)

Current model: rent = 2·sqft + 50·floor - 100·distance + 500

a) Create a new feature: sqft_per_floor = sqft/floor. Why might this help?
b) Create: near_subway = 1 if distance < 0.5, else 0. Why might this help?
c) If you add feature sqft², is the model still "linear regression"?
d) What's the difference between a linear model and a linear relationship?

**Problem 3.5: Residual Analysis**

Model predictions and true values:

| ŷᵢ | yᵢ | Residual |
|----|----|----|
| 100 | 105 | ? |
| 200 | 190 | ? |
| 150 | 155 | ? |
| 250 | 240 | ? |

a) Calculate all residuals
b) Calculate MSE
c) Do residuals sum to approximately zero? (They should!)
d) Plot would show residuals vs predictions. What pattern indicates good fit?

**Problem 3.6: Comparing Models**

Two models on same test set:
- Model A: MSE = 25, R² = 0.85
- Model B: MSE = 36, R² = 0.78

a) Which model has better fit by MSE?
b) Which by R²?
c) Do they agree? (They should!)
d) If ȳ = 50, calculate SS_tot
e) For Model A, calculate SS_res

**Problem 3.7: When Linear Regression Fails**

True relationship: y = x² for x ∈ [0, 4]

You fit linear model: ŷ = wx + b

a) Sketch what the data looks like
b) Where will linear model overpredict?
c) Where will it underpredict?
d) Will residuals be randomly scattered?
e) What should you do instead?

---

<a name="summary"></a>
# Chapter 2 Summary

## Key Concepts

### 1. Linear Functions
- **Definition:** f(**x**) = **a** · **x** (inner product!)
- **Properties:** Homogeneity and additivity
- **Affine:** f(**x**) = **a** · **x** + b (adds bias/intercept)
- **ML use:** Predictions, decision boundaries, feature combinations
- **Interpretation:** Coefficients show feature importance

### 2. Taylor Approximation
- **Idea:** Approximate complex functions with linear ones locally
- **Formula:** f(**x**) ≈ f(**a**) + ∇f(**a**) · (**x** - **a**)
- **Gradient:** Vector of partial derivatives
- **ML use:** Gradient descent, sensitivity analysis, local understanding
- **Limitation:** Accuracy decreases far from expansion point

### 3. Regression Models
- **Goal:** Find best linear function fitting data
- **Model:** ŷ = **w**ᵀ**x** + b
- **Loss:** Mean Squared Error (MSE)
- **Solution:** Normal equations or gradient descent
- **Evaluation:** MSE, RMSE, MAE, R²
- **Connection:** Linear regression finds best linear approximation to data

## How Everything Connects

```
Linear Functions
      ↓
   (define our model class)
      ↓
Taylor Approximation
      ↓
   (explains why linear works)
      ↓
   (even for nonlinear relationships)
      ↓
Regression
      ↓
   (finds best linear function for data)
      ↓
Machine Learning!
```

**The Big Picture:**
1. We want to learn from data
2. Linear functions are simplest, most interpretable
3. Taylor tells us: even complex relationships look linear locally
4. Regression finds the best linear approximation globally
5. This minimizes prediction errors (squared)
6. We get closed-form solution (fast!) or use gradient descent (scalable!)
7. Foundation for everything in supervised learning

## What Makes This Chapter Special

**Three fundamental concepts that power ALL of machine learning:**

1. **Linear functions** - The building blocks
2. **Taylor approximation** - Why linearization works
3. **Regression** - How to learn from data

**Every advanced ML technique builds on these:**
- Neural networks? Stacked linear functions + nonlinearities
- Logistic regression? Linear function + sigmoid
- SVM? Linear decision boundary (or kernel trick)
- Deep learning? Gradients via Taylor approximation

## Key Formulas to Remember

**Linear Function:**
```
f(x) = a·x                    (1D)
f(x) = w·x                    (vector, pure linear)
f(x) = w·x + b                (affine, with bias)
```

**Taylor Approximation:**
```
f(x) ≈ f(a) + f'(a)(x - a)              (1D)
f(x) ≈ f(a) + ∇f(a)·(x - a)             (multivariate)
```

**Linear Regression:**
```
ŷ = w·x + b                              (model)
L = (1/n)Σ(yi - ŷi)²                     (loss)
w = (X^T X)^(-1) X^T y                   (solution)
Δw = -α∇L                                (gradient descent)
```

## Common Pitfalls and Misconceptions

**1. "Linear" vs "Affine"**
- ❌ Wrong: y = 2x + 3 is linear
- ✓ Right: y = 2x + 3 is affine (linear + constant)
- True linear: f(0) must equal 0

**2. "Taylor only works for differentiable functions"**
- ✓ True! Need derivatives
- Many ML functions are designed to be differentiable
- Non-differentiable points (like ReLU at 0) handled carefully

**3. "Linear regression only works for linear relationships"**
- ❌ Wrong! Can model nonlinear relationships
- ✓ Right: Add polynomial/interaction features
- Model is "linear in parameters" not "linear in features"

**4. "More features always better"**
- ❌ Wrong! Can cause overfitting
- Too many features vs too little data → poor generalization
- Need regularization or feature selection

**5. "R² = 0.9 always means good model"**
- Depends on domain and use case
- Social sciences: R² = 0.3 might be good
- Physics: R² = 0.99 might be expected
- Always check residual plots!

## Practical Tips for ML

**When building models:**

1. **Start simple (linear) first**
   - Establishes baseline
   - Fast to train
   - Easy to interpret
   - Might be "good enough"

2. **Check your features**
   - Remove collinear features
   - Normalize/standardize scales
   - Consider polynomial/interaction terms

3. **Always visualize**
   - Plot predictions vs actual
   - Check residual plots
   - Look for patterns in errors

4. **Use appropriate metrics**
   - RMSE for same units as target
   - MAE for robustness to outliers
   - R² for proportion of variance explained

5. **Split your data**
   - Training set: Fit model
   - Validation set: Tune hyperparameters
   - Test set: Final evaluation
   - Never test on training data!

## What's Next?

**Chapter 3** will cover:
- Norm (vector length) - formalizes distance
- Distance between vectors - similarity measures
- Standard deviation - measuring spread
- Angles and cosine similarity - directional similarity

These build directly on inner products and enable clustering!

**Chapter 4** will cover:
- Clustering algorithms
- K-means in detail
- Objective functions
- Practical applications

**Chapter 5** will cover:
- Linear independence - when vectors give new information
- Basis - minimal spanning sets
- Orthogonality - perpendicular vectors
- Gram-Schmidt - making vectors orthogonal

All of these extend what you've learned here!

---

<a name="practice"></a>
# Comprehensive Practice Problems

## Section 1: Integrated Concepts

**Problem 4.1: Complete Linear Regression Pipeline**

You're hired to predict student grades based on study time.

**Data (hours studied, grade):**
(2, 65), (4, 75), (6, 85), (8, 95), (10, 98)

a) Calculate x̄ and ȳ
b) Calculate w (slope) using the formula
c) Calculate b (intercept)
d) Write your regression equation
e) Predict grade for someone who studies 5 hours
f) Calculate MSE on the training data
g) Calculate R² score
h) A new student studies 12 hours. Predict their grade. Should you trust this prediction?

**Problem 4.2: Taylor Approximation in Action**

Function: f(x, y) = x² + xy + y²

Current point: (2, 3)
- f(2, 3) = 4 + 6 + 9 = 19

Gradients:
- ∂f/∂x = 2x + y
- ∂f/∂y = x + 2y

a) Calculate ∇f(2, 3)
b) Approximate f(2.1, 2.9) using Taylor
c) Calculate true f(2.1, 2.9)
d) What's the error?
e) Would you expect larger error for f(3, 4)? Why?

**Problem 4.3: Feature Engineering Challenge**

You're predicting pizza delivery time with:
- x₁ = distance (km)
- x₂ = traffic (1-10 scale)
- x₃ = time of day (hour, 0-23)

Current model: time = 5·distance + 2·traffic + 0.5·hour + 10

a) Predict delivery time for: 3km, traffic=7, 6pm (hour=18)
b) Create new feature: rush_hour = 1 if 17 ≤ hour ≤ 19, else 0
c) Why might rush_hour be better than raw hour?
d) Create: distance_in_traffic = distance × traffic. Why might this help?
e) If you add distance², is the model still linear regression? Explain.

**Problem 4.4: Gradient Descent from Scratch**

Loss function: L(w) = (w - 5)² at w = 0
Learning rate: α = 0.3

a) Calculate gradient: dL/dw at w = 0
b) Calculate update: w_new = w - α·(dL/dw)
c) Calculate new loss L(w_new)
d) Repeat steps a-c for two more iterations
e) Are you getting closer to optimal w = 5?
f) What happens if α = 1.5? Try one iteration.

**Problem 4.5: Interpreting Multiple Regression**

Model: salary = 8·experience + 15·education + 5·location + 30

Where:
- salary in $1000s
- experience in years
- education: 1=HS, 2=Bachelor, 3=Master, 4=PhD
- location: 1=small town, 2=city, 3=metro

a) What's salary for fresh graduate (experience=0, Bachelor, metro)?
b) How much does each year of experience add?
c) Value of PhD (4) vs Bachelor (2)?
d) Moving from small town to metro?
e) Who earns more: (10 yrs, Bachelor, city) or (5 yrs, Master, metro)?

## Section 2: Real-World Applications

**Problem 4.6: House Price Prediction**

Data:

| sqft | bedrooms | age | price ($1000s) |
|------|----------|-----|----------------|
| 1500 | 2 | 20 | 300 |
| 2000 | 3 | 10 | 400 |
| 2500 | 4 | 5 | 500 |
| 1800 | 3 | 15 | 380 |

Fitted model: price = 0.15·sqft + 40·beds - 2·age + 100

a) Predict price for each house in the dataset
b) Calculate residuals for each
c) Calculate MSE
d) Calculate RMSE (in $1000s)
e) A 2200 sqft, 3 bed, 12 year old house. Predicted price?
f) Which feature has biggest impact per unit?

**Problem 4.7: Stock Price Movement**

You want to predict next-day stock price change (%) from:
- x₁ = today's volume change (%)
- x₂ = market sentiment (-1 to +1)
- x₃ = news mentions count

Model: change = 0.3·volume + 5·sentiment + 0.1·news + 0.5

a) Interpret each coefficient
b) If volume up 10%, sentiment +0.5, 20 news mentions, predict change
c) What needs to happen for >5% increase prediction?
d) If model has R² = 0.15, is it useful? (Consider stock market randomness)

**Problem 4.8: Medical Dosage**

Predicting effective drug dosage (mg) from:
- x₁ = patient weight (kg)
- x₂ = age (years)
- x₃ = severity (1-10)

Model: dosage = 0.5·weight + 0.2·age + 3·severity + 10

a) Calculate dosage for: 70kg, 45 years, severity=6
b) Two patients same weight/age, but severity differs by 3. Dosage difference?
c) Why might you want to be extra careful trusting this model?
d) What additional features might improve predictions?

**Problem 4.9: E-commerce Revenue**

Predicting daily revenue from:
- x₁ = website visitors (1000s)
- x₂ = average time on site (minutes)
- x₃ = email campaign sent (0 or 1)

Model: revenue = 5·visitors + 2·time + 15·email + 100

(Revenue in $1000s)

a) Revenue on a day with 10k visitors, 8 min avg time, no email?
b) Expected revenue boost from email campaign?
c) Hiring an SEO expert adds 2k visitors. Revenue impact?
d) Which is better: +1k visitors or +2 min avg time?

**Problem 4.10: Credit Risk Scoring**

Predicting default probability (0-1) using linear model:
- x₁ = credit score (300-850)
- x₂ = debt-to-income ratio (0-1)
- x₃ = number of late payments

Model: p(default) = -0.002·score + 0.5·debt_ratio + 0.1·late_payments + 1.5

a) Calculate probability for: score=650, ratio=0.4, 2 late payments
b) What credit score needed to offset 5 late payments?
c) Why might linear regression be problematic for probabilities?
d) What would you use instead? (Hint: next chapter material)

## Section 3: Debugging and Understanding

**Problem 4.11: Finding Errors in Reasoning**

A student says: "My linear regression has w = (5, -3, 2). The second feature is negative, so it's the least important."

a) What's wrong with this reasoning?
b) If features have different scales, what should you do first?
c) How do you properly assess feature importance?

**Problem 4.12: Overfitting Diagnosis**

You fit two models:
- Simple: 3 features, Train MSE = 25, Test MSE = 28
- Complex: 20 features, Train MSE = 5, Test MSE = 45

a) Which model performs better on training?
b) Which performs better on test?
c) Which model is overfitting?
d) What should you do?
e) What is "generalization gap"?

**Problem 4.13: Understanding Residuals**

Perfect model: ŷᵢ = yᵢ for all i

a) What would residual plot look like?
b) What would MSE be?
c) What would R² be?
d) Why is this impossible in practice?
e) What residual pattern indicates heteroscedasticity?

**Problem 4.14: Taylor Approximation Limitations**

Function: f(x) = e^x
Expansion at a = 0: f(x) ≈ 1 + x

a) Approximate e^0.1 and calculate error
b) Approximate e^0.5 and calculate error
c) Approximate e^2.0 and calculate error
d) What do you notice about errors?
e) How could you improve approximation accuracy?

**Problem 4.15: Collinearity Problem**

You're predicting house prices with:
- x₁ = square feet
- x₂ = square meters (x₂ = 0.0929·x₁)
- x₃ = bedrooms

a) Are x₁ and x₂ linearly independent?
b) What problems does this cause for regression?
c) What is multicollinearity?
d) How would you detect it?
e) How would you fix it?

## Section 4: Advanced Synthesis

**Problem 4.16: Polynomial Regression**

Data shows quadratic relationship: y = x²

You fit linear model: ŷ = wx + b

a) Where will model underpredict most?
b) Where will it overpredict?
c) Create feature x₂ = x². New model: ŷ = w₁x + w₂x² + b
d) Is this still "linear regression"? Why?
e) What's the difference between linear model and linear relationship?

**Problem 4.17: Gradient Descent Dynamics**

Loss landscape: L(w) = w² - 4w + 5 (parabola, minimum at w = 2)

Starting at w = 0, learning rate α = 0.1:

a) Calculate gradient dL/dw at w = 0
b) Calculate w after 1 step
c) Calculate w after 2 steps
d) Continue until convergence (< 5 steps)
e) What happens if α = 2.1? (Too large)

**Problem 4.18: Bias-Variance Tradeoff**

Model A: High bias, low variance (underfitting)
Model B: Low bias, high variance (overfitting)
Model C: Balanced

a) Which has lowest training error?
b) Which has lowest test error?
c) As model complexity increases, what happens to bias?
d) What happens to variance?
e) How do you find the "sweet spot"?

**Problem 4.19: Cross-Validation**

You have 100 data points. Use 5-fold cross-validation.

a) How many points in each fold?
b) How many times is each point used for testing?
c) Why is this better than single train/test split?
d) What is the final performance estimate?
e) What's the disadvantage of cross-validation?

**Problem 4.20: Feature Scaling Impact**

Two features:
- x₁ = income ($20k-$200k)
- x₂ = age (20-65 years)

Weights before scaling: w₁ = 0.0001, w₂ = 0.5

a) Which feature dominates purely by scale?
b) Apply standardization: x'ᵢ = (xᵢ - mean) / std
c) Why does this help gradient descent?
d) Do predictions change after scaling? Why/why not?
e) Do weights change after scaling? How should you interpret them?

## Section 5: Theoretical Understanding

**Problem 4.21: Proving Linear Properties**

For f(x) = ax, prove:

a) f(αx) = αf(x) for any scalar α
b) f(x + y) = f(x) + f(y)
c) If g(x) = ax + b where b ≠ 0, show g is NOT linear
d) Show that sum of linear functions is linear
e) Show that composition of linear functions is linear

**Problem 4.22: Least Squares Derivation**

Loss: L(w, b) = Σ(yᵢ - wxᵢ - b)²

a) Expand one term: (yᵢ - wxᵢ - b)²
b) Take ∂L/∂b and set to zero
c) Solve for b in terms of w, xᵢ, yᵢ
d) Interpret: what does this tell you about the fitted line?
e) Why does the line pass through (x̄, ȳ)?

**Problem 4.23: Gradient as Direction**

Function: f(x, y) = x² + 4y²

At point (1, 1): ∇f = (2, 8)

a) In which direction does f increase fastest?
b) What's the magnitude of this maximum increase rate?
c) In which direction does f decrease fastest?
d) What directions leave f unchanged (level curves)?
e) If you move in direction (1, 0), how does f change?

**Problem 4.24: Normal Equations Understanding**

Solution: w = (X^T X)^(-1) X^T y

a) What are dimensions of X if n = 100 examples, d = 5 features?
b) What are dimensions of X^T X?
c) When is X^T X not invertible?
d) What does this mean geometrically?
e) What do you do if X^T X is singular?

**Problem 4.25: R² Interpretation**

R² = 1 - (SS_res / SS_tot)

where SS_tot = Σ(yᵢ - ȳ)²

a) What does SS_tot measure?
b) What does SS_res measure?
c) If predictions are perfect, what's R²?
d) If predictions always equal ȳ, what's R²?
e) Can R² be negative? When?

---

# Answer Key (Selected Problems)

**Problem 4.1:**
a) x̄ = 6, ȳ = 83.6
b) w = 4 (exactly linear relationship!)
c) b = 83.6 - 4(6) = 59.6
d) grade = 4·hours + 59.6
e) grade = 4(5) + 59.6 = 79.6
h) 12 hours → 107.6% (> 100%!). Don't trust: extrapolating beyond data range!

**Problem 4.4:**
a) dL/dw = 2(w - 5) = 2(0 - 5) = -10
b) w_new = 0 - 0.3(-10) = 3
c) L(3) = (3-5)² = 4 (reduced from L(0) = 25!)
f) With α = 1.5: w_new = 0 - 1.5(-10) = 15. L(15) = 100 (diverged! Too large α)

**Problem 4.15:**
a) No! x₂ = 0.0929x₁ (perfect linear dependence)
b) X^T X becomes singular (not invertible)
c) When predictors are correlated
d) Check correlation matrix or compute condition number
e) Remove one of the correlated features

**Problem 4.17:**
dL/dw = 2w - 4
e) α = 2.1: Update by -α(2w-4) = -2.1(2w-4). At w=0: step to w=8.4. Next: w oscillates with growing amplitude (diverges!)

---

# Congratulations!

You've completed Chapter 2! You now understand:
- ✅ Linear functions and why they matter
- ✅ How Taylor approximation connects to linearization
- ✅ Linear regression and how to fit models to data
- ✅ The connection between all three concepts
- ✅ How these form the foundation of machine learning

**You're ready for Chapter 3: Norm, Distance, Standard Deviation, and Angles!**

---

*End of Chapter 2*
