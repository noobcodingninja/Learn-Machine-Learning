# Linear Algebra for Machine Learning
## Chapter 1: Vectors and Vector Operations

### A First-Principles Approach with Detailed Examples

---

# Table of Contents

1. [Vectors: The Foundation](#vectors)
2. [Vector Addition](#vector-addition)
3. [Scalar-Vector Multiplication](#scalar-multiplication)
4. [Inner Product (Dot Product)](#inner-product)
5. [Complexity of Vector Computations](#complexity)
6. [Chapter Summary](#summary)
7. [Comprehensive Practice Problems](#practice)

---

<a name="vectors"></a>
# 1. Vectors: The Foundation

## The Fundamental Problem: Representing Information Numerically

Imagine you're building a spam email detector. You receive an email:

**"FREE MONEY!!! CLICK HERE NOW!!!"**

**The core challenge:** Computers don't understand words—they only understand numbers. How do you describe this email mathematically?

You might extract numerical features:
- Number of capital letters: 15
- Number of exclamation marks: 3
- Word count: 4
- Contains "FREE": 1 (yes)
- Contains "MONEY": 1 (yes)

Now your email becomes: **(15, 3, 4, 1, 1)**

**This ordered list of numbers is called a vector!**

## What Exactly Is a Vector?

A **vector** is an ordered list of numbers. These numbers are called **components**, **elements**, or **entries**.

**Why is order important?** Because each position has a specific meaning. In our email example:
- Position 1 is always capital letters
- Position 2 is always exclamation marks
- Etc.

If we wrote (3, 15, 4, 1, 1), it would mean something completely different!

**Notation:**
- Vectors are typically written as **bold lowercase letters**: **v**, **x**, **a**, **w**
- Sometimes with an arrow: $\vec{v}$
- Individual components accessed with subscripts: v₁, v₂, v₃, etc.

**Example:** If **v** = (15, 3, 4, 1, 1), then:
- v₁ = 15
- v₂ = 3
- v₃ = 4
- v₄ = 1
- v₅ = 1

**Dimension:** The number of components in a vector is its **dimension**.

**Mathematical notation:** If vector **v** has n components:
**v** ∈ ℝⁿ (read as "v is an element of n-dimensional real space")

### Example Representations

**2D vector:** **v** = (3, 5) ∈ ℝ²
- First component (v₁): 3
- Second component (v₂): 5

**5D vector:** **email** = (15, 3, 4, 1, 1) ∈ ℝ⁵
- Five components representing five features

**100D vector:** **v** ∈ ℝ¹⁰⁰
- Has 100 components
- Can't visualize it, but the math works the same!

## Geometric Interpretation: Vectors as Arrows

While vectors are fundamentally lists of numbers, they have powerful geometric interpretations.

### The 2D Case: Points and Arrows

Consider the point (3, 5) on a coordinate plane:
- 3 units along the x-axis (horizontal)
- 5 units along the y-axis (vertical)

This can be interpreted as:
1. **A point:** The location (3, 5) in the plane
2. **A vector:** An arrow from the origin (0, 0) to the point (3, 5)

**Key insight:** A vector represents both:
- **Direction:** Where it points (in this case, northeast)
- **Magnitude:** How far it reaches (we'll calculate this later as √(3² + 5²) = √34 ≈ 5.83 units)

### Why Both Interpretations Matter

**Point interpretation:** Useful for representing locations, positions, data points

**Arrow interpretation:** Useful for representing movements, forces, changes

**Example: GPS Navigation**
- Your house: point (3, 5) = "3 km east, 5 km north from origin"
- Friend's house: point (7, 2) = "7 km east, 2 km north"
- Displacement vector: (7, 2) - (3, 5) = (4, -3) = "go 4 km east, 3 km south"

The **point** tells you where you are.
The **arrow** tells you how to get there.

### The 3D Case and Beyond

**3D vector:** **v** = (x, y, z) ∈ ℝ³
- x: position along east-west axis
- y: position along north-south axis
- z: position along up-down axis

**Example: Airplane Position**
**p** = (1000, 500, 10000) meters
- 1000 m east of origin
- 500 m north of origin
- 10,000 m altitude

**Higher Dimensions:** While we can't visualize 4D, 5D, or 100D space, the mathematics works identically!

A 100-dimensional vector **v** ∈ ℝ¹⁰⁰ still represents:
- A "direction" in 100D space
- A "magnitude" (length)

We just can't draw it on paper!

## Why Machine Learning Is Built on Vectors

Machine Learning fundamentally transforms all data into vectors. Here's why:

### 1. Unified Representation

No matter what type of data you have, convert it to a vector:
- Text → vector
- Images → vector
- Audio → vector
- Structured data → vector

**Once everything is a vector, you can use the same mathematical tools on all of them!**

### 2. Text as Vectors

**Bag of Words Representation:**

Vocabulary: ["cat", "dog", "bird", "fish", "run"]

Document 1: "cat cat dog"
→ Count each word: (2, 1, 0, 0, 0)

Document 2: "dog fish fish"
→ (0, 1, 0, 2, 0)

Now we can measure: "How similar are these documents?"

**Modern embeddings (Word2Vec, BERT):**
Each word becomes a dense vector, typically 100-300 dimensions:

**king** ≈ (0.25, -0.43, 0.18, ..., 0.67) ∈ ℝ³⁰⁰

Remarkably: **king** - **man** + **woman** ≈ **queen**

The vectors capture meaning!

### 3. Images as Vectors

**Grayscale image (8×8 pixels):**
```
[10, 20, 30, 40, 50, 60, 70, 80]
[15, 25, 35, 45, 55, 65, 75, 85]
...
```

**Flatten to vector:** (10, 20, 30, ..., 85) ∈ ℝ⁶⁴

**Color image (1920×1080, RGB):**
- Width: 1920 pixels
- Height: 1080 pixels
- Channels: 3 (Red, Green, Blue)
- Total: 1920 × 1080 × 3 = 6,220,800 dimensions!

**v** ∈ ℝ⁶'²²⁰'⁸⁰⁰

**Modern CNNs:** Don't use raw pixels directly, but still fundamentally operate on vectors.

### 4. Audio as Vectors

**Digital audio:** Sample the amplitude at regular intervals

CD quality: 44,100 samples per second (44.1 kHz)

1 second of mono audio: **v** ∈ ℝ⁴⁴'¹⁰⁰

3-minute song: **v** ∈ ℝ⁷'⁹³⁸'⁰⁰⁰

**Spectrograms:** Transform to frequency domain, still vectors!

### 5. Structured/Tabular Data as Vectors

**Customer record:**
- Age: 35
- Income: $75,000
- Years as customer: 5
- Number of purchases: 23
- Satisfaction rating (1-10): 8

**Customer vector:** **c** = (35, 75000, 5, 23, 8) ∈ ℝ⁵

Each row in your dataset is a vector!

### 6. Why Vectors Enable Machine Learning

Once data is vectors, we can:

**a) Measure Similarity**
- Which customers are similar? → Find vectors with small distance
- Which documents are about the same topic? → Calculate angle between vectors
- Which songs sound alike? → Compare their feature vectors

**b) Find Patterns**
- Where does the data cluster? → Group nearby vectors
- What direction captures most variation? → Find principal components
- What separates spam from non-spam? → Find a hyperplane boundary

**c) Make Predictions**
- Linear models: prediction = **w** · **x** + b (inner product!)
- Neural networks: layers of vector operations
- All predictions ultimately come from vector operations

## Detailed Real-World Examples

### Example 1.1: Song Feature Vector (Spotify-style)

A song might be characterized by:
- Tempo (BPM): 120
- Loudness (dB): -5
- Danceability (0-1): 0.8
- Energy (0-1): 0.9
- Valence/happiness (0-1): 0.7
- Acousticness (0-1): 0.2
- Instrumentalness (0-1): 0.1
- Duration (seconds): 210

**Song vector:** **s** = (120, -5, 0.8, 0.9, 0.7, 0.2, 0.1, 210) ∈ ℝ⁸

**Recommendation system:** Find songs with similar vectors!

If you like song **s₁**, recommend songs **s₂** where distance(**s₁**, **s₂**) is small.

### Example 1.2: Patient Health Record

Medical diagnosis system features:
- Age (years): 45
- Systolic blood pressure (mmHg): 130
- Total cholesterol (mg/dL): 200
- BMI: 26.5
- Fasting blood sugar (mg/dL): 95
- Resting heart rate (BPM): 72
- Exercise hours per week: 3
- Smoker (yes=1, no=0): 0

**Patient vector:** **p** = (45, 130, 200, 26.5, 95, 72, 3, 0) ∈ ℝ⁸

**Diagnosis:** Compare to vectors of patients with known conditions.

### Example 1.3: E-commerce Customer Profile

Online retailer tracking:
- Number of visits this month: 8
- Total spent this year ($): 1,250
- Average time per visit (minutes): 15
- Items in cart right now: 3
- Items in wishlist: 5
- Customer since (days): 730
- Number of reviews written: 12
- Number of returns: 2
- Email open rate (0-1): 0.6

**Customer vector:** **c** = (8, 1250, 15, 3, 5, 730, 12, 2, 0.6) ∈ ℝ⁹

**Segmentation:** Cluster customers into groups (loyal, at-risk, new, etc.)

### Example 1.4: Real Estate Property

House listing features:
- Square footage: 2,000
- Number of bedrooms: 3
- Number of bathrooms: 2.5
- Lot size (sq ft): 5,000
- Year built: 2005
- Distance to downtown (miles): 8.5
- School district rating (1-10): 8
- Property tax ($/year): 4,500
- HOA fees ($/month): 150

**House vector:** **h** = (2000, 3, 2.5, 5000, 2005, 8.5, 8, 4500, 150) ∈ ℝ⁹

**Price prediction:** Learn function f(**h**) = price using regression.

### Example 1.5: Social Media Post

Post features for engagement prediction:
- Hour of day posted (0-23): 14
- Day of week (1-7): 3
- Number of hashtags: 5
- Number of mentions: 2
- Character count: 180
- Has image (1/0): 1
- Has video (1/0): 0
- Has link (1/0): 1
- Follower count of poster: 5,200
- Historical engagement rate: 0.04

**Post vector:** **p** = (14, 3, 5, 2, 180, 1, 0, 1, 5200, 0.04) ∈ ℝ¹⁰

**Predict:** Engagement = f(**p**)

### Example 1.6: Network Traffic Packet

Cybersecurity features:
- Packet size (bytes): 1,024
- Duration (ms): 45
- Protocol type (encoded): 6
- Source port: 443
- Destination port: 8080
- Number of packets in flow: 15
- Bytes sent: 12,000
- Bytes received: 8,000
- Number of connections to same destination: 3
- Time since last connection (seconds): 2

**Packet vector:** **pkt** = (1024, 45, 6, 443, 8080, 15, 12000, 8000, 3, 2) ∈ ℝ¹⁰

**Anomaly detection:** Is this vector unusual compared to normal traffic?

### Example 1.7: Stock Market Data

Stock features for prediction:
- Current price ($): 150.25
- Daily volume (millions): 5.2
- Market cap (billions): 50
- P/E ratio: 22.5
- 50-day moving average: 148.50
- RSI (0-100): 65
- Beta: 1.15
- Dividend yield (%): 2.1
- YTD return (%): 12.5

**Stock vector:** **s** = (150.25, 5.2, 50, 22.5, 148.50, 65, 1.15, 2.1, 12.5) ∈ ℝ⁹

**Portfolio optimization:** Which combinations of stock vectors maximize return?

## The Zero Vector: A Special Case

The **zero vector** is the vector with all components equal to zero:

**0** = (0, 0, 0, ..., 0)

**Properties:**
- Adding zero vector doesn't change anything: **v** + **0** = **v**
- Multiplying any scalar by zero vector gives zero: α**0** = **0**
- It's the only vector with length zero
- Represents "no change" or "no information"

**Geometric meaning:** The zero vector is the point at the origin—it has no direction or magnitude.

## Practice Problems - Vectors

**Problem 1.1: Creating Vectors from Data**

You're analyzing movies with these features:
- Runtime (minutes): 142
- Budget (millions USD): 100
- Number of actors in cast: 50
- Average review score (0-10): 7.8
- Box office revenue (millions USD): 250
- Year released: 2022

a) Write this movie as a vector
b) What is its dimension? Write it in the form **v** ∈ ℝⁿ
c) If you have 1,000 movies in your dataset, what are the dimensions of your full dataset?

**Problem 1.2: RGB Color Representation**

A color in RGB format is represented as (R, G, B) where each value is 0-255.

a) Pure red is (255, 0, 0). Write pure blue and pure green as vectors.
b) White is (255, 255, 255) and black is (0, 0, 0). Why does this make sense?
c) Gray colors have R = G = B. Write a medium gray as a vector.
d) The color orange has RGB values (255, 165, 0). What dimension is this vector?

**Problem 1.3: Weather Tracking**

You track daily weather with: morning temp (°C), afternoon temp (°C), evening temp (°C), humidity (%), and rainfall (mm).

Yesterday's data: 15°C morning, 22°C afternoon, 18°C evening, 65% humidity, 2mm rain.

a) Write this as a vector
b) If you track weather for 30 days, and store all data in one long vector, what is its dimension?
c) Why might it be better to store each day as a separate vector rather than one huge vector?

**Problem 1.4: Image Dimensions**

a) A tiny 8×8 grayscale image (like old video game sprites) has one brightness value (0-255) per pixel. What dimension vector represents this image?

b) The same image in color (RGB) needs three values per pixel. What dimension now?

c) A smartphone photo is 4032×3024 pixels in color (RGB). What dimension vector?

d) Why don't we typically use raw pixel vectors for image recognition?

**Problem 1.5: Understanding Dimension**

a) You have a vector **v** ∈ ℝ⁵⁰⁰. How many numbers does it contain?

b) Can you visualize a vector in ℝ⁵⁰⁰? Why or why not?

c) Does the mathematics of vectors work the same in ℝ⁵⁰⁰ as in ℝ²? Explain.

d) Give an example of real data that might naturally be represented in ℝ⁵⁰⁰.

**Problem 1.6: Data Representation**

A fitness app tracks for each workout:
- Duration (minutes)
- Distance (km)
- Calories burned
- Average heart rate (BPM)
- Steps taken
- Floors climbed

a) Write a sample workout as a vector
b) You complete 100 workouts in a year. How many total numbers are you storing?
c) Why is it useful to represent each workout as a vector rather than just keeping them in a table?

**Problem 1.7: Zero Vector**

a) Write the zero vector in ℝ³
b) What does it mean geometrically?
c) In the context of our email spam detector, what would a zero vector (0, 0, 0, 0, 0) represent?
d) True or False: The zero vector has no dimension. Explain.

---

<a name="vector-addition"></a>
# 2. Vector Addition: Combining Information

## The Problem We're Solving

Imagine you're tracking your daily fitness activity with a smartwatch:

**Monday's activity:**
- Steps: 8,000
- Floors climbed: 5
- Exercise minutes: 30

We represent this as: **Monday** = (8000, 5, 30)

**Tuesday's activity:**
- Steps: 6,000
- Floors climbed: 3
- Exercise minutes: 45

**Tuesday** = (6000, 3, 45)

**Natural question:** What's your total activity over both days combined?

Your intuition says to add each category:
- Total steps: 8,000 + 6,000 = 14,000
- Total floors: 5 + 3 = 8
- Total minutes: 30 + 45 = 75

**Monday + Tuesday = (14000, 8, 75)**

This is **vector addition**!

## Why Do We Need Vector Addition?

In many situations, we need to combine information from multiple sources:
- Combining daily activities into weekly totals
- Merging data from different sensors
- Averaging vectors to find typical patterns
- Updating positions by adding displacement vectors
- Aggregating predictions from multiple models

**Vector addition is the fundamental way to combine vector-based information.**

## Formal Definition

To add two vectors, **add their corresponding components**.

If **u** = (u₁, u₂, ..., uₙ) and **v** = (v₁, v₂, ..., vₙ), then:

**u + v** = (u₁+v₁, u₂+v₂, u₃+v₃, ..., uₙ+vₙ)

**Critical requirement:** Both vectors must have the **same dimension**!

### Why the Same Dimension Requirement?

You cannot add vectors of different dimensions:

(3, 4) + (5, 6, 7) ← **ERROR!** Undefined!

**Think about why:** How would you add "apples and oranges" to "cats, dogs, and birds"? The categories don't match up!

**Category matching:**
- Component 1 of first vector only makes sense to add to component 1 of second vector
- They must measure the same thing
- If dimensions differ, there's no meaningful way to combine them

## Geometric Interpretation: Tip-to-Tail

Imagine walking in a city grid:

**First walk:** **u** = (3, 4)
- Move 3 blocks east
- Move 4 blocks north

**Second walk:** **v** = (2, 1)
- Move 2 blocks east
- Move 1 block north

**Where do you end up?**

**u + v** = (3+2, 4+1) = (5, 5)

You're now 5 blocks east and 5 blocks north from your starting point!

**Geometric visualization (tip-to-tail method):**
1. Draw vector **u** as an arrow from origin to point (3, 4)
2. Place the **tail** of vector **v** at the **tip** of **u**
3. Draw **v** from (3, 4) to (3+2, 4+1) = (5, 5)
4. The sum **u+v** is the direct arrow from origin to final point (5, 5)

**Parallelogram method (equivalent):**
1. Draw both **u** and **v** starting from the origin
2. Complete the parallelogram
3. The diagonal from origin is **u+v**

Both methods give the same result!

**Physical interpretation:** If **u** and **v** represent forces acting on an object, **u+v** is the net force (resultant force).

## Properties of Vector Addition

These properties ensure vector addition behaves "nicely":

**1. Commutative Property:** **u + v** = **v + u**

Order doesn't matter!

**Example:**
**u** = (2, 3)
**v** = (1, 4)

**u + v** = (2, 3) + (1, 4) = (3, 7)
**v + u** = (1, 4) + (2, 3) = (3, 7) ✓

**Geometric meaning:** Whether you walk path **u** then **v**, or **v** then **u**, you end up at the same place!

**2. Associative Property:** (**u + v**) + **w** = **u** + (**v + w**)

Grouping doesn't matter!

**Example:**
**u** = (1, 2)
**v** = (3, 4)
**w** = (5, 6)

(**u + v**) + **w** = [(1, 2) + (3, 4)] + (5, 6) = (4, 6) + (5, 6) = (9, 12)

**u** + (**v + w**) = (1, 2) + [(3, 4) + (5, 6)] = (1, 2) + (8, 10) = (9, 12) ✓

**Practical meaning:** When adding multiple vectors, you can group them any way you want!

**3. Identity Element:** **v + 0** = **v**

Adding the zero vector doesn't change anything!

**Example:**
**v** = (3, 5)
**0** = (0, 0)

**v + 0** = (3, 5) + (0, 0) = (3, 5) ✓

**4. Inverse Element:** **v + (-v)** = **0**

Every vector has an opposite that cancels it out!

**Example:**
**v** = (3, 5)
**-v** = (-3, -5)

**v + (-v)** = (3, 5) + (-3, -5) = (0, 0) ✓

**Geometric meaning:** Walking somewhere then walking back gets you to the origin!

## Why Machine Learning Cares About Vector Addition

Vector addition appears everywhere in ML:

### 1. Computing Averages and Centroids

**Example: Finding the "average" spam email**

You have 100 spam emails as vectors:
**spam₁**, **spam₂**, ..., **spam₁₀₀**

**Average spam pattern:**
**spam_avg** = (1/100)(**spam₁** + **spam₂** + ... + **spam₁₀₀**)

This "prototype" spam vector helps classify new emails!

**K-means clustering:** Cluster centers are averages:
**centroid** = (1/n)(**x₁** + **x₂** + ... + **xₙ**)

### 2. Gradient Descent Optimization

Training neural networks:
- Current weights: **w**
- Gradient (direction to improve): **∇L**
- Learning rate: α
- **Updated weights: w_new = w - α∇L**

This is vector addition! (**w** plus **-α∇L**)

### 3. Ensemble Learning

Combining predictions from multiple models:
- Model 1 prediction: **p₁**
- Model 2 prediction: **p₂**
- Model 3 prediction: **p₃**

**Ensemble prediction:**
**p_final** = (1/3)(**p₁** + **p₂** + **p₃**)

Or weighted average:
**p_final** = 0.5**p₁** + 0.3**p₂** + 0.2**p₃**

### 4. Data Augmentation

Creating new training examples:
- Original image vector: **img**
- Random noise vector: **noise**
- Augmented image: **img_new** = **img** + 0.1**noise**

Adds slight variation while preserving the main content!

### 5. Residual Connections (ResNets)

Deep learning architecture:
- Input to layer: **x**
- Transformation: f(**x**)
- **Output with skip connection: f(x) + x**

Vector addition allows information to flow through the network!

## Detailed Examples

**Example 2.1: Basic Vector Addition**

**u** = (1, 2, 3)
**v** = (4, 5, 6)

**u + v** = (1+4, 2+5, 3+6) = (5, 7, 9)

**Verify commutativity:**
**v + u** = (4+1, 5+2, 6+3) = (5, 7, 9) ✓

**Example 2.2: Nutritional Tracking**

Daily intake vectors:

**Day 1:** (2000 calories, 100g protein, 250g carbs, 50g fat)
**Day 2:** (2200 calories, 80g protein, 280g carbs, 60g fat)
**Day 3:** (1800 calories, 90g protein, 220g carbs, 45g fat)

**Three-day total:**
(2000, 100, 250, 50) + (2200, 80, 280, 60) + (1800, 90, 220, 45)
= (6000, 270, 750, 155)

**Daily average:**
(1/3) × (6000, 270, 750, 155) = (2000, 90, 250, 51.67)

**Example 2.3: Store Inventory Management**

**Store A inventory:** (100 apples, 50 oranges, 200 bananas, 75 pears)
**Store B inventory:** (150 apples, 75 oranges, 180 bananas, 90 pears)
**Store C inventory:** (80 apples, 60 oranges, 150 bananas, 85 pears)

**Combined inventory across all stores:**
(100, 50, 200, 75) + (150, 75, 180, 90) + (80, 60, 150, 85)
= (330, 185, 530, 250)

**Example 2.4: Financial Portfolio Returns**

Portfolio returns in three scenarios (bull market, normal, bear market):

**Stock portfolio:** (0.20, 0.10, -0.05) = 20% in bull, 10% in normal, -5% in bear
**Bond portfolio:** (0.05, 0.05, 0.03) = 5%, 5%, 3%
**Real estate:** (0.15, 0.08, 0.02) = 15%, 8%, 2%

**Combined portfolio (33.3% each):**
(1/3)(0.20, 0.10, -0.05) + (1/3)(0.05, 0.05, 0.03) + (1/3)(0.15, 0.08, 0.02)

First, scale each:
(0.0667, 0.0333, -0.0167) + (0.0167, 0.0167, 0.01) + (0.05, 0.0267, 0.0067)

Then add:
= (0.1334, 0.0767, 0) = 13.34%, 7.67%, 0%

**Interpretation:** Diversification smooths out returns!

**Example 2.5: Physics - Force Vectors**

Three forces acting on a spacecraft:

**Gravity:** (0, -9.8, 0) N = 9.8 N downward
**Thrust:** (5, 15, 0) N = 5 N east, 15 N up
**Wind:** (-2, 0, 3) N = 2 N west, 3 N north

**Net force:**
(0, -9.8, 0) + (5, 15, 0) + (-2, 0, 3)
= (3, 5.2, 3) N

**Resultant:** 3 N east, 5.2 N up, 3 N north

**Example 2.6: Sensor Fusion**

Three sensors measuring temperature, but each has noise:

**Sensor 1:** (22.1, 23.0, 21.8)°C over 3 time points
**Sensor 2:** (21.9, 23.2, 22.0)°C
**Sensor 3:** (22.2, 22.8, 21.9)°C

**Average reading (more accurate):**
(1/3) × [(22.1, 23.0, 21.8) + (21.9, 23.2, 22.0) + (22.2, 22.8, 21.9)]
= (1/3) × (66.2, 69.0, 65.7)
= (22.07, 23.0, 21.9)°C

**Example 2.7: Word Embeddings**

In NLP, adding word vectors captures semantic relationships:

**king** = (0.5, 0.3, 0.8, -0.2) [simplified to 4D for illustration]
**man** = (0.4, 0.2, 0.1, -0.1)
**woman** = (0.3, 0.2, -0.1, 0.2)

**Compute: king - man + woman**

First: **king - man**
= (0.5, 0.3, 0.8, -0.2) - (0.4, 0.2, 0.1, -0.1)
= (0.1, 0.1, 0.7, -0.1)

Then: **result + woman**
= (0.1, 0.1, 0.7, -0.1) + (0.3, 0.2, -0.1, 0.2)
= (0.4, 0.3, 0.6, 0.1)

This result would be close to the vector for **queen**! The vectors capture gender relationships.

## Practice Problems - Vector Addition

**Problem 2.1: Basic Addition**

Given:
**u** = (2, -1, 3, 5)
**v** = (1, 4, -2, 3)
**w** = (-3, 2, 1, -1)

Calculate:
a) **u + v**
b) **v + w**
c) **u + v + w**
d) Verify that (**u + v**) + **w** = **u** + (**v + w**)

**Problem 2.2: Weekly Activity Tracking**

Your smartwatch records daily activity vectors (steps, active minutes, calories):

- Monday: (8000, 45, 350)
- Tuesday: (6500, 30, 280)
- Wednesday: (10000, 60, 420)
- Thursday: (7200, 40, 310)
- Friday: (9100, 50, 380)
- Saturday: (12000, 75, 480)
- Sunday: (5000, 20, 200)

a) Calculate your total weekly activity
b) Calculate your average daily activity
c) On which day were you most active? (Hint: think about what "most active" means)

**Problem 2.3: Store Sales**

Three stores report weekly sales vectors (electronics, clothing, food, other):

- Store A: (15000, 25000, 35000, 10000) dollars
- Store B: (20000, 18000, 42000, 12000) dollars
- Store C: (12000, 22000, 38000, 9000) dollars

a) Calculate total sales across all stores
b) Calculate average per store
c) If each store gets a bonus equal to 5% of their total sales, calculate the bonus as a vector for Store A

**Problem 2.4: GPS Navigation**

You take a hike with the following displacement vectors (east, north) in km:

- Leg 1: (2.5, 3.0)
- Leg 2: (1.5, -1.0) [negative means west/south]
- Leg 3: (-1.0, 2.5)
- Leg 4: (-3.0, -1.5)

a) What is your final position relative to the starting point?
b) How far east/west and north/south are you from the start?
c) If you want to return directly to start, what displacement vector do you need?

**Problem 2.5: Climate Data**

Monthly average temperatures (morning, afternoon, evening) in Celsius:

- January: (5, 12, 8)
- February: (6, 14, 9)
- March: (10, 18, 13)

a) Calculate the total temperature readings for the quarter
b) Calculate the average temperature vector for the quarter
c) If temperatures increase by (2, 3, 2) degrees next year, what would March's vector be?

**Problem 2.6: Portfolio Management**

You have three investment portfolios with returns (year 1, year 2, year 3):

- Portfolio A: (0.08, 0.12, -0.03)
- Portfolio B: (0.05, 0.05, 0.05)
- Portfolio C: (0.15, -0.05, 0.20)

a) If you invest equally (1/3 each), what are your combined returns?
b) If you invest 50% in A, 30% in B, and 20% in C, what are your returns?
c) Which allocation gives you better returns in year 3?

**Problem 2.7: Error Analysis**

A sensor makes three measurements of the same quantity. The true value is (100, 200, 150).

The sensor readings are:
- Reading 1: (102, 198, 151)
- Reading 2: (99, 201, 149)
- Reading 3: (101, 199, 150)

a) Calculate the error vector for each reading (reading - true value)
b) Calculate the average of all three readings
c) Calculate the error of the average
d) Is the average more accurate than individual readings?

---

<a name="scalar-multiplication"></a>
# 3. Scalar-Vector Multiplication: Scaling Information

## The Problem We're Solving

Your fitness tracker shows:
**Monday** = (8000 steps, 5 floors, 30 minutes)

**Question 1:** What if you did exactly twice as much activity?

You'd multiply each component by 2:
- Steps: 8000 × 2 = 16,000
- Floors: 5 × 2 = 10
- Minutes: 30 × 2 = 60

**2 × Monday = (16000, 10, 60)**

**Question 2:** What if you did half as much?

**0.5 × Monday = (4000, 2.5, 15)**

**Question 3:** What if you went backwards (negative activity)?

This doesn't make physical sense for steps, but mathematically:

**-1 × Monday = (-8000, -5, -30)**

This could represent "undoing" the activity or going in the opposite direction.

## Formal Definition

To multiply a vector by a scalar (a single number), multiply **each component** by that scalar.

If **v** = (v₁, v₂, ..., vₙ) and α is a scalar, then:

**α · v** = (α·v₁, α·v₂, α·v₃, ..., α·vₙ)

**Note:** The dot (·) is often omitted: α**v** means the same as α·**v**

**Result:** You get a new vector of the same dimension!

## Geometric Interpretation: Stretching and Shrinking

Consider **v** = (3, 4):

**Positive scalars (α > 0):**
- **2v** = (6, 8) → Points in **same direction**, **twice as long**
- **3v** = (9, 12) → Points in **same direction**, **three times as long**
- **0.5v** = (1.5, 2) → Points in **same direction**, **half as long**
- **0.1v** = (0.3, 0.4) → Points in **same direction**, **much shorter**

**Negative scalars (α < 0):**
- **-v** = **-1v** = (-3, -4) → Points in **opposite direction**, **same length**
- **-2v** = (-6, -8) → Points in **opposite direction**, **twice as long**
- **-0.5v** = (-1.5, -2) → Points in **opposite direction**, **half as long**

**Zero scalar (α = 0):**
- **0v** = (0, 0) → Collapses to the **zero vector** (origin)

**Geometric summary:**
- **|α| > 1:** Stretches the vector (makes it longer)
- **0 < |α| < 1:** Shrinks the vector (makes it shorter)
- **α < 0:** Flips the direction
- **α = 0:** Collapses to origin

## Why Direction Is Preserved (or Exactly Reversed)

**Key insight:** Scalar multiplication affects **magnitude** but preserves the **direction** (or exactly reverses it if α < 0).

**Example:** 
- Original: **v** = (3, 4)
- Scaled: **2v** = (6, 8)

The ratio between components stays the same:
- Original ratio: 4/3 ≈ 1.33
- Scaled ratio: 8/6 ≈ 1.33 ✓

This means the angle of the vector doesn't change—just its length!

## Properties of Scalar Multiplication

**1. Associativity with scalars:** α(β**v**) = (αβ)**v**

**Example:**
**v** = (2, 3)
α = 3, β = 2

**Left side:** 3(2**v**) = 3(4, 6) = (12, 18)
**Right side:** (3×2)**v** = 6(2, 3) = (12, 18) ✓

**Meaning:** You can multiply scalars first, then apply to vector.

**2. Distributivity over vector addition:** α(**u + v**) = α**u** + α**v**

**Example:**
α = 2
**u** = (1, 2)
**v** = (3, 4)

**Left side:** 2[(1, 2) + (3, 4)] = 2(4, 6) = (8, 12)
**Right side:** 2(1, 2) + 2(3, 4) = (2, 4) + (6, 8) = (8, 12) ✓

**Meaning:** You can distribute scalar multiplication over addition.

**3. Distributivity over scalar addition:** (α + β)**v** = α**v** + β**v**

**Example:**
α = 2, β = 3
**v** = (1, 2)

**Left side:** (2+3)(1, 2) = 5(1, 2) = (5, 10)
**Right side:** 2(1, 2) + 3(1, 2) = (2, 4) + (3, 6) = (5, 10) ✓

**4. Identity:** 1·**v** = **v**

Multiplying by 1 doesn't change the vector.

**5. Zero:** 0·**v** = **0** and α·**0** = **0**

Multiplying by zero (or multiplying zero vector) gives zero vector.

## Why Machine Learning Cares About Scalar Multiplication

Scalar multiplication appears constantly in ML:

### 1. Learning Rates in Gradient Descent

**The core of neural network training:**

Current weights: **w**
Gradient (direction to improve): **∇L**
Learning rate: α (typically 0.001 to 0.1)

**Update rule:**
**w_new** = **w** - α**∇L**

**Why multiply by α?**
- The gradient **∇L** tells us the direction to move
- But it might be too large—taking full step could overshoot!
- α scales it down: α**∇L** is a small step in the right direction

**Example:**
**w** = (1.0, 2.0, 0.5)
**∇L** = (10, -5, 8) [large gradient!]
α = 0.01 [small learning rate]

α**∇L** = 0.01(10, -5, 8) = (0.1, -0.05, 0.08) [small update]

**w_new** = (1.0, 2.0, 0.5) - (0.1, -0.05, 0.08)
= (0.9, 2.05, 0.42)

Small, controlled update!

### 2. Feature Scaling and Normalization

**Problem:** Features have different scales
- Feature 1: House size (500-5000 sq ft)
- Feature 2: Number of bedrooms (1-5)

**Solution:** Scale each feature

**v_original** = (2000, 3)

**Normalize to [0, 1]:**
- Divide size by 5000: 2000/5000 = 0.4
- Divide bedrooms by 5: 3/5 = 0.6

**v_normalized** = (0.4, 0.6)

This is scalar multiplication: multiply each component by different scalars!

### 3. Regularization

**L2 regularization:** Penalize large weights

Loss = prediction_error + λ||**w**||²

where λ is a scalar controlling regularization strength.

**Weight decay:** Multiply weights by slightly less than 1

**w_new** = 0.999 × **w** - α**∇L**

The 0.999 factor slowly shrinks weights toward zero.

### 4. Data Augmentation

**Image augmentation:** Scale pixel values

**img_darkened** = 0.7 × **img** (70% brightness)
**img_brightened** = 1.3 × **img** (130% brightness)

### 5. Ensemble Weighting

**Combine model predictions with weights:**

**p_final** = 0.5**p₁** + 0.3**p₂** + 0.2**p₃**

This is scalar multiplication plus addition!

### 6. Momentum in Optimization

**Momentum term in SGD:**

**velocity** = β × **velocity_old** + **gradient**

where β ≈ 0.9 is a scalar that determines how much past velocity to keep.

## Detailed Examples

**Example 3.1: Recipe Scaling**

Recipe for 2 servings: (4 eggs, 2 cups flour, 1 cup milk, 0.5 tsp salt)

**For 6 servings (triple the recipe):**
3 × (4, 2, 1, 0.5) = (12, 6, 3, 1.5)

**For 1 serving (half the recipe):**
0.5 × (4, 2, 1, 0.5) = (2, 1, 0.5, 0.25)

**For 10 servings:**
5 × (4, 2, 1, 0.5) = (20, 10, 5, 2.5)

**Example 3.2: Price Changes**

Original prices: (10, 20, 30, 15) dollars

**20% discount (pay 80%):**
0.8 × (10, 20, 30, 15) = (8, 16, 24, 12) dollars

**50% markup (pay 150%):**
1.5 × (10, 20, 30, 15) = (15, 30, 45, 22.5) dollars

**10% tax (pay 110%):**
1.1 × (10, 20, 30, 15) = (11, 22, 33, 16.5) dollars

**Black Friday sale (pay 40%):**
0.4 × (10, 20, 30, 15) = (4, 8, 12, 6) dollars

**Example 3.3: Unit Conversions**

Distance vector in miles: (5, 10, 3, 7.5) miles

**Convert to kilometers (1 mile = 1.60934 km):**
1.60934 × (5, 10, 3, 7.5) 
= (8.047, 16.093, 4.828, 12.070) km

**Convert to feet (1 mile = 5280 feet):**
5280 × (5, 10, 3, 7.5)
= (26400, 52800, 15840, 39600) feet

**Example 3.4: Investment Returns**

Portfolio value: (10000, 5000, 8000, 3000) dollars in 4 assets

**After 15% gain (multiply by 1.15):**
1.15 × (10000, 5000, 8000, 3000)
= (11500, 5750, 9200, 3450) dollars

**After 8% loss (multiply by 0.92):**
0.92 × (10000, 5000, 8000, 3000)
= (9200, 4600, 7360, 2760) dollars

**After 25% loss (multiply by 0.75):**
0.75 × (10000, 5000, 8000, 3000)
= (7500, 3750, 6000, 2250) dollars

**Example 3.5: Temperature Conversion**

Temperatures in Celsius: (0, 10, 20, 30, 40)

**Convert to Fahrenheit: F = 1.8C + 32**

This isn't pure scalar multiplication because of the +32!

First, multiply:
1.8 × (0, 10, 20, 30, 40) = (0, 18, 36, 54, 72)

Then add 32 to each:
= (32, 50, 68, 86, 104) Fahrenheit

**Example 3.6: Image Brightness**

Pixel values: (128, 200, 64, 255, 100)

**Darken to 60% brightness:**
0.6 × (128, 200, 64, 255, 100)
= (76.8, 120, 38.4, 153, 60)

**Brighten to 140%:**
1.4 × (128, 200, 64, 255, 100)
= (179.2, 280, 89.6, 357, 140)

Note: Pixel values should be capped at 255, so (179.2, 255, 89.6, 255, 140)

**Example 3.7: Physics - Velocity**

Velocity vector: (30, 40, 10) m/s (east, north, up)

**Double the speed:**
2 × (30, 40, 10) = (60, 80, 20) m/s

**Opposite direction:**
-1 × (30, 40, 10) = (-30, -40, -10) m/s (west, south, down)

**Half speed in opposite direction:**
-0.5 × (30, 40, 10) = (-15, -20, -5) m/s

**Example 3.8: Gradient Descent Step**

Current weights: **w** = (1.5, -0.3, 2.1, 0.8)
Gradient: **∇L** = (20, -15, 30, 10)
Learning rate: α = 0.01

**Gradient step:**
α**∇L** = 0.01 × (20, -15, 30, 10) = (0.2, -0.15, 0.3, 0.1)

**Updated weights:**
**w_new** = **w** - α**∇L**
= (1.5, -0.3, 2.1, 0.8) - (0.2, -0.15, 0.3, 0.1)
= (1.3, -0.15, 1.8, 0.7)

## Linear Combinations: The Power of Both Operations

Combining scalar multiplication and vector addition gives us **linear combinations**:

α₁**v₁** + α₂**v₂** + ... + αₙ**vₙ**

**This is the foundation of linear algebra!**

### Example 3.9: Weighted Average

Three test score vectors:
**s₁** = (85, 90, 78)
**s₂** = (92, 88, 95)
**s₃** = (78, 85, 82)

**Weighted average (30%, 50%, 20%):**
0.3**s₁** + 0.5**s₂** + 0.2**s₃**

Step by step:
0.3(85, 90, 78) = (25.5, 27, 23.4)
0.5(92, 88, 95) = (46, 44, 47.5)
0.2(78, 85, 82) = (15.6, 17, 16.4)

Sum: (25.5 + 46 + 15.6, 27 + 44 + 17, 23.4 + 47.5 + 16.4)
= (87.1, 88, 87.3)

### Example 3.10: RGB Color Mixing

Base colors:
- Red: (255, 0, 0)
- Green: (0, 255, 0)
- Blue: (0, 0, 255)

**Yellow (equal red + green):**
0.5(255, 0, 0) + 0.5(0, 255, 0) = (127.5, 127.5, 0)

**Orange (70% red, 30% green):**
0.7(255, 0, 0) + 0.3(0, 255, 0) = (178.5, 76.5, 0)

**Purple (60% red, 40% blue):**
0.6(255, 0, 0) + 0.4(0, 0, 255) = (153, 0, 102)

**White (equal mix of all):**
(1/3)(255, 0, 0) + (1/3)(0, 255, 0) + (1/3)(0, 0, 255)
= (85, 85, 85) [gray, actually—pure white is (255, 255, 255)]

**Correct white:**
1(255, 0, 0) + 1(0, 255, 0) + 1(0, 0, 255) = (255, 255, 255)

## Practice Problems - Scalar Multiplication

**Problem 3.1: Basic Scalar Multiplication**

Given **v** = (3, -2, 5, 1):

a) Calculate 4**v**
b) Calculate -2**v**
c) Calculate 0.5**v**
d) Calculate 0**v**
e) Verify that 2(3**v**) = (2×3)**v** = 6**v**

**Problem 3.2: Recipe Scaling**

A recipe for 4 servings requires: (8 oz pasta, 2 cups sauce, 1 lb meat, 0.5 cup cheese)

a) Scale for 6 servings
b) Scale for 2 servings
c) Scale for 10 servings
d) If you only have 6 oz pasta, what fraction of the recipe should you make?

**Problem 3.3: Financial Planning**

Your monthly budget: (1500 rent, 400 food, 200 transport, 300 entertainment) dollars

a) Calculate your annual spending (12 months)
b) If you get a 20% raise and increase all categories by 20%, what's your new budget?
c) If you need to cut spending by 15%, what's your new budget?
d) Calculate spending for 3 months

**Problem 3.4: Learning Rate Experiment**

Gradient vector: **∇L** = (100, -50, 75, 25)

Calculate the update step α**∇L** for these learning rates:
a) α = 0.1 (large learning rate)
b) α = 0.01 (medium)
c) α = 0.001 (small)
d) Which learning rate gives the smallest update? Why might this be good or bad?

**Problem 3.5: Image Brightness**

Pixel vector: (128, 64, 200, 180, 100)

a) Create a darker version at 70% brightness
b) Create a brighter version at 130% brightness
c) Create an inverted version (hint: for pixels in 0-255, inverted pixel = 255 - pixel. Can you express this using scalar multiplication and addition?)

**Problem 3.6: Unit Conversion**

Weight vector in pounds: (150, 180, 200, 165) lbs

a) Convert to kilograms (1 lb = 0.453592 kg)
b) Convert to ounces (1 lb = 16 oz)
c) If each person gains 10% weight, what are the new weights in pounds?

**Problem 3.7: Linear Combinations**

Given:
**u** = (2, 3)
**v** = (1, -1)

Calculate:
a) 2**u** + 3**v**
b) -**u** + 2**v**
c) 0.5**u** + 0.5**v**
d) 3**u** - 2**v**
e) Can you make the vector (5, 7) using a linear combination of **u** and **v**? If so, find the coefficients.

---

<a name="inner-product"></a>
# 4. Inner Product (Dot Product): Measuring Alignment

## The Problem We're Solving

You're on a dating app and you create a preference vector for your ideal match:

**Your preferences:**
- Likes hiking: 10 (very important!)
- Likes reading: 7 (important)
- Likes cooking: 5 (somewhat important)
- Likes gaming: 2 (not very important)

**p** = (10, 7, 5, 2)

Someone's profile appears:

**Their interests:**
- Hiking: Yes = 1
- Reading: No = 0
- Cooking: Yes = 1
- Gaming: Yes = 1

**a** = (1, 0, 1, 1)

**Question:** How compatible are you?

You'd naturally calculate a match score by considering:
1. How much do they like each activity? (their profile)
2. How much do you care about each activity? (your preferences)
3. Multiply these together for each activity
4. Add up all the products

**Compatibility calculation:**
- Hiking: 10 × 1 = 10 (they like it AND you care a lot—perfect!)
- Reading: 7 × 0 = 0 (they don't like it but you wanted it—unfortunate)
- Cooking: 5 × 1 = 5 (they like it and you care some—good!)
- Gaming: 2 × 1 = 2 (they like it but you barely care—whatever)

**Total compatibility: 10 + 0 + 5 + 2 = 17**

**This is the inner product (also called dot product)!**

## What Is the Inner Product Doing?

The inner product answers: **"How much do these two vectors agree or align?"**

- When vectors point in similar directions → large positive inner product
- When vectors point in opposite directions → large negative inner product
- When vectors are perpendicular (unrelated) → inner product near zero

**It's measuring the "overlap" or "alignment" between vectors.**

## Formal Definition

The **inner product** (or **dot product**) of two vectors is the sum of the products of corresponding components.

If **u** = (u₁, u₂, ..., uₙ) and **v** = (v₁, v₂, ..., vₙ), then:

**u · v** = u₁v₁ + u₂v₂ + u₃v₃ + ... + uₙvₙ

**Alternative notations:**
- **u** · **v** (dot notation)
- **u**ᵀ**v** (transpose notation, used in Boyd's book)
- ⟨**u**, **v**⟩ (angle bracket notation)

**Important:** The result is a **single number** (scalar), not a vector!

**Requirements:**
- Both vectors must have the same dimension
- Otherwise, the operation is undefined

## Geometric Interpretation

### Example in 2D

**u** = (3, 4) and **v** = (6, 8)

**u · v** = 3(6) + 4(8) = 18 + 32 = 50

**Notice:** **v** = 2**u**, so they point in exactly the same direction!

The inner product is large and positive!

**Example: Perpendicular vectors**

**u** = (1, 0) [pointing east]
**v** = (0, 1) [pointing north]

**u · v** = 1(0) + 0(1) = 0

They're perpendicular (90° angle), so inner product is zero!

**Example: Opposite directions**

**u** = (1, 1)
**v** = (-1, -1)

**u · v** = 1(-1) + 1(-1) = -1 - 1 = -2

They point opposite ways, so inner product is negative!

### The Pattern

**Large positive value:** Vectors point in similar directions
- **u** and **v** "agree"
- They're "aligned"

**Large negative value:** Vectors point in opposite directions
- **u** and **v** "disagree"
- They're "anti-aligned"

**Zero (or close to zero):** Vectors are perpendicular
- **u** and **v** are "orthogonal" (perpendicular)
- They're "unrelated" or "independent"

## Properties of Inner Product

**1. Commutative:** **u · v** = **v · u**

Order doesn't matter!

**Example:**
**u** = (2, 3)
**v** = (4, 5)

**u · v** = 2(4) + 3(5) = 8 + 15 = 23
**v · u** = 4(2) + 5(3) = 8 + 15 = 23 ✓

**2. Distributive over addition:** **u** · (**v + w**) = **u · v** + **u · w**

**Example:**
**u** = (1, 2)
**v** = (3, 4)
**w** = (5, 6)

**Left side:**
**v + w** = (8, 10)
**u** · (8, 10) = 1(8) + 2(10) = 28

**Right side:**
**u · v** = 1(3) + 2(4) = 11
**u · w** = 1(5) + 2(6) = 17
Sum = 11 + 17 = 28 ✓

**3. Scalar multiplication:** (α**u**) · **v** = α(**u · v**) = **u** · (α**v**)

**Example:**
**u** = (2, 3)
**v** = (4, 5)
α = 3

(3**u**) · **v** = (6, 9) · (4, 5) = 24 + 45 = 69

3(**u · v**) = 3(8 + 15) = 3(23) = 69 ✓

**4. Positive definite:** **v · v** ≥ 0, with equality only if **v** = **0**

**v · v** is always non-negative!

**Example:**
**v** = (3, 4)
**v · v** = 3(3) + 4(4) = 9 + 16 = 25 > 0 ✓

This quantity **v · v** is special—it's related to the length of **v**!

## Connection to Vector Length (Norm)

The inner product of a vector with itself gives the **square of its length**:

**v · v** = ||**v**||²

**Example:**
**v** = (3, 4)

**v · v** = 3² + 4² = 9 + 16 = 25

||**v**|| = √(3² + 4²) = √25 = 5

||**v**||² = 25 ✓

This connects inner product to geometry! We'll explore this more in the Norm chapter.

## Why This Is THE Most Important Operation in Machine Learning

The inner product is absolutely fundamental to ML. Here's why:

### 1. Linear Models Make Predictions Using Inner Products

**Linear regression:**

prediction = **w · x** + b

where:
- **w** = weight vector (learned from data)
- **x** = input feature vector
- b = bias term

**Example: House Price Prediction**

Weights: **w** = (100, 50000, -1000, 20000)
- $100 per square foot
- $50,000 per bedroom
- -$1,000 per year of age
- $20,000 per school rating point

House features: **x** = (2000, 3, 10, 8)
- 2000 sq ft
- 3 bedrooms
- 10 years old
- School rating 8/10

**Price = w · x + b**
= 100(2000) + 50000(3) + (-1000)(10) + 20000(8) + b
= 200,000 + 150,000 - 10,000 + 160,000 + b
= 500,000 + b

If b = 50,000 (base price), then **Price = $550,000**

**The core prediction is just an inner product!**

### 2. Neural Networks Are Built on Inner Products

Every single neuron in a neural network computes:

output = activation(**w · x** + b)

where:
- **x** = inputs from previous layer
- **w** = weights (learned)
- **w · x** = inner product
- activation = nonlinear function (ReLU, sigmoid, etc.)

**A deep neural network is thousands of inner products stacked together!**

### 3. Similarity Measures

**Document similarity:**

Document 1: **d₁** = (5, 2, 0, 8, 3, ...) [word counts]
Document 2: **d₂** = (6, 1, 0, 9, 2, ...)

**Similarity score = d₁ · d₂**

**d₁ · d₂** = 5(6) + 2(1) + 0(0) + 8(9) + 3(2) + ...
= 30 + 2 + 0 + 72 + 6 + ...
= high value → similar documents!

**Why?** If both documents use the same words frequently, the products will be large!

### 4. Recommender Systems

**User-item ratings:**

User preference vector: **u** = (5, 1, 4, 2, 5) [ratings for 5 movie genres]
Movie vector: **m** = (0.9, 0.1, 0.8, 0.2, 0.9) [how much the movie belongs to each genre]

**Predicted rating = u · m**
= 5(0.9) + 1(0.1) + 4(0.8) + 2(0.2) + 5(0.9)
= 4.5 + 0.1 + 3.2 + 0.4 + 4.5
= 12.7

High score → recommend this movie!

### 5. Distance Calculations

Distance between vectors uses inner products:

distance(**u**, **v**) = ||**u** - **v**|| = √[(**u - v**) · (**u - v**)]

We'll explore this in the Distance chapter!

### 6. Cosine Similarity

The angle θ between vectors is given by:

cos(θ) = (**u · v**) / (||**u**|| ||**v**||)

This is used everywhere in ML:
- Document similarity
- Image recognition
- Face verification
- Semantic search

### 7. Attention Mechanisms (Transformers)

Modern NLP models like GPT use attention:

attention(**Q**, **K**) = softmax(**Q** · **Kᵀ**)

where **Q** and **K** are query and key vectors.

**The inner product determines which words pay attention to which other words!**

### 8. Support Vector Machines

SVM decision boundary:

decision = **w · x** + b

If **w · x** + b > 0 → class 1
If **w · x** + b < 0 → class 2

The inner product determines which side of the boundary you're on!

## Detailed Examples

**Example 4.1: Same Direction**

**u** = (3, 4)
**v** = (6, 8) = 2**u**

**u · v** = 3(6) + 4(8) = 18 + 32 = 50

They point in exactly the same direction, so inner product is large and positive!

**Example 4.2: Opposite Directions**

**u** = (3, 4)
**v** = (-3, -4) = -**u**

**u · v** = 3(-3) + 4(-4) = -9 - 16 = -25

They point in opposite directions, so inner product is negative!

**Example 4.3: Perpendicular**

**u** = (1, 0) [pointing east]
**v** = (0, 1) [pointing north]

**u · v** = 1(0) + 0(1) = 0

They're perpendicular (90° angle), so inner product is zero!

**Example 4.4: Another Perpendicular Pair**

**u** = (3, 4)
**v** = (4, -3)

**u · v** = 3(4) + 4(-3) = 12 - 12 = 0

Perpendicular! Not obvious from looking at coordinates, but inner product reveals it!

**Example 4.5: Spam Detection**

Spam pattern learned from training data:
**spam** = (12, 2.5, 8, 0.9, 0.8)

Represents weights for features:
- Capital letters: 12
- Exclamation marks: 2.5
- Short words: 8
- Contains "FREE": 0.9
- Contains "MONEY": 0.8

New email features:
**email** = (15, 3, 4, 1, 1)

**Spam score = spam · email**
= 12(15) + 2.5(3) + 8(4) + 0.9(1) + 0.8(1)
= 180 + 7.5 + 32 + 0.9 + 0.8
= 221.2

**High score → likely spam!**

If score < some threshold (say 50), classify as ham (not spam).

**Example 4.6: Movie Recommendations**

User preference: **u** = (5, 2, 4, 1, 5)
- Action: 5 (love it!)
- Comedy: 2 (meh)
- Drama: 4 (like it)
- Horror: 1 (hate it)
- Sci-fi: 5 (love it!)

Movie 1 feature vector: **m₁** = (0.8, 0.1, 0.2, 0, 0.7)
Movie 2 feature vector: **m₂** = (0.1, 0.9, 0.3, 0.5, 0)

**Rating prediction for Movie 1:**
**u · m₁** = 5(0.8) + 2(0.1) + 4(0.2) + 1(0) + 5(0.7)
= 4 + 0.2 + 0.8 + 0 + 3.5
= 8.5 (high score!)

**Rating prediction for Movie 2:**
**u · m₂** = 5(0.1) + 2(0.9) + 4(0.3) + 1(0.5) + 5(0)
= 0.5 + 1.8 + 1.2 + 0.5 + 0
= 4.0 (lower score)

**Recommendation: Watch Movie 1!**

**Example 4.7: Word Similarity**

Word embeddings (simplified to 4D):

**king** = (0.8, 0.3, 0.6, -0.2)
**queen** = (0.7, 0.3, 0.5, 0.1)
**man** = (0.5, 0.2, 0.1, -0.1)
**woman** = (0.4, 0.2, 0, 0.2)

**How similar are king and queen?**
**king · queen** = 0.8(0.7) + 0.3(0.3) + 0.6(0.5) + (-0.2)(0.1)
= 0.56 + 0.09 + 0.3 - 0.02
= 0.93 (high similarity!)

**How similar are king and woman?**
**king · woman** = 0.8(0.4) + 0.3(0.2) + 0.6(0) + (-0.2)(0.2)
= 0.32 + 0.06 + 0 - 0.04
= 0.34 (lower similarity)

King and queen are more similar to each other!

**Example 4.8: Linear Regression Prediction**

Weight vector: **w** = (2.5, -0.8, 1.2, 3.0)
Represents coefficients for features: study hours, distractions, sleep hours, attendance

Student features: **x** = (20, 5, 7, 0.9)
- 20 hours of study
- 5 hours of distractions
- 7 hours of sleep
- 0.9 (90%) attendance

**Grade prediction = w · x + b**

**w · x** = 2.5(20) + (-0.8)(5) + 1.2(7) + 3.0(0.9)
= 50 - 4 + 8.4 + 2.7
= 57.1

If b = 30, then **Grade = 57.1 + 30 = 87.1%**

**Example 4.9: Face Verification**

Two face embedding vectors (simplified to 5D):

**Face 1:** **f₁** = (0.8, 0.2, 0.5, 0.3, 0.9)
**Face 2:** **f₂** = (0.7, 0.3, 0.4, 0.3, 0.8)

**Similarity = f₁ · f₂**
= 0.8(0.7) + 0.2(0.3) + 0.5(0.4) + 0.3(0.3) + 0.9(0.8)
= 0.56 + 0.06 + 0.2 + 0.09 + 0.72
= 1.63

If similarity > threshold (say 1.5), they're the same person!

**Example 4.10: Search Engine Ranking**

Query vector: **q** = (2, 1, 0, 3, 1) [term frequencies]
Document vectors:
- **d₁** = (3, 1, 0, 4, 2)
- **d₂** = (1, 0, 5, 0, 1)
- **d₃** = (2, 2, 0, 3, 1)

**Relevance scores:**
**q · d₁** = 2(3) + 1(1) + 0(0) + 3(4) + 1(2) = 6 + 1 + 0 + 12 + 2 = 21
**q · d₂** = 2(1) + 1(0) + 0(5) + 3(0) + 1(1) = 2 + 0 + 0 + 0 + 1 = 3
**q · d₃** = 2(2) + 1(2) + 0(0) + 3(3) + 1(1) = 4 + 2 + 0 + 9 + 1 = 16

**Ranking:** d₁ (score 21) > d₃ (score 16) > d₂ (score 3)

Show results in this order!

## Special Case: Inner Product with Zero Vector

**v · 0** = v₁(0) + v₂(0) + ... + vₙ(0) = 0

The inner product with the zero vector is always zero!

**Geometric meaning:** The zero vector has no direction, so it can't "align" with anything.

## Practice Problems - Inner Product

**Problem 4.1: Basic Inner Products**

Given:
**u** = (2, -1, 3, 5)
**v** = (1, 4, -2, 3)
**w** = (-2, 1, -3, -5)

Calculate:
a) **u · v**
b) **v · u** (verify commutativity)
c) **u · w** (notice **w** = -**u**)
d) **u · u**
e) **v · 0** where **0** = (0, 0, 0, 0)

**Problem 4.2: Perpendicular Vectors**

Which of these pairs are perpendicular (orthogonal)?

a) **u** = (1, 2) and **v** = (2, -1)
b) **u** = (3, 4) and **v** = (4, -3)
c) **u** = (1, 1, 1) and **v** = (1, -1, 0)
d) **u** = (2, 3, 6) and **v** = (3, -2, 0)

**Problem 4.3: Dating App Compatibility**

You create preference weights: **p** = (10, 8, 6, 5, 3, 2)
For: outdoors, reading, cooking, music, sports, gaming

Three potential matches with interest levels (0-1):
- Person A: (1, 1, 0, 1, 0, 0)
- Person B: (0, 1, 1, 1, 1, 1)
- Person C: (1, 0, 1, 0, 1, 0)

a) Calculate compatibility scores for each person
b) Who is most compatible?
c) Why does Person A score higher than Person C even though both have 3 interests?

**Problem 4.4: Spam Classification**

Spam weight vector: **w** = (15, 3, 10, 2, 8)
For: capitals, exclamations, urgent_words, links, attachments

Classify these emails:
- Email 1: (20, 5, 8, 3, 2)
- Email 2: (5, 0, 2, 1, 0)
- Email 3: (25, 8, 15, 5, 3)

a) Calculate spam score for each
b) If threshold is 100, which are spam?
c) Why does Email 1 score higher than Email 2?

**Problem 4.5: Movie Recommendations**

User preference vector: **u** = (5, 1, 4, 0, 5)
For genres: Action, Comedy, Drama, Horror, Sci-Fi

Movie genre vectors:
- Movie A: (0.9, 0, 0.2, 0, 0.8)
- Movie B: (0.1, 0.8, 0.5, 0.3, 0)
- Movie C: (0.7, 0.2, 0.3, 0, 0.6)

a) Calculate predicted ratings for each movie
b) Which movie should be recommended first?
c) If the user rates Comedy as 4 instead of 1, recalculate ratings. Which movie benefits most?

**Problem 4.6: Linear Regression**

Weight vector: **w** = (50, 30000, -500, 10000)
For: square footage, bedrooms, age, school rating

Predict prices for:
- House 1: (1500, 2, 20, 6)
- House 2: (2500, 4, 5, 9)
- House 3: (1800, 3, 15, 7)

a) Calculate **w · x** for each house (ignore bias for now)
b) If bias b = 100000, what are the final prices?
c) Which features contribute most to House 2's high price?

**Problem 4.7: Document Similarity**

Two documents represented as word count vectors for vocabulary: [cat, dog, bird, fish, run, jump]

- Doc 1: (5, 2, 0, 0, 3, 1)
- Doc 2: (6, 1, 0, 0, 2, 2)
- Doc 3: (0, 0, 8, 5, 0, 0)

a) Calculate similarity **d₁ · d₂**
b) Calculate similarity **d₁ · d₃**
c) Which pair is more similar? Why does this make sense?

**Problem 4.8: Verifying Properties**

Given **u** = (1, 2, 3), **v** = (4, 5, 6), **w** = (7, 8, 9):

Verify:
a) Commutativity: **u · v** = **v · u**
b) Distributivity: **u** · (**v + w**) = **u · v** + **u · w**
c) Scalar multiplication: 2(**u · v**) = (2**u**) · **v**
d) **u · u** ≥ 0

**Problem 4.9: Geometric Interpretation**

**u** = (3, 4) and **v** = (-4, 3)

a) Calculate **u · v**
b) What does this tell you about the angle between them?
c) Verify by calculating ||**u**|| and ||**v**||, then checking if they're perpendicular using Pythagoras on **u** + **v**

---

<a name="complexity"></a>
# 5. Complexity of Vector Computations

## Why Care About Computational Complexity?

In machine learning, we often work with:
- **Large vectors:** Millions of dimensions (e.g., image pixels, word vocabularies)
- **Many vectors:** Billions of data points (e.g., user interactions, sensor readings)
- **Repeated operations:** Training iterates millions of times

**Understanding complexity helps us:**
1. Estimate how long algorithms will take
2. Compare different approaches
3. Identify bottlenecks
4. Scale to bigger problems
5. Choose appropriate hardware (CPU vs GPU)

## Big-O Notation: A Quick Review

**O(n)** means: time grows linearly with input size n
- Double the input → double the time

**O(n²)** means: time grows quadratically
- Double the input → 4× the time

**O(1)** means: constant time
- Doesn't depend on input size

## Complexity of Vector Operations

Let's analyze the operations we've learned, assuming vectors have dimension n.

### Vector Addition: **u + v**

**Operation:** Add n pairs of numbers

```
(u₁ + v₁, u₂ + v₂, ..., uₙ + vₙ)
```

**Complexity:** **O(n)**

**Why?**
- n components to add
- Each addition is O(1)
- Total: n × O(1) = O(n)

**Example:**
- n = 1,000: ~1,000 operations
- n = 1,000,000: ~1,000,000 operations

**Scaling:** Linear! Double the dimension → double the time.

### Scalar Multiplication: α**v**

**Operation:** Multiply n numbers by scalar α

```
(αv₁, αv₂, ..., αvₙ)
```

**Complexity:** **O(n)**

**Why?**
- n multiplications
- Each is O(1)
- Total: O(n)

### Inner Product: **u · v**

**Operation:** Multiply n pairs, then sum

```
u₁v₁ + u₂v₂ + ... + uₙvₙ
```

**Complexity:** **O(n)**

**Why?**
- n multiplications: O(n)
- n-1 additions: O(n)
- Total: O(n) + O(n) = O(n)

**This is the most expensive** of our basic operations (by a constant factor), but still linear!

### Linear Combination: α**u** + β**v**

**Operation:** Scalar multiply both vectors, then add

**Complexity:** **O(n)**

**Why?**
- α**u**: O(n)
- β**v**: O(n)
- Sum: O(n)
- Total: O(n) + O(n) + O(n) = O(n)

All constants factors are absorbed in Big-O!

## Detailed Complexity Analysis

### Example: Computing an Average

Find average of k vectors, each of dimension n:

**avg** = (1/k)(**v₁** + **v₂** + ... + **vₖ**)

**Step 1:** Add k vectors
- Each addition: O(n)
- k-1 additions total: (k-1) × O(n) = O(kn)

**Step 2:** Scalar multiply by 1/k
- O(n)

**Total:** O(kn) + O(n) = **O(kn)**

**Interpretation:**
- More vectors (larger k) → proportionally more time
- Higher dimensions (larger n) → proportionally more time

**Example with numbers:**
- k = 100 vectors, n = 1,000 dimensions
- Operations: ~100,000
- k = 1,000 vectors, n = 1,000,000 dimensions
- Operations: ~1,000,000,000 (1 billion!)

### Example: K-Means Clustering (Simplified)

One iteration of K-means on N data points, K clusters, dimension n:

**For each point (N times):**
1. Calculate distance to each centroid (K times)
   - Each distance: O(n) [inner product-based]
2. Assign to nearest: O(K)

**Complexity per iteration:** O(NKn)

**Full algorithm:** I iterations → **O(INKn)**

**Why this matters:**
- N = 1,000,000 data points
- K = 100 clusters
- n = 100 dimensions
- I = 50 iterations
- Total operations: ~500,000,000,000 (500 billion!)

**This is why distributed computing matters for large-scale ML!**

### Example: Linear Regression Training

Training set: N examples, each with n features

**Computing predictions:** **Xw** where X is N×n matrix
- N inner products, each O(n)
- **O(Nn)**

**Computing gradient:** involves **Xᵀr** where r is N×1
- n inner products, each O(N)
- **O(Nn)**

**One gradient descent step:** **O(Nn)**

**Full training:** T iterations → **O(TNn)**

**Example:**
- N = 1,000,000 training examples
- n = 1,000 features
- T = 1,000 iterations
- Operations: ~1,000,000,000,000 (1 trillion!)

**Modern approach:** Use mini-batches and GPUs to parallelize!

## Memory Complexity

### Storage Requirements

**Single vector** in ℝⁿ: **O(n)** memory

**k vectors:** **O(kn)** memory

**Matrix (N×n):** **O(Nn)** memory

**Example:**
- Vector of 1 million floats (n = 1,000,000)
- Each float: 4 bytes
- Total: 4 MB per vector
- 1000 vectors: 4 GB!

**This is why:**
- Feature selection matters (reduce n)
- Sampling matters (reduce N)
- Dimensionality reduction matters (reduce n while keeping information)

## Practical Implications for Machine Learning

### 1. Feature Engineering

**Before:** n = 10,000 features
**After:** n = 100 features (selected best ones)

**Speedup:** 100× faster for all vector operations!

**Why:** O(n) operations become O(100) instead of O(10000)

### 2. Mini-Batch Training

**Full batch:** Process all N examples at once
- One gradient: O(Nn)
- Slow but accurate

**Mini-batch:** Process m << N examples at a time
- One gradient: O(mn)
- Fast but noisier
- Can parallelize across batches!

**Example:**
- N = 1,000,000, n = 1000
- Full batch: 1 billion operations per update
- Mini-batch (m=100): 100,000 operations per update
- 10,000× faster per update!
- More frequent updates compensate for noise

### 3. GPU Acceleration

**Why GPUs help:**
- Vector operations are **parallelizable**
- Add 1 million pairs simultaneously!
- Inner product: multiply all pairs in parallel, then reduce

**CPU:** O(n) operations done sequentially
**GPU:** O(n) operations done in O(log n) time with n processors

**Practical speedup:** 10-100× for large vectors!

### 4. Sparse Vectors

Many real-world vectors are **sparse** (mostly zeros):
- Text: most words don't appear in a document
- Collaborative filtering: most users haven't rated most items

**Dense representation:** (0, 0, 5, 0, 0, 3, 0, ..., 0)
- Storage: O(n)
- Operations: O(n)

**Sparse representation:** [(2: 5), (5: 3)]
- Storage: O(k) where k = number of non-zeros
- Operations: O(k)

**Example:**
- n = 100,000 vocabulary size
- k = 100 non-zero words per document
- Sparse speedup: 1000×!

## Complexity Summary Table

| Operation | Complexity | Description |
|-----------|-----------|-------------|
| Vector addition: **u + v** | O(n) | Linear in dimension |
| Scalar multiplication: α**v** | O(n) | Linear in dimension |
| Inner product: **u · v** | O(n) | Linear in dimension |
| Linear combination: α**u** + β**v** | O(n) | Linear in dimension |
| Average of k vectors | O(kn) | Linear in both |
| Distance: \\|**u - v**\\| | O(n) | Subtraction + norm |
| K-means iteration | O(NKn) | N points, K clusters, n dims |
| Linear regression step | O(Nn) | N examples, n features |

## Practice Problems - Complexity

**Problem 5.1: Estimating Runtime**

A vector operation takes 1 microsecond per dimension.

a) How long to add two vectors of dimension 1,000?
b) How long for dimension 1,000,000?
c) How long to compute inner product of dimension 10,000?
d) If you double the dimension, how does runtime change?

**Problem 5.2: Comparing Approaches**

You need to find the average of 1000 vectors, each of dimension 10,000.

Approach A: Add all vectors, then divide
Approach B: Add pairs repeatedly (tree-like)

a) What is the complexity of Approach A?
b) What is the complexity of Approach B?
c) Which is faster? Why or why not?

**Problem 5.3: Memory Requirements**

You have 1 million training examples, each with 500 features.

a) How much memory to store as float32 (4 bytes per number)?
b) If you reduce to 50 features (via feature selection), how much memory?
c) What's the speedup for inner product operations?

**Problem 5.4: Sparse vs Dense**

A text document vector has dimension 100,000 (vocabulary size) but only 200 non-zero entries.

a) Dense storage: how many numbers stored?
b) Sparse storage: how many numbers stored?
c) Dense inner product complexity?
d) Sparse inner product complexity (if both vectors sparse with ~200 non-zeros)?
e) What's the speedup ratio?

**Problem 5.5: Scaling Analysis**

K-means clustering: N data points, K clusters, n dimensions, I iterations.

Current: N=10,000, K=10, n=100, I=50

a) Estimate total operations (use O(INKn))
b) You get 10× more data (N=100,000). How do operations scale?
c) You double dimensions (n=200). How do operations scale?
d) Which has bigger impact on runtime: 10× data or 2× dimensions?

**Problem 5.6: Mini-Batch Benefits**

Neural network training: N=1,000,000 examples, n=1,000 features

Full batch: process all N at once
Mini-batch: process m=1,000 at once

a) Complexity of one full batch update?
b) Complexity of one mini-batch update?
c) How many mini-batches to cover all data once?
d) Total complexity to process all data once with mini-batches?
e) Are mini-batches actually faster for one full pass? Why use them?

**Problem 5.7: GPU Parallelization**

Inner product of two 1,000,000-dimensional vectors.

CPU: sequential, 1 operation per cycle
GPU: parallel, 1000 processors

a) CPU time: 1,000,000 cycles (simplified model)
b) GPU time with perfect parallelism?
c) Actual GPU overhead means 100× speedup. What's actual GPU time?
d) For what vector size does GPU become worthwhile (assuming setup cost of 1000 cycles)?

---

<a name="summary"></a>
# Chapter 1 Summary

## Key Concepts We've Learned

### 1. Vectors: Foundation of ML
- **Definition:** Ordered list of numbers representing data
- **Dimension:** Number of components (n for vector in ℝⁿ)
- **Geometric interpretation:** Direction and magnitude
- **Why ML uses them:** Unified representation for all data types

### 2. Vector Addition
- **Operation:** Add corresponding components
- **Notation:** **u + v** = (u₁+v₁, u₂+v₂, ..., uₙ+vₙ)
- **Geometric meaning:** Tip-to-tail combination
- **ML applications:** Averaging, gradient updates, ensemble methods
- **Complexity:** O(n)

### 3. Scalar Multiplication
- **Operation:** Multiply each component by scalar
- **Notation:** α**v** = (αv₁, αv₂, ..., αvₙ)
- **Geometric meaning:** Stretching/shrinking and direction flipping
- **ML applications:** Learning rates, normalization, weighting
- **Complexity:** O(n)

### 4. Inner Product
- **Operation:** Sum of products of components
- **Notation:** **u · v** = u₁v₁ + u₂v₂ + ... + uₙvₙ
- **Geometric meaning:** Measures alignment/similarity
- **ML applications:** Predictions, similarity, attention, distance
- **Complexity:** O(n)
- **Key insight:** Returns a scalar, not a vector!

### 5. Linear Combinations
- **Form:** α₁**v₁** + α₂**v₂** + ... + αₖ**vₖ**
- **Foundation of linear algebra:** Everything builds on this
- **ML applications:** Everywhere! Predictions, transformations, weighted averages

### 6. Computational Complexity
- **All basic operations:** O(n) in dimension
- **Practical impact:** Billions of operations in real ML
- **Optimization strategies:** Feature selection, mini-batches, sparsity, GPUs

## Connections to Machine Learning

**Every ML algorithm fundamentally:**
1. Represents data as vectors
2. Combines them with linear combinations
3. Measures similarity with inner products
4. Updates parameters with vector addition
5. Scales updates with scalar multiplication

**Examples we've seen:**
- Linear regression: **w · x** + b
- Neural networks: activation(**w · x** + b) repeated
- K-means: averaging vectors to find centroids
- Recommender systems: **user · item**
- Spam detection: **spam_pattern · email**
- Face recognition: **face₁ · face₂**

## What's Next?

**Chapter 5** will cover:
- Linear independence and dependence
- Basis vectors
- Orthogonality
- Gram-Schmidt algorithm

These concepts reveal the structure of vector spaces!

---

<a name="practice"></a>
# Comprehensive Practice Problems

## Section 1: Integrated Concepts

**Problem 6.1: Complete Pipeline**

You're building a simple linear classifier for email spam detection.

Training data (4 emails, 5 features each):
- Spam 1: (20, 5, 3, 1, 1) → label: +1
- Spam 2: (18, 4, 5, 1, 1) → label: +1
- Ham 1: (2, 0, 10, 0, 0) → label: -1
- Ham 2: (3, 1, 8, 0, 0) → label: -1

a) Calculate average spam vector (add spam vectors, multiply by 1/2)
b) Calculate average ham vector
c) Calculate weight vector: **w** = spam_avg - ham_avg
d) New email arrives: (15, 3, 4, 1, 0). Calculate **w · email**
e) If **w · email** > 0 classify as spam, otherwise ham. What's your prediction?

**Problem 6.2: Recommendation System**

You have 3 users and 5 movies. User vectors represent genre preferences:
(action, comedy, drama, horror, sci-fi)

- User A: (5, 2, 4, 1, 5)
- User B: (1, 5, 3, 4, 2)
- User C: (4, 1, 5, 0, 4)

Movie vectors represent genre composition (0-1 scale):
- Movie 1: (0.9, 0.1, 0.2, 0, 0.8)
- Movie 2: (0.2, 0.8, 0.5, 0.3, 0.1)
- Movie 3: (0.5, 0.2, 0.8, 0, 0.5)

a) Calculate predicted rating for User A on all movies (use inner product)
b) Which movie should be recommended to User A?
c) Calculate all 9 ratings (3 users × 3 movies)
d) Which user pair is most similar? (Calculate inner products between user vectors)
e) Based on similarity, if User B loved Movie 2, what should you recommend to User A?

**Problem 6.3: Clustering Setup**

You have 6 data points in 2D that need clustering (shown as (x, y)):
- P1: (2, 10)
- P2: (2, 5)
- P3: (8, 4)
- P4: (5, 8)
- P5: (7, 5)
- P6: (6, 4)

You want K=2 clusters and initialize centers at:
- C1 = P1 = (2, 10)
- C2 = P3 = (8, 4)

a) Calculate distance (using inner product formula: ||**u-v**||² = (**u-v**)·(**u-v**)) from each point to each center
b) Assign each point to nearest center
c) Calculate new centers (average of assigned points)
d) Did the centers move? Would you do another iteration?

**Problem 6.4: Feature Engineering**

Original features for house: (square_feet, bedrooms, age_years)
- House 1: (2000, 3, 15)

Create new features:
a) sqft_per_bedroom = square_feet / bedrooms (express as scalar multiplication)
b) age_penalty = -0.01 × age_years (scalar multiplication)
c) combined_score = 0.5(sqft_per_bedroom) + age_penalty (linear combination)
d) Calculate combined_score for House 1
e) Why might linear combinations of features help ML models?

**Problem 6.5: Gradient Descent Simulation**

Simple 1D optimization: minimize f(w) = (w - 5)²

Starting point: w₀ = 0
Learning rate: α = 0.3
Gradient at w: ∇f(w) = 2(w - 5)

Simulate 3 steps of gradient descent:
a) Calculate gradient at w₀ = 0
b) Update: w₁ = w₀ - α × gradient
c) Calculate gradient at w₁
d) Update: w₂ = w₁ - α × gradient
e) Calculate gradient at w₂
f) Update: w₃ = w₂ - α × gradient
g) Are you getting closer to the minimum at w = 5?
h) What happens if α = 1.5 (too large)?

## Section 2: Real-World Applications

**Problem 6.6: Image Processing**

A 4×4 grayscale image (values 0-255):
```
[100, 120, 130, 140]
[110, 125, 135, 145]
[115, 130, 140, 150]
[120, 135, 145, 155]
```

Flattened to vector: **img** = (100, 120, 130, 140, 110, 125, 135, 145, 115, 130, 140, 150, 120, 135, 145, 155)

a) Darken by 20%: calculate 0.8 × **img**
b) Increase contrast: (1.5 × **img**) - mean, where mean = (1/16) × sum(components)
c) Average with another image: **img2** = (110, 130, ...). Calculate 0.5(**img** + **img2**)
d) What's the dimension of this vector?
e) For a 1920×1080 color image (RGB), what's the dimension?

**Problem 6.7: Stock Portfolio**

You have portfolio weights: **w** = (0.4, 0.3, 0.2, 0.1) for 4 stocks
Returns in 3 scenarios: bull, normal, bear

- Stock 1 returns: (0.20, 0.10, -0.05)
- Stock 2 returns: (0.15, 0.08, -0.02)
- Stock 3 returns: (0.05, 0.05, 0.03)
- Stock 4 returns: (0.25, 0.05, -0.10)

a) Calculate weighted average return in bull market: 0.4(0.20) + 0.3(0.15) + ...
b) Calculate for normal and bear markets
c) Portfolio return vector: **r** = (bull_return, normal_return, bear_return)
d) If you think probabilities are (0.2, 0.6, 0.2) for (bull, normal, bear), what's expected return? (Use inner product!)

**Problem 6.8: Natural Language Processing**

Word vectors (simplified to 4D):
- **king** = (0.8, 0.3, 0.6, -0.2)
- **queen** = (0.7, 0.3, 0.5, 0.1)
- **man** = (0.5, 0.2, 0.1, -0.1)
- **woman** = (0.4, 0.2, 0, 0.2)

Famous relationship: **king** - **man** + **woman** ≈ **queen**

a) Calculate **king** - **man**
b) Calculate (**king** - **man**) + **woman**
c) Calculate **queen** for comparison
d) Compute inner product between your result and **queen**. Is it close?
e) Try: **queen** - **woman** + **man**. Should get something close to **king**!

**Problem 6.9: Sensor Fusion**

Three temperature sensors (noisy) measure over 5 time points:
- Sensor 1: (20.5, 21.0, 21.5, 22.0, 22.5)
- Sensor 2: (20.8, 21.2, 21.3, 22.1, 22.3)
- Sensor 3: (20.3, 20.9, 21.7, 21.9, 22.7)

a) Calculate average reading: (1/3)(**s₁** + **s₂** + **s₃**)
b) Calculate reading if you trust Sensor 2 most: 0.2**s₁** + 0.6**s₂** + 0.2**s₃**
c) Calculate the "spread" from Sensor 1: **s₁** - **s_avg**
d) Which time point has the most disagreement between sensors?

**Problem 6.10: A/B Testing**

You test two website versions on user engagement (time spent, clicks, purchases):

- Version A results (10 users average): (120 seconds, 5 clicks, 0.3 purchases)
- Version B results (10 users average): (150 seconds, 6 clicks, 0.4 purchases)

Importance weights: **w** = (0.3, 0.3, 0.4) for (time, clicks, purchases)

a) Calculate weighted score for Version A: **w** · **v_A**
b) Calculate for Version B
c) Which version is better?
d) If you change weights to (0.1, 0.2, 0.7) (purchases matter most), does the answer change?
e) What does this tell you about the importance of choosing the right metric?

## Section 3: Debugging and Understanding

**Problem 6.11: Finding the Error**

A student calculates:
**u** = (2, 3, 4)
**v** = (1, 0, 2)

Inner product: **u** · **v** = 2 + 3 + 4 + 1 + 0 + 2 = 12

a) What's wrong with this calculation?
b) What's the correct answer?
c) What's the correct formula being violated?

**Problem 6.12: Dimension Mismatch**

A student tries:
**u** = (1, 2, 3)
**v** = (4, 5)
**u** + **v** = (1+4, 2+5, 3+?) = (5, 7, 3)

a) What's wrong?
b) Why doesn't this make sense mathematically?
c) Why doesn't this make sense conceptually (think about what each component represents)?

**Problem 6.13: Understanding Zero Inner Product**

**u** = (3, 4)
**v** = (-4, 3)
**u** · **v** = 3(-4) + 4(3) = -12 + 12 = 0

a) What does it mean geometrically when inner product is zero?
b) Calculate ||**u**|| and ||**v**||
c) Calculate ||**u** + **v**||² and compare to ||**u**||² + ||**v**||² (Pythagorean theorem!)
d) Does this confirm they're perpendicular?

**Problem 6.14: Scalar Multiplication Properties**

**v** = (2, -3, 1)

a) Calculate 3**v**
b) Calculate -2**v**
c) Verify: 3(-2**v**) = (3 × -2)**v** = -6**v**
d) Calculate 0 × **v**. What do you get?
e) For what scalar α does α**v** have the same length as **v**?

**Problem 6.15: Linear Combination Challenge**

Can you make **target** = (7, 11) using a linear combination of:
**v₁** = (2, 3)
**v₂** = (1, 1)

That is, find α and β such that: α**v₁** + β**v₂** = (7, 11)

a) Set up the equations: 2α + 1β = 7 and 3α + 1β = 11
b) Solve for α and β
c) Verify your answer
d) Can you always express any 2D vector using **v₁** and **v₂**?

## Section 4: Complexity and Scaling

**Problem 6.16: Scaling Analysis**

Operation: Calculate average of K vectors, each dimension N

Current: K=100, N=1000, takes 0.1 seconds

a) If you double K (K=200), how long should it take?
b) If you double N (N=2000), how long?
c) If you double both K and N, how long?
d) To keep time under 0.5 seconds, what's the maximum N if K=500?

**Problem 6.17: Sparse Vector Efficiency**

Document vector: dimension 50,000 (vocabulary size), only 100 non-zero entries

a) Dense storage: how many numbers stored?
b) Sparse storage (store only non-zero indices and values): how many numbers?
c) For inner product of two sparse documents (100 non-zeros each), how many multiplications in dense form?
d) How many in sparse form (assuming clever implementation)?
e) Speedup ratio?

**Problem 6.18: Mini-Batch Trade-offs**

Dataset: N=100,000 examples, n=500 features

Full batch gradient: Process all 100,000 at once
Mini-batch gradient: Process m=100 at once

a) Operations per full batch update?
b) Operations per mini-batch update?
c) How many mini-batch updates to see all data once (one "epoch")?
d) Total operations for one epoch with mini-batching?
e) Why use mini-batches if total operations per epoch are the same?
f) What's a good mini-batch size and why?

---

# Answer Key (Selected Problems)

**Problem 6.1:**
a) spam_avg = ((20,5,3,1,1) + (18,4,5,1,1))/2 = (19,4.5,4,1,1)
b) ham_avg = ((2,0,10,0,0) + (3,1,8,0,0))/2 = (2.5,0.5,9,0,0)
c) **w** = (19,4.5,4,1,1) - (2.5,0.5,9,0,0) = (16.5,4,-5,1,1)
d) **w** · (15,3,4,1,0) = 16.5(15) + 4(3) + (-5)(4) + 1(1) + 1(0) = 247.5 + 12 - 20 + 1 = 240.5
e) 240.5 > 0, so classify as SPAM

**Problem 6.13:**
a) Perpendicular (orthogonal) - they form a 90° angle
b) ||**u**|| = √(9+16) = 5, ||**v**|| = √(16+9) = 5
c) **u**+**v** = (-1,7), ||**u**+**v**||² = 1+49 = 50; ||**u**||²+||**v**||² = 25+25 = 50 ✓
d) Yes! Pythagorean theorem confirms perpendicularity

**Problem 6.15:**
a) 2α + β = 7; 3α + β = 11
b) Subtract first from second: α = 4; then β = 7 - 8 = -1
c) 4(2,3) + (-1)(1,1) = (8,12) + (-1,-1) = (7,11) ✓
d) Yes! These two vectors "span" all of 2D space (we'll formalize this in Chapter 5)

---

# Congratulations!

You've completed Chapter 1! You now understand:
- ✅ What vectors are and why ML uses them
- ✅ How to add vectors and why it matters
- ✅ How to scale vectors and when to use it
- ✅ How to compute inner products and why they're fundamental
- ✅ How to analyze computational complexity
- ✅ How these operations power real ML algorithms

**You're ready for Chapter 2: Linear Functions, Taylor Approximation, and Regression Models!**
