# Chapter 1: Python Basics & Variables
## A First-Principles Approach to Programming

---

## 📑 INDEX - Chapter 1

### [Part 1: The Problem We're Solving - Why Programming?](#part-1-the-problem-were-solving---why-programming)
- [The Real-World Problem](#the-real-world-problem)
- [Why Python Specifically?](#why-python-specifically)

### [Part 2: Variables - The Problem Before The Solution](#part-2-variables---the-problem-before-the-solution)
- [Why Do We Need Variables?](#why-do-we-need-variables)
- [What IS a Variable?](#what-is-a-variable)
- [The Assignment Process - What Actually Happens?](#the-assignment-process---what-actually-happens)
- [Why This Matters - The Power of Variables](#why-this-matters---the-power-of-variables)

### [Part 3: Data Types - Different Kinds of Information](#part-3-data-types---different-kinds-of-information)
- [The Core Problem: Not All Information Is The Same](#the-core-problem-not-all-information-is-the-same)
- [Data Type #1: Integers (int)](#data-type-1-integers-int)
- [Data Type #2: Floating-Point Numbers (float)](#data-type-2-floating-point-numbers-float)
- [Data Type #3: Strings (str)](#data-type-3-strings-str)
- [Data Type #4: Booleans (bool)](#data-type-4-booleans-bool)

### [Part 4: Type Conversion - Changing Data Types](#part-4-type-conversion---changing-data-types)
- [The Problem: Types Don't Mix Automatically](#the-problem-types-dont-mix-automatically)
- [Solution: Explicit Type Conversion](#solution-explicit-type-conversion)
- [Checking Types with type()](#checking-types-with-type)

### [Part 5: Input and Output - Interacting with Users](#part-5-input-and-output---interacting-with-users)
- [The Problem: Programs Need to Communicate](#the-problem-programs-need-to-communicate)
- [Output with print()](#output-with-print)
- [Input with input()](#input-with-input)

### [Part 6: Common Mistakes and How to Avoid Them](#part-6-common-mistakes-and-how-to-avoid-them)
- [Mistake #1: Confusing = with ==](#mistake-1-confusing--with-)
- [Mistake #2: Using quotes incorrectly with numbers](#mistake-2-using-quotes-incorrectly-with-numbers)
- [Mistake #3: Forgetting that input() returns strings](#mistake-3-forgetting-that-input-returns-strings)
- [Mistake #4: Variable naming errors](#mistake-4-variable-naming-errors)
- [Mistake #5: Case sensitivity](#mistake-5-case-sensitivity)
- [Mistake #6: Not understanding operator precedence](#mistake-6-not-understanding-operator-precedence)

### [Part 7: Detailed Worked Examples](#part-7-detailed-worked-examples)
- [Example 1: Restaurant Bill Calculator](#example-1-restaurant-bill-calculator)
- [Example 2: Temperature Converter](#example-2-temperature-converter)
- [Example 3: Time Calculator](#example-3-time-calculator)
- [Example 4: Username Generator](#example-4-username-generator)
- [Example 5: BMI Calculator](#example-5-bmi-calculator)
- [Example 6: String Information Display](#example-6-string-information-display)
- [Example 7: Compound Interest Calculator](#example-7-compound-interest-calculator)

### [Part 8: Practice Problems](#part-8-practice-problems)
- [Easy Problems (1-7)](#easy-problems-problems-1-7)
- [Medium Problems (8-15)](#medium-problems-problems-8-15)
- [Hard Problems (16-25)](#hard-problems-problems-16-25)
- [Challenge Problems (26-30)](#challenge-problems-problems-26-30)

### [Part 9: Mini-Project - Personal Budget Analyzer](#part-9-mini-project---personal-budget-analyzer)

### [Part 10: Key Takeaways](#part-10-key-takeaways)
- [What You've Learned](#what-youve-learned)
- [Best Practices Introduced](#best-practices-introduced)
- [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)
- [Next Steps](#next-steps)

### [Appendix: Operator Reference](#appendix-operator-reference)
- [Arithmetic Operators](#arithmetic-operators)
- [Comparison Operators](#comparison-operators)
- [Logical Operators](#logical-operators)
- [String Operators](#string-operators)

---

## Part 1: The Problem We're Solving - Why Programming?

### The Real-World Problem

Imagine you work at a coffee shop. Every day, you need to:
- Calculate the total sales
- Track inventory
- Apply discounts
- Generate reports
- Send receipts to customers

You *could* do all of this with pen, paper, and a calculator. But you'd make mistakes, it would take hours, and you'd be bored out of your mind doing the same calculations repeatedly.

**The fundamental problem**: Humans are terrible at repetitive tasks. We get tired, make mistakes, and hate doing the same thing over and over.

**What we need**: A way to tell a machine exactly what to do, once, and have it execute those instructions perfectly every time, at incredible speed.

This is what programming is: **giving instructions to a computer in a language it can understand**.

### Why Python Specifically?

Let's think about human languages. Some are more formal (legal documents), some are more conversational (casual speech). Programming languages are the same.

**The problem with older languages**: Languages like C or Java are like writing legal documents—very precise, but verbose and intimidating for beginners.

**Python's solution**: Python reads almost like English. Compare these:

```
Java (older language):
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}

Python:
print("Hello, World!")
```

Python solves the problem of making programming **accessible** without sacrificing power. It's used by:
- **Scientists** analyzing climate data
- **Financial analysts** building trading algorithms
- **Web developers** building Instagram and YouTube
- **Data scientists** training AI models
- **Automation engineers** eliminating repetitive tasks

---

## Part 2: Variables - The Problem Before The Solution

### Why Do We Need Variables?

Let's start with a question: **How does your brain remember things?**

You're at a party. Someone says, "Hi, I'm Sarah." Your brain stores this:
- The name "Sarah"
- Connected to this person's face
- Connected to this moment

Later, you see her again and think, "That's Sarah!"

**The problem computers face**: Computers work with billions of pieces of information. Without a way to *label* and *store* information, they'd be useless.

Think about it: If I tell you "Calculate 15% tip on $47.50," you naturally think:
- "bill_amount is $47.50"
- "tip_percentage is 15%"
- "tip_amount is bill_amount × tip_percentage"

You just created **variables** in your mind!

### What IS a Variable?

**A variable is a labeled box that holds a value.**

Imagine your computer's memory as a massive warehouse with millions of boxes. Variables let you:
1. **Put a value in a box** (assignment)
2. **Label the box** (naming)
3. **Find the box later** (retrieval)
4. **Change what's in the box** (reassignment)

```python
# This is your first Python code!
# The '#' symbol means this is a comment - Python ignores these lines
# Comments are notes to humans reading the code

# Creating a variable: putting a value (42) in a box (age)
age = 42

# The '=' is the assignment operator
# It means: "Store the value on the right into the variable on the left"
# Read it as: "age GETS the value 42" or "age is assigned 42"
```

### The Assignment Process - What Actually Happens?

When you write `age = 42`, here's what happens step by step:

1. **Python finds space in memory** (allocates a box)
2. **Stores the value 42** in that space
3. **Creates a label "age"** pointing to that space
4. **Later, when you use "age"**, Python looks up that label and retrieves 42

```python
# Let's see this in action
coffee_price = 4.50  # Store 4.50 in memory, label it coffee_price
quantity = 3          # Store 3 in memory, label it quantity

# Now we can USE these variables
total = coffee_price * quantity  # Python retrieves 4.50 and 3, multiplies them

print(total)  # Output: 13.5
# 'print' is a function that displays values to the screen
```

### Why This Matters - The Power of Variables

Without variables:
```python
# Calculating a 15% tip without variables (BAD!)
print(47.50 * 0.15)  # What does this mean? What are these numbers?
```

With variables:
```python
# The same calculation with variables (GOOD!)
bill_amount = 47.50
tip_percentage = 0.15
tip_amount = bill_amount * tip_percentage

print(tip_amount)  # Output: 7.125

# Now the code tells a STORY. Anyone can understand it!
```

**The key insight**: Variables make code **readable**, **reusable**, and **maintainable**.

---

## Part 3: Data Types - Different Kinds of Information

### The Core Problem: Not All Information Is The Same

Quick question: What's wrong with this?
- "Add 5 to 'hello'"
- "Divide 'pizza' by 3"

You immediately know these don't make sense! You can't do math with words.

**The problem**: Computers need to know what KIND of data they're working with to know what OPERATIONS are valid.

Think about the real world:
- **Numbers**: You can add, subtract, multiply, divide
- **Text**: You can combine, search, capitalize
- **True/False**: You can make decisions (if it's true, do this)

Python has **data types** to represent these different kinds of information.

### Data Type #1: Integers (int)

**What problem do integers solve?** We need to count discrete, whole things.

```python
# Integers are whole numbers (no decimal point)
students_in_class = 30
pages_read = 157
temperature = -5  # Can be negative!
bank_account_balance = 0  # Can be zero

# Why integers? Some things CAN'T be fractional:
# - You can't have 2.5 students
# - You can't read 3.7 pages (you read 3 or 4)
# - You can't watch 1.3 episodes of a show

# Operations with integers
x = 10
y = 3

addition = x + y        # 13
subtraction = x - y     # 7
multiplication = x * y  # 30
division = x / y        # 3.3333... (becomes a float!)
integer_division = x // y  # 3 (keeps only whole part)
remainder = x % y       # 1 (remainder when dividing 10 by 3)
power = x ** y          # 1000 (10 to the power of 3)

print(f"10 + 3 = {addition}")
print(f"10 // 3 = {integer_division}")
print(f"10 % 3 = {remainder}")
```

**Real-world example**:
```python
# Splitting a bill among friends
total_bill = 87
number_of_people = 4

# How much does each person pay?
per_person = total_bill / number_of_people  # 21.75

# But we can only pay in whole cents! (in reality)
# We might want whole dollars:
per_person_rounded = total_bill // number_of_people  # 21

# What's left over?
leftover = total_bill % number_of_people  # 3

print(f"Each person pays: ${per_person_rounded}")
print(f"Leftover: ${leftover}")
# Output: Each person pays: $21
#         Leftover: $3
```

### Data Type #2: Floating-Point Numbers (float)

**What problem do floats solve?** We need to represent continuous measurements and fractions.

```python
# Floats have decimal points
height_in_meters = 1.75
pi = 3.14159
temperature = 98.6
stock_price = 142.37
tax_rate = 0.07  # 7% represented as decimal

# Why floats? Some things ARE continuous:
# - Height varies continuously (1.75m, 1.76m, 1.751m...)
# - Money has cents
# - Scientific measurements
# - Percentages and ratios

# Operations with floats (same as integers)
price = 19.99
quantity = 3.5

total = price * quantity  # 69.965

# IMPORTANT: Floats can have precision issues!
a = 0.1
b = 0.2
c = a + b

print(c)  # Output: 0.30000000000000004 (weird, right?)
# This happens because computers store decimals in binary
# For money, use the 'decimal' module (we'll learn later)
```

**Real-world example**:
```python
# Calculate compound interest
principal = 1000.0      # Starting amount
rate = 0.05             # 5% annual interest
years = 3

# Amount after 3 years: principal × (1 + rate)^years
final_amount = principal * ((1 + rate) ** years)

print(f"Starting: ${principal}")
print(f"After {years} years: ${final_amount:.2f}")
# The :.2f formats to 2 decimal places
# Output: Starting: $1000.0
#         After 3 years: $1157.63
```

### Data Type #3: Strings (str)

**What problem do strings solve?** We need to work with text!

```python
# Strings are sequences of characters
# You can use single quotes or double quotes (be consistent!)
name = "Alice"
message = 'Hello, World!'
address = "123 Main St."
email = 'user@example.com'

# When do you need the other quote style?
# When your text contains quotes!
quote = "She said, 'Python is amazing!'"
quote2 = 'He replied, "I agree!"'

# Or use triple quotes for multi-line strings
paragraph = """
This is a longer piece of text.
It can span multiple lines.
Very useful for documentation!
"""

# Why strings? Text is everywhere:
# - Names, addresses, emails
# - Messages, posts, tweets
# - File contents
# - User input
```

**String Operations**:
```python
first_name = "John"
last_name = "Doe"

# Concatenation (combining strings) with +
full_name = first_name + " " + last_name  # "John Doe"

# Repetition with *
laugh = "ha" * 3  # "hahaha"
separator = "-" * 20  # "--------------------"

# Length with len()
name_length = len(full_name)  # 8 (counts characters including space)

# Accessing individual characters with indexing [position]
first_letter = full_name[0]  # "J" (positions start at 0!)
last_letter = full_name[-1]  # "e" (negative counts from end)

print(full_name)      # John Doe
print(first_letter)   # J
print(name_length)    # 8
```

**Real-world example**:
```python
# Building a personalized email
recipient_name = "Alice"
product = "Python Course"
discount = 20

# Old way (concatenation)
message = "Dear " + recipient_name + ", get " + str(discount) + "% off " + product + "!"

# Better way (f-strings - formatted string literals)
message = f"Dear {recipient_name}, get {discount}% off {product}!"

print(message)
# Output: Dear Alice, get 20% off Python Course!

# f-strings are AMAZING - you can put ANY expression inside {}
x = 10
y = 5
print(f"The sum of {x} and {y} is {x + y}")
# Output: The sum of 10 and 5 is 15
```

### Data Type #4: Booleans (bool)

**What problem do booleans solve?** We need to make decisions!

```python
# Booleans have only TWO values: True or False
# (Note the capital T and F!)
is_raining = True
is_sunny = False
has_umbrella = True
is_weekend = False

# Why booleans? Programming is full of yes/no decisions:
# - Is the user logged in?
# - Is the file empty?
# - Is the number positive?
# - Did the operation succeed?

# Comparison operators produce booleans
age = 25
is_adult = age >= 18  # True (25 is greater than or equal to 18)
is_teenager = 13 <= age <= 19  # False (25 is not between 13 and 19)

temperature = 72
is_hot = temperature > 80  # False

# Logical operators combine booleans
sunny = True
warm = True
perfect_day = sunny and warm  # True (both are True)

raining = True
has_umbrella = False
will_get_wet = raining and not has_umbrella  # True

print(f"Is it a perfect day? {perfect_day}")
print(f"Will I get wet? {will_get_wet}")
```

**Real-world example**:
```python
# E-commerce free shipping logic
order_total = 45.00
is_prime_member = True

# Free shipping if: order over $50 OR prime member
qualifies_for_free_shipping = (order_total >= 50) or is_prime_member

print(f"Order total: ${order_total}")
print(f"Prime member: {is_prime_member}")
print(f"Free shipping: {qualifies_for_free_shipping}")
# Output: Order total: $45.0
#         Prime member: True
#         Free shipping: True

# Another example: password validation
password = "Abc123"
has_minimum_length = len(password) >= 8
has_number = True  # (we'd check this properly later)

is_valid_password = has_minimum_length and has_number
print(f"Password valid: {is_valid_password}")  # False (too short)
```

---

## Part 4: Type Conversion - Changing Data Types

### The Problem: Types Don't Mix Automatically

What happens if you try to "add" different types?

```python
# This makes sense:
number = 5
another_number = 3
result = number + another_number  # 8

# This makes sense:
word = "Hello"
another_word = "World"
result = word + " " + another_word  # "Hello World"

# But this does NOT work:
age = 25
message = "I am " + age + " years old"  # ERROR!
# TypeError: can only concatenate str (not "int") to str

# Why? Python doesn't know if you want:
# - "I am 25 years old" (convert 25 to string)
# - Some weird math (convert "I am " to number?)
```

### Solution: Explicit Type Conversion

```python
# Converting to string with str()
age = 25
message = "I am " + str(age) + " years old"  # Works!
print(message)  # Output: I am 25 years old

# Or use f-strings (easier!)
message = f"I am {age} years old"  # Python converts automatically in f-strings
print(message)  # Output: I am 25 years old

# Converting to integer with int()
user_input = "42"  # This is a string (maybe from user input)
number = int(user_input)  # Converts "42" to 42
doubled = number * 2  # 84

print(f"String: {user_input}, Number: {number}, Doubled: {doubled}")
# Output: String: 42, Number: 42, Doubled: 84

# Converting to float with float()
price_string = "19.99"
price = float(price_string)  # Converts "19.99" to 19.99
tax = price * 0.07  # Can now do math!

print(f"Price: ${price}, Tax: ${tax:.2f}")
# Output: Price: $19.99, Tax: $1.40

# Converting to boolean with bool()
# Rules: 0, empty string, None → False; everything else → True
print(bool(0))      # False
print(bool(42))     # True
print(bool(""))     # False (empty string)
print(bool("Hi"))   # True
print(bool([]))     # False (empty list - we'll learn this later)
```

**Real-world example**:
```python
# Processing user input from a form
age_input = "25"        # User typed "25" (it's a string!)
salary_input = "50000"
employed_input = "yes"

# Convert to appropriate types
age = int(age_input)
salary = float(salary_input)
is_employed = employed_input.lower() == "yes"  # Convert to boolean

# Now we can do calculations
years_to_retirement = 65 - age
annual_savings = salary * 0.10

print(f"Age: {age}")
print(f"Years to retirement: {years_to_retirement}")
print(f"Annual savings: ${annual_savings}")
print(f"Currently employed: {is_employed}")
```

### Checking Types with type()

```python
# You can check what type something is
x = 42
y = 3.14
z = "Hello"
w = True

print(type(x))  # <class 'int'>
print(type(y))  # <class 'float'>
print(type(z))  # <class 'str'>
print(type(w))  # <class 'bool'>

# This is useful for debugging!
mystery_value = "123"
print(f"Value: {mystery_value}, Type: {type(mystery_value)}")
# Output: Value: 123, Type: <class 'str'>
# Aha! It's a string, not a number!
```

---

## Part 5: Input and Output - Interacting with Users

### The Problem: Programs Need to Communicate

So far, we've been hard-coding values:
```python
name = "Alice"  # We wrote this directly in code
```

But real programs need to:
- **Get information FROM users** (input)
- **Show information TO users** (output)

### Output with print()

```python
# Basic printing
print("Hello, World!")  # Output: Hello, World!

# Printing variables
name = "Alice"
age = 30
print(name)  # Output: Alice
print(age)   # Output: 30

# Printing multiple things (separated by spaces)
print("Name:", name, "Age:", age)
# Output: Name: Alice Age: 30

# Using f-strings (best practice!)
print(f"Name: {name}, Age: {age}")
# Output: Name: Alice, Age: 30

# Controlling separators
print("A", "B", "C")              # Output: A B C
print("A", "B", "C", sep="-")     # Output: A-B-C
print("A", "B", "C", sep="")      # Output: ABC

# Controlling line endings
print("First line")
print("Second line")
# Output:
# First line
# Second line

print("Same line", end=" ")
print("continued!")
# Output: Same line continued!
```

### Input with input()

```python
# input() always returns a STRING!
name = input("What is your name? ")
# Program pauses here, waiting for user to type and press Enter

print(f"Hello, {name}!")

# If you need a number, convert it!
age_string = input("How old are you? ")
age = int(age_string)  # Convert string to integer

# Or do it in one line:
age = int(input("How old are you? "))

# Be careful with float conversions:
height = float(input("Height in meters: "))
```

**Real-world example**: Interactive calculator
```python
# Simple calculator
print("=== Simple Calculator ===")

# Get input from user
first_number = float(input("Enter first number: "))
second_number = float(input("Enter second number: "))

# Perform calculations
sum_result = first_number + second_number
difference = first_number - second_number
product = first_number * second_number
quotient = first_number / second_number if second_number != 0 else "undefined"

# Display results
print(f"\nResults:")
print(f"{first_number} + {second_number} = {sum_result}")
print(f"{first_number} - {second_number} = {difference}")
print(f"{first_number} × {second_number} = {product}")
print(f"{first_number} ÷ {second_number} = {quotient}")

# Example run:
# Enter first number: 10
# Enter second number: 3
#
# Results:
# 10.0 + 3.0 = 13.0
# 10.0 - 3.0 = 7.0
# 10.0 × 3.0 = 30.0
# 10.0 ÷ 3.0 = 3.3333333333333335
```

---

## Part 6: Common Mistakes and How to Avoid Them

### Mistake #1: Confusing = with ==

```python
# WRONG: Using = when you mean ==
age = 25
if age = 25:  # ERROR! SyntaxError: invalid syntax
    print("You are 25")

# CORRECT: = is assignment, == is comparison
age = 25  # Assignment: age GETS 25
if age == 25:  # Comparison: is age EQUAL TO 25?
    print("You are 25")

# Remember:
# = means "store this value"
# == means "are these equal?"
```

### Mistake #2: Using quotes incorrectly with numbers

```python
# WRONG: Making a number into a string by accident
age = "25"  # This is a string, not a number!
next_year = age + 1  # ERROR! TypeError: can only concatenate str (not "int") to str

# CORRECT: No quotes for numbers
age = 25  # This is a number
next_year = age + 1  # 26 - works fine!

# OR: If you have a string, convert it
age = "25"
next_year = int(age) + 1  # 26
```

### Mistake #3: Forgetting that input() returns strings

```python
# WRONG: Trying to do math with input directly
age = input("Enter age: ")  # User types "25"
next_year = age + 1  # ERROR! Can't add string and number

# CORRECT: Convert to number first
age = int(input("Enter age: "))
next_year = age + 1  # Works!
```

### Mistake #4: Variable naming errors

```python
# WRONG: Using spaces in variable names
my age = 25  # ERROR! SyntaxError

# WRONG: Starting with numbers
2nd_place = "silver"  # ERROR! SyntaxError

# WRONG: Using Python keywords
class = "Math"  # ERROR! 'class' is a reserved word
for = 10  # ERROR! 'for' is a reserved word

# CORRECT: Use underscores, start with letter or underscore
my_age = 25
second_place = "silver"
class_name = "Math"
for_count = 10

# GOOD PRACTICE: descriptive names
# Bad
x = 25
y = "John"

# Good
age = 25
name = "John"
```

### Mistake #5: Case sensitivity

```python
# Python is case-sensitive!
name = "Alice"
Name = "Bob"  # This is a DIFFERENT variable!
NAME = "Charlie"  # Also DIFFERENT!

print(name)  # Alice
print(Name)  # Bob
print(NAME)  # Charlie

# This applies to everything:
print("Hello")  # Works
Print("Hello")  # ERROR! NameError: name 'Print' is not defined
```

### Mistake #6: Not understanding operator precedence

```python
# WRONG: Expecting left-to-right evaluation
result = 5 + 3 * 2  # What do you get?
print(result)  # 11, not 16!
# Why? Multiplication happens BEFORE addition

# CORRECT: Use parentheses to be explicit
result = (5 + 3) * 2  # 16
print(result)

# Order of operations (PEMDAS):
# 1. Parentheses ()
# 2. Exponents **
# 3. Multiplication/Division/Modulo *, /, //, %
# 4. Addition/Subtraction +, -

x = 2 + 3 * 4 ** 2  # What is this?
# Step 1: 4 ** 2 = 16
# Step 2: 3 * 16 = 48
# Step 3: 2 + 48 = 50

# When in doubt, use parentheses!
x = 2 + (3 * (4 ** 2))  # Much clearer!
```

---

## Part 7: Detailed Worked Examples

### Example 1: Restaurant Bill Calculator

**Problem**: Calculate the total cost of a meal including tip and tax.

```python
# Step 1: Define our constants (values that don't change)
TAX_RATE = 0.08  # 8% sales tax
TIP_RATE = 0.18  # 18% tip

# Step 2: Get the meal cost
meal_cost = float(input("Enter meal cost: $"))

# Step 3: Calculate tax (on meal only)
tax_amount = meal_cost * TAX_RATE

# Step 4: Calculate tip (on meal + tax)
subtotal = meal_cost + tax_amount
tip_amount = subtotal * TIP_RATE

# Step 5: Calculate total
total = meal_cost + tax_amount + tip_amount

# Step 6: Display breakdown
print("\n=== Bill Breakdown ===")
print(f"Meal cost:    ${meal_cost:.2f}")
print(f"Tax (8%):     ${tax_amount:.2f}")
print(f"Tip (18%):    ${tip_amount:.2f}")
print(f"{'─' * 30}")
print(f"TOTAL:        ${total:.2f}")

# Example run:
# Enter meal cost: $45.00
#
# === Bill Breakdown ===
# Meal cost:    $45.00
# Tax (8%):     $3.60
# Tip (18%):    $8.75
# ──────────────────────────────
# TOTAL:        $57.35
```

**What we learned**:
- Using descriptive variable names (TAX_RATE, not tr)
- Breaking calculations into steps
- Formatting currency with :.2f
- Building clear output

### Example 2: Temperature Converter

**Problem**: Convert between Fahrenheit and Celsius.

```python
# Formula: C = (F - 32) × 5/9
# Formula: F = C × 9/5 + 32

print("=== Temperature Converter ===")
print("1. Fahrenheit to Celsius")
print("2. Celsius to Fahrenheit")

choice = input("Enter choice (1 or 2): ")

if choice == "1":
    # Fahrenheit to Celsius
    fahrenheit = float(input("Enter temperature in °F: "))
    celsius = (fahrenheit - 32) * 5 / 9
    print(f"{fahrenheit}°F = {celsius:.1f}°C")
    
elif choice == "2":
    # Celsius to Fahrenheit
    celsius = float(input("Enter temperature in °C: "))
    fahrenheit = celsius * 9 / 5 + 32
    print(f"{celsius}°C = {fahrenheit:.1f}°F")
    
else:
    print("Invalid choice!")

# Example run:
# === Temperature Converter ===
# 1. Fahrenheit to Celsius
# 2. Celsius to Fahrenheit
# Enter choice (1 or 2): 1
# Enter temperature in °F: 98.6
# 98.6°F = 37.0°C
```

**What we learned**:
- Making interactive programs
- Using mathematical formulas
- Basic conditional logic (we'll cover this deeply later)
- Handling different user paths

### Example 3: Time Calculator

**Problem**: Convert total seconds into hours, minutes, and seconds.

```python
# Get total seconds from user
total_seconds = int(input("Enter total seconds: "))

# Calculate hours (3600 seconds in an hour)
hours = total_seconds // 3600  # Integer division

# Calculate remaining seconds after removing hours
remaining_seconds = total_seconds % 3600  # Modulo gets remainder

# Calculate minutes (60 seconds in a minute)
minutes = remaining_seconds // 60

# Calculate remaining seconds
seconds = remaining_seconds % 60

# Display result
print(f"{total_seconds} seconds = {hours}h {minutes}m {seconds}s")

# Example runs:
# Enter total seconds: 7384
# 7384 seconds = 2h 3m 4s
#
# Enter total seconds: 3661
# 3661 seconds = 1h 1m 1s
```

**What we learned**:
- Using // (integer division) and % (modulo)
- Breaking down a problem mathematically
- Working with remainders

### Example 4: Username Generator

**Problem**: Create a username from first name, last name, and birth year.

```python
# Get user information
first_name = input("First name: ")
last_name = input("Last name: ")
birth_year = input("Birth year: ")

# Extract parts
# first_name[:3] takes first 3 characters
# last_name[:3] takes first 3 characters  
# birth_year[-2:] takes last 2 characters
first_part = first_name[:3].lower()
last_part = last_name[:3].lower()
year_part = birth_year[-2:]

# Combine parts
username = first_part + last_part + year_part

print(f"Your username is: {username}")

# Example run:
# First name: Jennifer
# Last name: Anderson
# Birth year: 1995
# Your username is: jenand95
```

**What we learned**:
- String slicing [:3], [-2:]
- String methods .lower()
- Combining string operations

### Example 5: BMI Calculator

**Problem**: Calculate Body Mass Index from height and weight.

```python
# BMI Formula: weight (kg) / height (m)²

print("=== BMI Calculator ===")

# Get input
weight_kg = float(input("Weight in kg: "))
height_m = float(input("Height in meters: "))

# Calculate BMI
bmi = weight_kg / (height_m ** 2)

# Determine category
if bmi < 18.5:
    category = "Underweight"
elif bmi < 25:
    category = "Normal weight"
elif bmi < 30:
    category = "Overweight"
else:
    category = "Obese"

# Display results
print(f"\nYour BMI: {bmi:.1f}")
print(f"Category: {category}")

# Example run:
# === BMI Calculator ===
# Weight in kg: 70
# Height in meters: 1.75
#
# Your BMI: 22.9
# Category: Normal weight
```

**What we learned**:
- Using exponentiation (**)
- Multiple conditions with elif
- Categorizing numerical results

### Example 6: String Information Display

**Problem**: Show detailed information about a user's input.

```python
# Get input
text = input("Enter some text: ")

# Analyze the text
length = len(text)
uppercase_version = text.upper()
lowercase_version = text.lower()
first_char = text[0] if len(text) > 0 else ""
last_char = text[-1] if len(text) > 0 else ""

# Count specific characters
spaces = text.count(" ")
letter_a = text.lower().count("a")

# Display information
print(f"\n=== Text Analysis ===")
print(f"Original: '{text}'")
print(f"Length: {length} characters")
print(f"Uppercase: '{uppercase_version}'")
print(f"Lowercase: '{lowercase_version}'")
print(f"First character: '{first_char}'")
print(f"Last character: '{last_char}'")
print(f"Number of spaces: {spaces}")
print(f"Number of 'a's: {letter_a}")

# Example run:
# Enter some text: Hello World
#
# === Text Analysis ===
# Original: 'Hello World'
# Length: 11 characters
# Uppercase: 'HELLO WORLD'
# Lowercase: 'hello world'
# First character: 'H'
# Last character: 'd'
# Number of spaces: 1
# Number of 'a's: 0
```

**What we learned**:
- String methods: .upper(), .lower(), .count()
- String indexing: [0], [-1]
- Conditional expressions (if/else in one line)

### Example 7: Compound Interest Calculator

**Problem**: Calculate how much money grows with compound interest.

```python
# Get initial values
principal = float(input("Initial investment: $"))
annual_rate = float(input("Annual interest rate (e.g., 5 for 5%): "))
years = int(input("Number of years: "))

# Convert percentage to decimal
rate_decimal = annual_rate / 100

# Calculate compound interest
# Formula: A = P(1 + r)^t
final_amount = principal * ((1 + rate_decimal) ** years)

# Calculate total interest earned
interest_earned = final_amount - principal

# Calculate percentage growth
percent_growth = (interest_earned / principal) * 100

# Display results
print(f"\n=== Investment Growth ===")
print(f"Initial investment: ${principal:.2f}")
print(f"Interest rate: {annual_rate}% per year")
print(f"Time period: {years} years")
print(f"")
print(f"Final amount: ${final_amount:.2f}")
print(f"Interest earned: ${interest_earned:.2f}")
print(f"Total growth: {percent_growth:.1f}%")

# Example run:
# Initial investment: $1000
# Annual interest rate (e.g., 5 for 5%): 7
# Number of years: 10
#
# === Investment Growth ===
# Initial investment: $1000.00
# Interest rate: 7.0% per year
# Time period: 10 years
#
# Final amount: $1967.15
# Interest earned: $967.15
# Total growth: 96.7%
```

**What we learned**:
- Multi-step calculations
- Working with formulas
- Percentage conversions
- Professional output formatting

---

## Part 8: Practice Problems

### Easy Problems (Problems 1-7)

**Problem 1**: Create variables for your name, age, and favorite color, then print them in a sentence.

**Problem 2**: Calculate and print the area of a rectangle with width 15 and height 7.

**Problem 3**: Create a variable with your birth year, calculate your age in 2026, and print it.

**Problem 4**: Store two numbers in variables, then print their sum, difference, product, and quotient.

**Problem 5**: Create a string with your favorite quote, then print its length and uppercase version.

**Problem 6**: Calculate how many seconds are in 3 days (3 days × 24 hours × 60 minutes × 60 seconds).

**Problem 7**: Create variables for item price ($29.99) and quantity (4), calculate total cost, and print it.

### Medium Problems (Problems 8-15)

**Problem 8**: Ask the user for their name and age, then print "Hello [name], you are [age] years old!"

**Problem 9**: Write a program that asks for a number and prints whether it's positive, negative, or zero. (Hint: use comparison operators and print the result)

**Problem 10**: Calculate the average of three numbers entered by the user.

**Problem 11**: Convert a distance in miles to kilometers (1 mile = 1.60934 km). Get the miles from user input.

**Problem 12**: Ask for a product price and discount percentage, then calculate and show the discounted price.

**Problem 13**: Create a simple grade calculator: ask for points earned and total points, calculate and display the percentage.

**Problem 14**: Ask for a word and print it three times, each time on a new line.

**Problem 15**: Calculate how many weeks, days, and hours old you are. Ask the user for their age in years.

### Hard Problems (Problems 16-25)

**Problem 16**: Create a program that asks for hours and minutes, then calculates and displays the total minutes.

**Problem 17**: Write a program that asks for a number and displays its square, cube, and square root (hint: square root is ** 0.5).

**Problem 18**: Calculate paint needed for a room: ask for room dimensions (length, width, height), calculate wall area (ignore windows/doors), and determine gallons needed (1 gallon covers 350 sq ft).

**Problem 19**: Create a "name badge" program that asks for first name, last name, and company, then prints:
```
╔══════════════════╗
║ HELLO, I'M       ║
║ [FULL NAME]      ║
║ [Company]        ║
╚══════════════════╝
```

**Problem 20**: Calculate a "life in weeks" visualization: ask for current age and life expectancy, calculate weeks lived and weeks remaining, display as percentages.

**Problem 21**: Currency converter: Ask for amount in USD and exchange rate, convert to foreign currency, show both amounts with proper formatting.

**Problem 22**: Pizza party calculator: Ask for number of people, slices per person, and slices per pizza. Calculate how many pizzas to order (round up!).

**Problem 23**: Phone number formatter: Ask user to enter 10 digits, format as (XXX) XXX-XXXX.

**Problem 24**: Simple loan calculator: Ask for loan amount, annual interest rate, and loan term in years. Calculate monthly payment using: M = P × (r(1+r)^n) / ((1+r)^n - 1), where r is monthly rate and n is total months.

**Problem 25**: Create a "receipt generator": Ask for 3 items with names and prices, calculate subtotal, tax (8%), and total, display as a formatted receipt.

### Challenge Problems (Problems 26-30)

**Problem 26**: Build a "time until event" calculator: Ask for current date (year, month, day) and event date, calculate days between them. (Hint: you'll need to estimate, or look up how many days in each month)

**Problem 27**: Create a "password strength checker" that asks for a password and displays:
- Length
- Whether it contains numbers (you can manually check for a specific number)
- Whether it contains uppercase letters
- Whether it contains lowercase letters

**Problem 28**: Gas mileage tracker: Ask for starting odometer reading, ending reading, and gallons purchased. Calculate miles driven and miles per gallon.

**Problem 29**: Binary converter (limited): Ask for a decimal number (0-15 only), show its binary representation using mathematical operations. (Hint: use //, %, and division by powers of 2)

**Problem 30**: Create a "savings goal tracker": Ask for current savings, monthly contribution, annual interest rate, and goal amount. Calculate how many months to reach the goal (approximate, assume monthly compounding).

---

## Part 9: Mini-Project - Personal Budget Analyzer

**Objective**: Create a comprehensive budget analysis tool that uses all the concepts from this chapter.

**Requirements**:
1. Ask user for their monthly income
2. Ask for expenses in different categories (rent, food, transportation, entertainment)
3. Calculate total expenses
4. Calculate remaining money
5. Calculate percentage of income spent in each category
6. Provide a summary with recommendations

**Step-by-Step Build**:

```python
print("=" * 50)
print("      PERSONAL BUDGET ANALYZER")
print("=" * 50)

# Step 1: Get income
monthly_income = float(input("\nEnter your monthly income: $"))

# Step 2: Get expenses by category
print("\nEnter your monthly expenses:")
rent = float(input("  Rent/Mortgage: $"))
food = float(input("  Food/Groceries: $"))
transportation = float(input("  Transportation: $"))
entertainment = float(input("  Entertainment: $"))
utilities = float(input("  Utilities: $"))
other = float(input("  Other: $"))

# Step 3: Calculate totals
total_expenses = rent + food + transportation + entertainment + utilities + other
remaining = monthly_income - total_expenses

# Step 4: Calculate percentages
rent_percent = (rent / monthly_income) * 100
food_percent = (food / monthly_income) * 100
trans_percent = (transportation / monthly_income) * 100
entertain_percent = (entertainment / monthly_income) * 100
utilities_percent = (utilities / monthly_income) * 100
other_percent = (other / monthly_income) * 100
total_percent = (total_expenses / monthly_income) * 100

# Step 5: Display detailed report
print("\n" + "=" * 50)
print("            BUDGET SUMMARY")
print("=" * 50)
print(f"\nMonthly Income:          ${monthly_income:>10.2f}")
print(f"\nEXPENSE BREAKDOWN:")
print(f"  Rent/Mortgage:         ${rent:>10.2f}  ({rent_percent:>5.1f}%)")
print(f"  Food/Groceries:        ${food:>10.2f}  ({food_percent:>5.1f}%)")
print(f"  Transportation:        ${transportation:>10.2f}  ({trans_percent:>5.1f}%)")
print(f"  Entertainment:         ${entertainment:>10.2f}  ({entertain_percent:>5.1f}%)")
print(f"  Utilities:             ${utilities:>10.2f}  ({utilities_percent:>5.1f}%)")
print(f"  Other:                 ${other:>10.2f}  ({other_percent:>5.1f}%)")
print(f"  {'-' * 48}")
print(f"  TOTAL EXPENSES:        ${total_expenses:>10.2f}  ({total_percent:>5.1f}%)")
print(f"\n  REMAINING:             ${remaining:>10.2f}")

# Step 6: Provide analysis
print(f"\n" + "=" * 50)
print("              ANALYSIS")
print("=" * 50)

if remaining > 0:
    print(f"✓ You have ${remaining:.2f} left over!")
    savings_rate = (remaining / monthly_income) * 100
    print(f"  Savings rate: {savings_rate:.1f}%")
    
    if savings_rate >= 20:
        print("  Excellent! You're saving 20% or more.")
    elif savings_rate >= 10:
        print("  Good job! Try to increase savings to 20%.")
    else:
        print("  Consider increasing your savings rate.")
else:
    print(f"⚠ WARNING: You're overspending by ${abs(remaining):.2f}!")
    print("  You need to reduce expenses or increase income.")

# Specific recommendations
if rent_percent > 30:
    print(f"\n⚠ Housing costs ({rent_percent:.1f}%) exceed recommended 30%")
    
if food_percent > 15:
    print(f"\n⚠ Food costs ({food_percent:.1f}%) are high. Consider meal planning.")
    
if entertainment_percent > 10:
    print(f"\n⚠ Entertainment ({entertain_percent:.1f}%) is above recommended 10%")

print("\n" + "=" * 50)
print("Thank you for using the Budget Analyzer!")
print("=" * 50)
```

**What This Project Teaches**:
- Gathering multiple inputs
- Performing complex calculations
- Formatting output professionally
- Making comparisons and decisions
- Providing conditional feedback
- Building a complete, useful program

---

## Part 10: Key Takeaways

**What You've Learned**:

1. **Variables** are labeled containers for data
2. **Data types** (int, float, str, bool) represent different kinds of information
3. **Operators** (+, -, *, /, //, %, **) perform calculations
4. **Type conversion** (int(), float(), str()) changes data types
5. **input()** gets information from users (always returns strings!)
6. **print()** displays information to users
7. **f-strings** make formatting easy: f"Hello {name}"

**Best Practices Introduced**:
- Use descriptive variable names
- Comment your code
- Format currency with :.2f
- Check types with type()
- Convert input() to numbers when needed
- Use parentheses to clarify math operations

**Common Pitfalls to Avoid**:
- Confusing = (assignment) with == (comparison)
- Forgetting to convert input() strings to numbers
- Using reserved keywords as variable names
- Forgetting operator precedence
- Not handling decimal precision for money

---

## Next Steps

In **Chapter 2**, we'll learn about **Control Flow** - how to make programs that make decisions and repeat actions. You'll learn:
- How to make your programs intelligent (if/elif/else)
- How to repeat actions efficiently (loops)
- How to validate user input
- How to build interactive programs

**Before moving on, make sure you can**:
- Create and use variables confidently
- Understand all four data types
- Convert between types
- Use input() and print() effectively
- Write code with proper formatting
- Solve problems by breaking them into steps

**Practice Recommendation**: Spend time on the practice problems above. Programming is like learning an instrument—you need to practice, not just read about it!

---

## Appendix: Operator Reference

**Arithmetic Operators**:
```
+   Addition          5 + 3 = 8
-   Subtraction       5 - 3 = 2
*   Multiplication    5 * 3 = 15
/   Division          5 / 3 = 1.6666...
//  Floor Division    5 // 3 = 1
%   Modulo           5 % 3 = 2
**  Exponentiation   5 ** 3 = 125
```

**Comparison Operators**:
```
==  Equal to            5 == 5  → True
!=  Not equal to        5 != 3  → True
<   Less than           5 < 3   → False
>   Greater than        5 > 3   → True
<=  Less or equal       5 <= 5  → True
>=  Greater or equal    5 >= 3  → True
```

**Logical Operators**:
```
and   Both must be True      True and False → False
or    At least one is True   True or False → True
not   Reverses boolean       not True → False
```

**String Operators**:
```
+   Concatenation    "Hello" + " " + "World" → "Hello World"
*   Repetition       "Ha" * 3 → "HaHaHa"
```
