# Chapter 2: Control Flow
## Making Programs That Think and Repeat

---

## 📑 INDEX - Chapter 2

### [Part 1: The Problem We're Solving - Why Control Flow?](#part-1-the-problem-were-solving---why-control-flow)
- [The Real-World Problem](#the-real-world-problem)
- [Two Types of Control Flow](#two-types-of-control-flow)

### [Part 2: Conditional Logic - Making Decisions](#part-2-conditional-logic---making-decisions)
- [The if Statement - Basic Decision Making](#the-if-statement---basic-decision-making)
- [The else Statement - Alternative Paths](#the-else-statement---alternative-paths)
- [The elif Statement - Multiple Conditions](#the-elif-statement---multiple-conditions)
- [Nested if Statements - Decisions Within Decisions](#nested-if-statements---decisions-within-decisions)

### [Part 3: Comparison Operators - Testing Conditions](#part-3-comparison-operators---testing-conditions)
- [Equality and Inequality](#equality-and-inequality)
- [Relational Operators](#relational-operators)
- [Chaining Comparisons](#chaining-comparisons)
- [Common Pitfalls](#common-pitfalls)

### [Part 4: Logical Operators - Combining Conditions](#part-4-logical-operators---combining-conditions)
- [The and Operator](#the-and-operator)
- [The or Operator](#the-or-operator)
- [The not Operator](#the-not-operator)
- [Operator Precedence](#operator-precedence)
- [Short-Circuit Evaluation](#short-circuit-evaluation)

### [Part 5: while Loops - Repeating Until Done](#part-5-while-loops---repeating-until-done)
- [Why Loops? The Problem of Repetition](#why-loops-the-problem-of-repetition)
- [Basic while Loop](#basic-while-loop)
- [Loop Control Variables](#loop-control-variables)
- [Infinite Loops and How to Avoid Them](#infinite-loops-and-how-to-avoid-them)
- [Input Validation with while](#input-validation-with-while)

### [Part 6: for Loops - Iterating Over Sequences](#part-6-for-loops---iterating-over-sequences)
- [Why for Loops?](#why-for-loops)
- [Looping with range()](#looping-with-range)
- [Looping Over Strings](#looping-over-strings)
- [Looping Over Lists](#looping-over-lists)
- [When to Use for vs while](#when-to-use-for-vs-while)

### [Part 7: Loop Control Statements](#part-7-loop-control-statements)
- [break - Exit the Loop Early](#break---exit-the-loop-early)
- [continue - Skip to Next Iteration](#continue---skip-to-next-iteration)
- [pass - Do Nothing (Placeholder)](#pass---do-nothing-placeholder)

### [Part 8: Nested Loops - Loops Within Loops](#part-8-nested-loops---loops-within-loops)
- [Understanding Nested Loops](#understanding-nested-loops)
- [Common Patterns](#common-patterns)
- [Performance Considerations](#performance-considerations)

### [Part 9: Common Mistakes and How to Avoid Them](#part-9-common-mistakes-and-how-to-avoid-them)
- [Mistake #1: Forgetting Indentation](#mistake-1-forgetting-indentation)
- [Mistake #2: Using = Instead of ==](#mistake-2-using--instead-of-)
- [Mistake #3: Creating Infinite Loops](#mistake-3-creating-infinite-loops)
- [Mistake #4: Off-by-One Errors](#mistake-4-off-by-one-errors)
- [Mistake #5: Confusing and/or Logic](#mistake-5-confusing-andor-logic)

### [Part 10: Detailed Worked Examples](#part-10-detailed-worked-examples)
- [Example 1: Grade Calculator](#example-1-grade-calculator)
- [Example 2: Password Validator](#example-2-password-validator)
- [Example 3: Guess the Number Game](#example-3-guess-the-number-game)
- [Example 4: Multiplication Table Generator](#example-4-multiplication-table-generator)
- [Example 5: Prime Number Checker](#example-5-prime-number-checker)
- [Example 6: ATM Withdrawal System](#example-6-atm-withdrawal-system)
- [Example 7: Pattern Printer](#example-7-pattern-printer)

### [Part 11: Practice Problems](#part-11-practice-problems)
- [Easy Problems (1-7)](#easy-problems-problems-1-7)
- [Medium Problems (8-17)](#medium-problems-problems-8-17)
- [Hard Problems (18-27)](#hard-problems-problems-18-27)
- [Challenge Problems (28-35)](#challenge-problems-problems-28-35)

### [Part 12: Mini-Project - Interactive Menu System](#part-12-mini-project---interactive-menu-system)

### [Part 13: Key Takeaways](#part-13-key-takeaways)

### [Appendix: Control Flow Reference](#appendix-control-flow-reference)

---

## Part 1: The Problem We're Solving - Why Control Flow?

### The Real-World Problem

Think about your daily life. You constantly make decisions:

- **IF** it's raining, **THEN** take an umbrella
- **IF** you're hungry **AND** have money, **THEN** buy food
- **WHILE** there are dirty dishes, **KEEP** washing them
- **FOR EACH** email in your inbox, **READ** it

Your brain does this automatically. But computers are dumb—they only do **exactly** what we tell them, in the exact order we specify.

**The problem**: Without control flow, programs can only execute instructions sequentially, one after another. They can't:
- Make decisions based on conditions
- Repeat actions
- Handle different scenarios
- Respond to user input intelligently

```python
# Without control flow (BORING and LIMITED):
name = input("Enter name: ")
print(f"Hello, {name}")
# That's it. No intelligence. No choices. No repetition.

# With control flow (POWERFUL and INTELLIGENT):
name = input("Enter name: ")
if name == "":
    print("You didn't enter a name!")
elif len(name) < 2:
    print("Name too short!")
else:
    print(f"Hello, {name}!")
    
# Now the program is SMART - it handles different cases!
```

### Two Types of Control Flow

**1. Conditional Execution (Branching)**
- Making decisions: if, elif, else
- Different paths based on conditions
- Like a fork in the road

**2. Repetition (Looping)**
- Doing things multiple times: while, for
- Repeating until a condition is met
- Like a roundabout you keep circling

Think of a video game:
- **IF** player health = 0 **THEN** game over (conditional)
- **WHILE** enemies exist **KEEP** fighting (loop)
- **FOR EACH** item in inventory **SHOW** it (loop)

Control flow is what makes programs **interactive**, **intelligent**, and **useful**.

---

## Part 2: Conditional Logic - Making Decisions

### The if Statement - Basic Decision Making

**The problem**: How do we make a program do something only under certain conditions?

Think about a thermostat:
- IF temperature > 75°F → turn on AC
- IF temperature < 65°F → turn on heat
- Otherwise → do nothing

```python
# Basic if statement structure:
# if condition:
#     code to execute if condition is True

temperature = 78

if temperature > 75:
    print("Turning on AC")
    print("It's getting cool!")
    
# The code continues here regardless
print("Temperature check complete")

# Output:
# Turning on AC
# It's getting cool!
# Temperature check complete
```

**Critical insight**: The **indentation** (spaces) is how Python knows what's inside the if block!

```python
# Indentation matters!
age = 20

if age >= 18:
    print("You are an adult")  # This runs if age >= 18
    print("You can vote")       # This ALSO runs if age >= 18
    
print("Program continues")      # This ALWAYS runs (not indented)
```

**Real-world example**: Ticket pricing
```python
age = int(input("Enter your age: "))

if age < 3:
    print("Free admission!")
    ticket_price = 0
    
if age >= 3:
    print("Ticket required")
    ticket_price = 15
    
print(f"Your ticket costs: ${ticket_price}")

# Problem: We're checking conditions separately
# Better approach coming next...
```

### The else Statement - Alternative Paths

**The problem**: Often we want to do one thing if a condition is true, and something DIFFERENT if it's false.

Like a light switch: either ON or OFF, not both.

```python
# if-else structure: ONE path will execute
# if condition:
#     code if True
# else:
#     code if False

age = int(input("Enter age: "))

if age >= 18:
    print("You are an adult")
    can_vote = True
else:
    print("You are a minor")
    can_vote = False
    
print(f"Can vote: {can_vote}")

# Exactly ONE of the print statements runs, never both!
```

**Real-world example**: Login system
```python
username = input("Username: ")
password = input("Password: ")

correct_username = "admin"
correct_password = "secret123"

if username == correct_username and password == correct_password:
    print("✓ Login successful!")
    print("Welcome back!")
    is_logged_in = True
else:
    print("✗ Login failed!")
    print("Incorrect username or password")
    is_logged_in = False
```

### The elif Statement - Multiple Conditions

**The problem**: Sometimes we have MORE than two possibilities. We need to check multiple conditions in sequence.

Think about letter grades:
- 90-100 → A
- 80-89 → B
- 70-79 → C
- Below 70 → F

We can't do this with just if/else (only 2 options). We need **elif** (else if).

```python
# elif structure: Check conditions in order
# if condition1:
#     code
# elif condition2:
#     code
# elif condition3:
#     code
# else:
#     code

score = int(input("Enter your score: "))

if score >= 90:
    grade = "A"
    print("Excellent work!")
elif score >= 80:
    grade = "B"
    print("Good job!")
elif score >= 70:
    grade = "C"
    print("Passing grade")
elif score >= 60:
    grade = "D"
    print("Needs improvement")
else:
    grade = "F"
    print("Failed")
    
print(f"Your grade: {grade}")

# Key insight: Python checks from TOP to BOTTOM
# Once one condition is True, it skips the rest!
```

**Why the order matters**:
```python
# WRONG ORDER - Bug!
score = 85

if score >= 60:  # This is True for 85!
    grade = "D"  # So this executes and we're done
elif score >= 70:  # Never checked
    grade = "C"
elif score >= 80:  # Never checked
    grade = "B"
    
print(grade)  # Outputs: D (WRONG! Should be B)

# CORRECT ORDER - From highest to lowest
if score >= 90:
    grade = "A"
elif score >= 80:  # Checks this only if score < 90
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"
    
print(grade)  # Outputs: B (CORRECT!)
```

**Real-world example**: Shipping cost calculator
```python
order_total = float(input("Order total: $"))

if order_total >= 100:
    shipping = 0
    print("Free shipping!")
elif order_total >= 50:
    shipping = 5.99
    print("Reduced shipping: $5.99")
elif order_total >= 25:
    shipping = 9.99
    print("Standard shipping: $9.99")
else:
    shipping = 14.99
    print("Small order shipping: $14.99")
    
total_with_shipping = order_total + shipping
print(f"Total: ${total_with_shipping:.2f}")
```

### Nested if Statements - Decisions Within Decisions

**The problem**: Sometimes we need to check a condition, and THEN check another condition based on the result.

Think about going to a movie:
- First: Are you old enough? (age check)
  - If yes: Do you have money? (money check)
    - If yes: Buy ticket
    - If no: Can't buy
  - If no: Too young

```python
# Nested if structure
# if condition1:
#     if condition2:
#         code
#     else:
#         code
# else:
#     code

age = int(input("Age: "))

if age >= 18:
    # Inside the "adult" branch
    has_id = input("Do you have ID? (yes/no): ")
    
    if has_id == "yes":
        print("Entry granted!")
    else:
        print("Need ID to enter")
else:
    # Inside the "minor" branch
    print("Too young to enter")
```

**Real-world example**: Restaurant recommendations
```python
print("=== Restaurant Finder ===")
is_hungry = input("Are you hungry? (yes/no): ")

if is_hungry == "yes":
    budget = float(input("Budget per person: $"))
    
    if budget >= 50:
        cuisine = input("Preferred cuisine? (italian/japanese/american): ")
        
        if cuisine == "italian":
            print("Recommendation: Luigi's Fine Dining")
        elif cuisine == "japanese":
            print("Recommendation: Sakura Sushi")
        else:
            print("Recommendation: The Steakhouse")
    elif budget >= 20:
        print("Recommendation: Casual dining options")
        print("Try: Olive Garden, Chipotle, or Panera")
    else:
        print("Recommendation: Fast food")
        print("Try: McDonald's, Taco Bell, or Subway")
else:
    print("Come back when you're hungry!")

# This shows NESTED decisions:
# First: hungry or not?
# If hungry: what's the budget?
# If high budget: what cuisine?
```

**When to use nested vs elif**:
```python
# Use ELIF when conditions are mutually exclusive (can't be both)
if age < 13:
    category = "child"
elif age < 20:
    category = "teenager"
elif age < 65:
    category = "adult"
else:
    category = "senior"

# Use NESTED when conditions are independent (both could be true)
if has_license:
    if has_car:
        print("You can drive your own car")
    else:
        print("You can rent a car")
else:
    print("You need a license first")
```

---

## Part 3: Comparison Operators - Testing Conditions

### Equality and Inequality

Comparison operators return **boolean values** (True or False).

```python
# == (equal to) - checks if values are the same
x = 5
y = 5
z = 10

print(x == y)  # True (5 equals 5)
print(x == z)  # False (5 does not equal 10)

# COMMON MISTAKE: Using = instead of ==
if x = 5:  # ERROR! SyntaxError
    print("This won't work")
    
if x == 5:  # CORRECT! Comparison
    print("This works")

# != (not equal to) - checks if values are different
print(x != y)  # False (5 equals 5, so NOT different)
print(x != z)  # True (5 does not equal 10, so different)

# Works with strings too!
name = "Alice"
print(name == "Alice")  # True
print(name == "alice")  # False (case-sensitive!)
print(name != "Bob")    # True
```

### Relational Operators

```python
# < (less than)
print(5 < 10)   # True
print(10 < 5)   # False
print(5 < 5)    # False (not less than, it's equal)

# > (greater than)
print(10 > 5)   # True
print(5 > 10)   # False
print(5 > 5)    # False

# <= (less than or equal to)
print(5 <= 10)  # True (5 is less than 10)
print(5 <= 5)   # True (5 equals 5)
print(10 <= 5)  # False

# >= (greater than or equal to)
print(10 >= 5)  # True (10 is greater than 5)
print(5 >= 5)   # True (5 equals 5)
print(5 >= 10)  # False

# Real-world example
age = 25
min_age = 21

if age >= min_age:
    print("Can purchase alcohol")
else:
    print(f"Must be {min_age} or older")
```

### Chaining Comparisons

Python has a beautiful feature: **chained comparisons**!

```python
# Instead of using 'and'
x = 15
if x >= 10 and x <= 20:
    print("x is between 10 and 20")

# Python lets you write it naturally!
if 10 <= x <= 20:
    print("x is between 10 and 20")
    
# This is clearer and more readable!

# More examples
age = 25
if 18 <= age < 65:
    print("Working age")

score = 85
if 80 <= score < 90:
    print("Grade: B")
    
# You can even chain more!
if 0 < x < 10 < y < 20:
    print("Complex condition!")
```

### Common Pitfalls

```python
# Pitfall #1: Comparing floats
a = 0.1 + 0.2
b = 0.3

print(a == b)  # False! (Why? Floating point precision)
print(a)       # 0.30000000000000004

# Solution: Check if "close enough"
tolerance = 0.0001
if abs(a - b) < tolerance:
    print("Close enough!")

# Pitfall #2: Case sensitivity with strings
name = "Alice"
if name == "alice":  # False! 'A' != 'a'
    print("Won't print")
    
# Solution: Convert to same case
if name.lower() == "alice":  # True!
    print("Match!")

# Pitfall #3: Comparing different types
age = "25"  # String!
if age > 18:  # ERROR! Can't compare string and int
    print("Adult")
    
# Solution: Convert types
age = int("25")
if age > 18:  # Now it works!
    print("Adult")
```

---

## Part 4: Logical Operators - Combining Conditions

### The and Operator

**The problem**: Sometimes we need MULTIPLE conditions to be true.

Think: "I'll go to the beach if it's sunny **AND** it's warm **AND** I have time."

```python
# and: ALL conditions must be True for result to be True

# Truth table for 'and':
# True and True = True
# True and False = False
# False and True = False
# False and False = False

# Example: Club entry requirements
age = 25
has_id = True

if age >= 21 and has_id:
    print("Entry allowed")
else:
    print("Entry denied")

# Both must be True!
# age >= 21 is True (25 >= 21)
# has_id is True
# True and True = True, so "Entry allowed"

# Real-world example: Loan approval
credit_score = 720
annual_income = 60000
has_job = True

if credit_score >= 700 and annual_income >= 50000 and has_job:
    print("Loan approved!")
    print("You meet all requirements")
else:
    print("Loan denied")
    
# All three conditions must be True
```

**Multiple conditions**:
```python
# You can chain many 'and' operators
temperature = 72
humidity = 60
precipitation = 0

if temperature >= 70 and temperature <= 80 and humidity < 70 and precipitation == 0:
    print("Perfect day!")
    
# Cleaner with chained comparison:
if 70 <= temperature <= 80 and humidity < 70 and precipitation == 0:
    print("Perfect day!")
```

### The or Operator

**The problem**: Sometimes we need ONLY ONE condition to be true.

Think: "I'll be happy if I get an A **OR** a B" (don't need both!)

```python
# or: AT LEAST ONE condition must be True for result to be True

# Truth table for 'or':
# True or True = True
# True or False = True
# False or True = True
# False or False = False

# Example: Free shipping eligibility
order_total = 45
is_prime_member = True

if order_total >= 50 or is_prime_member:
    print("Free shipping!")
else:
    print(f"Add ${50 - order_total} for free shipping")

# Only ONE needs to be True!
# order_total >= 50 is False (45 < 50)
# is_prime_member is True
# False or True = True, so "Free shipping!"

# Real-world example: Weekend or holiday
day = "Saturday"
is_holiday = False

if day == "Saturday" or day == "Sunday" or is_holiday:
    print("Day off!")
    sleeping_late = True
else:
    print("Work day")
    sleeping_late = False
```

### The not Operator

**The problem**: Sometimes we want the OPPOSITE of a condition.

Think: "I'll go outside if it's **NOT** raining."

```python
# not: Reverses the boolean value

# Truth table for 'not':
# not True = False
# not False = True

# Example: Door access
is_locked = False

if not is_locked:
    print("Door is open, you can enter")
else:
    print("Door is locked")

# not False = True, so "Door is open"

# Real-world example: Form validation
has_error = False

if not has_error:
    print("Form submitted successfully!")
else:
    print("Please fix errors")

# Using 'not' with other operators
is_raining = True
is_cold = False

if not is_raining and not is_cold:
    print("Perfect weather!")
else:
    print("Stay inside")
    
# More readable alternative:
if not (is_raining or is_cold):
    print("Perfect weather!")
```

### Operator Precedence

**The problem**: When mixing operators, which evaluates first?

```python
# Order of operations for boolean logic:
# 1. not (highest priority)
# 2. and
# 3. or (lowest priority)

# Example 1:
result = True or False and False
# Step 1: False and False = False (and first)
# Step 2: True or False = True (or last)
print(result)  # True

# Example 2:
result = not True or False
# Step 1: not True = False (not first)
# Step 2: False or False = False (or last)
print(result)  # False

# Use parentheses for clarity!
age = 25
has_license = True
has_car = False

# Ambiguous:
if age >= 18 and has_license or has_car:
    print("Can drive")
    
# Clear:
if (age >= 18 and has_license) or has_car:
    print("Can drive")
    
# Different meaning:
if age >= 18 and (has_license or has_car):
    print("Adult with transportation option")
```

**Best practice**: Always use parentheses to make your intent clear!

```python
# Complex condition made clear with parentheses
is_weekend = True
is_holiday = False
has_work = False
is_sick = True

# Can stay home if:
# (weekend or holiday) AND (no work OR sick)
can_stay_home = (is_weekend or is_holiday) and (not has_work or is_sick)

if can_stay_home:
    print("Staying home!")
```

### Short-Circuit Evaluation

**Important concept**: Python stops evaluating as soon as it knows the answer!

```python
# With 'and': If first is False, don't check the rest
# (Because False and ANYTHING = False)

x = 5
y = 0

# This would cause division by zero:
# if x > 0 and (10 / y) > 1:  # ERROR if we evaluate 10/y

# But Python short-circuits:
if y != 0 and (10 / y) > 1:  # Safe!
    print("This won't run")
# Since y != 0 is False, Python never evaluates 10/y

# With 'or': If first is True, don't check the rest
# (Because True or ANYTHING = True)

has_admin = True
has_permission = False  # Imagine this takes time to check

if has_admin or has_permission:
    print("Access granted")
# Since has_admin is True, Python never checks has_permission

# Practical use:
username = input("Username: ")

# Check if string is empty BEFORE checking length
if username != "" and len(username) >= 3:
    print("Valid username")
else:
    print("Username too short")
```

---

## Part 5: while Loops - Repeating Until Done

### Why Loops? The Problem of Repetition

Imagine you need to print numbers 1 to 100. Without loops:

```python
# The painful way (don't do this!):
print(1)
print(2)
print(3)
# ... 97 more lines ...
print(100)

# What if you needed 1 to 1000? 10,000?
# What if the number changed based on user input?
# This is unsustainable!
```

**The problem**: We need a way to repeat code without writing it repeatedly.

**The solution**: Loops! Tell the computer "keep doing this until I tell you to stop."

### Basic while Loop

```python
# while loop structure:
# while condition:
#     code to repeat
#     (update something to eventually make condition False)

# Example: Count from 1 to 5
counter = 1  # Start value

while counter <= 5:  # Condition
    print(counter)
    counter = counter + 1  # Update (or counter += 1)
    
print("Done!")

# Output:
# 1
# 2
# 3
# 4
# 5
# Done!

# What's happening:
# Iteration 1: counter=1, 1<=5 is True, print 1, counter becomes 2
# Iteration 2: counter=2, 2<=5 is True, print 2, counter becomes 3
# Iteration 3: counter=3, 3<=5 is True, print 3, counter becomes 4
# Iteration 4: counter=4, 4<=5 is True, print 4, counter becomes 5
# Iteration 5: counter=5, 5<=5 is True, print 5, counter becomes 6
# Iteration 6: counter=6, 6<=5 is FALSE, exit loop
```

**Real-world example**: Countdown timer
```python
import time  # We'll learn about imports later, but this lets us pause

countdown = 10

print("Countdown starting...")
while countdown > 0:
    print(countdown)
    time.sleep(1)  # Pause for 1 second
    countdown -= 1  # Decrease by 1
    
print("Blast off! 🚀")
```

### Loop Control Variables

The variable that controls the loop (like `counter` above) is crucial!

```python
# Pattern 1: Counting up
i = 1
while i <= 10:
    print(i)
    i += 1  # Same as: i = i + 1

# Pattern 2: Counting down
i = 10
while i >= 1:
    print(i)
    i -= 1  # Same as: i = i - 1

# Pattern 3: Counting by steps
i = 0
while i <= 100:
    print(i)
    i += 10  # Count by 10s: 0, 10, 20, 30...

# Pattern 4: Until condition changes
user_input = ""
while user_input != "quit":
    user_input = input("Enter command (or 'quit' to exit): ")
    print(f"You entered: {user_input}")
```

### Infinite Loops and How to Avoid Them

**The danger**: If the condition never becomes False, the loop runs FOREVER!

```python
# INFINITE LOOP - Don't run this!
# counter = 1
# while counter <= 5:
#     print(counter)
#     # Forgot to update counter! It's always 1!
#     # This prints 1 forever until you force-quit

# Another infinite loop:
# while True:
#     print("This runs forever")
#     # No way to break out!

# How to avoid:
# 1. Always have a way for condition to become False
# 2. Make sure your update statement is correct
# 3. Use a maximum iteration counter as safety

# Safe loop with max iterations:
counter = 1
max_iterations = 1000

while counter <= 5 and counter < max_iterations:
    print(counter)
    counter += 1
```

### Input Validation with while

**Super useful pattern**: Keep asking until you get valid input!

```python
# Example 1: Get a positive number
number = -1  # Start with invalid value

while number < 0:
    number = int(input("Enter a positive number: "))
    if number < 0:
        print("Error: Must be positive!")
        
print(f"You entered: {number}")

# Example 2: Get yes/no response
response = ""

while response != "yes" and response != "no":
    response = input("Do you agree? (yes/no): ").lower()
    if response != "yes" and response != "no":
        print("Please enter 'yes' or 'no'")
        
print(f"Your answer: {response}")

# Example 3: Password verification
attempts = 0
max_attempts = 3
correct_password = "secret123"
password = ""

while password != correct_password and attempts < max_attempts:
    password = input("Enter password: ")
    attempts += 1
    
    if password != correct_password:
        remaining = max_attempts - attempts
        if remaining > 0:
            print(f"Incorrect! {remaining} attempts remaining")
        else:
            print("Account locked!")
            
if password == correct_password:
    print("Access granted!")
```

---

## Part 6: for Loops - Iterating Over Sequences

### Why for Loops?

**The problem**: `while` loops are great, but managing counter variables is tedious and error-prone.

When you know HOW MANY times to repeat, or want to go through each item in a collection, `for` loops are cleaner!

```python
# With while loop (more work):
counter = 0
while counter < 5:
    print(counter)
    counter += 1

# With for loop (cleaner!):
for counter in range(5):
    print(counter)
    
# Same result, less code, less chance of errors!
```

### Looping with range()

`range()` generates a sequence of numbers for us!

```python
# range(stop) - from 0 to stop-1
for i in range(5):
    print(i)
# Output: 0, 1, 2, 3, 4 (NOT 5!)

# range(start, stop) - from start to stop-1
for i in range(1, 6):
    print(i)
# Output: 1, 2, 3, 4, 5

# range(start, stop, step) - from start to stop-1, by step
for i in range(0, 10, 2):
    print(i)
# Output: 0, 2, 4, 6, 8 (even numbers)

for i in range(10, 0, -1):
    print(i)
# Output: 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 (countdown!)

# Why does range stop at stop-1?
# Because it makes calculations easier:
# range(0, 10) gives us 10 numbers (0-9)
# range(5, 15) gives us 10 numbers (5-14)
# The LENGTH is always stop - start
```

**Real-world examples**:
```python
# Example 1: Calculate sum of first N numbers
n = 10
total = 0

for i in range(1, n + 1):  # 1 to 10 inclusive
    total += i
    
print(f"Sum of 1 to {n} = {total}")  # 55

# Example 2: Print multiplication table
number = 7

print(f"Multiplication table for {number}:")
for i in range(1, 11):
    result = number * i
    print(f"{number} × {i} = {result}")

# Output:
# 7 × 1 = 7
# 7 × 2 = 14
# ... etc

# Example 3: Calculate factorial
n = 5
factorial = 1

for i in range(1, n + 1):
    factorial *= i  # Same as: factorial = factorial * i
    
print(f"{n}! = {factorial}")  # 5! = 120
```

### Looping Over Strings

**Key insight**: Strings are sequences of characters! We can loop through each character.

```python
# Loop through each character
word = "Python"

for letter in word:
    print(letter)

# Output:
# P
# y
# t
# h
# o
# n

# Real-world example: Count vowels
text = "Hello World"
vowels = "aeiouAEIOU"
vowel_count = 0

for char in text:
    if char in vowels:
        vowel_count += 1
        
print(f"Number of vowels: {vowel_count}")  # 3

# Example 2: Reverse a string
word = "Python"
reversed_word = ""

for char in word:
    reversed_word = char + reversed_word
    
print(reversed_word)  # nohtyP

# Example 3: Check if palindrome
word = "racecar"
reversed_word = ""

for char in word:
    reversed_word = char + reversed_word
    
if word == reversed_word:
    print(f"{word} is a palindrome!")
else:
    print(f"{word} is not a palindrome")
```

### Looping Over Lists

Lists are collections of items (we'll learn more in Chapter 3). For now, know you can loop through them!

```python
# Loop through a list
fruits = ["apple", "banana", "cherry", "date"]

for fruit in fruits:
    print(f"I like {fruit}")

# Output:
# I like apple
# I like banana
# I like cherry
# I like date

# Real-world example: Calculate total price
prices = [19.99, 5.50, 12.00, 8.75]
total = 0

for price in prices:
    total += price
    
print(f"Total: ${total:.2f}")  # $46.24

# Example 2: Find maximum value
numbers = [45, 23, 67, 12, 89, 34]
max_value = numbers[0]  # Start with first number

for num in numbers:
    if num > max_value:
        max_value = num
        
print(f"Maximum: {max_value}")  # 89

# Example 3: Count specific items
grades = ["A", "B", "A", "C", "A", "B", "A"]
a_count = 0

for grade in grades:
    if grade == "A":
        a_count += 1
        
print(f"Number of A's: {a_count}")  # 4
```

### When to Use for vs while

**Use `for` when**:
- You know how many times to loop
- You're iterating over a collection (string, list, etc.)
- You want cleaner, more readable code

**Use `while` when**:
- You don't know how many iterations needed
- You're waiting for a condition to change
- You're doing input validation

```python
# GOOD use of for:
for i in range(10):
    print(i)

# BAD use of while (for is better here):
i = 0
while i < 10:
    print(i)
    i += 1

# GOOD use of while:
password = ""
while password != "secret":
    password = input("Enter password: ")

# BAD use of for (while is better here):
# Can't easily do this with for!
```

---

## Part 7: Loop Control Statements

### break - Exit the Loop Early

**The problem**: Sometimes we want to stop a loop before it naturally finishes.

Think: Searching for something—once you find it, stop looking!

```python
# Example 1: Find a number
numbers = [5, 12, 8, 19, 3, 15]
target = 19

for num in numbers:
    print(f"Checking {num}...")
    if num == target:
        print(f"Found {target}!")
        break  # Exit the loop immediately
    print(f"{num} is not the target")
    
print("Search complete")

# Output:
# Checking 5...
# 5 is not the target
# Checking 12...
# 12 is not the target
# Checking 8...
# 8 is not the target
# Checking 19...
# Found 19!
# Search complete

# Notice: We never checked 3 or 15 after finding 19!

# Example 2: Shopping with budget
prices = [15, 25, 10, 30, 8]
budget = 50
spent = 0

print("Shopping spree!")
for price in prices:
    if spent + price > budget:
        print(f"Can't afford ${price} item")
        print("Budget exceeded, stopping!")
        break
    
    spent += price
    print(f"Bought item for ${price}, spent so far: ${spent}")
    
print(f"Final total: ${spent}")
```

**Real-world example**: ATM PIN validation
```python
correct_pin = "1234"
max_attempts = 3
attempts = 0

print("=== ATM ===")
while attempts < max_attempts:
    pin = input("Enter PIN: ")
    attempts += 1
    
    if pin == correct_pin:
        print("✓ Access granted!")
        print("Welcome!")
        break  # Exit loop on success
    else:
        remaining = max_attempts - attempts
        if remaining > 0:
            print(f"✗ Incorrect PIN. {remaining} attempts remaining")
        else:
            print("✗ Card locked!")
            
# If break was called, this runs immediately
# If loop finished naturally (3 wrong attempts), this also runs
print("Transaction complete")
```

### continue - Skip to Next Iteration

**The problem**: Sometimes we want to skip the rest of THIS iteration but continue the loop.

Think: Grading papers—skip the ones not submitted, grade the rest.

```python
# Example 1: Print only positive numbers
numbers = [5, -3, 8, -1, 12, -7, 4]

print("Positive numbers:")
for num in numbers:
    if num < 0:
        continue  # Skip this iteration, go to next number
    print(num)

# Output:
# Positive numbers:
# 5
# 8
# 12
# 4

# What's happening:
# num=5: not < 0, so print 5
# num=-3: < 0, continue (skip print)
# num=8: not < 0, so print 8
# ... etc

# Example 2: Process only even numbers
for i in range(1, 11):
    if i % 2 != 0:  # If odd
        continue  # Skip odd numbers
    
    # This only runs for even numbers
    print(f"{i} is even")
    square = i ** 2
    print(f"  {i}² = {square}")

# Output:
# 2 is even
#   2² = 4
# 4 is even
#   4² = 16
# ... etc
```

**Real-world example**: Email filter
```python
emails = [
    "work@company.com",
    "spam@ads.com",
    "friend@email.com",
    "spam@offers.com",
    "boss@company.com"
]

print("=== Inbox (No Spam) ===")
for email in emails:
    # Skip spam emails
    if "spam" in email or "ads" in email or "offers" in email:
        continue
    
    # Process legitimate email
    print(f"✉ New message from: {email}")
    
# Output:
# ✉ New message from: work@company.com
# ✉ New message from: friend@email.com
# ✉ New message from: boss@company.com
```

**break vs continue**:
```python
# break: EXIT the entire loop
for i in range(5):
    if i == 3:
        break
    print(i)
# Output: 0, 1, 2 (stops at 3)

# continue: SKIP to next iteration
for i in range(5):
    if i == 3:
        continue
    print(i)
# Output: 0, 1, 2, 4 (skips 3, but continues)
```

### pass - Do Nothing (Placeholder)

**The problem**: Sometimes you need a statement syntactically, but don't want to do anything yet.

```python
# pass is a placeholder that does literally nothing

# Example 1: Planning future code
for i in range(10):
    if i % 2 == 0:
        pass  # TODO: Handle even numbers later
    else:
        print(f"{i} is odd")

# Example 2: Empty conditional (for now)
age = int(input("Enter age: "))

if age < 18:
    pass  # Will add restriction later
elif age >= 65:
    print("Senior discount applied")
else:
    print("Regular price")

# Why not just leave it empty?
# This causes SyntaxError:
# if age < 18:
#     # Can't have nothing here!
# else:
#     print("Hello")

# This works:
# if age < 18:
#     pass  # Placeholder
# else:
#     print("Hello")

# In practice, you'll rarely use pass
# But it's good to know it exists!
```

---

## Part 8: Nested Loops - Loops Within Loops

### Understanding Nested Loops

**The problem**: Sometimes we need to repeat something, and for EACH repetition, repeat something ELSE.

Think: A clock—for each hour, the minute hand makes a full 60-minute rotation.

```python
# Basic nested loop structure:
for outer in range(3):
    print(f"Outer loop: {outer}")
    for inner in range(2):
        print(f"  Inner loop: {inner}")

# Output:
# Outer loop: 0
#   Inner loop: 0
#   Inner loop: 1
# Outer loop: 1
#   Inner loop: 0
#   Inner loop: 1
# Outer loop: 2
#   Inner loop: 0
#   Inner loop: 1

# Total iterations: 3 × 2 = 6

# What's happening:
# 1. outer=0, run entire inner loop (inner=0, then inner=1)
# 2. outer=1, run entire inner loop (inner=0, then inner=1)
# 3. outer=2, run entire inner loop (inner=0, then inner=1)
```

### Common Patterns

**Pattern 1: Multiplication table**
```python
# Print multiplication table (1-10)
print("Multiplication Table:")
print("-" * 40)

for i in range(1, 11):  # Rows (1 to 10)
    for j in range(1, 11):  # Columns (1 to 10)
        product = i * j
        print(f"{product:4}", end="")  # Print with spacing
    print()  # New line after each row

# Output:
#    1   2   3   4   5   6   7   8   9  10
#    2   4   6   8  10  12  14  16  18  20
#    3   6   9  12  15  18  21  24  27  30
# ... etc
```

**Pattern 2: Printing shapes**
```python
# Right triangle
rows = 5
for i in range(1, rows + 1):
    for j in range(i):
        print("*", end="")
    print()

# Output:
# *
# **
# ***
# ****
# *****

# Square
size = 4
for i in range(size):
    for j in range(size):
        print("#", end=" ")
    print()

# Output:
# # # # #
# # # # #
# # # # #
# # # # #

# Pyramid
rows = 5
for i in range(rows):
    # Print spaces
    for j in range(rows - i - 1):
        print(" ", end="")
    # Print stars
    for j in range(2 * i + 1):
        print("*", end="")
    print()

# Output:
#     *
#    ***
#   *****
#  *******
# *********
```

**Pattern 3: Checking all combinations**
```python
# Find all pairs that sum to target
numbers = [1, 3, 5, 7, 9]
target = 10

print(f"Pairs that sum to {target}:")
for i in range(len(numbers)):
    for j in range(i + 1, len(numbers)):  # Start from i+1 to avoid duplicates
        if numbers[i] + numbers[j] == target:
            print(f"{numbers[i]} + {numbers[j]} = {target}")

# Output:
# 1 + 9 = 10
# 3 + 7 = 10
```

**Pattern 4: Nested loops with different ranges**
```python
# Seating chart for a theater
rows = 3
seats_per_row = 5

print("Theater Seating Chart:")
for row in range(1, rows + 1):
    print(f"Row {row}: ", end="")
    for seat in range(1, seats_per_row + 1):
        seat_number = f"{row}{seat}"
        print(f"[{seat_number}]", end=" ")
    print()

# Output:
# Row 1: [11] [12] [13] [14] [15]
# Row 2: [21] [22] [23] [24] [25]
# Row 3: [31] [32] [33] [34] [35]
```

### Performance Considerations

**Important**: Nested loops multiply iterations!

```python
# Single loop: 100 iterations
for i in range(100):
    pass  # Some operation

# Nested loop: 100 × 100 = 10,000 iterations!
for i in range(100):
    for j in range(100):
        pass  # Some operation

# Triple nested: 100 × 100 × 100 = 1,000,000 iterations!
for i in range(100):
    for j in range(100):
        for k in range(100):
            pass  # Some operation

# Be careful with large ranges in nested loops!
# They can slow your program significantly
```

---

## Part 9: Common Mistakes and How to Avoid Them

### Mistake #1: Forgetting Indentation

```python
# WRONG: No indentation
# if age >= 18:
# print("Adult")  # IndentationError!

# CORRECT:
if age >= 18:
    print("Adult")

# WRONG: Inconsistent indentation
# if age >= 18:
#     print("Adult")
#   print("Can vote")  # IndentationError! (different spacing)

# CORRECT: Use same indentation (4 spaces is standard)
if age >= 18:
    print("Adult")
    print("Can vote")

# WRONG: Mixing tabs and spaces
# if age >= 18:
#     print("Adult")  # 4 spaces
# 	print("Can vote")  # Tab
# This can cause weird errors!

# CORRECT: Use spaces only (most editors convert Tab to 4 spaces)
```

### Mistake #2: Using = Instead of ==

```python
# WRONG: Assignment in condition
x = 10
# if x = 10:  # SyntaxError!
#     print("x is 10")

# CORRECT: Comparison
if x == 10:
    print("x is 10")

# This is especially tricky because some languages allow it!
# In Python, it's an error (which is good!)
```

### Mistake #3: Creating Infinite Loops

```python
# WRONG: Forgot to update counter
# counter = 0
# while counter < 5:
#     print(counter)
#     # Missing: counter += 1
#     # This prints 0 forever!

# CORRECT:
counter = 0
while counter < 5:
    print(counter)
    counter += 1  # Don't forget this!

# WRONG: Condition never becomes False
# while True:
#     print("Running...")
#     # No break statement, runs forever!

# CORRECT: Always have exit condition
while True:
    user_input = input("Enter 'quit' to exit: ")
    if user_input == "quit":
        break  # Exit the loop
```

### Mistake #4: Off-by-One Errors

```python
# WRONG: Forgetting range stops at n-1
# Want to print 1 to 10:
for i in range(1, 10):
    print(i)
# Output: 1, 2, 3, 4, 5, 6, 7, 8, 9 (missing 10!)

# CORRECT:
for i in range(1, 11):  # Need 11, not 10!
    print(i)
# Output: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

# WRONG: Starting from wrong index
numbers = [10, 20, 30, 40, 50]
# Want to print all except first:
for i in range(len(numbers)):  # Oops, starts at 0
    print(numbers[i])

# CORRECT:
for i in range(1, len(numbers)):  # Start from index 1
    print(numbers[i])
```

### Mistake #5: Confusing and/or Logic

```python
# WRONG: Using 'or' when you mean 'and'
age = 25
has_license = True

# Want: must be adult AND have license
if age >= 18 or has_license:  # WRONG! Only needs one
    print("Can drive")
# This allows driving if just have license, even if under 18!

# CORRECT:
if age >= 18 and has_license:  # Both required
    print("Can drive")

# WRONG: Checking multiple values incorrectly
x = 5
# if x == 3 or 4 or 5:  # WRONG! Always True
#     print("x is 3, 4, or 5")
# This is actually: (x == 3) or (4) or (5)
# 4 and 5 are "truthy" values, so always True!

# CORRECT:
if x == 3 or x == 4 or x == 5:
    print("x is 3, 4, or 5")

# Or even better:
if x in [3, 4, 5]:
    print("x is 3, 4, or 5")
```

---

## Part 10: Detailed Worked Examples

### Example 1: Grade Calculator

**Problem**: Create a complete grading system that takes multiple test scores and calculates letter grade, GPA, and pass/fail status.

```python
print("=== Grade Calculator ===")

# Get number of tests
num_tests = int(input("How many tests? "))

# Collect scores
total = 0
print(f"\nEnter {num_tests} test scores:")

for i in range(num_tests):
    score = float(input(f"  Test {i + 1}: "))
    total += score

# Calculate average
average = total / num_tests

# Determine letter grade
if average >= 90:
    letter_grade = "A"
    gpa = 4.0
elif average >= 80:
    letter_grade = "B"
    gpa = 3.0
elif average >= 70:
    letter_grade = "C"
    gpa = 2.0
elif average >= 60:
    letter_grade = "D"
    gpa = 1.0
else:
    letter_grade = "F"
    gpa = 0.0

# Determine pass/fail
if average >= 60:
    status = "PASS"
else:
    status = "FAIL"

# Display results
print("\n" + "=" * 40)
print("GRADE REPORT")
print("=" * 40)
print(f"Total Points:    {total:.1f}")
print(f"Average Score:   {average:.1f}%")
print(f"Letter Grade:    {letter_grade}")
print(f"GPA:             {gpa:.1f}")
print(f"Status:          {status}")
print("=" * 40)

# Additional feedback
if letter_grade == "A":
    print("🎉 Excellent work!")
elif letter_grade in ["B", "C"]:
    print("👍 Good job! Keep it up!")
elif letter_grade == "D":
    print("⚠ Passing, but needs improvement")
else:
    print("❌ Failed. Please see instructor")
```

**What this teaches**:
- Multiple input collection with loops
- Running totals
- Complex conditional logic
- Multi-level elif chains
- Professional output formatting

---

### Example 2: Password Validator

**Problem**: Create a password validator that checks multiple criteria with detailed feedback.

```python
print("=== Password Validator ===")
print("Requirements:")
print("  - At least 8 characters")
print("  - Contains uppercase letter")
print("  - Contains lowercase letter")
print("  - Contains number")
print("  - Contains special character (!@#$%)")
print()

password = input("Enter password to validate: ")

# Check length
has_min_length = len(password) >= 8

# Check for uppercase
has_uppercase = False
for char in password:
    if char.isupper():
        has_uppercase = True
        break

# Check for lowercase
has_lowercase = False
for char in password:
    if char.islower():
        has_lowercase = True
        break

# Check for digit
has_digit = False
for char in password:
    if char.isdigit():
        has_digit = True
        break

# Check for special character
special_chars = "!@#$%"
has_special = False
for char in password:
    if char in special_chars:
        has_special = True
        break

# Display results
print("\n" + "=" * 40)
print("VALIDATION RESULTS")
print("=" * 40)

# Check each requirement
print(f"{'✓' if has_min_length else '✗'} Length (8+ chars): {'PASS' if has_min_length else 'FAIL'}")
print(f"{'✓' if has_uppercase else '✗'} Uppercase letter: {'PASS' if has_uppercase else 'FAIL'}")
print(f"{'✓' if has_lowercase else '✗'} Lowercase letter: {'PASS' if has_lowercase else 'FAIL'}")
print(f"{'✓' if has_digit else '✗'} Number: {'PASS' if has_digit else 'FAIL'}")
print(f"{'✓' if has_special else '✗'} Special char (!@#$%): {'PASS' if has_special else 'FAIL'}")
print("=" * 40)

# Overall result
all_criteria_met = (has_min_length and has_uppercase and 
                   has_lowercase and has_digit and has_special)

if all_criteria_met:
    print("✓ Password is STRONG")
else:
    print("✗ Password is WEAK")
    print("Please fix the failed requirements")
```

**What this teaches**:
- Multiple boolean flags
- String methods (.isupper(), .islower(), .isdigit())
- Looping through characters
- Complex boolean logic
- User feedback with symbols

---

### Example 3: Guess the Number Game

**Problem**: Interactive number guessing game with hints and limited attempts.

```python
import random  # We'll learn more about this later

print("=== Guess the Number! ===")
print("I'm thinking of a number between 1 and 100")

# Game setup
secret_number = random.randint(1, 100)
max_attempts = 7
attempts = 0
guessed_correctly = False

# Game loop
while attempts < max_attempts and not guessed_correctly:
    attempts += 1
    remaining = max_attempts - attempts
    
    print(f"\nAttempt {attempts}/{max_attempts}")
    guess = int(input("Your guess: "))
    
    if guess == secret_number:
        guessed_correctly = True
        print(f"🎉 Correct! You guessed it in {attempts} attempts!")
    elif guess < secret_number:
        print("📈 Too low!")
        if remaining > 0:
            print(f"   Try a higher number ({remaining} attempts left)")
    else:  # guess > secret_number
        print("📉 Too high!")
        if remaining > 0:
            print(f"   Try a lower number ({remaining} attempts left)")
    
    # Give additional hints
    if not guessed_correctly and attempts == 3:
        if secret_number % 2 == 0:
            print("💡 Hint: The number is even")
        else:
            print("💡 Hint: The number is odd")

# Game over
if not guessed_correctly:
    print(f"\n❌ Game Over! The number was {secret_number}")
else:
    # Rating based on attempts
    if attempts <= 3:
        print("⭐⭐⭐ Amazing! You're a guessing master!")
    elif attempts <= 5:
        print("⭐⭐ Great job!")
    else:
        print("⭐ You got it!")
```

**What this teaches**:
- While loop with multiple conditions
- Boolean flags
- Nested conditionals
- User interaction
- Game logic patterns

---

### Example 4: Multiplication Table Generator

**Problem**: Generate and display a formatted multiplication table.

```python
print("=== Multiplication Table Generator ===")

# Get table size
size = int(input("Table size (e.g., 10 for 10x10): "))

# Print header row
print("\n" + " " * 4, end="")  # Space for row labels
for i in range(1, size + 1):
    print(f"{i:4}", end="")
print()  # New line

# Print separator
print("   " + "-" * (size * 4 + 1))

# Print table
for i in range(1, size + 1):
    # Print row label
    print(f"{i:3}|", end="")
    
    # Print row values
    for j in range(1, size + 1):
        product = i * j
        print(f"{product:4}", end="")
    
    print()  # New line after each row

# Example output for size=5:
#        1   2   3   4   5
#    ---------------------
#  1|   1   2   3   4   5
#  2|   2   4   6   8  10
#  3|   3   6   9  12  15
#  4|   4   8  12  16  20
#  5|   5  10  15  20  25
```

**What this teaches**:
- Nested for loops
- String formatting with field width
- Building tables and grids
- Aligned output

---

### Example 5: Prime Number Checker

**Problem**: Determine if a number is prime and show the checking process.

```python
print("=== Prime Number Checker ===")

number = int(input("Enter a number: "))

# Handle special cases
if number < 2:
    print(f"{number} is not prime (must be >= 2)")
elif number == 2:
    print(f"{number} is prime (the only even prime)")
elif number % 2 == 0:
    print(f"{number} is not prime (divisible by 2)")
else:
    # Check for divisors from 3 to sqrt(number)
    is_prime = True
    print(f"\nChecking divisors for {number}:")
    
    # We only need to check up to square root
    limit = int(number ** 0.5) + 1
    
    for divisor in range(3, limit, 2):  # Only odd numbers
        print(f"  Testing {divisor}...", end=" ")
        
        if number % divisor == 0:
            print(f"✗ {number} ÷ {divisor} = {number // divisor}")
            is_prime = False
            break
        else:
            print("✓")
    
    # Display result
    print()
    if is_prime:
        print(f"✓ {number} is PRIME!")
    else:
        print(f"✗ {number} is NOT prime")
```

**What this teaches**:
- Complex conditional logic
- Optimization (checking only to sqrt)
- Breaking out of loops early
- Mathematical algorithms

---

### Example 6: ATM Withdrawal System

**Problem**: Simulate an ATM with balance tracking and validation.

```python
print("=== ATM System ===")

# Initial setup
balance = 1000.00
pin = "1234"

# PIN verification
attempts = 0
max_attempts = 3
authenticated = False

while attempts < max_attempts and not authenticated:
    user_pin = input("Enter PIN: ")
    
    if user_pin == pin:
        authenticated = True
        print("✓ PIN accepted\n")
    else:
        attempts += 1
        remaining = max_attempts - attempts
        if remaining > 0:
            print(f"✗ Incorrect PIN. {remaining} attempts remaining\n")
        else:
            print("✗ Card locked. Contact your bank.")

# Main menu (only if authenticated)
if authenticated:
    continue_transactions = True
    
    while continue_transactions:
        print("=" * 30)
        print(f"Current Balance: ${balance:.2f}")
        print("=" * 30)
        print("1. Withdraw")
        print("2. Deposit")
        print("3. Check Balance")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ")
        
        if choice == "1":
            # Withdrawal
            amount = float(input("Withdrawal amount: $"))
            
            if amount <= 0:
                print("✗ Amount must be positive")
            elif amount > balance:
                print(f"✗ Insufficient funds (available: ${balance:.2f})")
            elif amount % 20 != 0:
                print("✗ Amount must be multiple of $20")
            else:
                balance -= amount
                print(f"✓ Dispensing ${amount:.2f}")
                print(f"  New balance: ${balance:.2f}")
        
        elif choice == "2":
            # Deposit
            amount = float(input("Deposit amount: $"))
            
            if amount <= 0:
                print("✗ Amount must be positive")
            else:
                balance += amount
                print(f"✓ Deposited ${amount:.2f}")
                print(f"  New balance: ${balance:.2f}")
        
        elif choice == "3":
            # Check balance
            print(f"Current balance: ${balance:.2f}")
        
        elif choice == "4":
            # Exit
            print("Thank you for using our ATM!")
            continue_transactions = False
        
        else:
            print("✗ Invalid option")
        
        print()  # Blank line for readability

What this teaches:

Menu-driven programs
State management (balance tracking)
Input validation
Multiple nested conditions
Real-world application logic


Example 7: Pattern Printer
Problem: Create various patterns using nested loops.
pythonprint("=== Pattern Printer ===")
print("1. Right Triangle")
print("2. Inverted Triangle")
print("3. Pyramid")
print("4. Diamond")
print("5. Number Pattern")

choice = input("\nSelect pattern (1-5): ")
size = int(input("Enter size: "))

print()

if choice == "1":
    # Right Triangle
    for i in range(1, size + 1):
        for j in range(i):
            print("*", end=" ")
        print()

elif choice == "2":
    # Inverted Triangle
    for i in range(size, 0, -1):
        for j in range(i):
            print("*", end=" ")
        print()

elif choice == "3":
    # Pyramid
    for i in range(size):
        # Print spaces
        for j in range(size - i - 1):
            print(" ", end=" ")
        # Print stars
        for j in range(2 * i + 1):
            print("*", end=" ")
        print()

elif choice == "4":
    # Diamond (pyramid + inverted)
    # Top half (pyramid)
    for i in range(size):
        for j in range(size - i - 1):
            print(" ", end=" ")
        for j in range(2 * i + 1):
            print("*", end=" ")
        print()
    
    # Bottom half (inverted pyramid)
    for i in range(size - 2, -1, -1):
        for j in range(size - i - 1):
            print(" ", end=" ")
        for j in range(2 * i + 1):
            print("*", end=" ")
        print()

elif choice == "5":
    # Number Pattern
    for i in range(1, size + 1):
        for j in range(1, i + 1):
            print(j, end=" ")
        print()

else:
    print("Invalid choice")
What this teaches:

Complex nested loops
Pattern recognition
Menu systems
Visual output formatting


Part 11: Practice Problems
Easy Problems (Problems 1-7)
Problem 1: Write a program that asks for a number and prints whether it's even or odd.
Problem 2: Create a program that asks for age and prints "Child" (0-12), "Teenager" (13-19), or "Adult" (20+).
Problem 3: Print all numbers from 1 to 20 using a for loop.
Problem 4: Ask the user for a string and print each character on a separate line.
Problem 5: Print all even numbers from 2 to 30 using a while loop.
Problem 6: Ask for a number N and print all numbers from 1 to N.
Problem 7: Create a simple menu that asks the user to choose between options A, B, or C, and prints their choice.

Medium Problems (Problems 8-17)
Problem 8: Write a program that asks for a password and keeps asking until they enter "secret123".
Problem 9: Calculate the sum of all numbers from 1 to 100 using a loop.
Problem 10: Ask for a word and count how many vowels (a, e, i, o, u) it contains.
Problem 11: Print a countdown from 10 to 1, then print "Blast off!".
Problem 12: Ask for a number and print its multiplication table (1 through 10).
Problem 13: Create a simple calculator that keeps running until the user types "quit". For each iteration, ask for two numbers and an operation (+, -, *, /).
Problem 14: Print all numbers from 1 to 50, but print "Fizz" for multiples of 3, "Buzz" for multiples of 5, and "FizzBuzz" for multiples of both.
Problem 15: Ask the user for 5 numbers and find the maximum value.
Problem 16: Print a rectangle of stars (*) where the user specifies width and height.
Problem 17: Ask for a positive number and keep asking until they provide one (input validation).

Hard Problems (Problems 18-27)
Problem 18: Create a number guessing game where the computer picks a random number (1-50) and the user has 7 attempts to guess it. Give "higher" or "lower" hints.
Problem 19: Calculate the factorial of a number (e.g., 5! = 5×4×3×2×1 = 120).
Problem 20: Check if a string is a palindrome (reads the same forwards and backwards, like "racecar").
Problem 21: Print the Fibonacci sequence up to N terms (each number is the sum of the previous two: 0, 1, 1, 2, 3, 5, 8, 13...).
Problem 22: Create a grade book program that asks for multiple student names and scores, then displays all students with their letter grades.
Problem 23: Simulate a simple bank account: Start with $1000, allow deposits and withdrawals with validation, and a menu to check balance or exit.
Problem 24: Print a chess board pattern using nested loops (alternating # and spaces in an 8×8 grid).
Problem 25: Find all prime numbers between 1 and 100.
Problem 26: Create a shopping cart system: allow user to add items with prices, remove items, and calculate total. Keep running until they choose to "checkout".
Problem 27: Print Floyd's Triangle (rows with consecutive numbers):
1
2 3
4 5 6
7 8 9 10

Challenge Problems (Problems 28-35)
Problem 28: Create a rock-paper-scissors game where the user plays against the computer. Best of 5 rounds, keep score, and declare a winner.
Problem 29: Calculate compound interest: Ask for principal, rate, and years. For each year, show the balance at end of year.
Problem 30: Create a "password strength meter" that analyzes a password and gives it a score based on: length (1 point per char), has uppercase (5 points), has lowercase (5 points), has number (5 points), has special char (10 points). Display rating: Weak (0-15), Medium (16-25), Strong (26+).
Problem 31: Print Pascal's Triangle up to N rows:
    1
   1 1
  1 2 1
 1 3 3 1
1 4 6 4 1
Problem 32: Create a simple encryption program using Caesar cipher: shift each letter by N positions. For example, with shift=3, 'A' becomes 'D', 'B' becomes 'E', etc.
Problem 33: Simulate a dice rolling game: Player rolls two dice, trying to reach exactly 50 points. If they roll doubles, roll again. If total goes over 50, they lose.
Problem 34: Create a "text analyzer" that takes a sentence and reports: word count, average word length, longest word, count of each letter (case-insensitive).
Problem 35: Build a "quiz game": Store 10 questions with answers. Ask questions one by one, keep score, give percentage at end, and allow review of missed questions.

Part 12: Mini-Project - Interactive Menu System
Objective: Build a complete text-based adventure game with multiple choices and outcomes.
pythonprint("=" * 50)
print("    THE ADVENTURE BEGINS")
print("=" * 50)

# Get player name
player_name = input("\nWhat is your name, adventurer? ")
print(f"\nWelcome, {player_name}!")

# Initialize game state
health = 100
gold = 50
has_sword = False
has_key = False
game_over = False

# Main game loop
while not game_over:
    print("\n" + "=" * 50)
    print(f"Health: {health} | Gold: {gold}")
    print("=" * 50)
    
    print("\nYou stand at a crossroads.")
    print("1. Enter the dark forest")
    print("2. Visit the village shop")
    print("3. Explore the ancient cave")
    print("4. Rest at the inn")
    print("5. Check inventory")
    print("6. Quit game")
    
    choice = input("\nWhat will you do? (1-6): ")
    
    if choice == "1":
        # Forest encounter
        print("\n🌲 You enter the dark forest...")
        print("A wild goblin appears!")
        
        if has_sword:
            print("You draw your sword and fight!")
            print("You defeated the goblin!")
            gold += 20
            print(f"You found 20 gold! (Total: {gold})")
        else:
            print("You have no weapon! The goblin attacks!")
            health -= 30
            print(f"You lost 30 health! (Remaining: {health})")
            
            if health <= 0:
                print("\n💀 You have been defeated...")
                print("GAME OVER")
                game_over = True
    
    elif choice == "2":
        # Village shop
        print("\n🏪 Welcome to the village shop!")
        print("1. Buy sword (50 gold)")
        print("2. Buy health potion (30 gold)")
        print("3. Buy key (100 gold)")
        print("4. Leave shop")
        
        shop_choice = input("\nWhat would you like? (1-4): ")
        
        if shop_choice == "1":
            if gold >= 50:
                if has_sword:
                    print("You already have a sword!")
                else:
                    gold -= 50
                    has_sword = True
                    print("⚔️ You bought a sword!")
            else:
                print("Not enough gold!")
        
        elif shop_choice == "2":
            if gold >= 30:
                gold -= 30
                health = min(100, health + 50)
                print(f"🧪 You bought a health potion! Health: {health}")
            else:
                print("Not enough gold!")
        
        elif shop_choice == "3":
            if gold >= 100:
                if has_key:
                    print("You already have the key!")
                else:
                    gold -= 100
                    has_key = True
                    print("🔑 You bought the mysterious key!")
            else:
                print("Not enough gold!")
        
        elif shop_choice == "4":
            print("You leave the shop.")
        else:
            print("Invalid choice!")
    
    elif choice == "3":
        # Ancient cave
        print("\n🏔️ You approach the ancient cave...")
        
        if has_key:
            print("You use the key to unlock the treasure room!")
            print("💎 You found the legendary treasure!")
            gold += 500
            print(f"You gained 500 gold! (Total: {gold})")
            print("\n🎉 CONGRATULATIONS! You won the game!")
            game_over = True
        else:
            print("The entrance is locked. You need a key.")
            print("A dragon guards the entrance!")
            
            if has_sword:
                print("You bravely fight the dragon!")
                health -= 40
                print(f"You lost 40 health! (Remaining: {health})")
                
                if health <= 0:
                    print("\n💀 The dragon defeated you...")
                    print("GAME OVER")
                    game_over = True
                else:
                    print("You managed to escape!")
            else:
                print("You flee from the dragon!")
    
    elif choice == "4":
        # Rest at inn
        print("\n🏨 You rest at the inn...")
        
        if gold >= 20:
            gold -= 20
            health = 100
            print(f"Cost: 20 gold (Remaining: {gold})")
            print(f"You feel refreshed! Health restored to 100!")
        else:
            print("Not enough gold for the inn (costs 20 gold)")
    
    elif choice == "5":
        # Check inventory
        print("\n🎒 INVENTORY:")
        print(f"Health: {health}/100")
        print(f"Gold: {gold}")
        print(f"Sword: {'Yes ⚔️' if has_sword else 'No'}")
        print(f"Key: {'Yes 🔑' if has_key else 'No'}")
    
    elif choice == "6":
        # Quit
        print(f"\nThanks for playing, {player_name}!")
        print(f"Final stats - Health: {health}, Gold: {gold}")
        game_over = True
    
    else:
        print("\n❌ Invalid choice! Please choose 1-6.")

print("\n" + "=" * 50)
print("    THE END")
print("=" * 50)
What This Project Teaches:

Complex state management (multiple variables)
Nested conditionals and menus
Game loop structure
Boolean flags for inventory
Win/lose conditions
Input validation
Interactive storytelling

Extensions You Can Add:

More locations and enemies
Experience points and leveling system
Multiple weapons and items
Save/load game state
Random encounters
Character classes (warrior, mage, etc.)


Part 13: Key Takeaways
What You've Learned:
Conditional Logic:

if - execute code when condition is True
else - alternative path when condition is False
elif - check multiple conditions in sequence
Nested conditions for complex decision making

Comparison Operators:

== equal, != not equal
< less than, > greater than
<= less or equal, >= greater or equal
Chaining: 10 <= x <= 20

Logical Operators:

and - all conditions must be True
or - at least one condition must be True
not - reverses boolean value
Short-circuit evaluation

Loops:

while - repeat while condition is True
for - iterate over sequences
range(start, stop, step) - generate number sequences
Loop over strings, lists, and other iterables

Loop Control:

break - exit loop immediately
continue - skip to next iteration
pass - do nothing (placeholder)

Best Practices:

Always indent consistently (4 spaces)
Use descriptive variable names
Check for edge cases
Validate user input
Avoid infinite loops
Use for when iterations are known
Use while for conditional repetition
Add comments for complex logic

Common Patterns:

Input validation with while
Counting and accumulation
Menu-driven programs
Search and find operations
Pattern printing with nested loops


Appendix: Control Flow Reference
Conditional Syntax
python# if statement
if condition:
    code

# if-else
if condition:
    code
else:
    code

# if-elif-else
if condition1:
    code
elif condition2:
    code
elif condition3:
    code
else:
    code

# Nested
if condition1:
    if condition2:
        code
Loop Syntax
python# while loop
while condition:
    code
    update_condition

# for loop with range
for i in range(stop):
    code

for i in range(start, stop):
    code

for i in range(start, stop, step):
    code

# for loop with sequence
for item in sequence:
    code
Operators
python# Comparison
==  !=  <  >  <=  >=

# Logical
and  or  not

# Loop Control
break     # exit loop
continue  # skip to next iteration
pass      # do nothing
Common Patterns
python# Counting pattern
count = 0
for item in sequence:
    if condition:
        count += 1

# Accumulation pattern
total = 0
for num in numbers:
    total += num

# Search pattern
found = False
for item in sequence:
    if item == target:
        found = True
        break

# Input validation
while True:
    value = input("Enter: ")
    if valid(value):
        break
    print("Invalid!")

Congratulations! You've completed Chapter 2. You now know how to make programs that think (conditionals) and repeat (loops). These are fundamental building blocks of programming.
Next Chapter Preview: In Chapter 3, we'll dive deep into Lists - Python's most versatile data structure. You'll learn how to store, manipulate, and process collections of data efficiently.
