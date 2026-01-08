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
for
