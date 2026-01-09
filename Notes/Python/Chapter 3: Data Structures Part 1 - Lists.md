# Chapter 3: Data Structures Part 1 - Lists
## Part 1: Foundation (Parts 1-6)

---

## Part 1: The Problem We're Solving - Why Lists?

### The Real-World Problem

Without lists, managing collections is impossible:

```python
# The nightmare approach (DON'T DO THIS!)
song1 = "Bohemian Rhapsody"
song2 = "Stairway to Heaven"
song3 = "Hotel California"
# What if you have 1000 songs?
```

**Lists solve**: Organization, scalability, flexibility, iteration, maintaining order.

```python
# With lists - elegant and scalable
songs = ["Bohemian Rhapsody", "Stairway to Heaven", "Hotel California"]
songs.append("Imagine")  # Easy to add!
```

---

## Part 2: Creating and Accessing Lists

### Creating Lists

```python
# Method 1: Literal syntax
fruits = ["apple", "banana", "cherry"]

# Method 2: Constructor
numbers = list(range(5))  # [0, 1, 2, 3, 4]
letters = list("hello")   # ['h', 'e', 'l', 'l', 'o']

# Method 3: Empty list
cart = []
cart.append("milk")

# Mixed types
student = ["Alice", 20, 3.8, True]
```

### Indexing - Zero-Based!

```python
fruits = ["apple", "banana", "cherry", "date"]
#         index 0   index 1    index 2   index 3

first = fruits[0]   # "apple"
third = fruits[2]   # "cherry"

# Negative indexing (from end)
last = fruits[-1]      # "date"
second_last = fruits[-2]  # "cherry"
```

### Index Errors

```python
numbers = [10, 20, 30]

# WRONG
# print(numbers[5])  # IndexError!

# CORRECT
if len(numbers) > 5:
    print(numbers[5])
```

---

## Part 3: List Slicing

### Basic Slicing

```python
# Syntax: list[start:stop]  (excludes stop)
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

subset = numbers[2:5]     # [2, 3, 4]
first_three = numbers[:3]  # [0, 1, 2]
from_five = numbers[5:]    # [5, 6, 7, 8, 9]
copy = numbers[:]          # Full copy
```

### Slicing with Step

```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

evens = numbers[::2]       # [0, 2, 4, 6, 8]
odds = numbers[1::2]       # [1, 3, 5, 7, 9]
reversed_list = numbers[::-1]  # [9, 8, 7, ..., 0]
```

### Slicing Tricks

```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

last_three = numbers[-3:]      # [7, 8, 9]
all_but_last = numbers[:-2]    # [0, 1, 2, 3, 4, 5, 6, 7]
middle = numbers[2:-2]         # [2, 3, 4, 5, 6, 7]
```

---

## Part 4: Modifying Lists

### Changing Elements

```python
fruits = ["apple", "banana", "cherry"]
fruits[1] = "blueberry"
# ["apple", "blueberry", "cherry"]

# Change multiple with slicing
numbers = [1, 2, 3, 4, 5]
numbers[1:4] = [20, 30, 40]
# [1, 20, 30, 40, 5]
```

### Adding Elements

**append()** - Add to end:
```python
fruits = ["apple", "banana"]
fruits.append("cherry")
# ["apple", "banana", "cherry"]
```

**insert()** - Add at position:
```python
fruits = ["apple", "cherry"]
fruits.insert(1, "banana")
# ["apple", "banana", "cherry"]
```

**extend()** - Add multiple:
```python
fruits = ["apple", "banana"]
fruits.extend(["cherry", "date"])
# ["apple", "banana", "cherry", "date"]

# Compare with append
list1 = [1, 2]
list1.append([3, 4])     # [1, 2, [3, 4]] - nested!
list1.extend([3, 4])     # [1, 2, 3, 4] - flat
```

### Removing Elements

**remove()** - Remove by value:
```python
fruits = ["apple", "banana", "cherry"]
fruits.remove("banana")
# ["apple", "cherry"]
```

**pop()** - Remove by index:
```python
numbers = [10, 20, 30, 40]
last = numbers.pop()      # Returns 40, list now [10, 20, 30]
second = numbers.pop(1)   # Returns 20, list now [10, 30]
```

**del** - Delete by index:
```python
numbers = [1, 2, 3, 4, 5]
del numbers[2]    # [1, 2, 4, 5]
del numbers[1:3]  # [1, 5]
```

**clear()** - Remove all:
```python
fruits = ["apple", "banana"]
fruits.clear()  # []
```

---

## Part 5: List Methods

### sort() and sorted()

```python
# sort() - modifies in place
numbers = [3, 1, 4, 1, 5]
numbers.sort()
print(numbers)  # [1, 1, 3, 4, 5]

# Descending
numbers.sort(reverse=True)  # [5, 4, 3, 1, 1]

# sorted() - returns new list
original = [3, 1, 4]
sorted_list = sorted(original)
print(original)      # [3, 1, 4] - unchanged
print(sorted_list)   # [1, 3, 4]
```

### reverse() and reversed()

```python
# reverse() - in place
numbers = [1, 2, 3]
numbers.reverse()  # [3, 2, 1]

# reversed() - returns iterator
original = [1, 2, 3]
rev = list(reversed(original))
print(original)  # [1, 2, 3] - unchanged
print(rev)       # [3, 2, 1]
```

### count() and index()

```python
numbers = [1, 2, 3, 2, 4, 2]

count = numbers.count(2)  # 3
position = numbers.index(2)  # 1 (first occurrence)

# Safe search
if 5 in numbers:
    pos = numbers.index(5)
else:
    print("Not found")
```

### copy()

```python
original = [1, 2, 3]
copy = original.copy()

copy[0] = 999
print(original)  # [1, 2, 3] - unchanged
print(copy)      # [999, 2, 3]
```

---

## Part 6: List Operations

### Concatenation

```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2  # [1, 2, 3, 4, 5, 6]
```

### Repetition

```python
zeros = [0] * 5  # [0, 0, 0, 0, 0]
pattern = [1, 2] * 3  # [1, 2, 1, 2, 1, 2]
```

### Membership Testing

```python
fruits = ["apple", "banana", "cherry"]

if "apple" in fruits:
    print("Found!")

if "grape" not in fruits:
    print("Not found!")
```

### Utility Functions

```python
numbers = [23, 45, 12, 67, 89]

length = len(numbers)    # 5
minimum = min(numbers)   # 12
maximum = max(numbers)   # 89
total = sum(numbers)     # 236
average = sum(numbers) / len(numbers)  # 47.2
```

---

**End of Part 1: Foundation**

# Chapter 3: Data Structures Part 1 - Lists
## Part 2: Advanced Concepts (Parts 7-10)

---

## Part 7: Iterating Over Lists

### Basic Iteration

```python
fruits = ["apple", "banana", "cherry"]

for fruit in fruits:
    print(fruit)
```

### Iterating with Indices

```python
fruits = ["apple", "banana", "cherry"]

for i in range(len(fruits)):
    print(f"Index {i}: {fruits[i]}")

# Modify in place
numbers = [1, 2, 3, 4, 5]
for i in range(len(numbers)):
    numbers[i] = numbers[i] * 2
# [2, 4, 6, 8, 10]
```

### enumerate() - Best Practice

```python
fruits = ["apple", "banana", "cherry"]

for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# Start from 1
for num, fruit in enumerate(fruits, start=1):
    print(f"#{num}: {fruit}")

# Real-world: menu
menu = ["Pizza", "Burger", "Salad"]
for num, item in enumerate(menu, start=1):
    print(f"{num}. {item}")
```

### zip() - Multiple Lists

```python
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
cities = ["NYC", "LA", "Chicago"]

for name, age, city in zip(names, ages, cities):
    print(f"{name} is {age} and lives in {city}")

# Stops at shortest list
list1 = [1, 2, 3, 4, 5]
list2 = ['a', 'b', 'c']
for num, letter in zip(list1, list2):
    print(num, letter)
# Only 3 pairs printed
```

---

## Part 8: List Comprehensions

### What Are They?

**Old way**:
```python
numbers = [1, 2, 3, 4, 5]
squares = []
for n in numbers:
    squares.append(n ** 2)
```

**New way**:
```python
squares = [n ** 2 for n in numbers]
```

### Basic Syntax

```python
# [expression for item in iterable]

numbers = [1, 2, 3, 4, 5]
doubled = [n * 2 for n in numbers]  # [2, 4, 6, 8, 10]

words = ["hello", "world"]
upper = [w.upper() for w in words]  # ['HELLO', 'WORLD']

fruits = ["apple", "banana"]
lengths = [len(f) for f in fruits]  # [5, 6]

# Generate from range
squares = [x**2 for x in range(10)]
```

### With Conditionals

```python
# Filter with if
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = [n for n in numbers if n % 2 == 0]
# [2, 4, 6, 8, 10]

# Filter and transform
numbers = [1, 2, 3, 4, 5, 6]
even_squares = [n**2 for n in numbers if n % 2 == 0]
# [4, 16, 36]

words = ["hi", "hello", "hey", "python"]
long_words = [w.upper() for w in words if len(w) > 4]
# ['HELLO', 'PYTHON']
```

### if-else (Ternary)

```python
# [true_expr if condition else false_expr for item in iterable]

numbers = [1, 2, 3, 4, 5]
labels = ["even" if n % 2 == 0 else "odd" for n in numbers]
# ['odd', 'even', 'odd', 'even', 'odd']

values = [5, -3, 8, -1]
absolutes = [v if v >= 0 else -v for v in values]
# [5, 3, 8, 1]
```

### Nested Comprehensions

```python
# Create matrix
matrix = [[i*3 + j for j in range(3)] for i in range(3)]
# [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

# Flatten nested list
nested = [[1, 2], [3, 4], [5, 6]]
flat = [num for row in nested for num in row]
# [1, 2, 3, 4, 5, 6]

# All pairs
list1 = [1, 2, 3]
list2 = ['a', 'b']
pairs = [(x, y) for x in list1 for y in list2]
# [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'), (3, 'a'), (3, 'b')]
```

### When to Use

**Use comprehensions**:
- Simple transformations
- Filtering
- One-liner makes sense

**Use loops**:
- Complex logic
- Multiple statements
- Readability suffers

```python
# GOOD
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]

# BAD (too complex)
# result = [f(x) if v(x) and c(x) else d(x) if x > 0 else fb(x) for x in data]

# BETTER - use regular loop
result = []
for x in data:
    if v(x) and c(x):
        result.append(f(x))
    elif x > 0:
        result.append(d(x))
    else:
        result.append(fb(x))
```

---

## Part 9: Nested Lists

### Lists Within Lists

```python
# 2D data
students = [
    ["Alice", 20, "A"],
    ["Bob", 22, "B"],
    ["Charlie", 21, "A"]
]

# 3D data
school = [
    [["Alice", 20], ["Bob", 22]],    # Class 1
    [["Charlie", 21], ["David", 23]]  # Class 2
]
```

### Accessing Nested Elements

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

first_row = matrix[0]     # [1, 2, 3]
element = matrix[1][2]    # 6 (row 1, col 2)

# Modify
matrix[0][0] = 99
# [[99, 2, 3], [4, 5, 6], [7, 8, 9]]
```

### 2D Lists (Matrices)

```python
# Tic-tac-toe board
board = [
    ['-', '-', '-'],
    ['-', '-', '-'],
    ['-', '-', '-']
]

board[1][1] = 'X'

# Display
for row in board:
    print(' '.join(row))
# - - -
# - X -
# - - -

# Student grades
grades = [
    ["Alice", 85, 92, 78],
    ["Bob", 78, 88, 91],
    ["Charlie", 92, 85, 89]
]

# Bob's English grade (index 1, then 2)
bobs_english = grades[1][2]  # 88

# Alice's average
alice_scores = grades[0][1:]  # [85, 92, 78]
alice_avg = sum(alice_scores) / len(alice_scores)
```

### Iterating Nested Lists

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Print all elements
for row in matrix:
    for element in row:
        print(element, end=' ')
    print()

# Sum all
total = sum([num for row in matrix for num in row])
# or
total = 0
for row in matrix:
    for num in row:
        total += num

# Find element
target = 5
for i, row in enumerate(matrix):
    for j, element in enumerate(row):
        if element == target:
            print(f"Found at ({i}, {j})")
            break
```

---

## Part 10: Common List Patterns

### Finding Elements

```python
# Pattern 1: Find first match
numbers = [10, 25, 30, 45, 50]
target = 30

if target in numbers:
    idx = numbers.index(target)
    print(f"Found at {idx}")

# Pattern 2: Find all matches
numbers = [1, 5, 3, 5, 7, 5]
indices = [i for i, n in enumerate(numbers) if n == 5]
print(indices)  # [1, 3, 5]

# Pattern 3: Find max with position
scores = [85, 92, 78, 95, 88]
max_score = max(scores)
max_idx = scores.index(max_score)
print(f"Highest: {max_score} at position {max_idx}")
```

### Filtering Lists

```python
# Pattern 1: Simple filter
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = [n for n in numbers if n % 2 == 0]

# Pattern 2: Multiple conditions
ages = [15, 22, 17, 30, 12, 25]
adults = [age for age in ages if 18 <= age < 65]

# Pattern 3: String filtering
words = ["apple", "banana", "apricot", "cherry"]
a_words = [w for w in words if w.startswith('a')]
```

### Transforming Lists

```python
# Pattern 1: Apply function
prices = [10.00, 15.50, 8.75]
with_tax = [price * 1.08 for price in prices]

# Pattern 2: Normalize
scores = [85, 92, 78, 95, 88]
max_score = max(scores)
normalized = [score / max_score for score in scores]

# Pattern 3: Type conversion
strings = ["1", "2", "3", "4"]
numbers = [int(s) for s in strings]
```

### Aggregating Data

```python
# Pattern 1: Count occurrences
votes = ["Alice", "Bob", "Alice", "Charlie", "Alice"]
alice_votes = votes.count("Alice")  # 3

# Pattern 2: Group by category
students = [
    ["Alice", "A"],
    ["Bob", "B"],
    ["Charlie", "A"]
]
a_students = [name for name, grade in students if grade == "A"]

# Pattern 3: Statistics
scores = [85, 92, 78, 95, 88]
avg = sum(scores) / len(scores)
min_score = min(scores)
max_score = max(scores)
range_score = max_score - min_score
```

---

**End of Part 2: Advanced Concepts**

Next section will cover:
- Part 3: Mistakes, Examples, and Practice (Parts 11-15)

# Chapter 3: Data Structures Part 1 - Lists
## Part 3: Mistakes, Examples & Practice (Parts 11-15)

---

## Part 11: Common Mistakes and How to Avoid Them

### Mistake #1: Modifying List While Iterating

```python
# WRONG: Removing items while iterating
numbers = [1, 2, 3, 4, 5, 6]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)  # BAD! Skips elements

print(numbers)  # [1, 3, 5, 6] - WRONG! 6 wasn't removed

# WHY? When you remove item, indices shift!

# CORRECT: Iterate over copy
numbers = [1, 2, 3, 4, 5, 6]
for num in numbers[:]:  # [:] creates copy
    if num % 2 == 0:
        numbers.remove(num)

print(numbers)  # [1, 3, 5] - Correct!

# BETTER: Use list comprehension
numbers = [1, 2, 3, 4, 5, 6]
numbers = [num for num in numbers if num % 2 != 0]
print(numbers)  # [1, 3, 5]
```

### Mistake #2: Shallow vs Deep Copy

```python
# Shallow copy with nested lists
original = [[1, 2], [3, 4]]
shallow = original[:]

shallow[0][0] = 999

print(original)  # [[999, 2], [3, 4]] - CHANGED!
print(shallow)   # [[999, 2], [3, 4]]

# WHY? Both point to same nested lists

# CORRECT: Deep copy
import copy
original = [[1, 2], [3, 4]]
deep = copy.deepcopy(original)

deep[0][0] = 999

print(original)  # [[1, 2], [3, 4]] - unchanged
print(deep)      # [[999, 2], [3, 4]]
```

### Mistake #3: Mutable Default Arguments

```python
# WRONG
def add_item(item, my_list=[]):
    my_list.append(item)
    return my_list

list1 = add_item(1)  # [1]
list2 = add_item(2)  # [1, 2] - UNEXPECTED!
# Same list is reused!

# CORRECT
def add_item(item, my_list=None):
    if my_list is None:
        my_list = []
    my_list.append(item)
    return my_list

list1 = add_item(1)  # [1]
list2 = add_item(2)  # [2] - Correct!
```

### Mistake #4: Index Out of Range

```python
# WRONG
numbers = [1, 2, 3]
# print(numbers[5])  # IndexError!

# CORRECT: Check first
if len(numbers) > 5:
    print(numbers[5])
else:
    print("Index out of range")

# BETTER: Avoid indexing
for num in numbers:
    print(num)
```

### Mistake #5: append() vs extend()

```python
# append() adds single item (even if it's a list)
list1 = [1, 2, 3]
list1.append([4, 5])
print(list1)  # [1, 2, 3, [4, 5]] - nested!

# extend() adds items individually
list2 = [1, 2, 3]
list2.extend([4, 5])
print(list2)  # [1, 2, 3, 4, 5] - flat

# Use append for single items
# Use extend for multiple items
```

---

## Part 12: Detailed Worked Examples

### Example 1: Grade Book Manager

**Problem**: Manage student grades with statistics.

```python
print("=== Grade Book Manager ===")

students = []  # Each student: [name, [grades]]

while True:
    print("\n1. Add student")
    print("2. Add grade")
    print("3. View all")
    print("4. Student stats")
    print("5. Class stats")
    print("6. Exit")
    
    choice = input("\nChoice: ")
    
    if choice == "1":
        name = input("Student name: ")
        students.append([name, []])
        print(f"✓ Added {name}")
    
    elif choice == "2":
        name = input("Student name: ")
        found = False
        
        for student in students:
            if student[0] == name:
                grade = float(input("Grade: "))
                student[1].append(grade)
                print(f"✓ Added grade for {name}")
                found = True
                break
        
        if not found:
            print("✗ Student not found")
    
    elif choice == "3":
        print("\n=== All Students ===")
        for name, grades in students:
            if grades:
                avg = sum(grades) / len(grades)
                print(f"{name}: {grades} (Avg: {avg:.1f})")
            else:
                print(f"{name}: No grades yet")
    
    elif choice == "4":
        name = input("Student name: ")
        
        for student_name, grades in students:
            if student_name == name:
                if grades:
                    print(f"\n=== {name}'s Statistics ===")
                    print(f"Grades: {grades}")
                    print(f"Average: {sum(grades)/len(grades):.1f}")
                    print(f"Highest: {max(grades)}")
                    print(f"Lowest: {min(grades)}")
                    print(f"Count: {len(grades)}")
                else:
                    print("No grades yet")
                break
        else:
            print("✗ Student not found")
    
    elif choice == "5":
        all_grades = []
        for name, grades in students:
            all_grades.extend(grades)
        
        if all_grades:
            print(f"\n=== Class Statistics ===")
            print(f"Total students: {len(students)}")
            print(f"Total grades: {len(all_grades)}")
            print(f"Class average: {sum(all_grades)/len(all_grades):.1f}")
            print(f"Highest: {max(all_grades)}")
            print(f"Lowest: {min(all_grades)}")
        else:
            print("No grades entered")
    
    elif choice == "6":
        print("Goodbye!")
        break
```

**What this teaches**:
- Nested lists for structured data
- List searching and modification
- Statistical calculations
- Menu-driven programs

---

### Example 2: Shopping List Manager

**Problem**: Interactive shopping list with quantities and prices.

```python
print("=== Shopping List Manager ===")

# Each item: [name, quantity, price_per_unit]
shopping_list = []

while True:
    print("\n1. Add item")
    print("2. Remove item")
    print("3. Update quantity")
    print("4. View list")
    print("5. Calculate total")
    print("6. Clear list")
    print("7. Exit")
    
    choice = input("\nChoice: ")
    
    if choice == "1":
        item = input("Item name: ")
        qty = int(input("Quantity: "))
        price = float(input("Price per unit: $"))
        shopping_list.append([item, qty, price])
        print(f"✓ Added {qty} {item}(s)")
    
    elif choice == "2":
        item = input("Item to remove: ")
        
        for i, [name, qty, price] in enumerate(shopping_list):
            if name.lower() == item.lower():
                removed = shopping_list.pop(i)
                print(f"✓ Removed {removed[0]}")
                break
        else:
            print("✗ Item not found")
    
    elif choice == "3":
        item = input("Item name: ")
        
        for i, [name, qty, price] in enumerate(shopping_list):
            if name.lower() == item.lower():
                new_qty = int(input(f"Current: {qty}. New quantity: "))
                shopping_list[i][1] = new_qty
                print(f"✓ Updated {name} to {new_qty}")
                break
        else:
            print("✗ Item not found")
    
    elif choice == "4":
        if not shopping_list:
            print("\nList is empty!")
        else:
            print("\n=== Shopping List ===")
            for i, [name, qty, price] in enumerate(shopping_list, 1):
                total = qty * price
                print(f"{i}. {name} - Qty: {qty} @ ${price:.2f} = ${total:.2f}")
    
    elif choice == "5":
        if shopping_list:
            total = sum(qty * price for name, qty, price in shopping_list)
            print(f"\n💰 Total: ${total:.2f}")
        else:
            print("List is empty")
    
    elif choice == "6":
        shopping_list.clear()
        print("✓ List cleared")
    
    elif choice == "7":
        print("Happy shopping!")
        break
```

**What this teaches**:
- Complex list operations
- List comprehensions for calculations
- Enumerate with unpacking
- Real-world data management

---

### Example 3: Temperature Tracker

**Problem**: Track daily temperatures and analyze trends.

```python
print("=== Temperature Tracker ===")

temperatures = []
days = []

while True:
    print("\n1. Add temperature")
    print("2. View all")
    print("3. Statistics")
    print("4. Find extremes")
    print("5. Trend analysis")
    print("6. Exit")
    
    choice = input("\nChoice: ")
    
    if choice == "1":
        day = input("Day (e.g., Monday): ")
        temp = float(input("Temperature (°F): "))
        days.append(day)
        temperatures.append(temp)
        print(f"✓ Added {temp}°F for {day}")
    
    elif choice == "2":
        if temperatures:
            print("\n=== Temperature Log ===")
            for day, temp in zip(days, temperatures):
                print(f"{day}: {temp}°F")
        else:
            print("No data yet")
    
    elif choice == "3":
        if temperatures:
            avg = sum(temperatures) / len(temperatures)
            print(f"\n=== Statistics ===")
            print(f"Average: {avg:.1f}°F")
            print(f"Highest: {max(temperatures)}°F")
            print(f"Lowest: {min(temperatures)}°F")
            print(f"Range: {max(temperatures) - min(temperatures):.1f}°F")
            print(f"Days recorded: {len(temperatures)}")
        else:
            print("No data yet")
    
    elif choice == "4":
        if temperatures:
            hot_idx = temperatures.index(max(temperatures))
            cold_idx = temperatures.index(min(temperatures))
            
            print(f"\n🔥 Hottest: {days[hot_idx]} at {temperatures[hot_idx]}°F")
            print(f"❄️ Coldest: {days[cold_idx]} at {temperatures[cold_idx]}°F")
        else:
            print("No data yet")
    
    elif choice == "5":
        if len(temperatures) >= 2:
            increasing = sum(1 for i in range(len(temperatures)-1) 
                           if temperatures[i+1] > temperatures[i])
            decreasing = sum(1 for i in range(len(temperatures)-1) 
                           if temperatures[i+1] < temperatures[i])
            
            print(f"\n=== Trend Analysis ===")
            print(f"Days warming: {increasing}")
            print(f"Days cooling: {decreasing}")
            
            if increasing > decreasing:
                print("📈 Overall warming trend")
            elif decreasing > increasing:
                print("📉 Overall cooling trend")
            else:
                print("➡️ Stable temperatures")
        else:
            print("Need at least 2 days of data")
    
    elif choice == "6":
        print("Goodbye!")
        break
```

**What this teaches**:
- Parallel lists (days and temperatures)
- zip() for combining lists
- List comprehensions with conditions
- Trend analysis

---

### Example 4: To-Do List with Priorities

**Problem**: Task manager with priority levels.

```python
print("=== Priority To-Do List ===")

# Each task: [description, priority, completed]
tasks = []

def display_tasks():
    if not tasks:
        print("\nNo tasks!")
        return
    
    print("\n=== Tasks ===")
    for i, (desc, priority, done) in enumerate(tasks, 1):
        status = "✓" if done else " "
        print(f"{i}. [{status}] ({priority}) {desc}")

while True:
    print("\n1. Add task")
    print("2. Complete task")
    print("3. Delete task")
    print("4. View tasks")
    print("5. Sort by priority")
    print("6. Filter by priority")
    print("7. Exit")
    
    choice = input("\nChoice: ")
    
    if choice == "1":
        desc = input("Task: ")
        priority = input("Priority (High/Medium/Low): ").capitalize()
        tasks.append([desc, priority, False])
        print("✓ Task added")
    
    elif choice == "2":
        display_tasks()
        if tasks:
            num = int(input("Task number to complete: "))
            if 1 <= num <= len(tasks):
                tasks[num-1][2] = True
                print("✓ Task completed!")
    
    elif choice == "3":
        display_tasks()
        if tasks:
            num = int(input("Task number to delete: "))
            if 1 <= num <= len(tasks):
                removed = tasks.pop(num-1)
                print(f"✓ Deleted: {removed[0]}")
    
    elif choice == "4":
        display_tasks()
    
    elif choice == "5":
        priority_order = {"High": 1, "Medium": 2, "Low": 3}
        tasks.sort(key=lambda x: priority_order[x[1]])
        print("✓ Sorted by priority")
        display_tasks()
    
    elif choice == "6":
        priority = input("Show (High/Medium/Low): ").capitalize()
        filtered = [t for t in tasks if t[1] == priority]
        
        if filtered:
            print(f"\n=== {priority} Priority Tasks ===")
            for i, (desc, pri, done) in enumerate(filtered, 1):
                status = "✓" if done else " "
                print(f"{i}. [{status}] {desc}")
        else:
            print(f"No {priority} priority tasks")
    
    elif choice == "7":
        total = len(tasks)
        completed = sum(1 for t in tasks if t[2])
        print(f"\nSummary: {completed}/{total} completed")
        print("Goodbye!")
        break
```

**What this teaches**:
- Nested data structures
- Lambda functions for sorting
- List filtering
- Status tracking

---

### Example 5: Number Analysis Tool

**Problem**: Analyze a list of numbers with various statistics.

```python
print("=== Number Analysis Tool ===")

print("Enter numbers (type 'done' to finish):")
numbers = []

while True:
    entry = input("Number: ")
    if entry.lower() == 'done':
        break
    try:
        numbers.append(float(entry))
    except ValueError:
        print("Invalid number, try again")

if not numbers:
    print("No numbers entered!")
else:
    # Basic stats
    total = sum(numbers)
    count = len(numbers)
    average = total / count
    
    print("\n=== Statistics ===")
    print(f"Count: {count}")
    print(f"Sum: {total:.2f}")
    print(f"Average: {average:.2f}")
    print(f"Min: {min(numbers):.2f}")
    print(f"Max: {max(numbers):.2f}")
    print(f"Range: {max(numbers) - min(numbers):.2f}")
    
    # Median
    sorted_nums = sorted(numbers)
    mid = len(sorted_nums) // 2
    if len(sorted_nums) % 2 == 0:
        median = (sorted_nums[mid-1] + sorted_nums[mid]) / 2
    else:
        median = sorted_nums[mid]
    print(f"Median: {median:.2f}")
    
    # Above/below average
    above = [n for n in numbers if n > average]
    below = [n for n in numbers if n < average]
    
    print(f"\nAbove average: {len(above)} numbers")
    print(f"Below average: {len(below)} numbers")
    
    # Frequency
    print("\n=== Value Frequency ===")
    unique = list(set(numbers))
    for val in sorted(unique):
        count = numbers.count(val)
        print(f"{val}: appears {count} time(s)")
    
    # Percentiles
    print("\n=== Percentiles ===")
    for p in [25, 50, 75]:
        idx = int(len(sorted_nums) * p / 100)
        print(f"{p}th percentile: {sorted_nums[idx]:.2f}")
```

**What this teaches**:
- User input collection
- Statistical calculations
- List comprehensions
- Set operations
- Error handling

---

### Example 6: Inventory System

**Problem**: Manage product inventory with stock tracking.

```python
print("=== Inventory Management ===")

# Each item: [name, quantity, price, category]
inventory = [
    ["Laptop", 5, 999.99, "Electronics"],
    ["Mouse", 15, 29.99, "Electronics"],
    ["Desk", 8, 299.99, "Furniture"],
    ["Chair", 12, 149.99, "Furniture"]
]

while True:
    print("\n1. View inventory")
    print("2. Add product")
    print("3. Update stock")
    print("4. Search product")
    print("5. Low stock alert")
    print("6. Category report")
    print("7. Total value")
    print("8. Exit")
    
    choice = input("\nChoice: ")
    
    if choice == "1":
        print("\n=== Inventory ===")
        print(f"{'Name':<15} {'Qty':<8} {'Price':<10} {'Category':<15} {'Value':<10}")
        print("-" * 65)
        
        for name, qty, price, cat in inventory:
            value = qty * price
            print(f"{name:<15} {qty:<8} ${price:<9.2f} {cat:<15} ${value:<9.2f}")
    
    elif choice == "2":
        name = input("Product name: ")
        qty = int(input("Quantity: "))
        price = float(input("Price: $"))
        cat = input("Category: ")
        
        inventory.append([name, qty, price, cat])
        print(f"✓ Added {name}")
    
    elif choice == "3":
        name = input("Product name: ")
        
        for item in inventory:
            if item[0].lower() == name.lower():
                print(f"Current stock: {item[1]}")
                change = int(input("Change (+/-): "))
                item[1] += change
                print(f"✓ New stock: {item[1]}")
                break
        else:
            print("✗ Product not found")
    
    elif choice == "4":
        term = input("Search: ").lower()
        found = [item for item in inventory if term in item[0].lower()]
        
        if found:
            print("\n=== Search Results ===")
            for name, qty, price, cat in found:
                print(f"{name}: {qty} units @ ${price} ({cat})")
        else:
            print("No results")
    
    elif choice == "5":
        threshold = int(input("Low stock threshold: "))
        low_stock = [item for item in inventory if item[1] < threshold]
        
        if low_stock:
            print("\n⚠️ Low Stock Alert:")
            for name, qty, price, cat in low_stock:
                print(f"{name}: Only {qty} left!")
        else:
            print("✓ All items well-stocked")
    
    elif choice == "6":
        categories = {}
        
        for name, qty, price, cat in inventory:
            if cat not in categories:
                categories[cat] = []
            categories[cat].append([name, qty, price])
        
        print("\n=== Category Report ===")
        for cat, items in categories.items():
            total_items = len(items)
            total_value = sum(qty * price for name, qty, price in items)
            print(f"\n{cat}:")
            print(f"  Products: {total_items}")
            print(f"  Total value: ${total_value:.2f}")
    
    elif choice == "7":
        total_value = sum(qty * price for name, qty, price, cat in inventory)
        total_items = sum(qty for name, qty, price, cat in inventory)
        
        print(f"\n💰 Total inventory value: ${total_value:,.2f}")
        print(f"📦 Total items: {total_items}")
    
    elif choice == "8":
        print("Goodbye!")
        break
```

**What this teaches**:
- Complex data structures
- Filtering and searching
- Grouping data by category
- Aggregation calculations

---

### Example 7: Quiz Game

**Problem**: Interactive quiz with scoring.

```python
print("=== Python Quiz Game ===")

# Each question: [question, options, correct_answer_index]
questions = [
    ["What is Python?", 
     ["A snake", "A programming language", "A game", "A calculator"],
     1],
    
    ["Which symbol starts a comment?",
     ["//", "#", "/*", "--"],
     1],
    
    ["What does len() do?",
     ["Lengthens strings", "Returns length", "Creates lists", "Deletes items"],
     1],
    
    ["How do you create a list?",
     ["{}", "()", "[]", "<>"],
     2],
    
    ["What is 5 // 2?",
     ["2.5", "2", "3", "Error"],
     1]
]

score = 0
wrong_answers = []

print(f"\nYou have {len(questions)} questions.\n")

for i, (question, options, correct) in enumerate(questions, 1):
    print(f"Question {i}: {question}")
    
    for j, option in enumerate(options):
        print(f"  {j+1}. {option}")
    
    while True:
        try:
            answer = int(input("\nYour answer (1-4): ")) - 1
            if 0 <= answer < 4:
                break
            print("Invalid choice!")
        except ValueError:
            print("Enter a number!")
    
    if answer == correct:
        print("✓ Correct!\n")
        score += 1
    else:
        print(f"✗ Wrong! Correct answer: {options[correct]}\n")
        wrong_answers.append(i)

# Results
print("=" * 40)
print("QUIZ COMPLETE!")
print("=" * 40)
print(f"Score: {score}/{len(questions)}")
percentage = (score / len(questions)) * 100
print(f"Percentage: {percentage:.1f}%")

if percentage >= 80:
    print("🎉 Excellent!")
elif percentage >= 60:
    print("👍 Good job!")
elif percentage >= 40:
    print("📚 Keep studying!")
else:
    print("💪 Practice more!")

if wrong_answers:
    print(f"\nYou missed questions: {', '.join(map(str, wrong_answers))}")
```

**What this teaches**:
- Nested lists for complex data
- Enumerate with unpacking
- Score tracking
- User interaction

---

## Part 13: Practice Problems

### Easy Problems (Problems 1-8)

**Problem 1**: Create a list of 5 fruits and print each one.

**Problem 2**: Make a list of numbers 1-10 and print the 3rd and 7th elements.

**Problem 3**: Create a list, add 3 items using append(), then print the list.

**Problem 4**: Make a list of 5 numbers and print the sum using sum().

**Problem 5**: Create a list and use len() to print how many items it has.

**Problem 6**: Make a list of names and check if "Alice" is in the list.

**Problem 7**: Create a list [1, 2, 3, 4, 5] and print it in reverse order.

**Problem 8**: Make two lists and combine them using +.

---

### Medium Problems (Problems 9-18)

**Problem 9**: Create a list of 10 numbers and print only the even numbers.

**Problem 10**: Make a list of words and print only those longer than 5 characters.

**Problem 11**: Create a list of numbers and find the maximum without using max().

**Problem 12**: Remove all occurrences of a specific value from a list.

**Problem 13**: Create a list of numbers and create a new list with each number squared using a list comprehension.

**Problem 14**: Merge two sorted lists into one sorted list.

**Problem 15**: Count how many times each element appears in a list.

**Problem 16**: Create a list of 10 random numbers and sort it in descending order.

**Problem 17**: Given a list of numbers, create two new lists: one with positives, one with negatives.

**Problem 18**: Flatten a nested list [[1,2],[3,4],[5,6]] into [1,2,3,4,5,6].

---

### Hard Problems (Problems 19-28)

**Problem 19**: Rotate a list by N positions (e.g., [1,2,3,4,5] rotated by 2 becomes [4,5,1,2,3]).

**Problem 20**: Find the second largest number in a list without sorting.

**Problem 21**: Remove duplicates from a list while preserving order.

**Problem 22**: Find all pairs of numbers in a list that sum to a target value.

**Problem 23**: Implement binary search on a sorted list.

**Problem 24**: Transpose a 2D matrix (swap rows and columns).

**Problem 25**: Find the longest consecutive sequence in an unsorted list.

**Problem 26**: Implement a function to merge two sorted lists without using built-in functions.

**Problem 27**: Given a list of lists, find the list with the maximum sum.

**Problem 28**: Implement a simple bubble sort algorithm to sort a list.

---

### Challenge Problems (Problems 29-35)

**Problem 29**: Create a function that returns the moving average of a list with window size N.

**Problem 30**: Implement a shopping cart where you can add items, remove items, apply discounts, and calculate total with tax.

**Problem 31**: Create a gradebook system that stores students and grades, calculates averages, and determines letter grades.

**Problem 32**: Build a simple contact manager with add, search, update, delete functionality.

**Problem 33**: Implement a function to find the longest increasing subsequence in a list.

**Problem 34**: Create a Tic-Tac-Toe board using a 2D list and implement win-checking logic.

**Problem 35**: Build a playlist manager where you can add songs, create playlists, shuffle, and play next/previous.

---

## Part 14: Mini-Project - Contact Management System

**Objective**: Build a complete contact manager with full CRUD operations.

```python
print("=" * 50)
print("    CONTACT MANAGEMENT SYSTEM")
print("=" * 50)

# Each contact: [name, phone, email, category]
contacts = []

def display_contacts(contact_list=None):
    if contact_list is None:
        contact_list = contacts
    
    if not contact_list:
        print("\nNo contacts!")
        return
    
    print("\n" + "=" * 70)
    print(f"{'#':<4} {'Name':<20} {'Phone':<15} {'Email':<20} {'Category':<10}")
    print("=" * 70)
    
    for i, (name, phone, email, cat) in enumerate(contact_list, 1):
        print(f"{i:<4} {name:<20} {phone:<15} {email:<20} {cat:<10}")
    print("=" * 70)

while True:
    print("\n=== MAIN MENU ===")
    print("1. Add contact")
    print("2. View all contacts")
    print("3. Search contact")
    print("4. Update contact")
    print("5. Delete contact")
    print("6. Filter by category")
    print("7. Sort contacts")
    print("8. Export contacts")
    print("9. Statistics")
    print("10. Exit")
    
    choice = input("\nChoice (1-10): ")
    
    if choice == "1":
        # Add contact
        print("\n=== Add New Contact ===")
        name = input("Name: ")
        phone = input("Phone: ")
        email = input("Email: ")
        category = input("Category (Family/Friend/Work/Other): ").capitalize()
        
        contacts.append([name, phone, email, category])
        print(f"✓ Added {name} to contacts")
    
    elif choice == "2":
        # View all
        print("\n=== All Contacts ===")
        display_contacts()
    
    elif choice == "3":
        # Search
        search_term = input("\nSearch (name/phone/email): ").lower()
        results = [c for c in contacts if search_term in c[0].lower() 
                  or search_term in c[1] or search_term in c[2].lower()]
        
        if results:
            print(f"\nFound {len(results)} result(s):")
            display_contacts(results)
        else:
            print("✗ No matches found")
    
    elif choice == "4":
        # Update
        display_contacts()
        if contacts:
            num = int(input("\nContact number to update: "))
            if 1 <= num <= len(contacts):
                contact = contacts[num-1]
                print(f"\nUpdating: {contact[0]}")
                print("(Press Enter to keep current value)")
                
                name = input(f"Name [{contact[0]}]: ") or contact[0]
                phone = input(f"Phone [{contact[1]}]: ") or contact[1]
                email = input(f"Email [{contact[2]}]: ") or contact[2]
                cat = input(f"Category [{contact[3]}]: ") or contact[3]
                
                contacts[num-1] = [name, phone, email, cat.capitalize()]
                print("✓ Contact updated")
            else:
                print("✗ Invalid number")
    
    elif choice == "5":
        # Delete
        display_contacts()
        if contacts:
            num = int(input("\nContact number to delete: "))
            if 1 <= num <= len(contacts):
                removed = contacts.pop(num-1)
                print(f"✓ Deleted {removed[0]}")
            else:
                print("✗ Invalid number")
    
    elif choice == "6":
        # Filter by category
        category = input("\nCategory (Family/Friend/Work/Other): ").capitalize()
        filtered = [c for c in contacts if c[3] == category]
        
        if filtered:
            print(f"\n=== {category} Contacts ===")
            display_contacts(filtered)
        else:
            print(f"✗ No {category} contacts")
    
    elif choice == "7":
        # Sort
        print("\nSort by:")
        print("1. Name")
        print("2. Category")
        
        sort_choice = input("Choice: ")
        
        if sort_choice == "1":
            contacts.sort(key=lambda x: x[0])
            print("✓ Sorted by name")
        elif sort_choice == "2":
            contacts.sort(key=lambda x: x[3])
            print("✓ Sorted by category")
        
        display_contacts()
    
    elif choice == "8":
        # Export
        if contacts:
            filename = input("\nFilename (without .txt): ") + ".txt"
            
            with open(filename, 'w') as f:
                f.write("Name,Phone,Email,Category\n")
                for name, phone, email, cat in contacts:
                    f.write(f"{name},{phone},{email},{cat}\n")
            
            print(f"✓ Exported {len(contacts)} contacts to {filename}")
        else:
            print("✗ No contacts to export")
    
    elif choice == "9":
        # Statistics
        if contacts:
            categories = {}
            for name, phone, email, cat in contacts:
                categories[cat] = categories.get(cat, 0) + 1
            
            print("\n=== Statistics ===")
            print(f"Total contacts: {len(contacts)}")
            print(f"\nBy category:")
            for cat, count in sorted(categories.items()):
                percentage = (count / len(contacts)) * 100
                print(f"  {cat}: {count} ({percentage:.1f}%)")
        else:
            print("No contacts yet")
    
    elif choice == "10":
        # Exit
        print(f"\n✓ {len(contacts)} contacts saved")
        print("Goodbye!")
        break
    
    else:
        print("✗ Invalid choice!")

print("\n" + "=" * 50)
print("    PROGRAM TERMINATED")
print("=" * 50)
```

**What This Project Teaches**:
- Complete CRUD operations (Create, Read, Update, Delete)
- List manipulation and searching
- Filtering and sorting
- File I/O (writing to files)
- Statistics and aggregation
- Menu-driven architecture
- Input validation
- Real-world application design

**Extensions You Can Add**:
1. Import contacts from file
2. Add favorite contacts feature
3. Validate phone and email formats
4. Birthday tracking and reminders
5. Multiple phone numbers per contact
6. Contact groups/tags
7. Backup and restore functionality
8. Search with multiple criteria

---

## Part 15: Key Takeaways

**What You've Learned**:

**List Fundamentals**:
- Lists store ordered collections: `my_list = [1, 2, 3]`
- Zero-based indexing: first element is `[0]`
- Negative indexing: last element is `[-1]`
- Lists are mutable (can be changed)

**Creating & Accessing**:
- Create: `[]`, `list()`, `[x for x in range(10)]`
- Index: `my_list[0]`, `my_list[-1]`
- Slice: `my_list[1:5]`, `my_list[::2]`, `my_list[::-1]`

**Modifying Lists**:
- Add: `append()`, `insert()`, `extend()`, `+=`
- Remove: `remove()`, `pop()`, `del`, `clear()`
- Change: `my_list[0] = new_value`

**Important Methods**:
- `sort()` - in-place sort
- `sorted()` - returns new sorted list
- `reverse()` - reverse in-place
- `count()` - count occurrences
- `index()` - find position
- `copy()` - create shallow copy

**Operations**:
- Concatenate: `list1 + list2`
- Repeat: `list1 * 3`
- Membership: `item in my_list`
- Length: `len(my_list)`
- Min/Max/Sum: `min()`, `max()`, `sum()`

**Iteration**:
- Basic: `for item in my_list:`
- With index: `for i in range(len(my_list)):`
- Best practice: `for i, item in enumerate(my_list):`
- Multiple lists: `for x, y in zip(list1, list2):`

**List Comprehensions**:
- Basic: `[x for x in range(10)]`
- Filter: `[x for x in list if x > 0]`
- Transform: `[x*2 for x in list]`
- Conditional: `[x if x>0 else 0 for x in list]`

**Nested Lists**:
- Create: `[[1,2], [3,4], [5,6]]`
- Access: `matrix[row][col]`
- Iterate: nested for loops
- Use for: tables, grids, matrices

**Common Patterns**:
- Find max: `max(my_list)` or manual loop
- Filter: `[x for x in list if condition]`
- Transform: `[f(x) for x in list]`
- Accumulate: `sum()`, `total += item`
- Search: `if item in list:` or `.index()`

**Common Mistakes to Avoid**:
1. ❌ Modifying list while iterating
2. ❌ Confusing shallow vs deep copy
3. ❌ Mutable default arguments
4. ❌ Index out of range errors
5. ❌ Confusing `append()` vs `extend()`

**Best Practices**:
- Use list comprehensions for simple transformations
- Use `enumerate()` when you need both index and value
- Use `zip()` for parallel iteration
- Copy lists with `[:]` or `.copy()` for shallow, `deepcopy()` for nested
- Prefer `in` operator over manual searching
- Use meaningful variable names
- Comment complex list operations

**When to Use Lists**:
- ✅ Ordered collection of items
- ✅ Need to modify contents
- ✅ Allow duplicates
- ✅ Need indexing/slicing
- ✅ Items of same or mixed types

**When NOT to Use Lists**:
- ❌ Need unique items only (use set)
- ❌ Need key-value pairs (use dict)
- ❌ Need immutable collection (use tuple)
- ❌ Need constant-time lookups (use dict/set)

---

## Appendix: List Method Reference

### Creation
```python
[]                    # Empty list
[1, 2, 3]            # Literal
list(range(10))      # From iterable
[x for x in range(10)]  # Comprehension
```

### Adding Elements
```python
my_list.append(x)      # Add x to end
my_list.insert(i, x)   # Insert x at index i
my_list.extend(list2)  # Add all from list2
my_list += list2       # Same as extend
```

### Removing Elements
```python
my_list.remove(x)      # Remove first x
my_list.pop()          # Remove & return last
my_list.pop(i)         # Remove & return at i
del my_list[i]         # Delete at index i
del my_list[i:j]       # Delete slice
my_list.clear()        # Remove all
```

### Searching
```python
x in my_list           # Check membership
my_list.index(x)       # Find position of x
my_list.count(x)       # Count occurrences
```

### Sorting
```python
my_list.sort()                    # Sort in-place
my_list.sort(reverse=True)        # Descending
sorted(my_list)                   # Return new sorted
my_list.sort(key=lambda x: x[1])  # Custom key
```

### Other Methods
```python
my_list.reverse()      # Reverse in-place
my_list.copy()         # Shallow copy
len(my_list)           # Length
min(my_list)           # Minimum
max(my_list)           # Maximum
sum(my_list)           # Sum (numbers only)
```

### Slicing
```python
my_list[i]             # Element at i
my_list[i:j]           # Slice from i to j-1
my_list[i:j:k]         # Slice with step k
my_list[:]             # Full copy
my_list[::-1]          # Reverse copy
my_list[-1]            # Last element
my_list[-3:]           # Last 3 elements
```

### Comprehensions
```python
[x for x in iterable]                    # Basic
[x for x in iterable if condition]       # Filter
[f(x) for x in iterable]                 # Transform
[x if cond else y for x in iterable]     # Conditional
[x for row in matrix for x in row]       # Flatten
```

---

**Congratulations!** You've completed Chapter 3 on Lists. You now understand Python's most versatile data structure and can build complex programs that manage collections of data efficiently.

**Next Chapter Preview**: Chapter 4 will cover **Tuples, Sets, and Dictionaries** - the other essential data structures in Python, each with unique properties and use cases.

---

**END OF CHAPTER 3 - PART 3**# Chapter 3: Data Structures Part 1 - Lists
## Part 3: Mistakes, Examples & Practice (Parts 11-15)

---

## Part 11: Common Mistakes and How to Avoid Them

### Mistake #1: Modifying List While Iterating

```python
# WRONG: Removing items while iterating
numbers = [1, 2, 3, 4, 5, 6]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)  # BAD! Skips elements

print(numbers)  # [1, 3, 5, 6] - WRONG! 6 wasn't removed

# CORRECT: Iterate over copy
numbers = [1, 2, 3, 4, 5, 6]
for num in numbers[:]:  # [:] creates copy
    if num % 2 == 0:
        numbers.remove(num)

print(numbers)  # [1, 3, 5] - Correct!

# BETTER: Use list comprehension
numbers = [1, 2, 3, 4, 5, 6]
numbers = [num for num in numbers if num % 2 != 0]
print(numbers)  # [1, 3, 5]
```

### Mistake #2: Shallow vs Deep Copy

```python
# Shallow copy with nested lists
original = [[1, 2], [3, 4]]
shallow = original[:]

shallow[0][0] = 999

print(original)  # [[999, 2], [3, 4]] - CHANGED!
print(shallow)   # [[999, 2], [3, 4]]

# CORRECT: Deep copy
import copy
original = [[1, 2], [3, 4]]
deep = copy.deepcopy(original)

deep[0][0] = 999

print(original)  # [[1, 2], [3, 4]] - unchanged
print(deep)      # [[999, 2], [3, 4]]
```

### Mistake #3: Mutable Default Arguments

```python
# WRONG
def add_item(item, my_list=[]):
    my_list.append(item)
    return my_list

list1 = add_item(1)  # [1]
list2 = add_item(2)  # [1, 2] - UNEXPECTED!

# CORRECT
def add_item(item, my_list=None):
    if my_list is None:
        my_list = []
    my_list.append(item)
    return my_list

list1 = add_item(1)  # [1]
list2 = add_item(2)  # [2] - Correct!
```

### Mistake #4: Index Out of Range

```python
# WRONG
numbers = [1, 2, 3]
# print(numbers[5])  # IndexError!

# CORRECT: Check first
if len(numbers) > 5:
    print(numbers[5])
else:
    print("Index out of range")

# BETTER: Avoid indexing when possible
for num in numbers:
    print(num)
```

### Mistake #5: append() vs extend()

```python
# append() adds single item (even if it's a list)
list1 = [1, 2, 3]
list1.append([4, 5])
print(list1)  # [1, 2, 3, [4, 5]] - nested!

# extend() adds items individually
list2 = [1, 2, 3]
list2.extend([4, 5])
print(list2)  # [1, 2, 3, 4, 5] - flat

# Remember:
# append() - for single items
# extend() - for multiple items
```

---

## Part 12: Detailed Worked Examples

### Example 1: Grade Book Manager

**Problem**: Manage student grades with full statistics.

```python
print("=== Grade Book Manager ===")

students = []  # Each: [name, [grades]]

while True:
    print("\n1. Add student")
    print("2. Add grade")
    print("3. View all")
    print("4. Student stats")
    print("5. Exit")
    
    choice = input("\nChoice: ")
    
    if choice == "1":
        name = input("Name: ")
        students.append([name, []])
        print(f"✓ Added {name}")
    
    elif choice == "2":
        name = input("Name: ")
        for student in students:
            if student[0] == name:
                grade = float(input("Grade: "))
                student[1].append(grade)
                print(f"✓ Added")
                break
        else:
            print("✗ Not found")
    
    elif choice == "3":
        for name, grades in students:
            if grades:
                avg = sum(grades) / len(grades)
                print(f"{name}: {grades} (Avg: {avg:.1f})")
            else:
                print(f"{name}: No grades")
    
    elif choice == "4":
        name = input("Name: ")
        for n, grades in students:
            if n == name and grades:
                print(f"Average: {sum(grades)/len(grades):.1f}")
                print(f"Max: {max(grades)}")
                print(f"Min: {min(grades)}")
                break
    
    elif choice == "5":
        break
```

### Example 2: Shopping Cart

**Problem**: Build a shopping cart system.

```python
cart = []  # [name, quantity, price]

while True:
    print("\n1. Add item")
    print("2. View cart")
    print("3. Total")
    print("4. Exit")
    
    choice = input("\nChoice: ")
    
    if choice == "1":
        name = input("Item: ")
        qty = int(input("Qty: "))
        price = float(input("Price: $"))
        cart.append([name, qty, price])
        print("✓ Added")
    
    elif choice == "2":
        for i, (name, qty, price) in enumerate(cart, 1):
            print(f"{i}. {name} x{qty} @ ${price}")
    
    elif choice == "3":
        total = sum(qty * price for n, qty, price in cart)
        print(f"Total: ${total:.2f}")
    
    elif choice == "4":
        break
```

### Example 3: Number Analyzer

**Problem**: Analyze a list of numbers.

```python
numbers = []

print("Enter numbers (type 'done'):")
while True:
    entry = input("Number: ")
    if entry == 'done':
        break
    numbers.append(float(entry))

if numbers:
    print(f"\nCount: {len(numbers)}")
    print(f"Sum: {sum(numbers):.2f}")
    print(f"Average: {sum(numbers)/len(numbers):.2f}")
    print(f"Max: {max(numbers):.2f}")
    print(f"Min: {min(numbers):.2f}")
    print(f"Range: {max(numbers)-min(numbers):.2f}")
```

### Example 4: To-Do List

**Problem**: Task manager with priorities.

```python
tasks = []  # [desc, priority, done]

while True:
    print("\n1. Add task")
    print("2. Complete")
    print("3. View")
    print("4. Exit")
    
    choice = input("\nChoice: ")
    
    if choice == "1":
        task = input("Task: ")
        priority = input("Priority (H/M/L): ")
        tasks.append([task, priority, False])
    
    elif choice == "2":
        for i, t in enumerate(tasks, 1):
            print(f"{i}. {t[0]}")
        num = int(input("Complete #: "))
        tasks[num-1][2] = True
    
    elif choice == "3":
        for task, pri, done in tasks:
            status = "✓" if done else " "
            print(f"[{status}] ({pri}) {task}")
    
    elif choice == "4":
        break
```

### Example 5: Temperature Tracker

**Problem**: Track and analyze temperatures.

```python
temps = []
days = []

while True:
    print("\n1. Add temp")
    print("2. Stats")
    print("3. Exit")
    
    choice = input("\nChoice: ")
    
    if choice == "1":
        day = input("Day: ")
        temp = float(input("Temp: "))
        days.append(day)
        temps.append(temp)
    
    elif choice == "2":
        if temps:
            print(f"Avg: {sum(temps)/len(temps):.1f}")
            print(f"Max: {max(temps)}")
            print(f"Min: {min(temps)}")
            hot_idx = temps.index(max(temps))
            print(f"Hottest: {days[hot_idx]}")
    
    elif choice == "3":
        break
```

### Example 6: Simple Inventory

**Problem**: Track product inventory.

```python
inventory = [
    ["Laptop", 5, 999.99],
    ["Mouse", 15, 29.99],
    ["Keyboard", 10, 79.99]
]

while True:
    print("\n1. View")
    print("2. Update stock")
    print("3. Total value")
    print("4. Exit")
    
    choice = input("\nChoice: ")
    
    if choice == "1":
        for name, qty, price in inventory:
            print(f"{name}: {qty} @ ${price}")
    
    elif choice == "2":
        name = input("Product: ")
        for item in inventory:
            if item[0] == name:
                change = int(input("Change: "))
                item[1] += change
                print(f"New stock: {item[1]}")
                break
    
    elif choice == "3":
        total = sum(qty * price for n, qty, price in inventory)
        print(f"Total: ${total:,.2f}")
    
    elif choice == "4":
        break
```

### Example 7: Quiz Game

**Problem**: Interactive quiz with scoring.

```python
questions = [
    ["What is 2+2?", ["3", "4", "5"], 1],
    ["Python is a?", ["Snake", "Language", "Food"], 1],
    ["What is 5*5?", ["20", "25", "30"], 1]
]

score = 0

for i, (q, opts, correct) in enumerate(questions, 1):
    print(f"\nQ{i}: {q}")
    for j, opt in enumerate(opts):
        print(f"{j+1}. {opt}")
    
    ans = int(input("Answer: ")) - 1
    
    if ans == correct:
        print("✓ Correct!")
        score += 1
    else:
        print(f"✗ Wrong! Answer: {opts[correct]}")

print(f"\nScore: {score}/{len(questions)}")
```

---

## Part 13: Practice Problems

### Easy (1-8)

1. Create list of 5 fruits, print each
2. List of 1-10, print 3rd and 7th
3. Create list, append 3 items
4. List of 5 numbers, print sum
5. List, use len() to count
6. Check if "Alice" in list of names
7. Reverse [1,2,3,4,5]
8. Combine two lists with +

### Medium (9-18)

9. Print only even numbers from list
10. Words longer than 5 chars
11. Find max without max()
12. Remove all occurrences of value
13. Square each number (comprehension)
14. Merge two sorted lists
15. Count occurrences of each element
16. Sort 10 random numbers descending
17. Split into positives/negatives
18. Flatten [[1,2],[3,4],[5,6]]

### Hard (19-28)

19. Rotate list by N positions
20. Second largest without sorting
21. Remove duplicates keeping order
22. Find pairs that sum to target
23. Binary search on sorted list
24. Transpose 2D matrix
25. Longest consecutive sequence
26. Merge sorted lists without builtins
27. List with maximum sum
28. Bubble sort implementation

### Challenge (29-35)

29. Moving average with window N
30. Shopping cart with discounts/tax
31. Gradebook with averages
32. Contact manager (CRUD)
33. Longest increasing subsequence
34. Tic-tac-toe with win checking
35. Playlist manager with shuffle

---

## Part 14: Mini-Project - Contact Manager

```python
print("=== CONTACT MANAGER ===")

contacts = []  # [name, phone, email, category]

while True:
    print("\n1. Add")
    print("2. View all")
    print("3. Search")
    print("4. Delete")
    print("5. Filter by category")
    print("6. Exit")
    
    choice = input("\nChoice: ")
    
    if choice == "1":
        name = input("Name: ")
        phone = input("Phone: ")
        email = input("Email: ")
        cat = input("Category (Work/Friend/Family): ")
        contacts.append([name, phone, email, cat])
        print("✓ Added")
    
    elif choice == "2":
        if not contacts:
            print("No contacts")
        else:
            for i, (n, p, e, c) in enumerate(contacts, 1):
                print(f"{i}. {n} | {p} | {e} | {c}")
    
    elif choice == "3":
        term = input("Search: ").lower()
        found = [c for c in contacts if term in c[0].lower()]
        
        if found:
            for n, p, e, c in found:
                print(f"{n} | {p} | {e}")
        else:
            print("Not found")
    
    elif choice == "4":
        for i, (n, p, e, c) in enumerate(contacts, 1):
            print(f"{i}. {n}")
        
        num = int(input("Delete #: "))
        if 1 <= num <= len(contacts):
            removed = contacts.pop(num-1)
            print(f"✓ Deleted {removed[0]}")
    
    elif choice == "5":
        cat = input("Category: ")
        filtered = [c for c in contacts if c[3] == cat]
        
        for n, p, e, c in filtered:
            print(f"{n} | {p}")
    
    elif choice == "6":
        print(f"Saved {len(contacts)} contacts")
        break

print("Goodbye!")
```

---

## Part 15: Key Takeaways

**List Fundamentals**:
- Ordered, mutable collections
- Zero-based indexing: `list[0]`
- Negative indexing: `list[-1]`
- Can hold any type

**Creating**:
- `[]` - empty list
- `[1, 2, 3]` - literal
- `list(range(10))` - from iterable
- `[x for x in range(10)]` - comprehension

**Accessing**:
- Index: `list[0]`, `list[-1]`
- Slice: `list[1:5]`, `list[::2]`, `list[::-1]`

**Modifying**:
- Add: `append()`, `insert()`, `extend()`
- Remove: `remove()`, `pop()`, `del`, `clear()`
- Change: `list[0] = value`

**Methods**:
- `sort()`, `reverse()`, `count()`, `index()`, `copy()`

**Operations**:
- `+` concatenate
- `*` repeat
- `in` membership
- `len()`, `min()`, `max()`, `sum()`

**Iteration**:
- `for item in list:`
- `for i, item in enumerate(list):`
- `for x, y in zip(list1, list2):`

**Comprehensions**:
- `[x for x in list]`
- `[x for x in list if condition]`
- `[f(x) for x in list]`

**Common Mistakes**:
1. Modifying while iterating
2. Shallow vs deep copy
3. Mutable defaults
4. Index out of range
5. append vs extend confusion

**Best Practices**:
- Use comprehensions for simple transforms
- Use `enumerate()` for index+value
- Copy with `[:]` or `.copy()`
- Prefer `in` over manual search
- Use meaningful names

---

## Appendix: Quick Reference

```python
# Creation
my_list = [1, 2, 3]
empty = []
from_range = list(range(10))

# Access
first = my_list[0]
last = my_list[-1]
slice = my_list[1:3]

# Modify
my_list.append(4)
my_list.insert(0, 0)
my_list.extend([5, 6])
my_list.remove(3)
popped = my_list.pop()
del my_list[0]

# Methods
my_list.sort()
my_list.reverse()
count = my_list.count(2)
index = my_list.index(2)

# Operations
combined = list1 + list2
repeated = list1 * 3
exists = 5 in my_list
length = len(my_list)

# Comprehensions
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]
```

---

**END OF CHAPTER 3**

**Next**: Chapter 4 - Tuples, Sets, and Dictionaries
