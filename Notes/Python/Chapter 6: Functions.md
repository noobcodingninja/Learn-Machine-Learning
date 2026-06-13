# Chapter 6: Functions

## Part 1: Why Do Functions Exist?

### The Problem: Code Without Functions Is a Nightmare

Imagine you're building an e-commerce app. Without functions, your code looks like this:

```python
# User 1 checkout
subtotal_1 = 0
for item in cart_1:
    subtotal_1 += item["price"] * item["quantity"]
tax_1 = subtotal_1 * 0.08
shipping_1 = 5.99 if subtotal_1 < 50 else 0
total_1 = subtotal_1 + tax_1 + shipping_1
print(f"User 1 total: ${total_1:.2f}")

# User 2 checkout — SAME CODE, copy-pasted
subtotal_2 = 0
for item in cart_2:
    subtotal_2 += item["price"] * item["quantity"]
tax_2 = subtotal_2 * 0.08
shipping_2 = 5.99 if subtotal_2 < 50 else 0
total_2 = subtotal_2 + tax_2 + shipping_2
print(f"User 2 total: ${total_2:.2f}")

# Now the tax rate changes to 0.09... 
# You have to find and fix EVERY copy. Miss one = bug!
```

**Problems with this approach:**
- **Repetition**: Same logic copy-pasted everywhere
- **Fragility**: Change tax rate → hunt through thousands of lines
- **No testing**: Hard to verify one piece works correctly
- **Hard to read**: What does this 10-line block do? You have to read all of it

**The core insight:** Any time you find yourself writing the same logic more than once, you need a function.

### What Is a Function?

A function is a **named, reusable block of code** that:
- Takes inputs (parameters)
- Does something with them
- Optionally returns an output

Think of it like a machine at a factory:
- **Input** → raw materials going in
- **Process** → what the machine does
- **Output** → finished product coming out

The machine can be used thousands of times with different raw materials, but you only build it once.

```python
# The "machine" — defined once
def calculate_checkout_total(cart, tax_rate=0.08, free_shipping_threshold=50):
    subtotal = sum(item["price"] * item["quantity"] for item in cart)
    tax = subtotal * tax_rate
    shipping = 0 if subtotal >= free_shipping_threshold else 5.99
    return subtotal + tax + shipping

# Used as many times as needed
total_1 = calculate_checkout_total(cart_1)
total_2 = calculate_checkout_total(cart_2)

# Tax rate changed? Fix it in ONE place
total_3 = calculate_checkout_total(cart_3, tax_rate=0.09)
```

---

## Part 2: Defining Functions

### Anatomy of a Function

```python
# def keyword — tells Python "I'm defining a function"
# function_name — what you call it (use snake_case)
# parameters — inputs the function expects (in parentheses)
# colon — marks start of function body
def greet(name, greeting="Hello"):      # ← signature
    """
    Returns a greeting message.         # ← docstring (documentation)
    
    Args:
        name: The person's name
        greeting: The greeting word (default: "Hello")
    
    Returns:
        A formatted greeting string
    """
    message = f"{greeting}, {name}!"    # ← function body (indented!)
    return message                       # ← return value

# Calling the function
result = greet("Alice")
print(result)  # Hello, Alice!

result = greet("Bob", "Hi")
print(result)  # Hi, Bob!
```

### The `return` Statement

```python
# Functions can return nothing (implicitly returns None)
def say_hello(name):
    print(f"Hello, {name}!")
    # No return statement — returns None

result = say_hello("Alice")  # Prints: Hello, Alice!
print(result)                # None

# Functions can return a single value
def square(n):
    return n * n

print(square(5))   # 25
print(square(10))  # 100

# Functions can return multiple values (as a tuple)
def min_max(numbers):
    return min(numbers), max(numbers)   # Returns tuple (min, max)

low, high = min_max([3, 1, 4, 1, 5, 9, 2, 6])
print(f"Min: {low}, Max: {high}")  # Min: 1, Max: 9

# Return exits the function immediately
def find_first_even(numbers):
    for num in numbers:
        if num % 2 == 0:
            return num          # Exits HERE, rest of function doesn't run
    return None                 # Only reached if no even number found

print(find_first_even([1, 3, 5, 4, 7]))  # 4
print(find_first_even([1, 3, 5, 7]))     # None
```

---

## Part 3: Parameters and Arguments

### The 4 Types of Parameters

This is where functions get powerful. Python gives you incredible flexibility in how you pass data to functions.

#### Type 1: Positional Parameters

```python
# Order matters — arguments match parameters by position
def describe_person(name, age, city):
    return f"{name} is {age} years old and lives in {city}."

# Must provide all 3, in order
print(describe_person("Alice", 28, "New York"))
# Alice is 28 years old and lives in New York.

# ❌ WRONG — wrong order gives wrong meaning
print(describe_person(28, "Alice", "New York"))
# 28 is Alice years old and lives in New York.  ← Nonsense!
```

#### Type 2: Keyword Arguments

```python
def describe_person(name, age, city):
    return f"{name} is {age} years old and lives in {city}."

# Use keyword arguments — order no longer matters!
print(describe_person(age=28, city="New York", name="Alice"))
# Alice is 28 years old and lives in New York.

# Mix positional and keyword (positional must come first!)
print(describe_person("Alice", city="New York", age=28))
# Alice is 28 years old and lives in New York.

# ❌ WRONG — positional after keyword
# print(describe_person(name="Alice", 28, "New York"))  # SyntaxError!
```

#### Type 3: Default Parameters

```python
# Parameters with default values — become optional when calling
def create_user(username, email, role="user", active=True):
    return {
        "username": username,
        "email": email,
        "role": role,
        "active": active
    }

# Minimal call — uses defaults for role and active
user1 = create_user("alice", "alice@email.com")
print(user1)
# {'username': 'alice', 'email': 'alice@email.com', 'role': 'user', 'active': True}

# Override specific defaults
user2 = create_user("admin", "admin@email.com", role="admin")
print(user2)
# {'username': 'admin', 'email': 'admin@email.com', 'role': 'admin', 'active': True}

# Override all defaults
user3 = create_user("bob", "bob@email.com", "moderator", False)
print(user3)
# {'username': 'bob', 'email': 'bob@email.com', 'role': 'moderator', 'active': False}

# ⚠️ IMPORTANT: Default parameters are evaluated ONCE at definition time
# NEVER use mutable objects (lists, dicts) as defaults!

# ❌ WRONG
def add_item_wrong(item, items=[]):
    items.append(item)
    return items

print(add_item_wrong("apple"))   # ['apple']
print(add_item_wrong("banana"))  # ['apple', 'banana'] ← Bug! Same list!

# ✓ CORRECT
def add_item(item, items=None):
    if items is None:
        items = []          # Create NEW list each time
    items.append(item)
    return items

print(add_item("apple"))   # ['apple']
print(add_item("banana"))  # ['banana'] ← Correct! Fresh list.
```

#### Type 4: `*args` — Collect Extra Positional Arguments

```python
# The problem: What if you don't know how many arguments you'll get?
# You want to sum any number of values

# Without *args — limited!
def add_two(a, b):
    return a + b

def add_three(a, b, c):
    return a + b + c
# What about 10 numbers? 100?

# With *args — unlimited!
def add_all(*numbers):
    # *numbers collects ALL positional arguments into a TUPLE
    print(f"Received: {numbers}")   # It's a tuple
    return sum(numbers)

print(add_all(1, 2))           # 3
print(add_all(1, 2, 3, 4, 5))  # 15
print(add_all(10, 20, 30))     # 60

# Real-world use: logging function with variable detail
def log(level, *messages):
    """Log multiple messages at once"""
    prefix = f"[{level.upper()}]"
    for msg in messages:
        print(f"{prefix} {msg}")

log("info", "Server started", "Listening on port 8000")
log("error", "Database connection failed")
# [INFO] Server started
# [INFO] Listening on port 8000
# [ERROR] Database connection failed

# Combining regular and *args
def make_sentence(verb, *nouns):
    noun_str = ", ".join(nouns)
    return f"{verb.title()}: {noun_str}"

print(make_sentence("likes", "Python", "coffee", "music"))
# Likes: Python, coffee, music
```

#### Type 5: `**kwargs` — Collect Extra Keyword Arguments

```python
# The problem: What if you want to accept any named options?

def display_user_info(**user_data):
    # **user_data collects ALL keyword arguments into a DICT
    print(f"User data: {user_data}")  # It's a dictionary
    for key, value in user_data.items():
        print(f"  {key}: {value}")

display_user_info(name="Alice", age=28, city="NYC", role="admin")
# User data: {'name': 'Alice', 'age': 28, 'city': 'NYC', 'role': 'admin'}
#   name: Alice
#   age: 28
#   city: NYC
#   role: admin

# Real-world: flexible API request builder
def build_api_request(endpoint, method="GET", **options):
    """Build an API request with any options"""
    request = {
        "endpoint": endpoint,
        "method": method,
    }
    request.update(options)    # Merge in any extra keyword args
    return request

req1 = build_api_request("/users", method="GET", page=1, limit=20)
req2 = build_api_request("/users", method="POST", data={"name": "Alice"}, auth_token="abc123")

print(req1)
# {'endpoint': '/users', 'method': 'GET', 'page': 1, 'limit': 20}
print(req2)
# {'endpoint': '/users', 'method': 'POST', 'data': {'name': 'Alice'}, 'auth_token': 'abc123'}
```

### Combining All Parameter Types

```python
# Order MUST be: positional, *args, keyword with defaults, **kwargs
def everything(pos1, pos2, *args, keyword1="default", **kwargs):
    print(f"pos1: {pos1}")
    print(f"pos2: {pos2}")
    print(f"args: {args}")
    print(f"keyword1: {keyword1}")
    print(f"kwargs: {kwargs}")

everything(1, 2, 3, 4, 5, keyword1="custom", extra="hello", more=42)
# pos1: 1
# pos2: 2
# args: (3, 4, 5)
# keyword1: custom
# kwargs: {'extra': 'hello', 'more': 42}

# Real-world: database query builder
def query_database(table, *fields, order_by=None, **filters):
    """
    Build a database query dynamically.
    table: table name (required)
    *fields: columns to select (optional, defaults to all)
    order_by: sort column (optional)
    **filters: WHERE clause conditions
    """
    field_str = ", ".join(fields) if fields else "*"
    query = f"SELECT {field_str} FROM {table}"
    
    if filters:
        conditions = [f"{k} = '{v}'" for k, v in filters.items()]
        query += " WHERE " + " AND ".join(conditions)
    
    if order_by:
        query += f" ORDER BY {order_by}"
    
    return query

print(query_database("users"))
# SELECT * FROM users

print(query_database("users", "name", "email", role="admin", active=True))
# SELECT name, email FROM users WHERE role = 'admin' AND active = 'True'

print(query_database("products", "name", "price", order_by="price", category="electronics"))
# SELECT name, price FROM products WHERE category = 'electronics' ORDER BY price
```

### Unpacking Into Functions

```python
# You can unpack lists/tuples into positional args with *
def add(a, b, c):
    return a + b + c

numbers = [1, 2, 3]
print(add(*numbers))    # Same as add(1, 2, 3) → 6

point = (10, 20, 30)
print(add(*point))      # Same as add(10, 20, 30) → 60

# You can unpack dicts into keyword args with **
def greet(name, greeting, punctuation):
    return f"{greeting}, {name}{punctuation}"

data = {"name": "Alice", "greeting": "Hello", "punctuation": "!"}
print(greet(**data))    # Same as greet(name="Alice", ...) → Hello, Alice!

# Powerful pattern: passing config dictionaries to functions
config = {
    "endpoint": "/api/users",
    "method": "POST",
    "timeout": 30
}
result = build_api_request(**config)
```

---

## Part 4: Scope and Namespaces

### The Problem: Where Does Python Look for Variables?

```python
# Variables live in "scopes" — think of them as rooms in a house
x = "global"  # In the global scope (hallway — accessible everywhere)

def outer():
    x = "outer"  # In outer's scope (living room)
    
    def inner():
        x = "inner"  # In inner's scope (bedroom)
        print(x)     # Looks in bedroom first → "inner"
    
    inner()
    print(x)         # Looks in living room → "outer"

outer()
print(x)             # Looks in hallway → "global"

# Output:
# inner
# outer
# global
```

### The LEGB Rule

Python looks for variables in this order:
**L**ocal → **E**nclosing → **G**lobal → **B**uilt-in

```python
# Built-in scope: Python's built-in functions (print, len, sum...)
# Global scope: Variables defined at the module level
# Enclosing scope: Variables in the outer function (for nested functions)
# Local scope: Variables defined inside the current function

name = "Global Alice"          # Global

def outer_function():
    name = "Enclosing Bob"     # Enclosing
    
    def inner_function():
        name = "Local Carol"   # Local
        print(name)            # L → found: Local Carol
    
    inner_function()
    print(name)                # E → found: Enclosing Bob

outer_function()
print(name)                    # G → found: Global Alice
```

### The `global` and `nonlocal` Keywords

```python
# Modifying global variables from inside a function

counter = 0  # Global

def increment():
    global counter          # Tell Python: use the GLOBAL counter
    counter += 1

increment()
increment()
increment()
print(counter)  # 3

# Without global, you'd get an error:
def bad_increment():
    counter += 1            # UnboundLocalError! Python sees assignment and
                            # assumes it's local, but it was never defined locally

# nonlocal — for modifying enclosing (not global) variables
def make_counter():
    count = 0               # Enclosing variable
    
    def increment():
        nonlocal count      # Modify the ENCLOSING count
        count += 1
        return count
    
    return increment        # Return the function itself!

counter = make_counter()
print(counter())  # 1
print(counter())  # 2
print(counter())  # 3
# Each call remembers the previous count — this is a CLOSURE (more on this soon!)

# Best practice: Avoid global whenever possible
# Better to pass values in and return them out
def increment_pure(counter):    # Takes current value
    return counter + 1          # Returns new value (no side effects)

count = 0
count = increment_pure(count)
count = increment_pure(count)
print(count)  # 2
```

---

## Part 5: Lambda Functions

### The Problem: Small Functions Feel Like Overkill

Sometimes you need a simple function just once — for sorting, filtering, or mapping. Writing a full `def` feels verbose.

```python
# Sorting a list of dictionaries by a field
users = [
    {"name": "Charlie", "age": 35},
    {"name": "Alice", "age": 28},
    {"name": "Bob", "age": 42},
]

# Without lambda — you need a named function just for this
def get_age(user):
    return user["age"]

sorted_users = sorted(users, key=get_age)

# With lambda — anonymous function defined inline
sorted_users = sorted(users, key=lambda user: user["age"])

for u in sorted_users:
    print(f"{u['name']}: {u['age']}")
# Alice: 28
# Charlie: 35
# Bob: 42
```

### Lambda Syntax

```python
# lambda parameters: expression
# Equivalent to: def anonymous(parameters): return expression

# Regular function
def square(x):
    return x * x

# Lambda equivalent
square = lambda x: x * x

print(square(5))   # 25

# Multi-parameter lambda
multiply = lambda x, y: x * y
print(multiply(3, 4))  # 12

# Lambda with condition
classify = lambda n: "even" if n % 2 == 0 else "odd"
print(classify(7))   # odd
print(classify(10))  # even
```

### When to Use Lambda vs `def`

```python
# ✓ USE LAMBDA: Short, single-expression functions passed to other functions

products = [
    {"name": "Laptop", "price": 999.99, "rating": 4.5},
    {"name": "Mouse", "price": 29.99, "rating": 4.8},
    {"name": "Monitor", "price": 399.99, "rating": 4.2},
]

# Sort by price
by_price = sorted(products, key=lambda p: p["price"])

# Sort by rating (descending)
by_rating = sorted(products, key=lambda p: p["rating"], reverse=True)

# Sort by value (price/rating ratio — best value first)
by_value = sorted(products, key=lambda p: p["price"] / p["rating"])

# Filter expensive products
expensive = list(filter(lambda p: p["price"] > 100, products))

# Get just names
names = list(map(lambda p: p["name"], products))
print(names)  # ['Laptop', 'Mouse', 'Monitor']

# ✓ USE LAMBDA: In one-liners with map/filter/sorted
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda n: n % 2 == 0, numbers))
squares = list(map(lambda n: n ** 2, numbers))
print(evens)    # [2, 4, 6, 8, 10]
print(squares)  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# ❌ AVOID LAMBDA: Complex logic — use def instead
# This is unreadable:
process = lambda x: x.strip().lower().replace(" ", "_") if x else ""

# This is clear:
def normalize_key(x):
    """Convert display name to dictionary key format."""
    if not x:
        return ""
    return x.strip().lower().replace(" ", "_")
```

---

## Part 6: Higher-Order Functions

### Functions That Take or Return Functions

```python
# Functions are "first-class" objects in Python — they can be:
# - Stored in variables
# - Passed as arguments
# - Returned from other functions

# Storing in a variable
def greet(name):
    return f"Hello, {name}!"

say_hello = greet           # No () — we're referencing the function, not calling it
print(say_hello("Alice"))   # Hello, Alice!

# Passing as an argument
def apply(func, value):
    return func(value)

print(apply(greet, "Bob"))       # Hello, Bob!
print(apply(str.upper, "hello")) # HELLO

# Returning from a function
def make_multiplier(factor):
    def multiplier(number):
        return number * factor  # Uses factor from enclosing scope — CLOSURE!
    return multiplier            # Returns the function

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15
print(double(10))  # 20

# Real-world: different discount calculators
def make_discount_calculator(discount_percent):
    def apply_discount(price):
        discount = price * (discount_percent / 100)
        return price - discount
    return apply_discount

student_discount = make_discount_calculator(15)   # 15% off
member_discount = make_discount_calculator(20)    # 20% off
staff_discount = make_discount_calculator(50)     # 50% off

price = 100.00
print(f"Student price: ${student_discount(price):.2f}")  # $85.00
print(f"Member price:  ${member_discount(price):.2f}")   # $80.00
print(f"Staff price:   ${staff_discount(price):.2f}")    # $50.00
```

### `map()`, `filter()`, `reduce()`

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# map(func, iterable) — apply function to every element
squares = list(map(lambda n: n ** 2, numbers))
print(squares)  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# Often a list comprehension is more readable:
squares = [n ** 2 for n in numbers]  # Same result, more Pythonic

# filter(func, iterable) — keep elements where function returns True
evens = list(filter(lambda n: n % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# List comprehension equivalent
evens = [n for n in numbers if n % 2 == 0]  # More Pythonic

# reduce() — accumulate values into a single result
from functools import reduce

total = reduce(lambda acc, n: acc + n, numbers)
print(total)   # 55  (sum of 1 to 10)

# Product of all numbers
product = reduce(lambda acc, n: acc * n, numbers)
print(product)  # 3628800

# Real-world: combining map + filter
transactions = [120.50, -45.00, 200.00, -30.00, 85.50, -10.00]

# Get positive transactions, doubled (simulate a 2x loyalty reward)
rewards = list(map(
    lambda t: t * 2,
    filter(lambda t: t > 0, transactions)
))
print(rewards)  # [241.0, 400.0, 171.0]

# Even more readable with comprehension:
rewards = [t * 2 for t in transactions if t > 0]
```

---

## Part 7: Closures

### The Problem: Functions That Remember State

```python
# A closure is a function that "remembers" variables from its enclosing scope
# even after the enclosing function has finished executing.

def make_greeting(language):
    # "language" lives in make_greeting's scope
    
    def greet(name):
        # greet() "closes over" language — it remembers it!
        if language == "english":
            return f"Hello, {name}!"
        elif language == "spanish":
            return f"¡Hola, {name}!"
        elif language == "french":
            return f"Bonjour, {name}!"
        return f"Hi, {name}!"
    
    return greet

english_greet = make_greeting("english")
spanish_greet = make_greeting("spanish")

print(english_greet("Alice"))   # Hello, Alice!
print(spanish_greet("Bob"))     # ¡Hola, Bob!

# make_greeting has finished — but the language is still remembered!
```

### Practical Closure: Memoization

```python
# Memoization: Remember the result of expensive computations
def make_memoized(func):
    cache = {}   # Cache lives in the closure!
    
    def memoized(*args):
        if args not in cache:
            cache[args] = func(*args)   # Calculate and store
            print(f"  Calculated {func.__name__}{args}")
        else:
            print(f"  Cache hit for {func.__name__}{args}")
        return cache[args]
    
    return memoized

def slow_square(n):
    # Imagine this is an expensive operation
    return n * n

fast_square = make_memoized(slow_square)

print(fast_square(5))   # Calculated → 25
print(fast_square(5))   # Cache hit → 25 (no recalculation!)
print(fast_square(10))  # Calculated → 100
print(fast_square(10))  # Cache hit → 100
```

---

## Part 8: Decorators

### The Problem: Adding Behaviour to Functions Without Changing Them

Imagine every function in your app needs:
- Timing (how long does it take?)
- Logging (what was called?)
- Error handling (what if it fails?)

You don't want to add this code to every function manually.

**Decorators solve this** — they wrap functions with extra behaviour.

```python
# Step-by-step: how decorators work

# Step 1: A function that wraps another function
def add_logging(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)                  # Call the original function
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper

def multiply(a, b):
    return a * b

# Apply the wrapper manually
logged_multiply = add_logging(multiply)
print(logged_multiply(3, 4))
# Calling multiply with args=(3, 4), kwargs={}
# multiply returned: 12
# 12

# Step 2: Using @ syntax (syntactic sugar — does the same thing!)
@add_logging
def add(a, b):
    return a + b

print(add(5, 3))
# Calling add with args=(5, 3), kwargs={}
# add returned: 8
# 8

# @add_logging is EXACTLY the same as: add = add_logging(add)
```

### Practical Decorators

```python
import time

# 1. TIMER DECORATOR
def timer(func):
    """Measure how long a function takes"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱  {func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_operation(n):
    total = 0
    for i in range(n):
        total += i
    return total

result = slow_operation(1_000_000)
print(f"Result: {result}")
# ⏱  slow_operation took 0.0523 seconds
# Result: 499999500000


# 2. RETRY DECORATOR
def retry(max_attempts=3, delay=1):
    """Retry a function if it fails"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise Exception(f"{func.__name__} failed after {max_attempts} attempts")
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.1)
def unstable_api_call(url):
    """Simulates an API that sometimes fails"""
    import random
    if random.random() < 0.7:    # 70% chance of failure
        raise ConnectionError("API unavailable")
    return {"status": "ok", "data": "response"}

# try:
#     result = unstable_api_call("https://api.example.com/data")
# except Exception as e:
#     print(f"Final error: {e}")


# 3. VALIDATE INPUT DECORATOR
def validate_positive(*param_names):
    """Ensure specified parameters are positive numbers"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            for i, arg in enumerate(args):
                if i < len(params) and params[i] in param_names:
                    if not isinstance(arg, (int, float)) or arg <= 0:
                        raise ValueError(f"Parameter '{params[i]}' must be positive, got {arg}")
            
            for name in param_names:
                if name in kwargs and (not isinstance(kwargs[name], (int, float)) or kwargs[name] <= 0):
                    raise ValueError(f"Parameter '{name}' must be positive, got {kwargs[name]}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_positive("price", "quantity")
def create_order_line(product, price, quantity):
    return {"product": product, "price": price, "quantity": quantity, "total": price * quantity}

print(create_order_line("Laptop", 999.99, 2))
# {'product': 'Laptop', 'price': 999.99, 'quantity': 2, 'total': 1999.98}

# create_order_line("Laptop", -100, 2)  # ValueError: price must be positive


# 4. CACHE DECORATOR (functools version — production-ready)
from functools import lru_cache

@lru_cache(maxsize=128)     # Cache up to 128 results
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(50))        # Instant with cache, painfully slow without
print(fibonacci.cache_info())  # CacheInfo(hits=48, misses=51, maxsize=128, currsize=51)
```

---

## Part 9: Docstrings and Documentation

### Writing Good Docstrings

```python
def calculate_compound_interest(principal, annual_rate, years, compounds_per_year=12):
    """
    Calculate compound interest on an investment.
    
    Compound interest grows your money faster than simple interest because
    interest is calculated on both principal AND accumulated interest.
    
    Args:
        principal (float): Initial investment amount in dollars
        annual_rate (float): Annual interest rate as a decimal (e.g., 0.05 for 5%)
        years (int): Number of years to invest
        compounds_per_year (int): How many times interest compounds per year.
                                  Default: 12 (monthly)
    
    Returns:
        dict: Dictionary containing:
            - 'final_amount' (float): Total value after investment period
            - 'interest_earned' (float): Total interest earned
            - 'growth_rate' (float): Percentage growth
    
    Raises:
        ValueError: If principal or annual_rate is negative
        ValueError: If years or compounds_per_year is less than 1
    
    Examples:
        >>> result = calculate_compound_interest(1000, 0.05, 10)
        >>> print(f"${result['final_amount']:,.2f}")
        $1,647.01
        
        >>> result = calculate_compound_interest(5000, 0.07, 20, compounds_per_year=4)
        >>> print(f"Growth: {result['growth_rate']:.1f}%")
        Growth: 287.4%
    """
    if principal < 0 or annual_rate < 0:
        raise ValueError("Principal and rate must be non-negative")
    if years < 1 or compounds_per_year < 1:
        raise ValueError("Years and compounds_per_year must be at least 1")
    
    n = compounds_per_year
    r = annual_rate
    t = years
    P = principal
    
    # Compound interest formula: A = P(1 + r/n)^(nt)
    final_amount = P * (1 + r / n) ** (n * t)
    interest_earned = final_amount - P
    growth_rate = (interest_earned / P) * 100
    
    return {
        "final_amount": round(final_amount, 2),
        "interest_earned": round(interest_earned, 2),
        "growth_rate": round(growth_rate, 2)
    }

# Accessing documentation
help(calculate_compound_interest)  # Prints the docstring in terminal

# Quick peek
print(calculate_compound_interest.__doc__[:100])  # First 100 chars of docstring
```

---

## Part 10: Worked Examples

### Worked Example 1: Data Pipeline

```python
def load_data(raw_data):
    """Parse raw CSV-like data into list of dicts"""
    lines = raw_data.strip().splitlines()
    headers = [h.strip() for h in lines[0].split(",")]
    return [
        {h: v.strip() for h, v in zip(headers, line.split(","))}
        for line in lines[1:]
        if line.strip()
    ]

def clean_data(records):
    """Clean and type-convert records"""
    cleaned = []
    for record in records:
        try:
            cleaned.append({
                "name": record["name"].strip().title(),
                "age": int(record["age"]),
                "salary": float(record["salary"]),
                "department": record["department"].strip().upper()
            })
        except (ValueError, KeyError):
            pass  # Skip malformed records
    return cleaned

def filter_data(records, **criteria):
    """Filter records by any combination of criteria"""
    filtered = records
    for field, value in criteria.items():
        if field.endswith("_min"):
            actual_field = field[:-4]
            filtered = [r for r in filtered if r.get(actual_field, 0) >= value]
        elif field.endswith("_max"):
            actual_field = field[:-4]
            filtered = [r for r in filtered if r.get(actual_field, 0) <= value]
        else:
            filtered = [r for r in filtered if r.get(field) == value]
    return filtered

def aggregate_data(records, group_by, *agg_fields):
    """Group records and calculate statistics"""
    groups = {}
    for record in records:
        key = record.get(group_by, "Unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(record)
    
    result = {}
    for group, group_records in groups.items():
        result[group] = {
            "count": len(group_records)
        }
        for field in agg_fields:
            values = [r[field] for r in group_records if field in r]
            if values:
                result[group][f"{field}_avg"] = sum(values) / len(values)
                result[group][f"{field}_total"] = sum(values)
    return result

def format_report(aggregated, title="Report"):
    """Format aggregated data into readable report"""
    lines = [f"\n{'=' * 50}", f"{title:^50}", "=" * 50]
    for group, stats in aggregated.items():
        lines.append(f"\n{group}:")
        for stat, value in stats.items():
            if isinstance(value, float):
                lines.append(f"  {stat:<20} ${value:>10,.2f}")
            else:
                lines.append(f"  {stat:<20} {value:>10}")
    return "\n".join(lines)

# Run the pipeline
raw_csv = """
name, age, salary, department
alice johnson, 28, 95000, engineering
bob smith, 35, 120000, engineering
carol davis, 42, 85000, marketing
dave wilson, 29, 78000, marketing
eve brown, 38, 110000, engineering
frank chen, 31, 92000, product
"""

# Each function does ONE thing — compose them together
records = load_data(raw_csv)
records = clean_data(records)
senior_records = filter_data(records, age_min=30)
by_department = aggregate_data(senior_records, "department", "salary")
print(format_report(by_department, "Senior Staff by Department"))
```

### Worked Example 2: Validator Framework

```python
def make_validator(*rules):
    """
    Create a validator function from multiple rule functions.
    Each rule should return (is_valid, error_message).
    """
    def validate(value):
        errors = []
        for rule in rules:
            valid, error = rule(value)
            if not valid:
                errors.append(error)
        return len(errors) == 0, errors
    return validate

# Define reusable rules
def min_length(n):
    return lambda v: (len(str(v)) >= n, f"Must be at least {n} characters")

def max_length(n):
    return lambda v: (len(str(v)) <= n, f"Must be at most {n} characters")

def contains_digit():
    return lambda v: (any(c.isdigit() for c in str(v)), "Must contain at least one digit")

def contains_upper():
    return lambda v: (any(c.isupper() for c in str(v)), "Must contain at least one uppercase letter")

def no_spaces():
    return lambda v: (" " not in str(v), "Must not contain spaces")

def is_email():
    return lambda v: ("@" in str(v) and "." in str(v).split("@")[-1], "Must be a valid email")

# Build specific validators
password_validator = make_validator(
    min_length(8),
    max_length(50),
    contains_digit(),
    contains_upper(),
    no_spaces()
)

email_validator = make_validator(
    min_length(5),
    max_length(100),
    is_email()
)

# Test
test_passwords = ["abc", "password", "Password1", "My Pass 1", "Secure@Pass1"]
for pwd in test_passwords:
    valid, errors = password_validator(pwd)
    status = "✓" if valid else "✗"
    print(f"{status} '{pwd}': {errors if errors else 'Valid!'}")

# Output:
# ✗ 'abc': ['Must be at least 8 characters', 'Must contain at least one digit', ...]
# ✗ 'password': ['Must contain at least one digit', 'Must contain at least one uppercase letter']
# ✓ 'Password1': Valid!
# ✗ 'My Pass 1': ['Must not contain spaces']
# ✓ 'Secure@Pass1': Valid!
```

### Worked Example 3: Event System

```python
class EventSystem:
    """Simple publish-subscribe event system using functions"""
    
    def __init__(self):
        self._handlers = {}   # event_name → list of handler functions
    
    def on(self, event_name):
        """Decorator to register a function as an event handler"""
        def decorator(func):
            if event_name not in self._handlers:
                self._handlers[event_name] = []
            self._handlers[event_name].append(func)
            return func
        return decorator
    
    def emit(self, event_name, **data):
        """Fire an event, calling all registered handlers"""
        if event_name in self._handlers:
            for handler in self._handlers[event_name]:
                handler(**data)
    
    def off(self, event_name, func):
        """Remove a specific handler"""
        if event_name in self._handlers:
            self._handlers[event_name].remove(func)

# Create the event system
events = EventSystem()

# Register handlers using the @events.on decorator
@events.on("user_signup")
def send_welcome_email(username, email, **kwargs):
    print(f"📧 Sending welcome email to {email}")

@events.on("user_signup")
def create_default_settings(username, **kwargs):
    print(f"⚙️  Creating default settings for {username}")

@events.on("user_signup")
def notify_admin(username, email, **kwargs):
    print(f"🔔 Admin notified: new user {username} ({email})")

@events.on("purchase")
def update_inventory(product, quantity, **kwargs):
    print(f"📦 Updating inventory: -{quantity} {product}")

@events.on("purchase")
def send_receipt(product, price, email, **kwargs):
    print(f"🧾 Receipt sent to {email}: {product} - ${price:.2f}")

# Fire events
print("=== User Signs Up ===")
events.emit("user_signup", username="alice", email="alice@example.com")

print("\n=== User Makes Purchase ===")
events.emit("purchase", product="Laptop", quantity=1, price=999.99, email="alice@example.com")

# Output:
# === User Signs Up ===
# 📧 Sending welcome email to alice@example.com
# ⚙️  Creating default settings for alice
# 🔔 Admin notified: new user alice (alice@example.com)
#
# === User Makes Purchase ===
# 📦 Updating inventory: -1 Laptop
# 🧾 Receipt sent to alice@example.com: Laptop - $999.99
```

---

## Practice Problems

### Easy (1–5)

**Problem 1: Temperature Converter**
Write functions to convert between Celsius, Fahrenheit, and Kelvin.

```python
def celsius_to_fahrenheit(c):
    pass  # F = (C × 9/5) + 32

def fahrenheit_to_celsius(f):
    pass  # C = (F − 32) × 5/9

def celsius_to_kelvin(c):
    pass  # K = C + 273.15

# Test
print(celsius_to_fahrenheit(100))  # 212.0
print(fahrenheit_to_celsius(32))   # 0.0
print(celsius_to_kelvin(0))        # 273.15
```

**Problem 2: Flexible Greeting**
Write a function that greets a person, with optional language and time-of-day.

```python
def greet(name, language="english", time_of_day="morning"):
    # Support: english, spanish, french
    # Time: morning, afternoon, evening
    pass

print(greet("Alice"))                            # Good morning, Alice!
print(greet("Bob", "spanish", "afternoon"))       # Buenas tardes, Bob!
print(greet("Carol", time_of_day="evening"))      # Good evening, Carol!
```

**Problem 3: Statistics Functions**
Write individual functions for mean, median, and mode. Then a summary function.

```python
def mean(numbers): pass
def median(numbers): pass
def mode(numbers): pass
def summary(numbers): pass  # Returns dict with all three

print(summary([1, 2, 2, 3, 4, 4, 4, 5]))
# {'mean': 3.125, 'median': 3.5, 'mode': 4}
```

**Problem 4: Repeater**
Write a higher-order function that takes a function and returns a new function that calls the original n times.

```python
def repeat(func, n):
    def wrapper(*args, **kwargs):
        for _ in range(n):
            func(*args, **kwargs)
    return wrapper

say_hello = lambda name: print(f"Hello, {name}!")
say_hello_3x = repeat(say_hello, 3)
say_hello_3x("Alice")
# Hello, Alice!
# Hello, Alice!
# Hello, Alice!
```

**Problem 5: Safe Divider**
Write a function that divides two numbers but handles errors gracefully.

```python
def safe_divide(a, b, default=None):
    pass

print(safe_divide(10, 2))       # 5.0
print(safe_divide(10, 0))       # None
print(safe_divide(10, 0, -1))   # -1
```

### Medium (6–12)

**Problem 6: Memoized Fibonacci**
Implement Fibonacci using a closure for caching (without using `lru_cache`).

```python
def make_fibonacci():
    cache = {0: 0, 1: 1}
    
    def fib(n):
        # Your code here
        pass
    
    return fib

fib = make_fibonacci()
print([fib(i) for i in range(10)])  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

**Problem 7: Function Pipeline**
Create a `pipeline` function that takes a list of functions and applies them in sequence.

```python
def pipeline(*functions):
    def execute(value):
        result = value
        for func in functions:
            result = func(result)
        return result
    return execute

clean_text = pipeline(
    str.strip,
    str.lower,
    lambda s: s.replace(" ", "_"),
    lambda s: "".join(c for c in s if c.isalnum() or c == "_")
)

print(clean_text("  Hello World! 2025  "))  # hello_world_2025
```

**Problem 8: Argument Logger**
Write a decorator that logs every call to a function, including its arguments and return value.

```python
def log_calls(func):
    # Your code here
    pass

@log_calls
def add(a, b):
    return a + b

@log_calls
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

result = add(3, 4)
result2 = greet("Alice", greeting="Hi")
# [LOG] Calling add(3, 4)
# [LOG] add returned: 7
# [LOG] Calling greet('Alice', greeting='Hi')
# [LOG] greet returned: 'Hi, Alice!'
```

**Problem 9: Flexible Formatter**
Write a function that returns a formatting function for different data types.

```python
def make_formatter(format_type, **options):
    pass

money_fmt = make_formatter("currency", symbol="$", decimals=2)
percent_fmt = make_formatter("percent", decimals=1)
date_fmt = make_formatter("date", separator="-")

print(money_fmt(1234567.89))   # $1,234,567.89
print(percent_fmt(0.8567))     # 85.7%
print(date_fmt((2025, 1, 15))) # 2025-01-15
```

**Problem 10: Rate Limiter**
Write a decorator that limits how often a function can be called.

```python
import time

def rate_limit(calls_per_second):
    # Your code here
    pass

@rate_limit(2)   # Max 2 calls per second
def fetch_data(url):
    print(f"Fetching {url}")
    return "data"

# Should work for first 2 calls, then wait
```

**Problem 11: Partial Application**
Implement your own version of `functools.partial` — a function that pre-fills some arguments.

```python
def partial(func, *preset_args, **preset_kwargs):
    # Your code here
    pass

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)
double = partial(lambda x, n: x * n, n=2)

print(square(5))   # 25
print(cube(3))     # 27
print(double(7))   # 14
```

**Problem 12: Type Checker Decorator**
Write a decorator that validates argument types based on annotations.

```python
def check_types(func):
    # Hint: use func.__annotations__ to get type hints
    pass

@check_types
def calculate_interest(principal: float, rate: float, years: int) -> float:
    return principal * (1 + rate) ** years

print(calculate_interest(1000.0, 0.05, 10))   # Works
# calculate_interest("1000", 0.05, 10)         # TypeError!
```

### Hard (13–20)

**Problem 13: Composable Sorter**
Build a sorting system where you can combine multiple sort criteria.

```python
def multi_sort(*criteria):
    """
    Sort by multiple criteria, in order.
    Each criterion is (field, reverse=False)
    """
    pass

employees = [
    {"name": "Alice", "dept": "Engineering", "salary": 95000},
    {"name": "Bob", "dept": "Marketing", "salary": 85000},
    {"name": "Carol", "dept": "Engineering", "salary": 105000},
    {"name": "Dave", "dept": "Marketing", "salary": 85000},
]

# Sort by dept (A-Z), then by salary (high to low)
sorted_employees = sorted(employees, key=multi_sort(("dept", False), ("salary", True)))
```

**Problem 14: Curry Function**
Implement currying — converting a multi-argument function into a chain of single-argument functions.

```python
def curry(func):
    pass

@curry
def add(a, b, c):
    return a + b + c

# Should work both ways:
print(add(1)(2)(3))    # 6
print(add(1, 2)(3))    # 6
print(add(1)(2, 3))    # 6
print(add(1, 2, 3))    # 6
```

**Problem 15: Memoize with TTL**
Extend memoization to expire cached results after a time-to-live period.

```python
def memoize_ttl(ttl_seconds):
    """Cache results but expire them after ttl_seconds"""
    pass

@memoize_ttl(5)
def fetch_stock_price(ticker):
    print(f"Fetching price for {ticker}...")
    return {"ticker": ticker, "price": 150.00}

print(fetch_stock_price("AAPL"))  # Fetches
print(fetch_stock_price("AAPL"))  # Cache hit
time.sleep(6)
print(fetch_stock_price("AAPL"))  # Cache expired, fetches again
```

**Problem 16: Function Signature Inspector**
Write a function that displays detailed info about any function.

```python
def inspect_function(func):
    """
    Print:
    - Function name and docstring
    - Parameters with types and defaults
    - Whether it has *args or **kwargs
    """
    pass

@inspect_function
def process_order(order_id: int, items: list, discount: float = 0.0, **metadata):
    """Process a customer order."""
    pass
```

**Problem 17: Retry with Exponential Backoff**
Enhance the retry decorator with exponential backoff.

```python
def retry_with_backoff(max_attempts=3, base_delay=1, exceptions=(Exception,)):
    """
    Retry with exponential backoff:
    Attempt 1: fails → wait 1s
    Attempt 2: fails → wait 2s
    Attempt 3: fails → wait 4s
    """
    pass
```

**Problem 18: Observer Pattern**
Implement the observer pattern using functions and closures.

```python
def make_observable(initial_value):
    """
    Create an observable value.
    Returns (getter, setter, subscribe) where:
    - getter() returns current value
    - setter(new_value) updates the value and notifies subscribers
    - subscribe(callback) registers a function to call on changes
    """
    pass

price, set_price, on_price_change = make_observable(100.0)

@on_price_change
def log_change(old_val, new_val):
    print(f"Price changed: ${old_val} → ${new_val}")

@on_price_change
def check_alert(old_val, new_val):
    if new_val > 150:
        print(f"⚠️ Price alert! ${new_val}")

set_price(120.0)   # Price changed: $100.0 → $120.0
set_price(160.0)   # Price changed: $120.0 → $160.0  AND  ⚠️ Price alert!
```

**Problem 19: Function Tracer**
Write a decorator that traces a function's execution, including recursive calls.

```python
def trace(func):
    """
    Print execution tree for recursive functions:
    fib(4)
      fib(3)
        fib(2)
          fib(1) → 1
          fib(0) → 0
        fib(2) → 1
      fib(1) → 1
    fib(4) → 3
    """
    pass

@trace
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)
```

**Problem 20: Mini Dependency Injection Container**
Build a simple DI container that manages function dependencies.

```python
class Container:
    """
    Dependency injection container.
    Register services, then resolve them automatically.
    """
    def __init__(self):
        self._services = {}
    
    def register(self, name, factory):
        """Register a service factory function"""
        pass
    
    def resolve(self, name):
        """Get an instance of a registered service"""
        pass
    
    def inject(self, func):
        """Decorator that automatically injects dependencies by parameter name"""
        pass

container = Container()
container.register("db", lambda: {"connection": "postgresql://localhost/mydb"})
container.register("logger", lambda: print)
container.register("email_service", lambda: lambda to, msg: print(f"Email to {to}: {msg}"))

@container.inject
def create_user(username, db, logger, email_service):
    logger(f"Creating user {username}")
    # db and email_service are automatically injected!
    email_service(f"{username}@example.com", "Welcome!")

create_user("alice")
```

---

## Answer Keys (Selected Problems)

### Problem 3 Solution:
```python
def mean(numbers):
    return sum(numbers) / len(numbers)

def median(numbers):
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
    return sorted_nums[mid]

def mode(numbers):
    freq = {}
    for n in numbers:
        freq[n] = freq.get(n, 0) + 1
    return max(freq, key=freq.get)

def summary(numbers):
    return {"mean": mean(numbers), "median": median(numbers), "mode": mode(numbers)}
```

### Problem 6 Solution:
```python
def make_fibonacci():
    cache = {0: 0, 1: 1}
    
    def fib(n):
        if n not in cache:
            cache[n] = fib(n - 1) + fib(n - 2)
        return cache[n]
    
    return fib
```

### Problem 11 Solution:
```python
def partial(func, *preset_args, **preset_kwargs):
    def wrapper(*args, **kwargs):
        combined_args = preset_args + args
        combined_kwargs = {**preset_kwargs, **kwargs}
        return func(*combined_args, **combined_kwargs)
    return wrapper
```

### Problem 14 Solution:
```python
import inspect

def curry(func):
    n_args = len(inspect.signature(func).parameters)
    
    def curried(*args):
        if len(args) >= n_args:
            return func(*args[:n_args])
        return lambda *more: curried(*(args + more))
    
    return curried
```

---

## Mini-Project: Functional Data Processing Library

### Project Goal
Build a mini library of composable, reusable data processing functions — inspired by how real data engineering pipelines work.

```python
"""
mini_pipeline.py — A tiny functional data processing library
"""

# ── CORE UTILITIES ─────────────────────────────────────────────────────────

def compose(*functions):
    """Right-to-left function composition: compose(f, g)(x) = f(g(x))"""
    from functools import reduce
    return reduce(lambda f, g: lambda x: f(g(x)), functions)

def pipe(*functions):
    """Left-to-right function application: pipe(f, g)(x) = g(f(x))"""
    from functools import reduce
    return reduce(lambda f, g: lambda x: g(f(x)), functions)

def memoize(func):
    """Cache function results"""
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

def timer(func):
    """Measure execution time"""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"⏱  {func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper

# ── COLLECTION TRANSFORMERS ────────────────────────────────────────────────

def select(*fields):
    """Return a transformer that keeps only specified fields"""
    def transform(records):
        return [{f: r[f] for f in fields if f in r} for r in records]
    return transform

def where(**conditions):
    """Return a transformer that filters records by conditions"""
    def transform(records):
        result = records
        for field, value in conditions.items():
            if callable(value):
                result = [r for r in result if value(r.get(field))]
            else:
                result = [r for r in result if r.get(field) == value]
        return result
    return transform

def add_field(field_name, compute_func):
    """Return a transformer that adds a computed field"""
    def transform(records):
        return [{**r, field_name: compute_func(r)} for r in records]
    return transform

def sort_by(field, reverse=False):
    """Return a transformer that sorts by a field"""
    def transform(records):
        return sorted(records, key=lambda r: r.get(field, 0), reverse=reverse)
    return transform

def group_by(field, *agg_specs):
    """
    Group records by a field and apply aggregations.
    agg_spec: (output_name, source_field, agg_func)
    """
    def transform(records):
        groups = {}
        for record in records:
            key = record.get(field, "Unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(record)
        
        result = []
        for group_key, group_records in groups.items():
            row = {field: group_key, "count": len(group_records)}
            for out_name, src_field, agg_func in agg_specs:
                values = [r[src_field] for r in group_records if src_field in r]
                row[out_name] = agg_func(values) if values else None
            result.append(row)
        
        return result
    return transform

def top_n(n):
    """Return only the first n records"""
    return lambda records: records[:n]

# ── AGGREGATION FUNCTIONS ──────────────────────────────────────────────────

def agg_sum(values): return sum(values)
def agg_mean(values): return sum(values) / len(values) if values else 0
def agg_max(values): return max(values)
def agg_min(values): return min(values)
def agg_count(values): return len(values)

# ── DEMO ───────────────────────────────────────────────────────────────────

# Sample dataset
employees = [
    {"name": "Alice",   "dept": "Engineering", "salary": 95000,  "years": 4},
    {"name": "Bob",     "dept": "Marketing",   "salary": 72000,  "years": 2},
    {"name": "Carol",   "dept": "Engineering", "salary": 110000, "years": 7},
    {"name": "Dave",    "dept": "Marketing",   "salary": 68000,  "years": 1},
    {"name": "Eve",     "dept": "Engineering", "salary": 88000,  "years": 3},
    {"name": "Frank",   "dept": "Product",     "salary": 95000,  "years": 5},
    {"name": "Grace",   "dept": "Product",     "salary": 102000, "years": 6},
    {"name": "Henry",   "dept": "Marketing",   "salary": 76000,  "years": 3},
]

# ── PIPELINE 1: Top earners ────────────────────────────────────────────────
print("=== TOP 3 EARNERS ===")
top_earners = pipe(
    sort_by("salary", reverse=True),
    top_n(3),
    select("name", "dept", "salary")
)(employees)

for emp in top_earners:
    print(f"  {emp['name']:<10} {emp['dept']:<15} ${emp['salary']:,.0f}")

# ── PIPELINE 2: Department summary ────────────────────────────────────────
print("\n=== DEPT SALARY SUMMARY ===")
dept_summary = pipe(
    group_by("dept",
        ("avg_salary", "salary", agg_mean),
        ("total_salary", "salary", agg_sum),
        ("max_salary", "salary", agg_max)
    ),
    sort_by("avg_salary", reverse=True)
)(employees)

for dept in dept_summary:
    print(f"  {dept['dept']:<15} "
          f"avg: ${dept['avg_salary']:>8,.0f}  "
          f"total: ${dept['total_salary']:>9,.0f}  "
          f"max: ${dept['max_salary']:>8,.0f}")

# ── PIPELINE 3: Senior engineers ──────────────────────────────────────────
print("\n=== SENIOR ENGINEERS (4+ years) ===")
senior_engineers = pipe(
    where(dept="Engineering", years=lambda y: y >= 4),
    add_field("bonus", lambda r: r["salary"] * 0.10),
    add_field("total_comp", lambda r: r["salary"] + r["bonus"]),
    sort_by("total_comp", reverse=True),
    select("name", "salary", "bonus", "total_comp")
)(employees)

for emp in senior_engineers:
    print(f"  {emp['name']:<10} "
          f"base: ${emp['salary']:,.0f}  "
          f"bonus: ${emp['bonus']:,.0f}  "
          f"total: ${emp['total_comp']:,.0f}")
```

---

## Chapter Summary

You've mastered functions — one of the most important concepts in all of programming.

✅ **Function Basics**: `def`, parameters, `return`, docstrings
✅ **All Parameter Types**: positional, keyword, default, `*args`, `**kwargs`
✅ **Scope**: LEGB rule, `global`, `nonlocal`
✅ **Lambda**: Anonymous functions for short one-liners
✅ **Higher-Order Functions**: Functions that take/return functions, `map`, `filter`
✅ **Closures**: Functions that remember their enclosing scope
✅ **Decorators**: Wrapping functions with extra behaviour using `@`
✅ **Documentation**: Writing meaningful docstrings

**Key Takeaways:**
- Functions should do **one thing** and do it well
- If code is repeated more than once, it belongs in a function
- `*args` and `**kwargs` make functions maximally flexible
- Decorators are just functions that return functions — elegant and powerful
- Closures let you create function factories and maintain state

**Next Chapter Preview:**
Chapter 7 covers **File I/O and Exception Handling** — how to read/write files, handle errors gracefully, and work with real data formats like CSV and JSON. This is where your programs start interacting with the real world!
