# Chapter 10: Advanced Python Features

## Part 1: Why Go Advanced?

### The Problem: Working Code Isn't Always Good Code

By now, you can solve almost any problem in Python. But there's a gap between *working* code and *elegant, efficient* code.

```python
# ── WORKING but inefficient ────────────────────────────────────────────
# Generate first 10 million squares — holds ALL in memory!
squares = []
for i in range(10_000_000):
    squares.append(i ** 2)
# Uses ~400 MB of RAM just to hold the list
# What if you only need to process them one at a time?

# ── WORKING but verbose ────────────────────────────────────────────────
# Filter and transform a list
result = []
for x in range(100):
    if x % 2 == 0:
        result.append(x ** 2)

# ── WORKING but repetitive ────────────────────────────────────────────
# Same boilerplate pattern repeated across 20 functions
def func1():
    print("Starting func1")
    # ... actual logic ...
    print("Done func1")

def func2():
    print("Starting func2")
    # ... actual logic ...
    print("Done func2")
```

This chapter teaches you the tools that turn these into:

```python
# Memory-efficient: generates values ONE AT A TIME
squares = (i ** 2 for i in range(10_000_000))  # Uses ~200 bytes!

# Concise: one expressive line
result = [x ** 2 for x in range(100) if x % 2 == 0]

# DRY: behaviour added once, applied everywhere
@log_calls
def func1(): ...

@log_calls
def func2(): ...
```

---

## Part 2: Comprehensions — Deep Dive

### You Already Know the Basics — Let's Go Deeper

Comprehensions are syntactic sugar for creating collections using a concise, readable expression.

#### List Comprehensions

```python
# ── ANATOMY ───────────────────────────────────────────────────────────────
# [expression   for item in iterable   if condition]
#  ↑ transform   ↑ loop                 ↑ filter (optional)

# Equivalent to:
result = []
for item in iterable:
    if condition:
        result.append(expression)

# ── EXAMPLES ──────────────────────────────────────────────────────────────

# Basic: squares of numbers 0-9
squares = [x ** 2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With filter: only even squares
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]

# With transformation: celsius to fahrenheit
temps_c = [0, 20, 37, 100]
temps_f = [(c * 9/5) + 32 for c in temps_c]
print(temps_f)  # [32.0, 68.0, 98.6, 212.0]

# With method calls: clean and normalize strings
raw = ["  Alice  ", "BOB", "  carol  ", "DAVE"]
cleaned = [name.strip().title() for name in raw]
print(cleaned)  # ['Alice', 'Bob', 'Carol', 'Dave']

# With conditions on expression (not filter — ternary in expression)
numbers = [-3, -1, 0, 2, 5, -4, 8]
abs_or_zero = [x if x > 0 else 0 for x in numbers]
# Note: this is [expression_if_true if condition else expression_if_false]
print(abs_or_zero)  # [0, 0, 0, 2, 5, 0, 8]

# With multiple conditions
divisible_by_3_and_5 = [x for x in range(100) if x % 3 == 0 if x % 5 == 0]
print(divisible_by_3_and_5)  # [0, 15, 30, 45, 60, 75, 90]
# Multiple `if` clauses are joined by AND

# Calling functions
import math
log_values = [round(math.log(x), 2) for x in range(1, 11)]
print(log_values)  # [0.0, 0.69, 1.1, 1.39, 1.61, 1.79, 1.95, 2.08, 2.2, 2.3]
```

#### Nested Comprehensions — Flattening and Matrices

```python
# ── FLATTENING a nested list ──────────────────────────────────────────────
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# The inner loop goes in the ORDER you'd write a for loop
flat = [x for row in nested for x in row]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Equivalent to:
flat = []
for row in nested:
    for x in row:
        flat.append(x)

# ── BUILDING a matrix ─────────────────────────────────────────────────────
# Outer list = rows, inner list = columns
matrix_3x3 = [[i * 3 + j for j in range(3)] for i in range(3)]
print(matrix_3x3)
# [[0, 1, 2],
#  [3, 4, 5],
#  [6, 7, 8]]

# Transpose a matrix (rows become columns)
original = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]

transposed = [[original[row][col] for row in range(3)] for col in range(3)]
print(transposed)
# [[1, 4, 7],
#  [2, 5, 8],
#  [3, 6, 9]]

# Cleaner with zip:
transposed = [list(row) for row in zip(*original)]
print(transposed)  # Same result

# ── REAL-WORLD: flatten API response ─────────────────────────────────────
api_response = {
    "users": [
        {"name": "Alice", "tags": ["admin", "user"]},
        {"name": "Bob", "tags": ["user", "premium"]},
        {"name": "Carol", "tags": ["admin"]},
    ]
}

# All unique tags across all users
all_tags = {tag for user in api_response["users"] for tag in user["tags"]}
print(all_tags)  # {'admin', 'user', 'premium'}

# ── WHEN NOT TO USE NESTED COMPREHENSIONS ────────────────────────────────
# Two levels: OK
# Three or more levels: Use regular loops — readability suffers badly!

# ❌ Too complex — hard to read
result = [x*y for sublist in nested for x in sublist for y in range(x)]

# ✓ Use regular loops when nesting exceeds 2 levels
result = []
for sublist in nested:
    for x in sublist:
        for y in range(x):
            result.append(x * y)
```

#### Dictionary Comprehensions

```python
# ── ANATOMY ───────────────────────────────────────────────────────────────
# {key_expr: value_expr   for item in iterable   if condition}

# Basic: word → length mapping
words = ["python", "is", "amazing", "and", "powerful"]
word_lengths = {word: len(word) for word in words}
print(word_lengths)
# {'python': 6, 'is': 2, 'amazing': 7, 'and': 3, 'powerful': 8}

# Invert a dictionary (swap keys and values)
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
print(inverted)  # {1: 'a', 2: 'b', 3: 'c'}

# Filter: only keep entries where value > 5
long_words = {word: length for word, length in word_lengths.items() if length > 5}
print(long_words)  # {'python': 6, 'amazing': 7, 'powerful': 8}

# Transform: apply function to all values
prices = {"apple": 1.20, "banana": 0.50, "cherry": 3.00}
discounted = {item: round(price * 0.90, 2) for item, price in prices.items()}
print(discounted)  # {'apple': 1.08, 'banana': 0.45, 'cherry': 2.7}

# From two lists (zip)
keys = ["name", "age", "city"]
values = ["Alice", 28, "New York"]
profile = {k: v for k, v in zip(keys, values)}
print(profile)  # {'name': 'Alice', 'age': 28, 'city': 'New York'}

# ── REAL-WORLD: grouping records ──────────────────────────────────────────
employees = [
    {"name": "Alice", "dept": "Engineering"},
    {"name": "Bob", "dept": "Marketing"},
    {"name": "Carol", "dept": "Engineering"},
    {"name": "Dave", "dept": "Marketing"},
]

# Group names by department
dept_to_names = {}
for emp in employees:
    dept = emp["dept"]
    dept_to_names.setdefault(dept, []).append(emp["name"])

print(dept_to_names)
# {'Engineering': ['Alice', 'Carol'], 'Marketing': ['Bob', 'Dave']}

# Index by unique key
by_name = {emp["name"]: emp for emp in employees}
print(by_name["Alice"])  # {'name': 'Alice', 'dept': 'Engineering'}
```

#### Set Comprehensions

```python
# ── ANATOMY ───────────────────────────────────────────────────────────────
# {expression   for item in iterable   if condition}
# Same as list comprehension but with {} → produces a set (unique, unordered)

words = "the quick brown fox jumps over the lazy dog".split()

# Unique words
unique_words = {word for word in words}
print(unique_words)  # {'the', 'quick', 'brown', ...} — no duplicates

# Unique first letters
first_letters = {word[0] for word in words}
print(first_letters)  # {'t', 'q', 'b', 'f', 'j', 'o', 'l', 'd'}

# Unique word lengths
lengths = {len(word) for word in words}
print(sorted(lengths))  # [2, 3, 4, 5] — sorted for readability

# ── REAL-WORLD: find unique domains from emails ───────────────────────────
emails = ["alice@gmail.com", "bob@yahoo.com", "carol@gmail.com", "dave@outlook.com"]
domains = {email.split("@")[1] for email in emails}
print(domains)  # {'gmail.com', 'yahoo.com', 'outlook.com'}
```

#### Generator Expressions — Lazy Comprehensions

```python
# ── THE KEY DIFFERENCE ────────────────────────────────────────────────────
# List comprehension  → builds the ENTIRE list in memory immediately
# Generator expression → computes values ONE AT A TIME, on demand

# List: all values computed NOW, stored in memory
squares_list = [x ** 2 for x in range(10_000_000)]  # ~400 MB!

# Generator: computes next value only when asked
squares_gen = (x ** 2 for x in range(10_000_000))   # ~200 bytes!

# Syntax: () instead of []
gen = (x ** 2 for x in range(5))
print(gen)           # <generator object ...>  (not a list!)
print(next(gen))     # 0  (computed now)
print(next(gen))     # 1  (computed now)
print(next(gen))     # 4  (computed now)
# Values are consumed — you can only iterate once!

# ── WHEN TO USE GENERATORS ────────────────────────────────────────────────
# ✓ When you only need to iterate once
# ✓ When the sequence is very large (or infinite)
# ✓ When you're passing to a function that consumes an iterable

# sum(), max(), min(), any(), all() work perfectly with generators
numbers = range(1, 1_000_001)
total = sum(x ** 2 for x in numbers)      # No huge list created!
print(f"Sum of squares: {total:,}")

# any() — short-circuits (stops when first True found)
has_even = any(x % 2 == 0 for x in [1, 3, 4, 7])  # Stops at 4
print(f"Has even: {has_even}")  # True

# all() — short-circuits (stops when first False found)
all_positive = all(x > 0 for x in [1, 2, -3, 4])  # Stops at -3
print(f"All positive: {all_positive}")  # False

# ── CHAINING GENERATOR EXPRESSIONS ───────────────────────────────────────
# No intermediate lists created at all!
data = range(1, 1001)
result = sum(x ** 2 for x in data if x % 2 == 0 if x % 3 == 0)
# Reads: sum of squares of numbers 1-1000 divisible by both 2 and 3
print(result)  # 6049796  (no list created)
```

---

## Part 3: Generators and `yield`

### The Problem: Memory and Lazy Evaluation

```python
# ── THE PROBLEM: reading a huge file ─────────────────────────────────────
# This loads the ENTIRE file into RAM — bad for 10GB log files!
def read_file_bad(filepath):
    with open(filepath) as f:
        return f.readlines()   # Entire file in memory!

lines = read_file_bad("huge.log")
for line in lines:
    process(line)

# ── GENERATOR SOLUTION: one line at a time ───────────────────────────────
def read_file_lazy(filepath):
    with open(filepath) as f:
        for line in f:
            yield line         # Produce one line, pause, wait for next()

# Memory usage: one line at a time, regardless of file size!
for line in read_file_lazy("huge.log"):
    process(line)
```

### `yield` — The Heart of Generators

```python
# A generator function contains yield
# When called, it returns a GENERATOR OBJECT (doesn't run yet!)
# Each next() call runs until the next yield, then PAUSES

def count_up(start, stop):
    print(f"Starting from {start}")    # Only runs when first next() is called
    current = start
    while current <= stop:
        print(f"About to yield {current}")
        yield current                  # Produce value, PAUSE here
        print(f"Resumed after yielding {current}")
        current += 1
    print("Done!")                     # Runs when generator is exhausted

gen = count_up(1, 3)
print("Generator created — nothing ran yet!")
print(next(gen))  # Starting from 1 / About to yield 1 → 1
print(next(gen))  # Resumed after 1 / About to yield 2 → 2
print(next(gen))  # Resumed after 2 / About to yield 3 → 3
# next(gen)       # Done! → StopIteration exception

# for loops handle StopIteration automatically
for n in count_up(1, 5):
    print(n)
```

### Generator Patterns

```python
# ── PATTERN 1: Infinite sequences ────────────────────────────────────────
def fibonacci():
    """Infinite Fibonacci sequence."""
    a, b = 0, 1
    while True:          # Forever!
        yield a
        a, b = b, a + b

# Take first 10 Fibonacci numbers
import itertools
first_10 = list(itertools.islice(fibonacci(), 10))
print(first_10)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Find first Fibonacci > 1000
gen = fibonacci()
result = next(x for x in gen if x > 1000)
print(result)  # 1597


# ── PATTERN 2: Pipelines ──────────────────────────────────────────────────
# Chain generators for memory-efficient data processing

def read_log_lines(filepath):
    """Source: generate lines from file"""
    with open(filepath) as f:
        for line in f:
            yield line.strip()

def filter_errors(lines):
    """Stage 1: keep only ERROR lines"""
    for line in lines:
        if "ERROR" in line:
            yield line

def parse_log_line(lines):
    """Stage 2: parse each line into structured dict"""
    for line in lines:
        parts = line.split(" ", 3)
        if len(parts) == 4:
            yield {
                "date": parts[0],
                "time": parts[1],
                "level": parts[2],
                "message": parts[3]
            }

def enrich_with_severity(records):
    """Stage 3: add severity score"""
    keywords = {"timeout": 3, "crash": 5, "failed": 2, "warning": 1}
    for record in records:
        score = sum(v for k, v in keywords.items() if k in record["message"].lower())
        record["severity"] = score
        yield record

# The pipeline — NOTHING runs until you iterate!
# Each record flows through ALL stages before the next is read
pipeline = enrich_with_severity(
    parse_log_line(
        filter_errors(
            read_log_lines("server.log")
        )
    )
)

# Only now does data actually flow through the pipeline
for record in pipeline:
    if record["severity"] >= 3:
        print(f"HIGH SEVERITY: {record['message']}")


# ── PATTERN 3: yield from — delegating to sub-generators ─────────────────
def chain(*iterables):
    """Our own version of itertools.chain"""
    for iterable in iterables:
        yield from iterable   # Equivalent to: for item in iterable: yield item

result = list(chain([1, 2], [3, 4], [5, 6]))
print(result)  # [1, 2, 3, 4, 5, 6]

def flatten(nested):
    """Recursively flatten arbitrarily nested lists"""
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)   # Recurse into sub-list
        else:
            yield item

deep = [1, [2, [3, [4, [5]]]]]
print(list(flatten(deep)))  # [1, 2, 3, 4, 5]


# ── PATTERN 4: Sending values INTO generators ─────────────────────────────
def running_average():
    """Generator that receives values and yields running average."""
    total = 0
    count = 0
    average = None
    
    while True:
        value = yield average    # yield produces average, receives next value
        if value is None:
            break
        total += value
        count += 1
        average = total / count

gen = running_average()
next(gen)          # Prime the generator (advance to first yield)
print(gen.send(10))   # 10.0
print(gen.send(20))   # 15.0
print(gen.send(30))   # 20.0
print(gen.send(40))   # 25.0


# ── PATTERN 5: Generator-based context manager ────────────────────────────
from contextlib import contextmanager
import time

@contextmanager
def timed_section(name):
    """Time a block of code using a generator-based context manager."""
    start = time.time()
    print(f"▶ Starting: {name}")
    try:
        yield                   # Code inside 'with' block runs here
    finally:
        elapsed = time.time() - start
        print(f"◀ Finished: {name} ({elapsed:.4f}s)")

with timed_section("data processing"):
    result = sum(x ** 2 for x in range(1_000_000))
    print(f"  Result: {result:,}")

# ▶ Starting: data processing
#   Result: 333,332,833,333,500,000
# ◀ Finished: data processing (0.1234s)
```

### Worked Example 1: Streaming CSV Processor

```python
import csv
from pathlib import Path

def stream_csv(filepath, type_hints=None):
    """
    Generator: stream rows from a CSV one at a time.
    Never loads entire file into memory.
    """
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = dict(row)
            if type_hints:
                for field, converter in type_hints.items():
                    if field in record:
                        try:
                            record[field] = converter(record[field])
                        except (ValueError, TypeError):
                            pass
            yield record

def batch(iterable, size):
    """
    Generator: group items from iterable into batches of given size.
    Useful for bulk database inserts, API calls, etc.
    """
    batch_items = []
    for item in iterable:
        batch_items.append(item)
        if len(batch_items) == size:
            yield batch_items
            batch_items = []
    if batch_items:         # Don't forget the last partial batch!
        yield batch_items

def filter_stream(iterable, **conditions):
    """Generator: filter records by field values."""
    for record in iterable:
        if all(record.get(k) == v for k, v in conditions.items()):
            yield record

def transform_stream(iterable, **transformers):
    """Generator: apply transformers to specific fields."""
    for record in iterable:
        new_record = dict(record)
        for field, func in transformers.items():
            if field in new_record:
                new_record[field] = func(new_record[field])
        yield new_record

# ── Usage: process 1M-row CSV in constant memory ─────────────────────────
rows = stream_csv(
    "sales.csv",
    type_hints={"amount": float, "quantity": int}
)

# Build a pipeline — nothing executes yet!
pipeline = transform_stream(
    filter_stream(rows, region="North"),
    amount=lambda x: round(x * 1.1, 2)    # Apply 10% markup
)

# Process in batches of 1000 (e.g., for bulk DB insert)
total_processed = 0
for batch_records in batch(pipeline, 1000):
    # Process batch (e.g., insert into database)
    total_processed += len(batch_records)
    # db.bulk_insert(batch_records)

print(f"Processed {total_processed} records")
```

---

## Part 4: Iterators — Understanding the Protocol

### What Makes Something Iterable?

```python
# The Iterator Protocol:
# __iter__(self) → returns the iterator object (usually self)
# __next__(self) → returns next value, raises StopIteration when done

# How `for x in obj:` works under the hood:
iterator = iter(obj)         # Calls obj.__iter__()
while True:
    try:
        x = next(iterator)   # Calls iterator.__next__()
        # ... loop body ...
    except StopIteration:
        break

# ── BUILD YOUR OWN ITERATOR ───────────────────────────────────────────────
class CountDown:
    """Counts from n down to 0."""
    
    def __init__(self, start):
        self.start = start
        self.current = start
    
    def __iter__(self):
        # Reset and return self — this object IS the iterator
        self.current = self.start
        return self
    
    def __next__(self):
        if self.current < 0:
            raise StopIteration
        value = self.current
        self.current -= 1
        return value

countdown = CountDown(5)
for n in countdown:
    print(n, end=" ")  # 5 4 3 2 1 0

print()

# Can iterate AGAIN because __iter__ resets state
for n in countdown:
    print(n, end=" ")  # 5 4 3 2 1 0

print()

# Also works with all iterator consumers
print(list(CountDown(3)))     # [3, 2, 1, 0]
print(sum(CountDown(10)))     # 55
print(max(CountDown(100)))    # 100


# ── BUILD AN ITERABLE (separate iterator object) ──────────────────────────
class NumberRange:
    """An iterable that creates a fresh iterator each time."""
    
    def __init__(self, start, stop, step=1):
        self.start = start
        self.stop = stop
        self.step = step
    
    def __iter__(self):
        return NumberRangeIterator(self.start, self.stop, self.step)
    
    def __len__(self):
        return max(0, (self.stop - self.start) // self.step)


class NumberRangeIterator:
    """The actual iterator — has the mutable current state."""
    
    def __init__(self, start, stop, step):
        self.current = start
        self.stop = stop
        self.step = step
    
    def __iter__(self):
        return self    # Iterator is its own iterator
    
    def __next__(self):
        if self.current >= self.stop:
            raise StopIteration
        value = self.current
        self.current += self.step
        return value

r = NumberRange(0, 10, 2)
print(list(r))     # [0, 2, 4, 6, 8]
print(list(r))     # [0, 2, 4, 6, 8] — fresh iterator each time!
print(len(r))      # 5
```

### Worked Example 2: Paginated API Iterator

```python
class PaginatedAPI:
    """
    Iterates over paginated API results transparently.
    Caller just `for item in api:` — pagination is hidden!
    """
    
    def __init__(self, base_url, page_size=20):
        self.base_url = base_url
        self.page_size = page_size
    
    def __iter__(self):
        return PaginatedAPIIterator(self.base_url, self.page_size)


class PaginatedAPIIterator:
    def __init__(self, base_url, page_size):
        self.base_url = base_url
        self.page_size = page_size
        self.page = 1
        self.buffer = []         # Current page's items
        self.exhausted = False
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # If buffer is empty, fetch next page
        if not self.buffer:
            if self.exhausted:
                raise StopIteration
            self._fetch_page()
        
        if not self.buffer:
            raise StopIteration
        
        return self.buffer.pop(0)
    
    def _fetch_page(self):
        """Simulate fetching a page from an API."""
        url = f"{self.base_url}?page={self.page}&limit={self.page_size}"
        print(f"Fetching: {url}")
        
        # Simulate API response (in reality: requests.get(url).json())
        # Total of 55 items, pages of 20
        total_items = 55
        start = (self.page - 1) * self.page_size
        end = min(start + self.page_size, total_items)
        
        if start >= total_items:
            self.exhausted = True
            return
        
        self.buffer = [{"id": i, "value": i * 10} for i in range(start, end)]
        
        if end >= total_items:
            self.exhausted = True
        
        self.page += 1


# Usage — caller doesn't need to know about pages at all!
api = PaginatedAPI("https://api.example.com/items", page_size=20)

# Just iterate normally
all_items = list(api)
print(f"Total items retrieved: {len(all_items)}")
# Fetching: ...?page=1&limit=20
# Fetching: ...?page=2&limit=20
# Fetching: ...?page=3&limit=20
# Total items retrieved: 55

# Or process lazily
for item in PaginatedAPI("https://api.example.com/items"):
    if item["id"] > 10:
        break  # Stop early — remaining pages never fetched!
```

---

## Part 5: Advanced Decorators

### Decorators with Arguments — Factory Pattern

```python
import time
import functools

# ── DECORATOR WITHOUT ARGS (simple) ──────────────────────────────────────
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper

# ── DECORATOR WITH ARGS (factory pattern) ────────────────────────────────
# To add args, you need THREE levels of nesting:
# Level 1: accepts decorator arguments
# Level 2: accepts the function
# Level 3: accepts the function's arguments

def retry(max_attempts=3, delay=1.0, exceptions=(Exception,)):
    """
    Retry a function on failure.
    
    @retry(max_attempts=5, delay=2.0, exceptions=(ConnectionError,))
    def unstable_function(): ...
    """
    def decorator(func):          # Level 2: receives the function
        @functools.wraps(func)
        def wrapper(*args, **kwargs):   # Level 3: receives func's args
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        print(f"Attempt {attempt}/{max_attempts} failed: {e}. "
                              f"Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"All {max_attempts} attempts failed.")
            raise last_exception
        return wrapper
    return decorator              # Level 1 returns the decorator


# Usage
@retry(max_attempts=3, delay=0.1, exceptions=(ValueError, ConnectionError))
def flaky_api_call(url):
    import random
    if random.random() < 0.7:
        raise ConnectionError("Connection refused")
    return {"data": "success"}

# try:
#     result = flaky_api_call("https://api.example.com")
# except ConnectionError as e:
#     print(f"Failed permanently: {e}")
```

### Stacking Decorators

```python
import time
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"⏱ {func.__name__}: {time.time() - start:.4f}s")
        return result
    return wrapper

def log_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"→ Calling {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"← {func.__name__} returned: {result!r}")
        return result
    return wrapper

def validate_args(*types):
    """Validate argument types at runtime."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i, (arg, expected_type) in enumerate(zip(args, types)):
                if not isinstance(arg, expected_type):
                    raise TypeError(
                        f"Argument {i} must be {expected_type.__name__}, "
                        f"got {type(arg).__name__}"
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Stacking decorators — applied BOTTOM UP
@timer           # Applied third (outermost)
@log_calls       # Applied second
@validate_args(int, int)   # Applied first (innermost, closest to function)
def add(a, b):
    return a + b

# Equivalent to: add = timer(log_calls(validate_args(int, int)(add)))
# Execution order: timer's wrapper → log_calls' wrapper → validate_args' wrapper → add

result = add(3, 4)
# → Calling add((3, 4), {})
# ← add returned: 7
# ⏱ add: 0.0001s
```

### Class-Based Decorators

```python
class memoize:
    """
    Memoization decorator as a class.
    Stores results in instance dict — persistent across calls.
    """
    
    def __init__(self, func):
        self.func = func
        self.cache = {}
        functools.update_wrapper(self, func)  # Preserve metadata
    
    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.func(*args)
        return self.cache[args]
    
    def clear_cache(self):
        self.cache.clear()
    
    @property
    def cache_size(self):
        return len(self.cache)

@memoize
def expensive_computation(n):
    print(f"Computing for {n}...")
    return sum(range(n))

print(expensive_computation(1000))   # Computing for 1000...
print(expensive_computation(1000))   # (cached — no print)
print(expensive_computation(2000))   # Computing for 2000...
print(f"Cache size: {expensive_computation.cache_size}")  # 2
expensive_computation.clear_cache()


class rate_limit:
    """
    Rate limiter as a class decorator.
    Limit function calls per time window.
    """
    
    def __init__(self, calls_per_second):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_called = 0
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - self.last_called
            remaining = self.min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self.last_called = time.time()
            return func(*args, **kwargs)
        return wrapper

@rate_limit(calls_per_second=2)
def api_call(endpoint):
    print(f"Calling {endpoint} at {time.time():.2f}")

# These will be spaced at least 0.5s apart
api_call("/users")
api_call("/products")
api_call("/orders")
```

### Decorator Recipes for Real Projects

```python
import functools
import time
import json

# ── 1. CACHE WITH TTL ─────────────────────────────────────────────────────
def cache_ttl(seconds):
    """Cache results, expire after `seconds`."""
    def decorator(func):
        cache = {}        # {args: (result, timestamp)}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = args + tuple(sorted(kwargs.items()))
            now = time.time()
            
            if key in cache:
                result, cached_at = cache[key]
                if now - cached_at < seconds:
                    return result           # Cache hit
            
            result = func(*args, **kwargs)  # Cache miss
            cache[key] = (result, now)
            return result
        
        wrapper.clear = lambda: cache.clear()
        return wrapper
    return decorator

@cache_ttl(seconds=60)
def get_stock_price(ticker):
    print(f"Fetching live price for {ticker}...")
    return 150.00  # Simulate API call


# ── 2. DEPRECATED ─────────────────────────────────────────────────────────
import warnings

def deprecated(reason="", replacement=None):
    """Mark a function as deprecated."""
    def decorator(func):
        msg = f"{func.__name__} is deprecated"
        if reason: msg += f": {reason}"
        if replacement: msg += f". Use {replacement} instead."
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        
        wrapper.__doc__ = f"[DEPRECATED] {func.__doc__ or ''}"
        return wrapper
    return decorator

@deprecated(reason="Use batch_process instead", replacement="batch_process()")
def old_process(data):
    """Old processing function."""
    return data


# ── 3. SINGLETON ──────────────────────────────────────────────────────────
def singleton(cls):
    """Ensure only one instance of a class exists."""
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self, url="sqlite:///db.sqlite"):
        print(f"Connecting to {url}...")
        self.url = url
        self.connected = True

db1 = DatabaseConnection()   # Connecting to sqlite:///db.sqlite
db2 = DatabaseConnection()   # No print — same instance returned!
print(db1 is db2)            # True


# ── 4. BENCHMARK ──────────────────────────────────────────────────────────
def benchmark(runs=100):
    """Run a function N times and report average/total time."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            result = None
            for _ in range(runs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                times.append(time.perf_counter() - start)
            
            avg = sum(times) / len(times)
            total = sum(times)
            print(f"Benchmark '{func.__name__}' ({runs} runs):")
            print(f"  Average: {avg*1000:.3f}ms")
            print(f"  Total:   {total*1000:.3f}ms")
            print(f"  Min:     {min(times)*1000:.3f}ms")
            print(f"  Max:     {max(times)*1000:.3f}ms")
            return result
        return wrapper
    return decorator

@benchmark(runs=1000)
def fast_function():
    return sum(range(100))

fast_function()
```

---

## Part 6: Advanced Context Managers

### Building Robust Context Managers

```python
from contextlib import contextmanager, suppress, ExitStack
import threading

# ── suppress — ignore specific exceptions ────────────────────────────────
# Instead of:
try:
    os.remove("might_not_exist.txt")
except FileNotFoundError:
    pass

# Use:
import os
with suppress(FileNotFoundError):
    os.remove("might_not_exist.txt")


# ── ExitStack — dynamic context managers ─────────────────────────────────
# Problem: you don't know at runtime how many context managers you need

# Without ExitStack — can't dynamically enter multiple CMs
def process_files_bad(filepaths):
    # Can't do: with open(f1) as f, open(f2) as g, ... (unknown count)
    pass

# With ExitStack — perfect for dynamic CMs
def process_files(filepaths):
    """Open and process any number of files."""
    with ExitStack() as stack:
        handles = [
            stack.enter_context(open(path))
            for path in filepaths
            # If any open() fails, all previously opened files are closed!
        ]
        
        # All files open, process them
        for handle in handles:
            yield handle.readline()
        
    # All files automatically closed here


# ── Thread-safe database connection pool ─────────────────────────────────
import threading
import queue

class ConnectionPool:
    """Context manager for database connection pooling."""
    
    def __init__(self, max_connections=5):
        self.pool = queue.Queue(max_connections)
        self.lock = threading.Lock()
        
        for i in range(max_connections):
            self.pool.put({"id": i, "connection": f"conn_{i}", "in_use": False})
    
    def __enter__(self):
        """Get a connection from the pool."""
        self.conn = self.pool.get(timeout=5)   # Wait up to 5s
        self.conn["in_use"] = True
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Return connection to pool."""
        self.conn["in_use"] = False
        self.pool.put(self.conn)
        return False   # Don't suppress exceptions

pool = ConnectionPool(max_connections=3)

def do_query(query):
    with pool as conn:
        print(f"Running '{query}' on {conn['connection']}")
        return f"Results of {query}"

print(do_query("SELECT * FROM users"))
print(do_query("INSERT INTO orders VALUES (...)"))


# ── Nested context managers ────────────────────────────────────────────────
@contextmanager
def transaction(db):
    """Database transaction as context manager."""
    connection = db.connect()
    try:
        yield connection
        connection.commit()
        print("Transaction committed")
    except Exception as e:
        connection.rollback()
        print(f"Transaction rolled back: {e}")
        raise
    finally:
        connection.close()


# ── Reentrant context manager ─────────────────────────────────────────────
class ReentrantLock:
    """A lock that can be acquired multiple times by the same thread."""
    
    def __init__(self):
        self._lock = threading.RLock()  # Reentrant lock
        self._depth = 0
    
    def __enter__(self):
        self._lock.acquire()
        self._depth += 1
        return self
    
    def __exit__(self, *args):
        self._depth -= 1
        self._lock.release()
        return False

lock = ReentrantLock()

def outer():
    with lock:               # Acquires (depth: 1)
        print("Outer acquired")
        inner()              # Works — reentrant!
        print("Outer releasing")

def inner():
    with lock:               # Acquires again (depth: 2) — no deadlock!
        print("Inner acquired")
    # Releases (depth: 1)

outer()
```

---

## Part 7: Worked Examples

### Worked Example 3: Lazy Data Pipeline Framework

```python
from functools import reduce

class Pipeline:
    """
    Lazy evaluation pipeline using generators.
    Values flow through stages one at a time.
    """
    
    def __init__(self, source):
        if callable(source):
            self._gen = source()
        else:
            self._gen = iter(source)
    
    def map(self, func):
        """Transform each item."""
        self._gen = (func(item) for item in self._gen)
        return self    # Return self for chaining!
    
    def filter(self, predicate):
        """Keep items where predicate returns True."""
        self._gen = (item for item in self._gen if predicate(item))
        return self
    
    def take(self, n):
        """Keep only first n items."""
        import itertools
        self._gen = itertools.islice(self._gen, n)
        return self
    
    def skip(self, n):
        """Skip first n items."""
        import itertools
        self._gen = itertools.islice(self._gen, n, None)
        return self
    
    def batch(self, size):
        """Group items into batches."""
        def _batch(gen, size):
            batch_items = []
            for item in gen:
                batch_items.append(item)
                if len(batch_items) == size:
                    yield batch_items
                    batch_items = []
            if batch_items:
                yield batch_items
        self._gen = _batch(self._gen, size)
        return self
    
    def flatten(self):
        """Flatten one level of nesting."""
        self._gen = (item for sublist in self._gen for item in sublist)
        return self
    
    def tap(self, func):
        """Side effect — call func without changing the stream."""
        def _tap(gen):
            for item in gen:
                func(item)
                yield item
        self._gen = _tap(self._gen)
        return self
    
    # Terminal operations — consume the pipeline
    def to_list(self): return list(self._gen)
    def to_dict(self, key): return {key(item): item for item in self._gen}
    def to_set(self): return set(self._gen)
    def count(self): return sum(1 for _ in self._gen)
    def first(self): return next(self._gen, None)
    def reduce(self, func, initial=None):
        return reduce(func, self._gen, initial) if initial is not None else reduce(func, self._gen)
    def sum(self): return sum(self._gen)
    def __iter__(self): return self._gen


# ── Demo ──────────────────────────────────────────────────────────────────
import random

# Generate sales data
def sales_source():
    products = ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"]
    regions = ["North", "South", "East", "West"]
    for _ in range(10000):
        yield {
            "product": random.choice(products),
            "region": random.choice(regions),
            "amount": round(random.uniform(10, 2000), 2),
            "quantity": random.randint(1, 10)
        }

# Fluent pipeline — reads almost like English!
top_sales = (
    Pipeline(sales_source)
    .filter(lambda s: s["region"] == "North")          # Only North
    .filter(lambda s: s["amount"] > 500)                # High value
    .map(lambda s: {**s, "revenue": s["amount"] * s["quantity"]})  # Add revenue
    .filter(lambda s: s["product"] in ["Laptop", "Monitor"])  # Premium products
    .tap(lambda s: None)                                # Could log here
    .take(5)                                            # Top 5
    .to_list()
)

print(f"Found {len(top_sales)} matching sales")
for sale in top_sales:
    print(f"  {sale['product']} - {sale['region']}: ${sale['revenue']:.2f}")
```

### Worked Example 4: Decorator System for Web Framework

```python
import functools
import time
import json
from typing import Callable

# Simulated web framework decorators

class Router:
    """Simple request router using decorators."""
    
    def __init__(self):
        self.routes = {}
        self.middleware = []
    
    def route(self, path, methods=("GET",)):
        """Register a route handler."""
        def decorator(func):
            for method in methods:
                self.routes[(method.upper(), path)] = func
            return func
        return decorator
    
    def middleware_decorator(self, func):
        """Register middleware (runs before every request)."""
        self.middleware.append(func)
        return func
    
    def dispatch(self, method, path, **request_data):
        """Process a request through middleware and handler."""
        handler = self.routes.get((method.upper(), path))
        if not handler:
            return {"status": 404, "body": "Not Found"}
        
        # Apply middleware chain
        def call_with_middleware(index, request):
            if index >= len(self.middleware):
                return handler(**request)
            return self.middleware[index](request, lambda req: call_with_middleware(index + 1, req))
        
        return call_with_middleware(0, {"method": method, "path": path, **request_data})


# ── Authentication decorator ──────────────────────────────────────────────
def requires_auth(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        token = kwargs.get("auth_token") or (args[0].get("auth_token") if args else None)
        if not token or not token.startswith("valid_"):
            return {"status": 401, "body": "Unauthorized"}
        return func(*args, **kwargs)
    return wrapper


# ── JSON response decorator ───────────────────────────────────────────────
def json_response(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, dict) and "status" not in result:
            return {"status": 200, "body": json.dumps(result), "content_type": "application/json"}
        return result
    return wrapper


# ── Rate limiting decorator ───────────────────────────────────────────────
def rate_limit(calls_per_minute):
    def decorator(func):
        call_times = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            call_times[:] = [t for t in call_times if now - t < 60]
            if len(call_times) >= calls_per_minute:
                return {"status": 429, "body": "Too Many Requests"}
            call_times.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ── Caching decorator ─────────────────────────────────────────────────────
def cached(ttl_seconds=60):
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()
            if key in cache and now - cache[key]["time"] < ttl_seconds:
                return {**cache[key]["value"], "x-cache": "HIT"}
            result = func(*args, **kwargs)
            cache[key] = {"value": result, "time": now}
            return {**result, "x-cache": "MISS"}
        return wrapper
    return decorator


# ── Build the app ─────────────────────────────────────────────────────────
app = Router()

@app.route("/users", methods=["GET"])
@requires_auth
@rate_limit(calls_per_minute=30)
@cached(ttl_seconds=30)
@json_response
def get_users(**kwargs):
    """Get all users."""
    return {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}

@app.route("/users", methods=["POST"])
@requires_auth
@json_response
def create_user(**kwargs):
    """Create a new user."""
    return {"user": {"id": 3, "name": "Carol"}, "created": True}


# ── Test the app ──────────────────────────────────────────────────────────
print("GET /users (no auth):")
print(app.dispatch("GET", "/users"))
# {'status': 401, 'body': 'Unauthorized'}

print("\nGET /users (with auth):")
print(app.dispatch("GET", "/users", auth_token="valid_token_abc"))
# {'status': 200, 'body': '...', 'x-cache': 'MISS'}

print("\nGET /users again (cached):")
print(app.dispatch("GET", "/users", auth_token="valid_token_abc"))
# {'status': 200, 'body': '...', 'x-cache': 'HIT'}
```

---

## Practice Problems

### Easy (1–5)

**Problem 1: Comprehension Warmup**
Convert these loops to comprehensions.

```python
# Convert to list comprehension
result1 = []
for x in range(20):
    if x % 3 == 0:
        result1.append(x * x)

# Convert to dict comprehension
result2 = {}
for word in ["hello", "world", "python"]:
    result2[word] = len(word)

# Convert to set comprehension
result3 = set()
for sentence in ["the cat sat", "the dog ran", "a cat ran"]:
    for word in sentence.split():
        result3.add(word)

# Convert to generator expression (use sum)
result4 = 0
for i in range(1000):
    if i % 7 == 0:
        result4 += i ** 2
```

**Problem 2: Infinite Sequence Generator**
Write generators for common infinite sequences.

```python
def naturals(start=1): pass       # 1, 2, 3, 4, ...
def evens(): pass                  # 0, 2, 4, 6, ...
def powers_of_two(): pass          # 1, 2, 4, 8, 16, ...
def cycle(items): pass             # Cycle through items forever

# Test: first 10 items from each
import itertools
print(list(itertools.islice(naturals(), 10)))    # [1, 2, 3, ...]
print(list(itertools.islice(evens(), 5)))         # [0, 2, 4, 6, 8]
print(list(itertools.islice(powers_of_two(), 8))) # [1, 2, 4, 8, ...]
print(list(itertools.islice(cycle(["R","G","B"]), 7)))  # [R,G,B,R,G,B,R]
```

**Problem 3: Decorator Timer**
Write a `@timer` decorator that measures and prints execution time.

```python
def timer(func): pass

@timer
def slow_sum(n):
    return sum(range(n))

result = slow_sum(10_000_000)
# slow_sum took 0.3241s, returned 49999995000000
```

**Problem 4: Dictionary Inversion**
Use a dict comprehension to invert a dictionary. Handle duplicate values.

```python
def invert_dict(d):
    pass

def invert_dict_grouped(d):
    """For duplicate values, group keys into a list."""
    pass

print(invert_dict({"a": 1, "b": 2, "c": 3}))
# {1: 'a', 2: 'b', 3: 'c'}

print(invert_dict_grouped({"a": 1, "b": 2, "c": 1}))
# {1: ['a', 'c'], 2: ['b']}
```

**Problem 5: Context Manager for Temp Directory**
Write a context manager that creates a temporary directory and cleans up after.

```python
from contextlib import contextmanager
from pathlib import Path
import shutil

@contextmanager
def temp_directory(prefix="tmp_"):
    pass

with temp_directory() as tmpdir:
    (tmpdir / "data.txt").write_text("hello")
    print(f"Working in {tmpdir}")
# Directory and all contents deleted automatically
```

### Medium (6–12)

**Problem 6: Sliding Window Generator**
Write a generator that yields overlapping windows of size n.

```python
def sliding_window(iterable, n):
    pass

data = [1, 2, 3, 4, 5, 6]
print(list(sliding_window(data, 3)))
# [(1,2,3), (2,3,4), (3,4,5), (4,5,6)]

# Application: detect trends in stock prices
prices = [100, 102, 99, 105, 103, 108, 112]
for window in sliding_window(prices, 3):
    trend = "↑" if window[-1] > window[0] else "↓"
    print(f"{window} {trend}")
```

**Problem 7: Memoization with Cache Stats**
Write a memoize decorator that tracks hit/miss rates.

```python
def memoize_stats(func): pass

@memoize_stats
def fib(n):
    if n < 2: return n
    return fib(n-1) + fib(n-2)

fib(30)
print(fib.stats)
# {"calls": 59, "hits": 28, "misses": 31, "hit_rate": "47.5%"}
```

**Problem 8: Lazy Range with Steps**
Build a lazy range-like class supporting slicing and negative steps.

```python
class LazyRange:
    def __init__(self, start, stop=None, step=1): pass
    def __iter__(self): pass
    def __len__(self): pass
    def __getitem__(self, index): pass   # Support r[2], r[-1], r[1:4]
    def __contains__(self, item): pass   # "5 in r" without iterating!
    def __reversed__(self): pass

r = LazyRange(0, 20, 2)
print(list(r))         # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
print(10 in r)         # True (O(1)!)
print(7 in r)          # False
print(r[3])            # 6
print(list(reversed(r)))  # [18, 16, ...]
```

**Problem 9: Composable Decorators**
Write a `compose_decorators` function that applies a list of decorators.

```python
def compose_decorators(*decorators):
    """Apply multiple decorators as one."""
    pass

# These should be equivalent:
@timer
@log_calls
@validate_args(int, int)
def add(a, b): return a + b

# And:
my_decorator = compose_decorators(timer, log_calls, validate_args(int, int))

@my_decorator
def add2(a, b): return a + b
```

**Problem 10: Chunked File Reader**
Write a generator that reads a file in chunks of n lines.

```python
def read_in_chunks(filepath, chunk_size=100):
    """
    Generator: yield lists of chunk_size lines.
    Memory-efficient for very large files.
    """
    pass

for chunk in read_in_chunks("huge.log", chunk_size=500):
    print(f"Processing {len(chunk)} lines...")
    # process(chunk)
```

**Problem 11: Lazy JSON Lines Reader**
Write a generator that reads a JSON Lines file (one JSON object per line).

```python
import json

def read_jsonl(filepath):
    """
    Generator: yield one parsed JSON object per line.
    Handles blank lines and malformed JSON gracefully.
    """
    pass

def filter_jsonl(filepath, **criteria):
    """Filter JSON Lines by field values."""
    pass

# JSON Lines format (each line is valid JSON):
# {"id": 1, "name": "Alice", "dept": "Engineering"}
# {"id": 2, "name": "Bob", "dept": "Marketing"}

for record in filter_jsonl("employees.jsonl", dept="Engineering"):
    print(record)
```

**Problem 12: Retry with Circuit Breaker**
Combine retry logic with a circuit breaker pattern.

```python
class CircuitBreaker:
    """
    After `failure_threshold` consecutive failures, open the circuit
    (stop calling for `timeout` seconds). Then try again.
    
    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
    """
    
    def __init__(self, failure_threshold=5, timeout=30):
        pass
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pass
        return wrapper

@CircuitBreaker(failure_threshold=3, timeout=10)
def unstable_service():
    import random
    if random.random() < 0.8:
        raise ConnectionError("Service unavailable")
    return "OK"
```

### Hard (13–20)

**Problem 13: Observable Generator**
Write a generator wrapper that notifies observers when values are produced.

```python
class ObservableGenerator:
    """
    Wrap a generator so observers can watch values as they flow.
    Support: on_value, on_complete, on_error callbacks.
    """
    
    def __init__(self, generator):
        self._gen = generator
        self._on_value = []
        self._on_complete = []
        self._on_error = []
    
    def subscribe(self, on_value=None, on_complete=None, on_error=None):
        if on_value: self._on_value.append(on_value)
        if on_complete: self._on_complete.append(on_complete)
        if on_error: self._on_error.append(on_error)
        return self
    
    def __iter__(self): pass
```

**Problem 14: Async-Style Coroutine Scheduler**
Build a simple coroutine scheduler using generators (pre-async/await style).

```python
class Scheduler:
    """
    Cooperative multitasking using generators.
    Each task yields to give other tasks a turn.
    """
    
    def __init__(self):
        self.tasks = []
    
    def add_task(self, generator): pass
    def run(self): pass

def task_a():
    for i in range(5):
        print(f"Task A: step {i}")
        yield    # Give other tasks a turn

def task_b():
    for i in range(3):
        print(f"Task B: step {i}")
        yield

scheduler = Scheduler()
scheduler.add_task(task_a())
scheduler.add_task(task_b())
scheduler.run()
# Task A: step 0
# Task B: step 0
# Task A: step 1
# Task B: step 1
# ...
```

**Problem 15: Comprehension-Based Query Engine**
Build a mini query engine using comprehensions and generators.

```python
class QueryEngine:
    def __init__(self, data): pass
    def select(self, *fields): return self     # Keep only these fields
    def where(self, predicate): return self    # Filter rows
    def order_by(self, key, desc=False): return self
    def group_by(self, key): return self       # Returns grouped dict
    def limit(self, n): return self
    def aggregate(self, **agg_funcs): return self  # sum, avg, count
    def execute(self): pass                    # Return result

result = (
    QueryEngine(employees)
    .where(lambda e: e["salary"] > 80000)
    .select("name", "dept", "salary")
    .order_by("salary", desc=True)
    .limit(5)
    .execute()
)
```

**Problem 16: Generator Debugger**
Write a decorator that lets you step through generator values interactively.

```python
def debuggable(func):
    """
    Wrap a generator function so you can inspect values as they're produced.
    Press Enter to get next value, 'q' to quit, 'a' to get all remaining.
    """
    pass

@debuggable
def count_up(n):
    for i in range(n):
        yield i

count_up(10)
# Value 0: (press Enter for next, 'q' to quit, 'a' for all)
```

**Problem 17: Type-Safe Comprehensions**
Write comprehension wrappers that enforce type consistency.

```python
def typed_list_comprehension(expr_func, iterable, condition_func=None, expected_type=None):
    """
    Like a list comprehension but validates all results are of expected_type.
    Raises TypeError with context if any value is wrong type.
    """
    pass
```

**Problem 18: Memory-Bounded Generator Cache**
Write a generator wrapper that caches values up to a memory limit.

```python
import sys

def bounded_cache(generator, max_bytes=1_000_000):
    """
    Cache generator values up to max_bytes.
    Once full, fetches remaining values fresh from generator.
    On re-iteration: use cache first, then continue from generator.
    """
    pass
```

**Problem 19: Reactive Pipeline**
Build a reactive data pipeline where changes propagate automatically.

```python
class ReactiveValue:
    """A value that triggers recomputation of dependent pipelines."""
    
    def __init__(self, value):
        self._value = value
        self._dependents = []
    
    @property
    def value(self): return self._value
    
    @value.setter
    def value(self, new_val): pass   # Trigger recomputation

class ComputedValue:
    """A value computed from ReactiveValues — updates automatically."""
    
    def __init__(self, func, *dependencies): pass
    
    @property
    def value(self): pass

price = ReactiveValue(100)
tax_rate = ReactiveValue(0.1)
total = ComputedValue(lambda p, t: p * (1 + t), price, tax_rate)

print(total.value)   # 110.0
price.value = 200    # total auto-updates!
print(total.value)   # 220.0
tax_rate.value = 0.2
print(total.value)   # 240.0
```

**Problem 20: DSL Using Comprehensions and Generators**
Build a domain-specific language for data transformation using Python's operator overloading.

```python
class Column:
    """Represents a column in a data transformation DSL."""
    
    def __init__(self, name, data=None):
        self.name = name
        self._data = data or []
    
    # Arithmetic — create new derived columns
    def __add__(self, other): pass   # col1 + col2 or col + scalar
    def __mul__(self, other): pass
    def __truediv__(self, other): pass
    
    # Comparison — create boolean masks
    def __gt__(self, other): pass    # col > 5 → boolean column
    def __lt__(self, other): pass
    def __eq__(self, other): pass
    
    # Aggregation
    def sum(self): pass
    def mean(self): pass
    def max(self): pass

# Should work like this (inspired by Pandas):
price = Column("price", [10, 20, 30, 40, 50])
quantity = Column("qty", [2, 1, 3, 2, 1])

revenue = price * quantity           # Element-wise multiplication
expensive = price > 25               # Boolean mask
total = revenue.sum()                # Aggregation
```

---

## Answer Keys (Selected Problems)

### Problem 1 Solution:
```python
# List comprehension
result1 = [x * x for x in range(20) if x % 3 == 0]

# Dict comprehension
result2 = {word: len(word) for word in ["hello", "world", "python"]}

# Set comprehension
result3 = {word for sentence in ["the cat sat", "the dog ran", "a cat ran"]
           for word in sentence.split()}

# Generator expression
result4 = sum(i ** 2 for i in range(1000) if i % 7 == 0)
```

### Problem 6 Solution:
```python
from collections import deque

def sliding_window(iterable, n):
    window = deque(maxlen=n)
    for item in iterable:
        window.append(item)
        if len(window) == n:
            yield tuple(window)
```

### Problem 7 Solution:
```python
def memoize_stats(func):
    cache = {}
    calls = [0]
    hits = [0]
    
    @functools.wraps(func)
    def wrapper(*args):
        calls[0] += 1
        if args in cache:
            hits[0] += 1
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    
    @property
    def stats(_):
        hit_rate = hits[0] / calls[0] * 100 if calls[0] else 0
        return {
            "calls": calls[0],
            "hits": hits[0],
            "misses": calls[0] - hits[0],
            "hit_rate": f"{hit_rate:.1f}%"
        }
    
    wrapper.stats = property(lambda self: {
        "calls": calls[0], "hits": hits[0],
        "misses": calls[0] - hits[0],
        "hit_rate": f"{hits[0]/calls[0]*100:.1f}%" if calls[0] else "0%"
    })
    
    # Simpler version:
    wrapper._calls = calls
    wrapper._hits = hits
    
    def get_stats():
        return {
            "calls": calls[0], "hits": hits[0],
            "misses": calls[0] - hits[0],
            "hit_rate": f"{hits[0]/calls[0]*100:.1f}%" if calls[0] else "0%"
        }
    wrapper.stats = get_stats
    return wrapper
```

### Problem 14 Solution:
```python
from collections import deque

class Scheduler:
    def __init__(self):
        self.tasks = deque()
    
    def add_task(self, generator):
        self.tasks.append(generator)
    
    def run(self):
        while self.tasks:
            task = self.tasks.popleft()
            try:
                next(task)           # Run until next yield
                self.tasks.append(task)  # Back to end of queue
            except StopIteration:
                pass                 # Task done, don't re-add
```

---

## Mini-Project: Data Stream Processing Engine

```python
"""
stream_engine.py — A complete lazy data stream processing library

Combines generators, comprehensions, decorators, and context managers
into a cohesive, production-style processing engine.
"""

import time
import json
import functools
import itertools
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, deque

# ── DECORATORS ─────────────────────────────────────────────────────────────

def measure(func):
    """Measure and log generator throughput."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        gen = func(*args, **kwargs)
        count = 0
        start = time.perf_counter()
        
        for item in gen:
            count += 1
            yield item
        
        elapsed = time.perf_counter() - start
        rate = count / elapsed if elapsed > 0 else float("inf")
        print(f"[{func.__name__}] {count:,} items in {elapsed:.3f}s ({rate:,.0f}/s)")
    
    return wrapper

def log_errors(func):
    """Skip and log errors in generators."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for item in func(*args, **kwargs):
            try:
                yield item
            except Exception as e:
                print(f"[ERROR in {func.__name__}] {e}: {item}")
    return wrapper


# ── CONTEXT MANAGERS ───────────────────────────────────────────────────────

@contextmanager
def pipeline_context(name):
    """Context manager for a named pipeline with timing."""
    print(f"\n{'='*50}")
    print(f"Pipeline: {name}")
    print(f"{'='*50}")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"\nPipeline '{name}' completed in {elapsed:.3f}s")


# ── SOURCE GENERATORS ──────────────────────────────────────────────────────

def from_csv(filepath, type_hints=None):
    """Source: stream rows from CSV file."""
    import csv
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = dict(row)
            if type_hints:
                for field, conv in type_hints.items():
                    if field in record:
                        try: record[field] = conv(record[field])
                        except: pass
            yield record

def from_jsonl(filepath):
    """Source: stream records from JSON Lines file."""
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass

def from_list(data):
    """Source: stream from an in-memory list."""
    yield from data

def infinite_sequence(func, *args, **kwargs):
    """Source: infinite sequence from a function."""
    while True:
        yield func(*args, **kwargs)


# ── TRANSFORM STAGES ───────────────────────────────────────────────────────

def map_stage(iterable, func):
    """Transform each item."""
    return (func(item) for item in iterable)

def filter_stage(iterable, predicate):
    """Keep items matching predicate."""
    return (item for item in iterable if predicate(item))

def add_field(iterable, field, compute):
    """Add a computed field to each record."""
    for record in iterable:
        yield {**record, field: compute(record)}

def rename_fields(iterable, **mapping):
    """Rename fields: rename_fields(stream, old_name='new_name')"""
    for record in iterable:
        yield {mapping.get(k, k): v for k, v in record.items()}

def select_fields(iterable, *fields):
    """Keep only specified fields."""
    return ({f: item[f] for f in fields if f in item} for item in iterable)

def batch_stage(iterable, size):
    """Group items into batches."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

def window_stage(iterable, size, step=1):
    """Sliding window over items."""
    window = deque(maxlen=size)
    for item in iterable:
        window.append(item)
        if len(window) == size:
            yield tuple(window)

def sample(iterable, every_n):
    """Take every nth item (downsampling)."""
    for i, item in enumerate(iterable):
        if i % every_n == 0:
            yield item


# ── SINK / TERMINAL OPERATIONS ─────────────────────────────────────────────

def to_list(iterable):
    return list(iterable)

def to_csv(iterable, filepath, fieldnames=None):
    """Write stream to CSV file."""
    import csv
    records = list(iterable)
    if not records:
        return 0
    
    if fieldnames is None:
        fieldnames = list(records[0].keys())
    
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    
    return len(records)

def aggregate(iterable, group_by_field, **agg_specs):
    """
    Aggregate stream by a field.
    agg_specs: field_name=(source_field, func)
    """
    groups = defaultdict(list)
    for record in iterable:
        key = record.get(group_by_field, "Unknown")
        groups[key].append(record)
    
    result = []
    for key, records in groups.items():
        row = {group_by_field: key, "count": len(records)}
        for out_field, (src_field, func) in agg_specs.items():
            values = [r[src_field] for r in records if src_field in r]
            row[out_field] = func(values) if values else None
        result.append(row)
    
    return result

def count_items(iterable):
    return sum(1 for _ in iterable)

def collect_stats(iterable, field):
    """Compute running statistics on a numeric field."""
    n = total = 0
    min_val = float("inf")
    max_val = float("-inf")
    
    for record in iterable:
        value = record.get(field)
        if value is not None:
            n += 1
            total += value
            min_val = min(min_val, value)
            max_val = max(max_val, value)
    
    return {
        "count": n,
        "sum": total,
        "mean": total / n if n else 0,
        "min": min_val if n else None,
        "max": max_val if n else None
    }


# ── THE STREAM CLASS — Fluent API ──────────────────────────────────────────

class Stream:
    """
    Fluent interface for lazy stream processing.
    Every operation returns self for chaining.
    Nothing executes until a terminal operation is called.
    """
    
    def __init__(self, source):
        self._source = source if hasattr(source, "__iter__") else iter([source])
    
    # ── Sources ──────────────────────────────────────────────────────────
    @classmethod
    def of(cls, *items): return cls(iter(items))
    
    @classmethod
    def from_csv(cls, filepath, **type_hints):
        return cls(from_csv(filepath, type_hints or None))
    
    @classmethod
    def from_jsonl(cls, filepath):
        return cls(from_jsonl(filepath))
    
    # ── Transforms ───────────────────────────────────────────────────────
    def map(self, func):
        self._source = map_stage(self._source, func)
        return self
    
    def filter(self, predicate):
        self._source = filter_stage(self._source, predicate)
        return self
    
    def add_field(self, name, compute):
        self._source = add_field(self._source, name, compute)
        return self
    
    def select(self, *fields):
        self._source = select_fields(self._source, *fields)
        return self
    
    def rename(self, **mapping):
        self._source = rename_fields(self._source, **mapping)
        return self
    
    def take(self, n):
        self._source = itertools.islice(self._source, n)
        return self
    
    def skip(self, n):
        self._source = itertools.islice(self._source, n, None)
        return self
    
    def batch(self, size):
        self._source = batch_stage(self._source, size)
        return self
    
    def window(self, size, step=1):
        self._source = window_stage(self._source, size, step)
        return self
    
    def sample(self, every_n):
        self._source = sample(self._source, every_n)
        return self
    
    def tap(self, func):
        """Side effect without changing stream."""
        def _tap(source):
            for item in source:
                func(item)
                yield item
        self._source = _tap(self._source)
        return self
    
    # ── Terminal operations ───────────────────────────────────────────────
    def to_list(self): return to_list(self._source)
    def count(self): return count_items(self._source)
    def first(self): return next(iter(self._source), None)
    def last(self): return functools.reduce(lambda _, x: x, self._source, None)
    
    def to_csv(self, filepath, fieldnames=None):
        return to_csv(self._source, filepath, fieldnames)
    
    def to_jsonl(self, filepath):
        count = 0
        with open(filepath, "w") as f:
            for record in self._source:
                f.write(json.dumps(record) + "\n")
                count += 1
        return count
    
    def aggregate(self, group_by, **agg_specs):
        return aggregate(self._source, group_by, **agg_specs)
    
    def stats(self, field):
        return collect_stats(self._source, field)
    
    def sum(self, field=None):
        if field:
            return sum(r[field] for r in self._source if field in r)
        return sum(self._source)
    
    def __iter__(self): return iter(self._source)


# ── DEMO ───────────────────────────────────────────────────────────────────

import random

# Generate sample data
def generate_sales():
    products = ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard", "Mouse"]
    regions = ["North", "South", "East", "West"]
    return [
        {
            "product": random.choice(products),
            "region": random.choice(regions),
            "amount": round(random.uniform(10, 2000), 2),
            "quantity": random.randint(1, 10),
            "month": random.randint(1, 12)
        }
        for _ in range(10_000)
    ]

sales_data = generate_sales()

with pipeline_context("Sales Analytics"):
    
    # Pipeline 1: Top North region products
    print("\n── Top 5 North Region Sales ──")
    top_north = (
        Stream(sales_data)
        .filter(lambda s: s["region"] == "North")
        .add_field("revenue", lambda s: round(s["amount"] * s["quantity"], 2))
        .filter(lambda s: s["revenue"] > 1000)
        .select("product", "region", "amount", "quantity", "revenue")
        .take(5)
        .to_list()
    )
    for sale in top_north:
        print(f"  {sale['product']:<12} ${sale['revenue']:>8,.2f}")
    
    # Pipeline 2: Revenue by region
    print("\n── Revenue by Region ──")
    by_region = (
        Stream(sales_data)
        .add_field("revenue", lambda s: s["amount"] * s["quantity"])
        .aggregate(
            "region",
            total_revenue=("revenue", sum),
            avg_sale=("revenue", lambda v: round(sum(v)/len(v), 2)),
            num_sales=("revenue", len)
        )
    )
    for row in sorted(by_region, key=lambda x: x["total_revenue"], reverse=True):
        print(f"  {row['region']:<8} ${row['total_revenue']:>10,.2f} "
              f"({row['num_sales']} sales, avg ${row['avg_sale']:,.2f})")
    
    # Pipeline 3: Monthly trend (sample every 10th item for speed)
    print("\n── Revenue Statistics (full dataset) ──")
    stats = (
        Stream(sales_data)
        .add_field("revenue", lambda s: s["amount"] * s["quantity"])
        .stats("revenue")
    )
    print(f"  Count: {stats['count']:,}")
    print(f"  Total: ${stats['sum']:,.2f}")
    print(f"  Mean:  ${stats['mean']:,.2f}")
    print(f"  Min:   ${stats['min']:,.2f}")
    print(f"  Max:   ${stats['max']:,.2f}")
```

---

## Chapter Summary

You've unlocked the most expressive features in Python!

✅ **Comprehensions**: List, dict, set, and generator expressions — the right tool for each
✅ **Generators**: `yield`, infinite sequences, pipelines, `yield from`, `send()`
✅ **Iterators**: The `__iter__`/`__next__` protocol, building custom iterables
✅ **Decorators (Advanced)**: Factory pattern, stacking, class-based, practical recipes
✅ **Context Managers (Advanced)**: `suppress`, `ExitStack`, reentrant, thread-safe
✅ **Lazy Evaluation**: Processing massive data in constant memory

**Key Takeaways:**
- Use list comprehensions for collections you need fully in memory; generator expressions when iterating once or data is large
- Generators let you build memory-efficient pipelines — data flows through stages one item at a time
- Decorators are just functions — once you see the three-level factory pattern, any decorator makes sense
- `contextlib` gives you `@contextmanager`, `suppress`, and `ExitStack` — use them liberally
- The iterator protocol underpins all of Python's iteration — understanding it makes everything else click

**Next Chapter Preview:**
Chapter 11 dives into **NumPy** — why pure Python is slow for numerical work, how arrays differ from lists, broadcasting, vectorization, and performance comparisons!
