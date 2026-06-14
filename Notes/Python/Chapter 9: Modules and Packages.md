# Chapter 9: Modules and Packages

## Part 1: Why Do Modules Exist?

### The Problem: One Giant File Is Unmanageable

Imagine you've been building your e-commerce app for 6 months. Without modules:

```
app.py  ← 8,000 lines of code
         ← User management (lines 1–800)
         ← Product catalog (lines 801–2000)
         ← Shopping cart (lines 2001–3200)
         ← Payment processing (lines 3201–4500)
         ← Order management (lines 4501–6000)
         ← Email notifications (lines 6001–7200)
         ← Analytics (lines 7201–8000)
```

**Problems:**
- Can't find anything — "where is the `calculate_tax` function?"
- Two developers edit the same file → constant merge conflicts
- Can't reuse the payment logic in another project without copy-pasting
- One bug in line 4500 could have side effects on line 200
- Testing any single piece requires loading everything

**The solution:** Split code into **modules** (individual files) and **packages** (folders of modules).

```
ecommerce/
├── users/
│   ├── __init__.py
│   ├── models.py
│   └── auth.py
├── products/
│   ├── __init__.py
│   ├── catalog.py
│   └── inventory.py
├── payments/
│   ├── __init__.py
│   ├── stripe.py
│   └── paypal.py
└── main.py
```

Each file is focused, independently testable, and reusable.

---

## Part 2: Importing Modules

### The `import` Statement — Four Ways

```python
# ── METHOD 1: Import the whole module ────────────────────────────────────
import math

# Access with dot notation
print(math.pi)          # 3.141592653589793
print(math.sqrt(16))    # 4.0
print(math.floor(3.7))  # 3

# ✓ Best when: you use many things from the module
# ✓ Makes it clear WHERE each name comes from


# ── METHOD 2: Import specific names ──────────────────────────────────────
from math import pi, sqrt, floor

# Use directly — no prefix needed
print(pi)          # 3.141592653589793
print(sqrt(16))    # 4.0
print(floor(3.7))  # 3

# ✓ Best when: you only need a few things
# ✓ Cleaner for heavily used functions
# ⚠ Risk: name collisions if two modules have same name


# ── METHOD 3: Import with alias ──────────────────────────────────────────
import numpy as np          # Convention: numpy → np
import pandas as pd         # Convention: pandas → pd
import matplotlib.pyplot as plt  # Convention

# Use the alias
arr = np.array([1, 2, 3])
df  = pd.DataFrame({"col": [1, 2, 3]})

# ✓ Best when: module name is long or has a community-standard alias
# ✓ Also useful for avoiding name collisions

import math as m
print(m.pi)


# ── METHOD 4: Import everything (AVOID!) ─────────────────────────────────
from math import *    # Imports ALL public names from math

print(pi)      # Works, but...
print(sqrt(9)) # Where does sqrt come from? You'd have to check!

# ❌ Avoid because:
# - Pollutes your namespace with hundreds of names
# - Causes silent name collisions
# - Makes code hard to read (where does this name come from?)
# - Only acceptable in interactive REPL sessions
```

### How Python Finds Modules

```python
import sys

# Python searches for modules in this order:
# 1. sys.modules cache (already imported? use that)
# 2. Built-in modules (math, os, sys...)
# 3. sys.path directories (in order)

print(sys.path)
# ['', '/usr/lib/python310', '/usr/lib/python310/lib-dynload', ...]
# '' = current directory  ← your own modules live here!

# You can add directories to sys.path:
sys.path.insert(0, "/path/to/my/modules")

# Or use environment variable PYTHONPATH
```

---

## Part 3: Creating Your Own Modules

### Step 1: A Simple Module

```python
# ── FILE: math_utils.py ──────────────────────────────────────────────────

"""
math_utils.py — Utility functions for mathematical operations.

This module provides helper functions for common math tasks
not covered by the standard library.
"""

PI = 3.141592653589793    # Module-level constant

def circle_area(radius):
    """Calculate the area of a circle."""
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return PI * radius ** 2

def circle_perimeter(radius):
    """Calculate the circumference of a circle."""
    return 2 * PI * radius

def is_prime(n):
    """Return True if n is a prime number."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def primes_up_to(n):
    """Return list of all primes up to n (Sieve of Eratosthenes)."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def clamp(value, min_val, max_val):
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))

# This block only runs when the file is run DIRECTLY
# NOT when it's imported — this is the key pattern!
if __name__ == "__main__":
    print("Testing math_utils...")
    print(f"Circle area (r=5): {circle_area(5):.2f}")
    print(f"Primes up to 20: {primes_up_to(20)}")
    print(f"Is 17 prime? {is_prime(17)}")
```

```python
# ── FILE: main.py ────────────────────────────────────────────────────────

import math_utils

print(math_utils.PI)                    # 3.141592653589793
print(math_utils.circle_area(5))        # 78.54
print(math_utils.is_prime(17))          # True
print(math_utils.primes_up_to(30))      # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
print(math_utils.clamp(150, 0, 100))    # 100 (clamped to max)

# Or import specific names
from math_utils import circle_area, is_prime
print(circle_area(10))    # 314.16
print(is_prime(7))         # True
```

### The `__name__` Guard — The Most Important Pattern

```python
# ── FILE: greet.py ───────────────────────────────────────────────────────

def greet(name):
    return f"Hello, {name}!"

def farewell(name):
    return f"Goodbye, {name}!"

# Without the guard — this runs EVERY TIME greet is imported
# Even when you don't want it to!
print("Greet module loaded!")   # Runs on import — usually not what you want

# ── The fix: ─────────────────────────────────────────────────────────────
# __name__ == "__main__"  ONLY when file is run directly
# __name__ == "greet"     when file is imported

if __name__ == "__main__":
    # This ONLY runs when you do: python greet.py
    # NOT when you do: import greet
    print("Running greet.py directly:")
    print(greet("Alice"))
    print(farewell("Bob"))
```

```python
# ── FILE: main.py ────────────────────────────────────────────────────────
import greet
# "Greet module loaded!" would print if there's no guard
# With guard, only the functions are imported silently

print(greet.greet("World"))  # Hello, World!
```

### Module-Level `__all__` — Controlling What Gets Exported

```python
# ── FILE: mymodule.py ────────────────────────────────────────────────────

# __all__ defines what gets exported with "from mymodule import *"
# Also serves as documentation: "these are the public API"
__all__ = ["public_function", "PublicClass", "IMPORTANT_CONSTANT"]

IMPORTANT_CONSTANT = 42

def public_function():
    """This is part of the public API."""
    return _helper()

def _helper():
    """Private by convention — starts with underscore."""
    return "internal work done"

class PublicClass:
    """This is part of the public API."""
    pass

class _InternalClass:
    """Not meant for external use."""
    pass

# __all__ doesn't prevent direct access — it's a GUIDE, not a wall:
# from mymodule import _helper  # Still works, just discouraged
# from mymodule import *        # Only imports what's in __all__
```

---

## Part 4: Packages — Organizing Multiple Modules

### Creating a Package

A **package** is a directory with an `__init__.py` file. This file marks the directory as a Python package and can contain initialization code.

```
myproject/
├── main.py                 ← Entry point
└── ecommerce/              ← Package
    ├── __init__.py         ← Makes this a package
    ├── products.py         ← Module
    ├── cart.py             ← Module
    ├── users.py            ← Module
    └── payments/           ← Sub-package
        ├── __init__.py
        ├── stripe.py
        └── paypal.py
```

```python
# ── FILE: ecommerce/__init__.py ──────────────────────────────────────────

"""
ecommerce — Online store package.

This package provides all components needed to run
an e-commerce platform.

Usage:
    from ecommerce import Product, Cart, User
    from ecommerce.payments import StripeProcessor
"""

# Version info
__version__ = "1.0.0"
__author__ = "Your Name"

# Re-export the most important classes for convenience
# This lets users do: from ecommerce import Product
# instead of: from ecommerce.products import Product
from .products import Product
from .cart import Cart, CartItem
from .users import User

# What's available with "from ecommerce import *"
__all__ = ["Product", "Cart", "CartItem", "User"]
```

```python
# ── FILE: ecommerce/products.py ──────────────────────────────────────────

class Product:
    def __init__(self, name, price, stock=0):
        self.name = name
        self.price = price
        self.stock = stock
    
    def __repr__(self):
        return f"Product('{self.name}', {self.price})"

class ProductCatalog:
    def __init__(self):
        self._products = {}
    
    def add(self, product):
        self._products[product.name] = product
    
    def get(self, name):
        return self._products.get(name)
    
    def search(self, query):
        return [p for p in self._products.values()
                if query.lower() in p.name.lower()]
    
    def __len__(self):
        return len(self._products)
```

```python
# ── FILE: ecommerce/cart.py ──────────────────────────────────────────────

# Relative import — imports from WITHIN the same package
from .products import Product   # "." means "current package"

class CartItem:
    def __init__(self, product, quantity=1):
        self.product = product
        self.quantity = quantity
    
    @property
    def subtotal(self):
        return self.product.price * self.quantity

class Cart:
    def __init__(self):
        self._items = {}
    
    def add(self, product, qty=1):
        if product.name in self._items:
            self._items[product.name].quantity += qty
        else:
            self._items[product.name] = CartItem(product, qty)
    
    @property
    def total(self):
        return sum(item.subtotal for item in self._items.values())
    
    def __len__(self):
        return sum(item.quantity for item in self._items.values())
```

```python
# ── FILE: ecommerce/payments/__init__.py ─────────────────────────────────

from .stripe import StripeProcessor
from .paypal import PayPalProcessor

__all__ = ["StripeProcessor", "PayPalProcessor"]
```

```python
# ── FILE: ecommerce/payments/stripe.py ───────────────────────────────────

class StripeProcessor:
    def __init__(self, api_key):
        self.api_key = api_key
    
    def charge(self, amount, customer_id):
        print(f"[Stripe] Charging ${amount:.2f} for customer {customer_id}")
        return {"status": "success", "processor": "stripe", "amount": amount}
```

```python
# ── FILE: main.py ────────────────────────────────────────────────────────

# Multiple import styles all work
import ecommerce
from ecommerce import Product, Cart
from ecommerce.payments import StripeProcessor
from ecommerce.products import ProductCatalog

# Use the package
laptop = Product("Laptop", 999.99, stock=10)
mouse  = Product("Mouse",   29.99, stock=50)

catalog = ProductCatalog()
catalog.add(laptop)
catalog.add(mouse)

print(f"Catalog size: {len(catalog)}")
print(f"Search 'lap': {catalog.search('lap')}")

cart = Cart()
cart.add(laptop, 1)
cart.add(mouse, 2)
print(f"Cart total: ${cart.total:.2f}")

stripe = StripeProcessor("sk_test_xxx")
result = stripe.charge(cart.total, "CUST001")
print(result)

# Package version from __init__.py
print(f"Package version: {ecommerce.__version__}")
```

### Relative vs Absolute Imports

```python
# ── FILE: ecommerce/cart.py ──────────────────────────────────────────────

# ABSOLUTE IMPORT — full path from project root
from ecommerce.products import Product      # Always works
from ecommerce.users import User            # Clear and unambiguous

# RELATIVE IMPORT — relative to current package
from .products import Product               # "." = current package (ecommerce)
from ..utils import format_currency         # ".." = parent package
from .payments.stripe import StripeProcessor  # Subpackage

# When to use which:
# Relative: within the same package — easier to rename/move the package
# Absolute: from outside the package, or cross-package imports
#           more readable in large projects

# ⚠ Relative imports only work INSIDE packages
# Running a file directly breaks relative imports:
# python ecommerce/cart.py   ← Relative imports fail here!
# python main.py             ← Works, because main.py uses absolute imports
```

---

## Part 5: The Standard Library — Python's Batteries

### Why "Batteries Included"?

Python ships with an enormous standard library. Before reaching for third-party packages, check if Python already has what you need.

### `os` — Operating System Interface

```python
import os

# ── PATHS ────────────────────────────────────────────────────────────────
print(os.getcwd())              # Current working directory
os.chdir("/tmp")                # Change directory

# Build paths safely (handles / vs \ automatically)
path = os.path.join("data", "2025", "january", "sales.csv")
print(path)     # data/2025/january/sales.csv  (or data\2025\... on Windows)

# Path info
print(os.path.exists("data.csv"))    # True/False
print(os.path.isfile("data.csv"))    # True if it's a file
print(os.path.isdir("data"))         # True if it's a directory
print(os.path.basename("/home/alice/file.txt"))  # file.txt
print(os.path.dirname("/home/alice/file.txt"))   # /home/alice
print(os.path.splitext("document.pdf"))          # ('document', '.pdf')

# ── DIRECTORY OPERATIONS ──────────────────────────────────────────────
os.makedirs("data/reports/2025", exist_ok=True)  # Create nested dirs safely
os.rename("old.txt", "new.txt")
os.remove("unwanted.txt")   # Delete file
os.rmdir("empty_dir")       # Delete empty directory

# List directory contents
files = os.listdir(".")     # Returns list of names
print(files)

# Walk directory tree recursively
for root, dirs, files in os.walk("myproject"):
    level = root.replace("myproject", "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for file in files:
        print(f"{indent}  {file}")

# ── ENVIRONMENT VARIABLES ─────────────────────────────────────────────
db_url = os.environ.get("DATABASE_URL", "sqlite:///default.db")
api_key = os.environ.get("API_KEY")  # Returns None if not set
secret = os.environ["SECRET_KEY"]    # Raises KeyError if not set

# Set for current process
os.environ["MY_VAR"] = "my_value"
```

### `pathlib` — Modern Path Handling (Prefer Over `os.path`)

```python
from pathlib import Path

# Create paths
home = Path.home()                          # /home/alice
cwd = Path.cwd()                            # /home/alice/project
data_dir = Path("data") / "2025" / "jan"   # data/2025/jan  (/ operator!)

# Path info
p = Path("data/sales.csv")
print(p.name)           # sales.csv
print(p.stem)           # sales
print(p.suffix)         # .csv
print(p.parent)         # data
print(p.exists())       # True/False
print(p.is_file())      # True
print(p.is_dir())       # False

# Creating directories
Path("output/reports").mkdir(parents=True, exist_ok=True)

# Iterating
data_path = Path("data")
for csv_file in data_path.glob("*.csv"):    # All .csv files
    print(csv_file)

for py_file in Path(".").rglob("*.py"):     # Recursive .py search
    print(py_file)

# Reading and writing (Path objects work directly with open!)
content = p.read_text(encoding="utf-8")     # Read entire file
p.write_text("new content", encoding="utf-8")

# More path operations
new_path = p.with_suffix(".json")           # data/sales.json
new_path = p.with_name("revenue.csv")       # data/revenue.csv
absolute = p.resolve()                      # Full absolute path

# Real-world: Find all Python files in a project
project = Path("myproject")
py_files = list(project.rglob("*.py"))
total_lines = sum(
    len(f.read_text().splitlines())
    for f in py_files
    if f.is_file()
)
print(f"Total Python lines: {total_lines}")
```

### `datetime` — Working with Dates and Times

```python
from datetime import datetime, date, time, timedelta
import datetime as dt

# ── CREATING DATES/TIMES ─────────────────────────────────────────────────
now = datetime.now()                          # Current local time
today = date.today()                          # Today's date only
utc_now = datetime.utcnow()                   # UTC time

specific = datetime(2025, 1, 15, 14, 30, 0)  # 2025-01-15 14:30:00

# ── FORMATTING ───────────────────────────────────────────────────────────
print(now.strftime("%Y-%m-%d"))               # 2025-01-15
print(now.strftime("%B %d, %Y"))             # January 15, 2025
print(now.strftime("%I:%M %p"))              # 02:30 PM
print(now.strftime("%Y-%m-%dT%H:%M:%S"))     # ISO format: 2025-01-15T14:30:00

# ── PARSING ──────────────────────────────────────────────────────────────
parsed = datetime.strptime("2025-01-15", "%Y-%m-%d")
parsed2 = datetime.strptime("January 15, 2025", "%B %d, %Y")

# ── ARITHMETIC ───────────────────────────────────────────────────────────
tomorrow = today + timedelta(days=1)
next_week = today + timedelta(weeks=1)
two_hours_later = now + timedelta(hours=2)
thirty_mins_ago = now - timedelta(minutes=30)

# Difference between dates
d1 = date(2025, 1, 1)
d2 = date(2025, 3, 15)
diff = d2 - d1
print(f"Days between: {diff.days}")    # 73

# ── PRACTICAL EXAMPLES ────────────────────────────────────────────────────
# Check if something expired
expiry = datetime(2025, 6, 30, 23, 59, 59)
is_expired = datetime.now() > expiry
print(f"Expired: {is_expired}")

# Format for display vs storage
display = now.strftime("%B %d, %Y at %I:%M %p")  # January 15, 2025 at 02:30 PM
storage = now.isoformat()                          # 2025-01-15T14:30:00.123456

# Age calculator
birth_date = date(1995, 6, 15)
today = date.today()
age = today.year - birth_date.year - (
    (today.month, today.day) < (birth_date.month, birth_date.day)
)
print(f"Age: {age}")
```

### `collections` — Specialized Container Types

```python
from collections import defaultdict, Counter, OrderedDict, namedtuple, deque

# ── COUNTER ─ count occurrences ──────────────────────────────────────────
words = "the quick brown fox jumps over the lazy dog the".split()
word_count = Counter(words)

print(word_count)                     # Counter({'the': 3, 'quick': 1, ...})
print(word_count.most_common(3))      # [('the', 3), ('quick', 1), ('brown', 1)]
print(word_count["the"])              # 3
print(word_count["missing"])          # 0 (not KeyError!)

# Combine counters
counter1 = Counter("aabbc")
counter2 = Counter("bbbcc")
print(counter1 + counter2)            # Counter({'b': 5, 'c': 3, 'a': 2})
print(counter1 - counter2)            # Counter({'a': 2})  (only positives)

# ── DEFAULTDICT ─ dict with default value ────────────────────────────────
# Problem: d["new_key"] raises KeyError
# defaultdict automatically creates a default value

# Group words by first letter
words = ["apple", "avocado", "banana", "blueberry", "cherry"]

# Without defaultdict — cumbersome
grouped = {}
for word in words:
    key = word[0]
    if key not in grouped:
        grouped[key] = []
    grouped[key].append(word)

# With defaultdict — clean!
grouped = defaultdict(list)  # Default value: empty list
for word in words:
    grouped[word[0]].append(word)   # No need to check if key exists

print(dict(grouped))
# {'a': ['apple', 'avocado'], 'b': ['banana', 'blueberry'], 'c': ['cherry']}

# Count with defaultdict
word_count2 = defaultdict(int)    # Default value: 0
for word in words:
    word_count2[word] += 1        # No KeyError — int() = 0

# Nested defaultdict
nested = defaultdict(lambda: defaultdict(int))
nested["alice"]["math"] += 95
nested["alice"]["science"] += 88
nested["bob"]["math"] += 72
print(dict(nested["alice"]))      # {'math': 95, 'science': 88}

# ── DEQUE ─ double-ended queue ────────────────────────────────────────────
# Lists are slow for insertions at the front (O(n))
# Deques are O(1) for both ends

from collections import deque

dq = deque([1, 2, 3, 4, 5])

dq.appendleft(0)    # Add to LEFT (front)
dq.append(6)        # Add to RIGHT (back)
print(dq)           # deque([0, 1, 2, 3, 4, 5, 6])

dq.popleft()        # Remove from LEFT
dq.pop()            # Remove from RIGHT
print(dq)           # deque([1, 2, 3, 4, 5])

dq.rotate(2)        # Rotate right by 2
print(dq)           # deque([4, 5, 1, 2, 3])

# Fixed-size deque (sliding window)
recent = deque(maxlen=3)    # Only keeps last 3 items
for i in range(7):
    recent.append(i)
    print(list(recent))
# [0], [0,1], [0,1,2], [1,2,3], [2,3,4], [3,4,5], [4,5,6]

# ── NAMEDTUPLE ─ already covered in Chapter 4, quick reminder ────────────
Point = namedtuple("Point", ["x", "y", "z"])
p = Point(1, 2, 3)
print(p.x, p.y, p.z)  # 1 2 3
print(p[0])            # 1  (positional still works)
```

### `itertools` — Powerful Iteration Tools

```python
import itertools

# ── CHAIN ─ iterate multiple iterables as one ────────────────────────────
chain = list(itertools.chain([1, 2], [3, 4], [5, 6]))
print(chain)      # [1, 2, 3, 4, 5, 6]

# Useful for flattening one level
nested = [[1, 2], [3, 4], [5, 6]]
flat = list(itertools.chain.from_iterable(nested))
print(flat)       # [1, 2, 3, 4, 5, 6]

# ── PRODUCT ─ Cartesian product ──────────────────────────────────────────
colors = ["red", "blue"]
sizes = ["S", "M", "L"]
variants = list(itertools.product(colors, sizes))
print(variants)
# [('red','S'),('red','M'),('red','L'),('blue','S'),('blue','M'),('blue','L')]

# Dice combinations
two_dice = list(itertools.product(range(1, 7), repeat=2))
print(f"Total outcomes: {len(two_dice)}")  # 36

# ── COMBINATIONS and PERMUTATIONS ────────────────────────────────────────
players = ["Alice", "Bob", "Carol", "Dave"]

# Combinations: order doesn't matter (AB = BA)
teams = list(itertools.combinations(players, 2))
print(f"Possible 2-person teams: {len(teams)}")  # 6
print(teams[:3])    # [('Alice','Bob'), ('Alice','Carol'), ('Alice','Dave')]

# Permutations: order matters (AB ≠ BA)
arrangements = list(itertools.permutations(players, 2))
print(f"Ordered pairs: {len(arrangements)}")  # 12

# ── GROUPBY ─ group consecutive elements ─────────────────────────────────
# IMPORTANT: data must be SORTED by the grouping key first!
data = [
    {"dept": "Engineering", "name": "Alice"},
    {"dept": "Engineering", "name": "Bob"},
    {"dept": "Marketing", "name": "Carol"},
    {"dept": "Marketing", "name": "Dave"},
    {"dept": "Product", "name": "Eve"},
]

for dept, members in itertools.groupby(data, key=lambda x: x["dept"]):
    names = [m["name"] for m in members]
    print(f"{dept}: {names}")

# Engineering: ['Alice', 'Bob']
# Marketing: ['Carol', 'Dave']
# Product: ['Eve']

# ── ISLICE ─ slice an iterator ────────────────────────────────────────────
import itertools

def infinite_counter(start=0):
    n = start
    while True:
        yield n
        n += 1

# Take first 10 from infinite generator
first_10 = list(itertools.islice(infinite_counter(), 10))
print(first_10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# ── ACCUMULATE ─ running totals ───────────────────────────────────────────
import operator

sales = [100, 200, 150, 300, 250]
cumulative = list(itertools.accumulate(sales))
print(cumulative)   # [100, 300, 450, 750, 1000]

# Running maximum
running_max = list(itertools.accumulate(sales, func=max))
print(running_max)  # [100, 200, 200, 300, 300]
```

### `functools` — Functional Programming Tools

```python
from functools import partial, reduce, lru_cache, wraps

# ── PARTIAL ─ pre-fill arguments ─────────────────────────────────────────
def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))    # 25
print(cube(3))      # 27

# Pre-configured print
verbose_print = partial(print, sep=" | ", end="\n---\n")
verbose_print("Alice", 28, "Engineer")
# Alice | 28 | Engineer
# ---

# ── LRU_CACHE ─ automatic memoization ────────────────────────────────────
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2: return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(100))    # Instant! Without cache: impossibly slow
print(fibonacci.cache_info())
# CacheInfo(hits=98, misses=101, maxsize=128, currsize=101)

fibonacci.cache_clear()  # Clear the cache if needed

# ── REDUCE ─ accumulate a sequence to a single value ─────────────────────
numbers = [1, 2, 3, 4, 5]

total = reduce(lambda a, b: a + b, numbers)     # 15
product = reduce(lambda a, b: a * b, numbers)   # 120
maximum = reduce(lambda a, b: a if a > b else b, numbers)  # 5

# ── WRAPS ─ preserve function metadata in decorators ─────────────────────
def my_decorator(func):
    @wraps(func)    # Without this, decorated function loses its __name__, __doc__
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def my_function():
    """This is my function's docstring."""
    pass

print(my_function.__name__)  # my_function  (not "wrapper"!)
print(my_function.__doc__)   # This is my function's docstring.
```

### `re` — Regular Expressions

```python
import re

text = "Contact us: alice@example.com or bob@company.org. Call: (555) 123-4567"

# ── BASIC PATTERNS ────────────────────────────────────────────────────────
# .       Any character except newline
# *       0 or more of previous
# +       1 or more of previous
# ?       0 or 1 of previous
# ^       Start of string
# $       End of string
# \d      Digit [0-9]
# \w      Word char [a-zA-Z0-9_]
# \s      Whitespace
# [abc]   Character class: a or b or c
# (...)   Capture group
# |       Alternation: a or b

# ── SEARCH ── Find first match ────────────────────────────────────────────
match = re.search(r"\d{3}-\d{4}", text)
if match:
    print(match.group())    # 123-4567
    print(match.start())    # Position where match starts
    print(match.end())      # Position where match ends

# ── FINDALL ── Find all matches ───────────────────────────────────────────
emails = re.findall(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b", text)
print(emails)   # ['alice@example.com', 'bob@company.org']

numbers = re.findall(r"\d+", text)
print(numbers)  # ['555', '123', '4567']

# ── GROUPS ── Capture parts of a pattern ─────────────────────────────────
log = "2025-01-15 14:23:01 ERROR Database timeout"
pattern = r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) (\w+) (.+)"
match = re.search(pattern, log)

if match:
    date, time, level, message = match.groups()
    print(f"Date: {date}, Level: {level}")  # Date: 2025-01-15, Level: ERROR

# Named groups — more readable
pattern = r"(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}) (?P<level>\w+)"
match = re.search(pattern, log)
if match:
    print(match.group("date"))    # 2025-01-15
    print(match.group("level"))   # ERROR

# ── SUB ── Replace with pattern ────────────────────────────────────────────
# Anonymize emails
anonymized = re.sub(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b", "[EMAIL]", text)
print(anonymized)
# Contact us: [EMAIL] or [EMAIL]. Call: (555) 123-4567

# ── COMPILE ── Pre-compile for reuse ─────────────────────────────────────
# If using the same pattern many times, compile it first
email_pattern = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b")
phone_pattern = re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")

emails = email_pattern.findall(text)
phones = phone_pattern.findall(text)
print(emails)   # ['alice@example.com', 'bob@company.org']
print(phones)   # ['(555) 123-4567']
```

### `json`, `csv`, `logging` — Quick Reference

```python
# json — already covered in Chapter 7, key additions:
import json

# Custom serializer for types json doesn't handle (datetime, etc.)
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

data = {"name": "Alice", "created_at": datetime.now()}
json_str = json.dumps(data, cls=DateTimeEncoder)
print(json_str)  # {"name": "Alice", "created_at": "2025-01-15T14:30:00"}

# ── LOGGING — professional logging ────────────────────────────────────────
import logging

# Configure logging (do this once at app startup)
logging.basicConfig(
    level=logging.DEBUG,         # Minimum level to show
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("app.log"),    # Write to file
        logging.StreamHandler()            # Also print to console
    ]
)

# Get a logger for your module (best practice)
logger = logging.getLogger(__name__)

# Five levels: DEBUG < INFO < WARNING < ERROR < CRITICAL
logger.debug("Variable x = 42")                    # For development detail
logger.info("User alice logged in")                 # Normal events
logger.warning("Memory usage at 85%")              # Something to watch
logger.error("Failed to connect to database")      # Errors
logger.critical("System out of disk space!")        # App might crash

# Logger with context
try:
    result = 10 / 0
except ZeroDivisionError:
    logger.exception("Division error occurred")     # Includes stack trace!
```

---

## Part 6: Virtual Environments

### The Problem: Package Version Conflicts

```
Project A needs Django 3.2
Project B needs Django 4.1
System Python can only have ONE version installed

Without virtual environments: Projects fight over the same packages!
```

```bash
# ── CREATING A VIRTUAL ENVIRONMENT ───────────────────────────────────────

# Create (built into Python 3.3+)
python -m venv myenv

# What gets created:
# myenv/
# ├── bin/           (Scripts/ on Windows)
# │   ├── python     ← Isolated Python interpreter
# │   ├── pip        ← Isolated pip
# │   └── activate   ← Activation script
# ├── lib/
# │   └── python3.x/
# │       └── site-packages/  ← Packages installed HERE, not globally
# └── pyvenv.cfg


# ── ACTIVATING ───────────────────────────────────────────────────────────
# macOS/Linux:
source myenv/bin/activate

# Windows:
myenv\Scripts\activate

# Your prompt changes: (myenv) $


# ── USING ────────────────────────────────────────────────────────────────
# Now pip installs INTO the venv, not globally
pip install requests pandas numpy

# Check installed packages
pip list
pip freeze              # With exact versions — for reproducibility


# ── REQUIREMENTS FILES ────────────────────────────────────────────────────
# Save your requirements (share with teammates)
pip freeze > requirements.txt

# requirements.txt looks like:
# numpy==1.24.0
# pandas==2.0.1
# requests==2.31.0

# Restore on another machine
pip install -r requirements.txt


# ── DEACTIVATING ─────────────────────────────────────────────────────────
deactivate
```

```python
# ── .gitignore — ALWAYS add venv to .gitignore ───────────────────────────
# You DON'T commit the venv folder — it's huge and machine-specific
# You ONLY commit requirements.txt

# .gitignore:
# myenv/
# __pycache__/
# *.pyc
# .env
```

---

## Part 7: Popular Third-Party Packages

### A Tour of the Ecosystem

```python
# ── REQUESTS ─ HTTP made human ────────────────────────────────────────────
# pip install requests
import requests

response = requests.get("https://api.github.com/users/python")
print(response.status_code)      # 200
print(response.json()["name"])   # Python

# POST request
response = requests.post(
    "https://api.example.com/users",
    json={"name": "Alice", "email": "alice@example.com"},
    headers={"Authorization": "Bearer token123"}
)

# Error handling
try:
    response = requests.get("https://api.example.com/data", timeout=5)
    response.raise_for_status()    # Raises exception for 4xx/5xx
    data = response.json()
except requests.Timeout:
    print("Request timed out")
except requests.HTTPError as e:
    print(f"HTTP error: {e.response.status_code}")
except requests.ConnectionError:
    print("No internet connection")


# ── CLICK ─ command-line interfaces ──────────────────────────────────────
# pip install click
import click

@click.command()
@click.argument("name")
@click.option("--greeting", "-g", default="Hello", help="Greeting to use")
@click.option("--count", "-n", default=1, type=int, help="How many times")
@click.option("--upper", is_flag=True, help="Uppercase the output")
def greet(name, greeting, count, upper):
    """Greet NAME with a custom message."""
    message = f"{greeting}, {name}!"
    if upper:
        message = message.upper()
    for _ in range(count):
        click.echo(message)

# python cli.py Alice --greeting "Hi" --count 3 --upper
# HI, ALICE!
# HI, ALICE!
# HI, ALICE!


# ── PYDANTIC ─ data validation ────────────────────────────────────────────
# pip install pydantic
from pydantic import BaseModel, validator, Field
from typing import Optional

class User(BaseModel):
    name: str
    email: str
    age: int = Field(ge=0, le=150)      # ge=greater_equal, le=less_equal
    bio: Optional[str] = None
    
    @validator("email")
    def email_must_have_at(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email")
        return v.lower()

# Automatic validation!
user = User(name="Alice", email="Alice@Example.COM", age=28)
print(user.email)   # alice@example.com  (normalized by validator)

# user = User(name="Bob", email="not-an-email", age=200)  # ValidationError!


# ── RICH ─ beautiful terminal output ─────────────────────────────────────
# pip install rich
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

# Colored output with markup
console.print("[bold green]Success![/bold green] File saved.")
console.print("[red]Error:[/red] Connection failed.")

# Beautiful tables
table = Table(title="Employees")
table.add_column("Name", style="cyan")
table.add_column("Dept", style="magenta")
table.add_column("Salary", justify="right", style="green")

table.add_row("Alice", "Engineering", "$95,000")
table.add_row("Bob", "Marketing", "$72,000")

console.print(table)

# Progress bars for loops
for item in track(range(100), description="Processing..."):
    pass  # Your work here
```

---

## Part 8: Worked Examples

### Worked Example 1: A Fully Structured Package

```
analytics/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── loader.py
│   └── cleaner.py
├── analysis/
│   ├── __init__.py
│   ├── stats.py
│   └── trends.py
└── reports/
    ├── __init__.py
    └── formatter.py
```

```python
# ── analytics/__init__.py ────────────────────────────────────────────────
"""
Analytics package — data loading, analysis, and reporting.
"""
__version__ = "0.1.0"

from .data.loader import load_csv, load_json
from .analysis.stats import describe, correlate
from .reports.formatter import to_text_report

__all__ = ["load_csv", "load_json", "describe", "correlate", "to_text_report"]


# ── analytics/data/loader.py ─────────────────────────────────────────────
import csv
import json
from pathlib import Path

def load_csv(filepath, type_hints=None):
    """Load CSV file into list of dicts with optional type conversion."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    records = []
    with open(path, "r", newline="") as f:
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
            records.append(record)
    return records

def load_json(filepath):
    """Load JSON file and return parsed data."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(path, "r") as f:
        return json.load(f)


# ── analytics/analysis/stats.py ──────────────────────────────────────────
def describe(data, field):
    """Compute descriptive statistics for a numeric field."""
    values = [r[field] for r in data if field in r and r[field] is not None]
    if not values:
        return {}
    
    n = len(values)
    sorted_vals = sorted(values)
    mean = sum(values) / n
    
    mid = n // 2
    median = sorted_vals[mid] if n % 2 else (sorted_vals[mid-1] + sorted_vals[mid]) / 2
    
    variance = sum((v - mean)**2 for v in values) / n
    std_dev = variance ** 0.5
    
    return {
        "count": n,
        "mean": round(mean, 2),
        "median": median,
        "std_dev": round(std_dev, 2),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values)
    }

def correlate(data, field1, field2):
    """Calculate Pearson correlation between two numeric fields."""
    pairs = [(r[field1], r[field2]) for r in data
             if field1 in r and field2 in r]
    n = len(pairs)
    if n < 2:
        return None
    
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n
    
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    x_std = sum((x - x_mean)**2 for x in x_vals) ** 0.5
    y_std = sum((y - y_mean)**2 for y in y_vals) ** 0.5
    
    if x_std == 0 or y_std == 0:
        return None
    return round(numerator / (x_std * y_std), 4)


# ── analytics/reports/formatter.py ───────────────────────────────────────
def to_text_report(title, sections):
    """
    Format a dictionary of sections into a text report.
    sections: {"Section Name": dict_of_stats}
    """
    lines = ["=" * 50, f"{title:^50}", "=" * 50]
    
    for section_name, stats in sections.items():
        lines.append(f"\n{section_name}:")
        lines.append("-" * 30)
        for key, value in stats.items():
            if isinstance(value, float):
                lines.append(f"  {key:<20} {value:>10.2f}")
            else:
                lines.append(f"  {key:<20} {value:>10}")
    
    lines.append("=" * 50)
    return "\n".join(lines)


# ── main.py ──────────────────────────────────────────────────────────────
import analytics

# High-level API (from __init__.py)
data = analytics.load_csv("employees.csv", type_hints={"salary": float, "age": int})

salary_stats = analytics.describe(data, "salary")
age_stats = analytics.describe(data, "age")

report = analytics.to_text_report("Employee Analytics", {
    "Salary Statistics": salary_stats,
    "Age Statistics": age_stats
})
print(report)

# Also available
corr = analytics.correlate(data, "age", "salary")
print(f"Age-Salary correlation: {corr}")
```

### Worked Example 2: Plugin Architecture Using Modules

```python
# ── FILE: plugin_manager.py ───────────────────────────────────────────────
import importlib
import os
from pathlib import Path

class PluginManager:
    """
    Dynamically load plugins from a plugins/ directory.
    Each plugin is a Python file with a run(data) function.
    """
    
    def __init__(self, plugin_dir="plugins"):
        self.plugin_dir = Path(plugin_dir)
        self._plugins = {}
    
    def discover(self):
        """Find and load all plugins from the plugin directory."""
        if not self.plugin_dir.exists():
            return
        
        for path in self.plugin_dir.glob("*.py"):
            if path.stem.startswith("_"):
                continue  # Skip __init__.py etc.
            self._load_plugin(path.stem)
    
    def _load_plugin(self, name):
        """Load a single plugin by name."""
        try:
            # Dynamic import: import the module by string name
            module = importlib.import_module(f"{self.plugin_dir}.{name}")
            
            # Validate it has required interface
            if not hasattr(module, "run"):
                print(f"  ⚠ Plugin '{name}' missing run() function — skipped")
                return
            
            self._plugins[name] = module
            print(f"  ✓ Loaded plugin: {name}")
        
        except ImportError as e:
            print(f"  ✗ Failed to load plugin '{name}': {e}")
    
    def run(self, plugin_name, data):
        """Run a specific plugin."""
        if plugin_name not in self._plugins:
            raise KeyError(f"Plugin '{plugin_name}' not found")
        return self._plugins[plugin_name].run(data)
    
    def run_all(self, data):
        """Run all plugins on the data."""
        results = {}
        for name, plugin in self._plugins.items():
            try:
                results[name] = plugin.run(data)
            except Exception as e:
                results[name] = {"error": str(e)}
        return results
    
    def reload(self, plugin_name):
        """Hot-reload a plugin without restarting."""
        if plugin_name in self._plugins:
            importlib.reload(self._plugins[plugin_name])
            print(f"Reloaded: {plugin_name}")
    
    @property
    def available_plugins(self):
        return list(self._plugins.keys())


# ── FILE: plugins/word_count.py ───────────────────────────────────────────
"""Word count plugin."""
plugin_name = "word_count"
plugin_version = "1.0"

def run(data):
    """Count words in the text data."""
    text = data.get("text", "")
    words = text.split()
    return {
        "word_count": len(words),
        "unique_words": len(set(w.lower() for w in words)),
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0
    }


# ── FILE: plugins/sentiment.py ────────────────────────────────────────────
"""Simple rule-based sentiment plugin."""
POSITIVE = {"great", "excellent", "love", "amazing", "wonderful", "good"}
NEGATIVE = {"terrible", "awful", "hate", "bad", "horrible", "poor"}

def run(data):
    text = data.get("text", "").lower().split()
    pos = sum(1 for w in text if w in POSITIVE)
    neg = sum(1 for w in text if w in NEGATIVE)
    score = (pos - neg) / max(len(text), 1)
    
    if score > 0.05: sentiment = "Positive"
    elif score < -0.05: sentiment = "Negative"
    else: sentiment = "Neutral"
    
    return {"sentiment": sentiment, "score": round(score, 3),
            "positive_words": pos, "negative_words": neg}


# ── Demo ──────────────────────────────────────────────────────────────────
pm = PluginManager("plugins")
pm.discover()

sample = {"text": "Python is great and amazing. I love using it every day!"}
results = pm.run_all(sample)

for plugin, result in results.items():
    print(f"\n{plugin}: {result}")
```

---

## Practice Problems

### Easy (1–5)

**Problem 1: Module Inventory**
Write a script that lists all modules in the standard library.

```python
import sys

def list_builtin_modules():
    """Return sorted list of all built-in module names."""
    pass

modules = list_builtin_modules()
print(f"Total built-in modules: {len(modules)}")
print(modules[:10])  # First 10
```

**Problem 2: Path Explorer**
Using `pathlib`, write a function that prints a directory tree with file sizes.

```python
from pathlib import Path

def print_tree(directory, max_depth=3):
    """
    Print directory tree like:
    myproject/
      main.py (1.2 KB)
      utils/
        helpers.py (3.4 KB)
    """
    pass
```

**Problem 3: Date Calculator**
Write functions using `datetime` for common date operations.

```python
from datetime import date, timedelta

def days_until(target_date_str):
    """Return days until a future date (format: YYYY-MM-DD)"""
    pass

def is_weekend(date_str):
    """Return True if given date falls on Saturday or Sunday"""
    pass

def workdays_between(start_str, end_str):
    """Count working days (Mon-Fri) between two dates"""
    pass

print(days_until("2025-12-25"))   # Days until Christmas
print(is_weekend("2025-01-18"))   # Is Jan 18 a weekend?
print(workdays_between("2025-01-01", "2025-01-31"))  # Working days in Jan
```

**Problem 4: Word Frequency with Counter**
Use `collections.Counter` to analyze text and answer questions.

```python
from collections import Counter

def analyze_text(filepath):
    """
    Read a text file and return:
    - 10 most common words (excluding stop words)
    - Number of unique words
    - Most common letter
    """
    pass
```

**Problem 5: Environment Config Loader**
Load configuration from environment variables with defaults and type conversion.

```python
import os

def load_config():
    """
    Load app config from environment variables.
    Return typed values, not just strings.
    """
    return {
        "debug": os.environ.get("DEBUG", "false").lower() == "true",
        "port": int(os.environ.get("PORT", "8000")),
        "db_url": os.environ.get("DATABASE_URL", "sqlite:///dev.db"),
        "max_workers": int(os.environ.get("MAX_WORKERS", "4")),
        "allowed_hosts": os.environ.get("ALLOWED_HOSTS", "localhost").split(",")
    }
```

### Medium (6–12)

**Problem 6: Module Dependency Analyzer**
Given a Python file, extract all its imports.

```python
import ast
from pathlib import Path

def get_imports(filepath):
    """
    Parse a Python file and return:
    - standard library imports
    - third-party imports (anything not in stdlib)
    - relative imports
    """
    pass

imports = get_imports("myapp/main.py")
print(f"Stdlib: {imports['stdlib']}")
print(f"Third-party: {imports['third_party']}")
```

**Problem 7: Smart Logger**
Build a module that provides pre-configured logging with rotation.

```python
# logger_setup.py — importable module
import logging
from logging.handlers import RotatingFileHandler

def get_logger(name, log_file=None, level="INFO", max_mb=10, backup_count=5):
    """
    Return a configured logger.
    Writes to file (with rotation) and console.
    
    Usage:
        from logger_setup import get_logger
        logger = get_logger("myapp", log_file="app.log")
        logger.info("App started")
    """
    pass
```

**Problem 8: Config File Watcher**
Using `pathlib` and `json`, watch a config file and reload when it changes.

```python
import time
from pathlib import Path
import json

def watch_config(filepath, on_change, poll_interval=1.0):
    """
    Monitor a JSON config file for changes.
    Call on_change(new_config) when file is modified.
    Run until KeyboardInterrupt.
    """
    pass

# Usage:
# watch_config("config.json", lambda cfg: print(f"Config updated: {cfg}"))
```

**Problem 9: CLI Tool with Click**
Build a file analysis CLI tool.

```python
# cli.py — run as: python cli.py analyze report.csv --top 10 --field salary

import click
from pathlib import Path

@click.group()
def cli(): pass

@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--field", "-f", help="Field to analyze")
@click.option("--top", "-n", default=5, help="Show top N values")
@click.option("--output", "-o", type=click.Path(), help="Save report to file")
def analyze(filepath, field, top, output):
    """Analyze a CSV file and show statistics."""
    pass

@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--encoding", default="utf-8")
def validate(filepath, encoding):
    """Validate a CSV file for common issues."""
    pass

if __name__ == "__main__":
    cli()
```

**Problem 10: Regex Toolkit**
Build a module of reusable regex utilities.

```python
# regex_utils.py

import re

def extract_emails(text): pass
def extract_phones(text): pass
def extract_urls(text): pass
def extract_dates(text): pass
def extract_ips(text): pass
def is_valid_password(password, min_length=8): pass

# Test
text = """
Contact alice@example.com or visit https://example.com.
Call (555) 123-4567. IP: 192.168.1.1. Date: 2025-01-15
"""

print(extract_emails(text))  # ['alice@example.com']
print(extract_urls(text))    # ['https://example.com']
print(extract_phones(text))  # ['(555) 123-4567']
print(extract_dates(text))   # ['2025-01-15']
print(extract_ips(text))     # ['192.168.1.1']
```

**Problem 11: Sliding Window Analytics**
Use `deque` from `collections` to compute rolling statistics.

```python
from collections import deque

def rolling_average(data, window=3):
    """Compute rolling average with given window size."""
    pass

def rolling_max(data, window=3):
    """Compute rolling maximum."""
    pass

def detect_spikes(data, window=5, threshold=2.0):
    """Detect values more than `threshold` std_devs from rolling mean."""
    pass

prices = [100, 102, 98, 105, 103, 99, 150, 101, 98, 100]
print(rolling_average(prices, 3))
# [100.0, 101.67, 102.0, 102.33, 102.33, 116.67, 116.67, 99.67]

spikes = detect_spikes(prices, window=5, threshold=2.0)
print(f"Spike detected at index {spikes}")   # [6]  (value 150)
```

**Problem 12: Package Version Checker**
Write a script that checks if installed packages have updates.

```python
# version_checker.py
import subprocess
import json

def get_installed_packages():
    """Return dict of {package_name: installed_version}"""
    pass

def check_for_updates(packages):
    """
    For each package, check if a newer version exists on PyPI.
    Return {package: {"installed": v, "latest": v, "outdated": bool}}
    """
    pass

def print_update_report(update_info):
    """Print a formatted report of outdated packages."""
    pass
```

### Hard (13–20)

**Problem 13: Mini Import System**
Implement a simple module loader that loads Python files from a custom directory.

```python
import importlib.util
from pathlib import Path

class ModuleLoader:
    """Load Python modules from a custom directory at runtime."""
    
    def __init__(self, modules_dir):
        self.modules_dir = Path(modules_dir)
        self._cache = {}
    
    def load(self, name):
        """Load and return module by name."""
        pass
    
    def reload(self, name):
        """Force reload a cached module."""
        pass
    
    def list_available(self):
        """List all loadable modules in the directory."""
        pass
```

**Problem 14: Namespace Package**
Create a namespace package that can be extended by third parties.

```python
# myframework/
# ├── core.py
# └── plugins/      ← namespace package (no __init__.py!)
#     └── builtin/
#         ├── __init__.py
#         └── text_plugin.py
#
# third_party_extension/
# └── myframework/
#     └── plugins/
#         └── advanced/
#             └── ml_plugin.py
#
# Both myframework/plugins/ paths are merged automatically!
```

**Problem 15: Configuration System**
Build a layered configuration system that merges settings from multiple sources.

```python
class Config:
    """
    Layered config: defaults < file < environment < runtime
    Later layers override earlier ones.
    Access with dot notation: config.database.host
    """
    
    def __init__(self):
        self._layers = []
    
    def add_defaults(self, defaults: dict): pass
    def add_file(self, filepath: str): pass      # JSON or YAML
    def add_env(self, prefix="APP_"): pass        # APP_DB_HOST → db.host
    def add_dict(self, override: dict): pass
    
    def get(self, path, default=None): pass        # "database.host"
    def __getattr__(self, name): pass              # config.database.host

config = Config()
config.add_defaults({"server": {"port": 8000, "debug": False}})
config.add_file("config.json")
config.add_env(prefix="APP_")

print(config.server.port)      # 8000 (or from file/env if set)
print(config.get("server.debug", False))
```

**Problem 16: Lazy Module Importer**
Create a module that only imports submodules when they're actually accessed.

```python
class LazyModule:
    """
    Delays actual import until the attribute is first accessed.
    Saves startup time for modules with expensive imports.
    """
    def __init__(self, module_name): pass
    def __getattr__(self, name): pass

# Usage:
numpy = LazyModule("numpy")     # numpy not actually imported yet

# ... lots of other code ...

arr = numpy.array([1, 2, 3])    # numpy is imported HERE, on first use
```

**Problem 17: Inter-Module Event Bus**
Build an event bus that allows modules to communicate without direct imports.

```python
# events.py — shared event bus
class EventBus:
    _instance = None
    
    @classmethod
    def get(cls):
        """Singleton — always returns the same instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def subscribe(self, event, handler): pass
    def publish(self, event, **data): pass
    def unsubscribe(self, event, handler): pass

# Module A — publishes events without knowing who listens
# from events import EventBus
# bus = EventBus.get()
# bus.publish("user_created", user_id=42, email="alice@example.com")

# Module B — listens for events without knowing who published
# from events import EventBus
# bus = EventBus.get()
# bus.subscribe("user_created", lambda **data: print(f"New user: {data}"))
```

**Problem 18: Hot Module Reloading**
Implement a development server that reloads modules when their files change.

```python
import importlib
import time
from pathlib import Path

class HotReloader:
    """
    Watch Python files for changes.
    Reload modules automatically when changed.
    Call registered callbacks after reload.
    """
    
    def __init__(self): pass
    def watch(self, module_name): pass
    def on_reload(self, callback): pass  # Decorator
    def start(self): pass               # Blocking loop
    def stop(self): pass
```

**Problem 19: Distributed Config with Fallbacks**
Build a config system that reads from multiple sources with fallback chain.

```python
class DistributedConfig:
    """
    Try each source in order, use first that works.
    Sources: Redis → Environment → File → Defaults
    """
    
    def __init__(self):
        self.sources = []
    
    def add_source(self, source, priority=0): pass
    def get(self, key, default=None): pass
    def set(self, key, value, source=None): pass  # Write to specified source

class FileSource:
    def get(self, key): pass
    def set(self, key, value): pass

class EnvSource:
    def __init__(self, prefix=""): pass
    def get(self, key): pass
    def set(self, key, value): pass
```

**Problem 20: Package Builder**
Write a script that creates a new Python package with proper structure.

```python
def create_package(name, author, description, with_tests=True, with_cli=False):
    """
    Create a new Python package with:
    - Proper directory structure
    - __init__.py with version and __all__
    - setup.py / pyproject.toml
    - README.md
    - .gitignore
    - requirements.txt
    - tests/ directory (if with_tests)
    - cli.py with click (if with_cli)
    """
    pass

create_package(
    name="mypackage",
    author="Alice Johnson",
    description="A sample Python package",
    with_tests=True,
    with_cli=True
)
```

---

## Answer Keys (Selected Problems)

### Problem 3 Solution:
```python
from datetime import date, timedelta

def days_until(target_date_str):
    target = date.fromisoformat(target_date_str)
    delta = target - date.today()
    return delta.days

def is_weekend(date_str):
    d = date.fromisoformat(date_str)
    return d.weekday() >= 5   # 5=Saturday, 6=Sunday

def workdays_between(start_str, end_str):
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    count = 0
    current = start
    while current <= end:
        if current.weekday() < 5:   # Mon-Fri
            count += 1
        current += timedelta(days=1)
    return count
```

### Problem 10 Solution:
```python
import re

def extract_emails(text):
    return re.findall(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b", text)

def extract_phones(text):
    return re.findall(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)

def extract_urls(text):
    return re.findall(r"https?://[\w\-._~:/?#\[\]@!$&'()*+,;=%]+", text)

def extract_dates(text):
    return re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)

def extract_ips(text):
    return re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)

def is_valid_password(password, min_length=8):
    if len(password) < min_length: return False
    if not re.search(r"[A-Z]", password): return False
    if not re.search(r"[a-z]", password): return False
    if not re.search(r"\d", password): return False
    if not re.search(r"[^a-zA-Z0-9]", password): return False
    return True
```

---

## Mini-Project: Task Runner Package

### Project Goal
Build a complete task runner package — like a mini version of `make` or `invoke`.

```
taskrunner/
├── __init__.py
├── core.py       ← Task definition and registry
├── runner.py     ← Task execution engine
├── cli.py        ← Command-line interface
└── utils.py      ← Shared utilities
```

```python
# ── taskrunner/__init__.py ────────────────────────────────────────────────
"""
TaskRunner — A simple task automation package.

Usage:
    from taskrunner import task, run

    @task
    def build():
        "Compile the project"
        run("python setup.py build")
    
    @task(depends_on=["build"])
    def test():
        "Run the test suite"
        run("pytest tests/")
"""

__version__ = "1.0.0"

from .core import task, TaskRegistry
from .runner import run, TaskRunner

__all__ = ["task", "run", "TaskRegistry", "TaskRunner"]


# ── taskrunner/core.py ────────────────────────────────────────────────────
import functools
from typing import List, Callable, Optional
from datetime import datetime

class Task:
    def __init__(self, func: Callable, name: str = None,
                 depends_on: List[str] = None, description: str = None):
        self.func = func
        self.name = name or func.__name__
        self.depends_on = depends_on or []
        self.description = description or func.__doc__ or ""
        self.last_run = None
        self.last_duration = None
        self.run_count = 0
    
    def execute(self, *args, **kwargs):
        start = datetime.now()
        try:
            result = self.func(*args, **kwargs)
            self.last_run = datetime.now()
            self.last_duration = (self.last_run - start).total_seconds()
            self.run_count += 1
            return result
        except Exception as e:
            raise RuntimeError(f"Task '{self.name}' failed: {e}") from e
    
    def __repr__(self):
        return f"Task('{self.name}', depends_on={self.depends_on})"


class TaskRegistry:
    """Central registry of all defined tasks."""
    _tasks = {}
    
    @classmethod
    def register(cls, task: Task):
        cls._tasks[task.name] = task
    
    @classmethod
    def get(cls, name: str) -> Optional[Task]:
        return cls._tasks.get(name)
    
    @classmethod
    def all_tasks(cls):
        return dict(cls._tasks)
    
    @classmethod
    def clear(cls):
        cls._tasks.clear()


def task(_func=None, *, depends_on=None, name=None, description=None):
    """
    Decorator to define a task.
    
    Usage:
        @task
        def build(): ...
        
        @task(depends_on=["build"], description="Run all tests")
        def test(): ...
    """
    def decorator(func):
        t = Task(
            func=func,
            name=name or func.__name__,
            depends_on=depends_on or [],
            description=description or func.__doc__ or ""
        )
        TaskRegistry.register(t)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return t.execute(*args, **kwargs)
        
        wrapper.task = t     # Expose task metadata
        return wrapper
    
    if _func is not None:
        return decorator(_func)  # Called as @task (no args)
    return decorator             # Called as @task(...) (with args)


# ── taskrunner/runner.py ──────────────────────────────────────────────────
import subprocess
import sys
from collections import deque
from .core import TaskRegistry

def run(command, shell=True, capture=False):
    """Run a shell command from within a task."""
    print(f"  $ {command}")
    result = subprocess.run(
        command, shell=shell,
        capture_output=capture,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {command}")
    return result.stdout if capture else None


class TaskRunner:
    """Execute tasks with dependency resolution."""
    
    def __init__(self):
        self.registry = TaskRegistry
        self._executed = set()
    
    def _resolve_order(self, task_name: str) -> list:
        """Topological sort to determine execution order."""
        order = []
        visiting = set()
        
        def visit(name):
            if name in self._executed:
                return
            if name in visiting:
                raise ValueError(f"Circular dependency detected: {name}")
            
            task = self.registry.get(name)
            if task is None:
                raise KeyError(f"Unknown task: '{name}'")
            
            visiting.add(name)
            for dep in task.depends_on:
                visit(dep)
            visiting.discard(name)
            
            if name not in order:
                order.append(name)
        
        visit(task_name)
        return order
    
    def execute(self, task_name: str, *args, **kwargs):
        """Execute a task and all its dependencies."""
        order = self._resolve_order(task_name)
        
        print(f"\n🚀 Running task: {task_name}")
        if len(order) > 1:
            print(f"   Execution order: {' → '.join(order)}")
        print("=" * 50)
        
        results = {}
        for name in order:
            task = self.registry.get(name)
            print(f"\n▶ {name}: {task.description}")
            print("-" * 30)
            
            try:
                kw = kwargs if name == task_name else {}
                result = task.execute(**kw)
                results[name] = {"status": "success", "result": result,
                                 "duration": task.last_duration}
                print(f"✓ {name} completed in {task.last_duration:.2f}s")
                self._executed.add(name)
            except RuntimeError as e:
                results[name] = {"status": "failed", "error": str(e)}
                print(f"✗ {name} FAILED: {e}")
                break
        
        return results
    
    def list_tasks(self):
        """Print all available tasks."""
        tasks = self.registry.all_tasks()
        print(f"\nAvailable tasks ({len(tasks)}):")
        print("-" * 40)
        for name, t in sorted(tasks.items()):
            deps = f" (depends: {', '.join(t.depends_on)})" if t.depends_on else ""
            print(f"  {name:<20} {t.description}{deps}")


# ── taskrunner/cli.py ─────────────────────────────────────────────────────
import sys
import argparse
from .runner import TaskRunner

def main():
    parser = argparse.ArgumentParser(prog="task", description="Task Runner")
    parser.add_argument("task", nargs="?", help="Task to run")
    parser.add_argument("--list", "-l", action="store_true", help="List tasks")
    parser.add_argument("--dry-run", action="store_true", help="Show order only")
    
    args = parser.parse_args()
    runner = TaskRunner()
    
    if args.list or not args.task:
        runner.list_tasks()
        return
    
    if args.dry_run:
        order = runner._resolve_order(args.task)
        print(f"Dry run — execution order: {' → '.join(order)}")
        return
    
    results = runner.execute(args.task)
    failed = [name for name, r in results.items() if r["status"] == "failed"]
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()


# ── tasks.py (user-defined tasks for a Python project) ───────────────────
from taskrunner import task, run

@task
def clean():
    """Remove build artifacts and cache files."""
    run("find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true")
    run("rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ 2>/dev/null || true")

@task(depends_on=["clean"])
def install():
    """Install dependencies from requirements.txt."""
    run("pip install -r requirements.txt")

@task(depends_on=["install"])
def lint():
    """Run code quality checks."""
    run("python -m py_compile **/*.py")   # Check syntax

@task(depends_on=["lint"])
def test():
    """Run test suite."""
    run("python -m pytest tests/ -v")

@task(depends_on=["test"])
def build():
    """Build distribution packages."""
    run("python setup.py sdist bdist_wheel")

@task(depends_on=["build"])
def publish():
    """Publish to PyPI (requires credentials)."""
    run("python -m twine upload dist/*")

@task
def docs():
    """Build documentation."""
    run("sphinx-build docs/ docs/_build/")

@task(depends_on=["test", "docs"])
def release():
    """Full release: test, build docs, publish."""
    pass  # Dependencies handle everything


# ── DEMO ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from taskrunner import TaskRunner
    
    runner = TaskRunner()
    runner.list_tasks()
    
    print("\n" + "=" * 50)
    print("Simulating: python tasks.py test")
    results = runner.execute("test")
    
    print("\nResults summary:")
    for name, result in results.items():
        status = "✓" if result["status"] == "success" else "✗"
        duration = result.get("duration", 0)
        print(f"  {status} {name} ({duration:.2f}s)")
```

---

## Chapter Summary

You can now organize professional Python projects!

✅ **Why Modules**: Split code by responsibility, enable reuse, avoid conflicts
✅ **Import Styles**: `import X`, `from X import Y`, `import X as alias` — when to use each
✅ **Creating Modules**: `__name__`, `__all__`, `__doc__`, `_private` conventions
✅ **Packages**: `__init__.py`, relative imports, sub-packages, re-exporting
✅ **Standard Library**: `os`, `pathlib`, `datetime`, `collections`, `itertools`, `functools`, `re`, `logging`
✅ **Virtual Environments**: Isolated dependencies, `requirements.txt`
✅ **Third-Party Packages**: `requests`, `click`, `pydantic`, `rich`

**Key Takeaways:**
- One module = one responsibility — keep files focused
- Always use `if __name__ == "__main__":` to separate runnable vs importable code
- Use `pathlib.Path` instead of `os.path` — it's cleaner and more modern
- `collections` and `itertools` solve many common problems elegantly
- Virtual environments are non-negotiable for real projects

**Next Chapter Preview:**
Chapter 10 dives into **Advanced Python Features** — list/dict/set comprehensions (deep dive), generators and `yield`, decorators (advanced), context managers (advanced), and iterators. This is where Python code goes from "works" to "elegant"!
