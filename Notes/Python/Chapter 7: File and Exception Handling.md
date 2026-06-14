# Chapter 7: File I/O and Exception Handling

## Part 1: Why Do We Need File I/O and Exception Handling?

### The Problem: Programs That Forget Everything

Every program we've written so far has a critical flaw — it has **amnesia**.

```python
# You collect 10,000 user responses in a survey
responses = []
for i in range(10000):
    responses.append({"id": i, "answer": "some data"})

# Program ends → ALL of that data is gone forever
# RAM is temporary — it only exists while the program runs
```

**The real world requires persistence:**
- A bank can't forget transactions when the server restarts
- A game can't forget your progress when you close it
- A data pipeline can't re-download 50GB of data every run

**Files solve this** — they live on disk and survive program restarts.

### The Problem: The World Is Unpredictable

```python
# What if the file doesn't exist?
data = open("users.csv")  # FileNotFoundError — program CRASHES

# What if the network is down?
response = requests.get("https://api.example.com")  # ConnectionError — CRASH

# What if the user types letters where you expect a number?
age = int(input("Your age: "))  # ValueError if they type "twenty" — CRASH
```

**Exception handling lets your program survive real-world chaos** — handle errors gracefully instead of crashing and burning.

---

## Part 2: Reading Files

### Opening a File — The Basics

```python
# The basic way (don't do this — see context managers below)
file = open("data.txt", "r")   # "r" = read mode
content = file.read()
file.close()                    # MUST close — or data can be lost/corrupted!

# The problem: If an error happens between open() and close(),
# close() never runs. File stays open. Bad things happen.
```

### Context Managers — The Right Way

```python
# with statement automatically closes the file, even if an error occurs
with open("data.txt", "r") as file:
    content = file.read()
# File is automatically closed here — guaranteed!

print(content)

# The with statement calls file.__enter__() on opening
# and file.__exit__() on leaving (even if there's an error)
# Think of it as: "open the file, do stuff, ALWAYS close it"
```

### File Modes

```python
# "r"  → Read (default). Error if file doesn't exist.
# "w"  → Write. Creates file if needed. OVERWRITES existing content!
# "a"  → Append. Creates file if needed. Adds to end of file.
# "x"  → Exclusive create. Error if file already exists.
# "r+" → Read and write.
# "rb" → Read in binary mode (for images, PDFs, etc.)
# "wb" → Write in binary mode.

# Reading a file
with open("story.txt", "r") as f:
    content = f.read()          # Reads entire file as one string

# Reading line by line (memory-efficient for large files)
with open("large_file.txt", "r") as f:
    for line in f:              # File object is iterable!
        print(line.strip())     # strip() removes the \n at end of each line

# Read all lines into a list
with open("data.txt", "r") as f:
    lines = f.readlines()       # Returns list of strings, each with \n

# Read one line at a time manually
with open("data.txt", "r") as f:
    first_line = f.readline()   # Reads one line, moves cursor forward
    second_line = f.readline()  # Reads the next line
```

### Worked Example 1: Reading and Analyzing a Log File

```python
def analyze_log_file(filepath):
    """
    Read a server log file and extract statistics.
    Log format: "YYYY-MM-DD HH:MM:SS LEVEL message"
    """
    stats = {
        "total_lines": 0,
        "by_level": {},
        "errors": [],
        "warnings": []
    }
    
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            stats["total_lines"] += 1
            
            parts = line.split(" ", 3)  # Split into at most 4 parts
            if len(parts) < 4:
                continue
            
            date, time_str, level, message = parts
            
            # Count by level
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
            
            # Collect errors and warnings
            if level == "ERROR":
                stats["errors"].append({
                    "timestamp": f"{date} {time_str}",
                    "message": message
                })
            elif level == "WARNING":
                stats["warnings"].append({
                    "timestamp": f"{date} {time_str}",
                    "message": message
                })
    
    return stats

# Create a sample log file to test with
sample_log = """2025-01-15 10:23:01 INFO Server started on port 8000
2025-01-15 10:23:15 INFO User alice logged in
2025-01-15 10:24:02 WARNING High memory usage: 87%
2025-01-15 10:24:45 ERROR Database connection timeout
2025-01-15 10:24:46 INFO Retrying database connection
2025-01-15 10:24:47 INFO Database connection restored
2025-01-15 10:25:30 WARNING Request queue depth: 450
2025-01-15 10:26:00 ERROR Disk write failed: /var/log/app.log
"""

# Write the sample file
with open("server.log", "w") as f:
    f.write(sample_log)

# Now analyze it
results = analyze_log_file("server.log")
print(f"Total log entries: {results['total_lines']}")
print(f"By level: {results['by_level']}")
print(f"\nErrors ({len(results['errors'])}):")
for err in results["errors"]:
    print(f"  [{err['timestamp']}] {err['message']}")
```

---

## Part 3: Writing Files

```python
# Writing — creates file or OVERWRITES existing content
with open("output.txt", "w") as f:
    f.write("Hello, World!\n")      # \n = newline (write doesn't add it)
    f.write("Second line\n")

# Appending — adds to end without overwriting
with open("output.txt", "a") as f:
    f.write("This is appended\n")

# Writing multiple lines at once
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
with open("output.txt", "w") as f:
    f.writelines(lines)             # Writes each item in the list

# Better pattern — join with newlines
lines = ["Line 1", "Line 2", "Line 3"]
with open("output.txt", "w") as f:
    f.write("\n".join(lines))

# Reading what we wrote
with open("output.txt", "r") as f:
    print(f.read())
```

### Worked Example 2: Simple Data Logger

```python
import datetime

def log_event(filepath, level, message):
    """Append a timestamped log entry to a file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} {level:<8} {message}\n"
    
    with open(filepath, "a") as f:   # "a" = append, doesn't overwrite
        f.write(log_entry)

def read_recent_logs(filepath, n=10):
    """Read the last n lines from a log file"""
    with open(filepath, "r") as f:
        lines = f.readlines()
    return lines[-n:]  # Last n lines

# Usage
log_file = "app.log"
log_event(log_file, "INFO", "Application started")
log_event(log_file, "INFO", "Connected to database")
log_event(log_file, "WARNING", "Cache miss rate above 20%")
log_event(log_file, "ERROR", "Failed to send email to user@example.com")
log_event(log_file, "INFO", "Retry successful")

print("Recent logs:")
for line in read_recent_logs(log_file, n=5):
    print(" ", line.strip())
```

---

## Part 4: Working with CSV Files

### The Problem: Tabular Data Is Everywhere

CSV (Comma-Separated Values) is the most common format for tabular data — spreadsheets, database exports, analytics tools all use it.

```python
import csv

# ── WRITING CSV ───────────────────────────────────────────────────────────

employees = [
    {"name": "Alice Johnson", "age": 28, "salary": 95000, "dept": "Engineering"},
    {"name": "Bob Smith",     "age": 35, "salary": 72000, "dept": "Marketing"},
    {"name": "Carol Davis",   "age": 42, "salary": 110000, "dept": "Engineering"},
]

# Write using DictWriter (recommended for dicts)
with open("employees.csv", "w", newline="") as f:
    # newline="" is important on Windows — prevents extra blank lines
    fieldnames = ["name", "age", "salary", "dept"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    writer.writeheader()        # Writes the header row
    writer.writerows(employees) # Writes all rows at once

# ── READING CSV ───────────────────────────────────────────────────────────

with open("employees.csv", "r", newline="") as f:
    reader = csv.DictReader(f)  # Reads rows as dictionaries
    
    for row in reader:
        # row is an OrderedDict: {'name': 'Alice', 'age': '28', ...}
        # Note: ALL values are STRINGS — you must convert types!
        name = row["name"]
        age = int(row["age"])        # Convert string to int
        salary = float(row["salary"]) # Convert string to float
        print(f"{name}: ${salary:,.0f} (age {age})")
```

### Worked Example 3: CSV Data Analysis Pipeline

```python
import csv

def load_csv(filepath):
    """Load CSV into list of dicts with type conversion"""
    records = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records

def save_csv(records, filepath, fieldnames=None):
    """Save list of dicts to CSV"""
    if not records:
        return
    
    if fieldnames is None:
        fieldnames = list(records[0].keys())
    
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

def analyze_sales_csv():
    """Full pipeline: read, process, write report"""
    
    # Create sample sales data
    sales_data = [
        {"date": "2025-01-10", "product": "Laptop", "quantity": "3", "price": "999.99", "region": "North"},
        {"date": "2025-01-11", "product": "Mouse", "quantity": "10", "price": "29.99", "region": "South"},
        {"date": "2025-01-12", "product": "Laptop", "quantity": "2", "price": "999.99", "region": "South"},
        {"date": "2025-01-13", "product": "Keyboard", "quantity": "5", "price": "79.99", "region": "North"},
        {"date": "2025-01-14", "product": "Mouse", "quantity": "8", "price": "29.99", "region": "North"},
        {"date": "2025-01-15", "product": "Monitor", "quantity": "1", "price": "399.99", "region": "East"},
    ]
    save_csv(sales_data, "sales.csv")
    
    # Load and process
    records = load_csv("sales.csv")
    
    # Add computed column
    for r in records:
        r["quantity"] = int(r["quantity"])
        r["price"] = float(r["price"])
        r["total"] = r["quantity"] * r["price"]
    
    # Aggregate by product
    product_totals = {}
    for r in records:
        prod = r["product"]
        if prod not in product_totals:
            product_totals[prod] = {"revenue": 0, "units": 0}
        product_totals[prod]["revenue"] += r["total"]
        product_totals[prod]["units"] += r["quantity"]
    
    # Build report rows
    report = [
        {"product": prod, "units_sold": stats["units"], "revenue": stats["revenue"]}
        for prod, stats in sorted(product_totals.items(), key=lambda x: x[1]["revenue"], reverse=True)
    ]
    
    save_csv(report, "sales_report.csv")
    
    print("Sales Report:")
    for row in report:
        print(f"  {row['product']:<12} {row['units_sold']:>5} units  ${row['revenue']:>8,.2f}")

analyze_sales_csv()
```

---

## Part 5: Working with JSON Files

### Why JSON?

CSV is great for flat tabular data. But what about nested data — a user with multiple addresses, orders with multiple line items? That's where **JSON** shines.

```python
import json

# ── WRITING JSON ──────────────────────────────────────────────────────────

data = {
    "users": [
        {
            "id": 1,
            "name": "Alice Johnson",
            "email": "alice@example.com",
            "preferences": {
                "theme": "dark",
                "notifications": True,
                "language": "en"
            },
            "tags": ["premium", "early_adopter"]
        },
        {
            "id": 2,
            "name": "Bob Smith",
            "email": "bob@example.com",
            "preferences": {
                "theme": "light",
                "notifications": False,
                "language": "es"
            },
            "tags": ["standard"]
        }
    ],
    "total_count": 2,
    "last_updated": "2025-01-15"
}

# Write JSON to file
with open("users.json", "w") as f:
    json.dump(data, f, indent=2)    # indent=2 makes it human-readable
    # indent=None → compact (smaller file, not readable)

# ── READING JSON ──────────────────────────────────────────────────────────

with open("users.json", "r") as f:
    loaded_data = json.load(f)      # Parses JSON → Python dict/list

# All Python types are preserved
print(type(loaded_data))                          # <class 'dict'>
print(type(loaded_data["users"]))                 # <class 'list'>
print(type(loaded_data["users"][0]["preferences"]["notifications"]))  # <class 'bool'>

# Access nested data naturally
for user in loaded_data["users"]:
    print(f"{user['name']}: theme={user['preferences']['theme']}")

# json.dumps() / json.loads() — work with STRINGS instead of files
json_string = json.dumps(data, indent=2)  # Python → JSON string
parsed = json.loads(json_string)           # JSON string → Python
```

### Worked Example 4: Config File Manager

```python
import json
import os

class ConfigManager:
    """
    Manages application configuration stored as JSON.
    Supports reading, writing, and updating nested config values.
    """
    
    DEFAULT_CONFIG = {
        "app": {
            "name": "MyApp",
            "version": "1.0.0",
            "debug": False
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "myapp_db"
        },
        "api": {
            "timeout": 30,
            "retries": 3,
            "base_url": "https://api.example.com"
        }
    }
    
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self._load()
    
    def _load(self):
        """Load config from file, or create default if missing"""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                loaded = json.load(f)
            # Merge with defaults (loaded values override defaults)
            return self._deep_merge(self.DEFAULT_CONFIG.copy(), loaded)
        else:
            self._save(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG.copy()
    
    def _save(self, config):
        """Save config to file"""
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    def _deep_merge(self, base, override):
        """Recursively merge override into base"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def get(self, path, default=None):
        """Get a config value by dot-notation path: 'database.host'"""
        keys = path.split(".")
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, path, value):
        """Set a config value by dot-notation path and save"""
        keys = path.split(".")
        target = self.config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
        self._save(self.config)
    
    def __repr__(self):
        return f"ConfigManager({json.dumps(self.config, indent=2)})"


# Usage
config = ConfigManager("app_config.json")

print(config.get("database.host"))    # localhost
print(config.get("api.timeout"))      # 30
print(config.get("app.debug"))        # False

config.set("app.debug", True)
config.set("database.host", "production-db.example.com")
config.set("features.dark_mode", True)   # Adds new nested key!

print(config.get("database.host"))    # production-db.example.com
print(config.get("features.dark_mode"))  # True
```

---

## Part 6: Exception Handling

### The Problem: Reality Is Messy

```python
# Without exception handling — program crashes on ANY error
def get_user_age():
    age = int(input("Enter your age: "))  # What if they type "abc"?
    return age

# With exception handling — program survives and responds gracefully
def get_user_age_safe():
    try:
        age = int(input("Enter your age: "))
        return age
    except ValueError:
        print("Please enter a valid number!")
        return None
```

### The try/except Structure

```python
# Full structure:
try:
    # Code that might fail
    result = risky_operation()

except SpecificError as e:
    # Handle a specific type of error
    print(f"Specific error occurred: {e}")

except (AnotherError, YetAnotherError) as e:
    # Handle multiple error types the same way
    print(f"One of two errors: {e}")

except Exception as e:
    # Catch-all for any other exception (use sparingly!)
    print(f"Unexpected error: {e}")

else:
    # Runs ONLY if no exception occurred in try block
    print("Everything worked!")

finally:
    # Runs ALWAYS — whether or not there was an exception
    # Perfect for cleanup (closing files, DB connections, etc.)
    print("This always runs")
```

### Common Built-in Exceptions

```python
# ValueError — wrong type of value
int("hello")           # ValueError: invalid literal for int()
int("3.14")            # ValueError: invalid literal for int() with base 10
[1,2,3].remove(99)     # ValueError: list.remove(x): x not in list

# TypeError — wrong type entirely
"hello" + 5            # TypeError: can only concatenate str (not "int") to str
len(42)                # TypeError: object of type 'int' has no len()

# KeyError — dict key doesn't exist
d = {"a": 1}
d["b"]                 # KeyError: 'b'

# IndexError — list index out of range
lst = [1, 2, 3]
lst[10]                # IndexError: list index out of range

# FileNotFoundError — file doesn't exist
open("missing.txt")    # FileNotFoundError: No such file or directory

# ZeroDivisionError — dividing by zero
10 / 0                 # ZeroDivisionError: division by zero

# AttributeError — object doesn't have attribute
"hello".explode()      # AttributeError: 'str' object has no attribute 'explode'

# ImportError — can't import module
import nonexistent     # ModuleNotFoundError: No module named 'nonexistent'
```

### Worked Example 5: Robust File Reader

```python
def safe_read_file(filepath, encoding="utf-8"):
    """
    Read a file with comprehensive error handling.
    Returns (content, error_message) tuple.
    """
    try:
        with open(filepath, "r", encoding=encoding) as f:
            content = f.read()
        return content, None
    
    except FileNotFoundError:
        return None, f"File not found: '{filepath}'"
    
    except PermissionError:
        return None, f"Permission denied: cannot read '{filepath}'"
    
    except UnicodeDecodeError:
        return None, f"File '{filepath}' contains invalid characters for {encoding} encoding"
    
    except IsADirectoryError:
        return None, f"'{filepath}' is a directory, not a file"
    
    except OSError as e:
        return None, f"OS error reading '{filepath}': {e}"

# Usage
files_to_read = ["server.log", "missing.txt", "/root/secret.txt", "data.csv"]

for filepath in files_to_read:
    content, error = safe_read_file(filepath)
    if error:
        print(f"⚠  {error}")
    else:
        lines = content.splitlines()
        print(f"✓  Read '{filepath}': {len(lines)} lines")
```

### Worked Example 6: Safe Type Conversion Utilities

```python
def safe_int(value, default=0):
    """Convert to int, return default if impossible"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_float(value, default=0.0):
    """Convert to float, return default if impossible"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_json_parse(text, default=None):
    """Parse JSON string, return default if invalid"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default

# These make data processing pipelines much more robust
raw_data = ["42", "3.14", "N/A", "", None, "100", "bad"]

ints = [safe_int(v, default=-1) for v in raw_data]
print(ints)    # [42, 3, -1, -1, -1, 100, -1]

floats = [safe_float(v) for v in raw_data]
print(floats)  # [42.0, 3.14, 0.0, 0.0, 0.0, 100.0, 0.0]

json_inputs = ['{"a": 1}', "not json", None, '{"b": [1,2,3]}']
parsed = [safe_json_parse(j, default={}) for j in json_inputs]
print(parsed)  # [{'a': 1}, {}, {}, {'b': [1, 2, 3]}]
```

### Raising Exceptions

```python
# You can raise exceptions yourself to signal errors in your code
def set_age(age):
    if not isinstance(age, int):
        raise TypeError(f"Age must be an integer, got {type(age).__name__}")
    if age < 0:
        raise ValueError(f"Age cannot be negative, got {age}")
    if age > 150:
        raise ValueError(f"Age {age} is unrealistic")
    return age

# Test
try:
    set_age(-5)
except ValueError as e:
    print(f"Error: {e}")   # Error: Age cannot be negative, got -5

try:
    set_age("twenty")
except TypeError as e:
    print(f"Error: {e}")   # Error: Age must be an integer, got str

# Re-raising exceptions
def process_file(filepath):
    try:
        with open(filepath) as f:
            data = f.read()
        return data
    except FileNotFoundError:
        print(f"Logging: file {filepath} not found")
        raise   # Re-raise the same exception after logging
```

---

## Part 7: Custom Exceptions

### Why Create Custom Exceptions?

```python
# Generic errors are hard to handle specifically
# If everything raises ValueError, how do you know which error?

# Custom exceptions create a clear, specific vocabulary for your errors
class AppError(Exception):
    """Base class for all app-specific errors"""
    pass

class ValidationError(AppError):
    """Raised when input data fails validation"""
    def __init__(self, field, value, message):
        self.field = field
        self.value = value
        self.message = message
        super().__init__(f"Validation error on '{field}': {message} (got: {value!r})")

class DatabaseError(AppError):
    """Raised when database operations fail"""
    def __init__(self, operation, detail):
        self.operation = operation
        self.detail = detail
        super().__init__(f"Database error during {operation}: {detail}")

class NotFoundError(AppError):
    """Raised when a requested resource doesn't exist"""
    def __init__(self, resource_type, identifier):
        self.resource_type = resource_type
        self.identifier = identifier
        super().__init__(f"{resource_type} not found: {identifier!r}")

class AuthError(AppError):
    """Raised for authentication/authorization failures"""
    pass


# Using custom exceptions
def get_user(user_id, database):
    if not isinstance(user_id, int) or user_id <= 0:
        raise ValidationError("user_id", user_id, "Must be a positive integer")
    
    user = database.get(user_id)
    if user is None:
        raise NotFoundError("User", user_id)
    
    return user

def update_user_email(user_id, new_email, database):
    if "@" not in new_email:
        raise ValidationError("email", new_email, "Must contain @")
    
    user = get_user(user_id, database)  # Can raise NotFoundError
    user["email"] = new_email
    return user


# Handling custom exceptions — you can be precise!
fake_db = {1: {"name": "Alice", "email": "alice@example.com"}}

try:
    user = update_user_email(99, "bob@example.com", fake_db)
except ValidationError as e:
    print(f"Validation failed on field '{e.field}': {e.message}")
except NotFoundError as e:
    print(f"Could not find {e.resource_type} with id {e.identifier}")
except AppError as e:
    print(f"Application error: {e}")

# Output: Could not find User with id 99

try:
    user = update_user_email(1, "not-an-email", fake_db)
except ValidationError as e:
    print(f"Invalid {e.field}: {e.message}")

# Output: Invalid email: Must contain @
```

---

## Part 8: Context Managers — Going Deeper

### Creating Your Own Context Managers

```python
# Context managers aren't just for files.
# Anything that needs "setup → do stuff → cleanup" can be one.

# Method 1: Using a class with __enter__ and __exit__
class Timer:
    """Context manager that times a block of code"""
    import time
    
    def __enter__(self):
        import time
        self.start = time.time()
        return self  # "as" variable gets this
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.elapsed = time.time() - self.start
        print(f"⏱  Elapsed: {self.elapsed:.4f}s")
        return False  # False = don't suppress exceptions

with Timer() as t:
    # Simulate some work
    total = sum(range(1_000_000))
# ⏱  Elapsed: 0.0523s

print(f"Result: {total}, Time: {t.elapsed:.4f}s")


# Method 2: Using contextlib.contextmanager (simpler!)
from contextlib import contextmanager

@contextmanager
def timer(description=""):
    import time
    start = time.time()
    try:
        yield   # Code inside the with block runs here
    finally:
        elapsed = time.time() - start
        label = f" [{description}]" if description else ""
        print(f"⏱{label} {elapsed:.4f}s")

with timer("Sorting 1M numbers"):
    numbers = sorted(range(1_000_000, 0, -1))

# ⏱ [Sorting 1M numbers] 0.1842s


@contextmanager
def temporary_file(suffix=".tmp"):
    """Create a temp file, clean it up when done"""
    import tempfile, os
    
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w")
    try:
        yield tmp           # Give the caller the file object
    finally:
        tmp.close()
        os.unlink(tmp.name) # Delete file after with block exits
        print(f"Cleaned up {tmp.name}")

with temporary_file(".txt") as f:
    f.write("Temporary data")
    filepath = f.name
    print(f"Working with: {filepath}")
# Cleaned up /tmp/tmpXXXXXX.txt
# File is gone now


@contextmanager
def transaction(database):
    """Simulate a database transaction"""
    print("BEGIN TRANSACTION")
    try:
        yield database
        print("COMMIT")
    except Exception as e:
        print(f"ROLLBACK (due to: {e})")
        raise

fake_db = {"users": []}
try:
    with transaction(fake_db) as db:
        db["users"].append({"name": "Alice"})
        # Simulate an error:
        raise ValueError("Something went wrong!")
except ValueError:
    pass  # Transaction was rolled back

print(fake_db)  # ROLLBACK means users is still []
```

---

## Part 9: Worked Examples

### Worked Example 7: Robust CSV ETL Pipeline

```python
import csv
import json
import os
from datetime import datetime

class ETLPipeline:
    """
    Extract → Transform → Load pipeline for CSV data.
    Handles errors gracefully, logs everything.
    """
    
    def __init__(self, log_file="etl.log"):
        self.log_file = log_file
        self.stats = {
            "extracted": 0,
            "transformed": 0,
            "failed": 0,
            "loaded": 0
        }
    
    def _log(self, level, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{timestamp} {level:<8} {message}\n"
        with open(self.log_file, "a") as f:
            f.write(entry)
        print(f"[{level}] {message}")
    
    def extract(self, filepath):
        """Read raw records from CSV"""
        self._log("INFO", f"Extracting from {filepath}")
        
        try:
            records = []
            with open(filepath, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(dict(row))
                    self.stats["extracted"] += 1
            
            self._log("INFO", f"Extracted {len(records)} records")
            return records
        
        except FileNotFoundError:
            self._log("ERROR", f"Source file not found: {filepath}")
            return []
        except csv.Error as e:
            self._log("ERROR", f"CSV parsing error: {e}")
            return []
    
    def transform(self, records, transformers):
        """Apply transformation functions, skip bad records"""
        self._log("INFO", f"Transforming {len(records)} records")
        results = []
        
        for i, record in enumerate(records):
            try:
                transformed = record.copy()
                for transformer in transformers:
                    transformed = transformer(transformed)
                results.append(transformed)
                self.stats["transformed"] += 1
            
            except (ValueError, KeyError, TypeError) as e:
                self.stats["failed"] += 1
                self._log("WARNING", f"Skipping record {i+1}: {e}")
        
        self._log("INFO", f"Transformed: {len(results)}, Failed: {self.stats['failed']}")
        return results
    
    def load(self, records, filepath, fieldnames=None):
        """Write transformed records to output CSV"""
        self._log("INFO", f"Loading {len(records)} records to {filepath}")
        
        if not records:
            self._log("WARNING", "No records to load")
            return
        
        try:
            if fieldnames is None:
                fieldnames = list(records[0].keys())
            
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)
                self.stats["loaded"] += len(records)
            
            self._log("INFO", f"Successfully loaded {len(records)} records")
        
        except PermissionError:
            self._log("ERROR", f"Permission denied writing to {filepath}")
        except OSError as e:
            self._log("ERROR", f"OS error writing file: {e}")
    
    def run(self, source, destination, transformers):
        """Run the full ETL pipeline"""
        self._log("INFO", "=== ETL PIPELINE STARTED ===")
        
        records = self.extract(source)
        if records:
            transformed = self.transform(records, transformers)
            self.load(transformed, destination)
        
        self._log("INFO", f"=== PIPELINE COMPLETE: {self.stats} ===")
        return self.stats


# --- DEMO ---

# Create sample input CSV
sample_data = """employee_id,first_name,last_name,age,salary,department
E001,Alice,Johnson,28,95000,Engineering
E002,Bob,Smith,35,72000,Marketing
E003,,Davis,42,110000,Engineering
E004,Dave,Wilson,abc,68000,Marketing
E005,Eve,Brown,38,88000,Engineering
E006,Frank,Chen,31,92000,Product
"""
with open("raw_employees.csv", "w") as f:
    f.write(sample_data)


# Define transformation functions
def convert_types(record):
    """Convert string fields to proper types"""
    record["age"] = int(record["age"])          # Raises ValueError if "abc"
    record["salary"] = float(record["salary"])
    return record

def validate_required_fields(record):
    """Ensure required fields are present"""
    if not record.get("first_name", "").strip():
        raise ValueError(f"Missing first_name for {record.get('employee_id')}")
    return record

def add_computed_fields(record):
    """Add derived fields"""
    record["full_name"] = f"{record['first_name']} {record['last_name']}"
    record["monthly_salary"] = round(record["salary"] / 12, 2)
    return record

def normalize_department(record):
    """Standardize department names"""
    record["department"] = record["department"].strip().title()
    return record


# Run it!
pipeline = ETLPipeline()
pipeline.run(
    source="raw_employees.csv",
    destination="clean_employees.csv",
    transformers=[
        convert_types,
        validate_required_fields,
        add_computed_fields,
        normalize_department
    ]
)
```

### Worked Example 8: Checkpoint System for Long-Running Tasks

```python
import json
import os

class Checkpoint:
    """
    Save progress for long-running tasks.
    If interrupted, resume from last checkpoint.
    """
    
    def __init__(self, checkpoint_file="checkpoint.json"):
        self.file = checkpoint_file
        self.data = self._load()
    
    def _load(self):
        try:
            with open(self.file, "r") as f:
                data = json.load(f)
                print(f"Resuming from checkpoint: {data}")
                return data
        except FileNotFoundError:
            return {"processed": [], "last_index": 0, "results": {}}
        except json.JSONDecodeError:
            print("Corrupt checkpoint file, starting fresh")
            return {"processed": [], "last_index": 0, "results": {}}
    
    def save(self):
        with open(self.file, "w") as f:
            json.dump(self.data, f, indent=2)
    
    def mark_done(self, item_id, result):
        self.data["processed"].append(item_id)
        self.data["results"][str(item_id)] = result
        self.data["last_index"] += 1
        self.save()
    
    def is_done(self, item_id):
        return item_id in self.data["processed"]
    
    def clear(self):
        if os.path.exists(self.file):
            os.remove(self.file)
        self.data = {"processed": [], "last_index": 0, "results": {}}

def process_with_checkpoint(items, process_func, checkpoint_file="checkpoint.json"):
    """Process a list of items, saving progress after each one"""
    cp = Checkpoint(checkpoint_file)
    
    for i, item in enumerate(items):
        if cp.is_done(item["id"]):
            print(f"Skipping {item['id']} (already processed)")
            continue
        
        try:
            result = process_func(item)
            cp.mark_done(item["id"], result)
            print(f"Processed {item['id']}: {result}")
        except Exception as e:
            print(f"Failed {item['id']}: {e} — will retry next run")
    
    cp.clear()
    return cp.data["results"]

# Example usage
import time, random

def simulate_api_call(item):
    time.sleep(0.1)  # Simulate work
    if random.random() < 0.1:   # 10% chance of failure
        raise ConnectionError("API temporarily unavailable")
    return {"processed": True, "value": item["data"] * 2}

items = [{"id": f"item_{i}", "data": i} for i in range(20)]
results = process_with_checkpoint(items, simulate_api_call)
print(f"\nTotal processed: {len(results)}")
```

---

## Practice Problems

### Easy (1–5)

**Problem 1: File Word Counter**
Read a text file and count total words, lines, and characters.

```python
def count_file_stats(filepath):
    # Your code here
    # Return: {"words": n, "lines": n, "characters": n}
    pass

stats = count_file_stats("story.txt")
print(stats)  # {'words': 342, 'lines': 28, 'characters': 1893}
```

**Problem 2: Safe File Reader**
Write a function that reads a file and returns its contents, or a default if the file doesn't exist.

```python
def read_or_default(filepath, default=""):
    pass

content = read_or_default("config.txt", default="debug=true\nport=8080")
```

**Problem 3: CSV Column Extractor**
Read a CSV file and return a specific column as a list.

```python
def extract_column(filepath, column_name):
    pass

names = extract_column("employees.csv", "name")
print(names)  # ['Alice Johnson', 'Bob Smith', ...]
```

**Problem 4: JSON Config Reader**
Read a JSON config file and return the value at a dot-notation path.

```python
def get_config(filepath, path, default=None):
    pass

host = get_config("config.json", "database.host", "localhost")
```

**Problem 5: Number Input Validator**
Write a function that keeps asking until the user enters a valid number in a range.

```python
def get_number_in_range(prompt, min_val, max_val):
    pass

age = get_number_in_range("Enter your age (1-120): ", 1, 120)
```

### Medium (6–12)

**Problem 6: CSV Merger**
Read multiple CSV files with the same structure and merge them into one.

```python
def merge_csv_files(input_files, output_file):
    pass

merge_csv_files(["sales_jan.csv", "sales_feb.csv", "sales_mar.csv"], "sales_q1.csv")
```

**Problem 7: Log Rotator**
Implement log rotation — when a log file exceeds a size limit, rename it and start fresh.

```python
def rotate_log(filepath, max_size_kb=100):
    """
    If filepath > max_size_kb, rename to filepath.1, filepath.1 → filepath.2, etc.
    Then start a fresh log file.
    """
    pass
```

**Problem 8: JSON Database**
Implement a simple key-value store backed by a JSON file.

```python
class JsonDB:
    def __init__(self, filepath):
        pass
    
    def get(self, key, default=None): pass
    def set(self, key, value): pass
    def delete(self, key): pass
    def all(self): pass

db = JsonDB("store.json")
db.set("users:alice", {"email": "alice@example.com", "score": 100})
print(db.get("users:alice"))
```

**Problem 9: CSV to JSON Converter**
Convert a CSV file to a JSON file.

```python
def csv_to_json(csv_filepath, json_filepath, type_map=None):
    """
    type_map: {'age': int, 'salary': float} — convert string columns to types
    """
    pass

csv_to_json("employees.csv", "employees.json", type_map={"age": int, "salary": float})
```

**Problem 10: File Diff**
Compare two text files and report which lines were added, removed, or unchanged.

```python
def diff_files(file1, file2):
    """
    Return dict with:
    - 'added': lines in file2 but not file1
    - 'removed': lines in file1 but not file2
    - 'unchanged': lines in both
    """
    pass
```

**Problem 11: Retry File Operation**
Write a decorator that retries a file operation if it fails, with configurable attempts.

```python
def retry_on_error(max_attempts=3, delay=1, exceptions=(OSError,)):
    pass

@retry_on_error(max_attempts=3, delay=0.5)
def write_critical_data(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f)
```

**Problem 12: Smart CSV Reader**
Read a CSV and automatically detect and convert column types.

```python
def smart_read_csv(filepath):
    """
    Auto-detect column types:
    - If all values are integers → int
    - If all values are floats → float
    - If "true"/"false" → bool
    - Otherwise → str
    """
    pass
```

### Hard (13–20)

**Problem 13: Append-Only Log Store**
Build a log store where entries can only be appended (never modified), with querying by time range and level.

```python
class LogStore:
    def __init__(self, filepath): pass
    def append(self, level, message): pass
    def query(self, level=None, start_time=None, end_time=None): pass
    def tail(self, n=10): pass
```

**Problem 14: File Watcher**
Write a function that monitors a file for changes and calls a callback when it changes.

```python
import time, os

def watch_file(filepath, callback, interval=1.0):
    """
    Poll filepath every interval seconds.
    Call callback(new_content) whenever file changes.
    """
    pass

def on_change(content):
    print(f"File changed! New content:\n{content[:100]}")

# watch_file("config.json", on_change)  # Runs forever
```

**Problem 15: CSV Report Generator**
Read sales data from CSV and generate a formatted text report with summary statistics.

```python
def generate_report(csv_filepath, output_filepath):
    """
    Read sales CSV with columns: date, product, quantity, price, region
    Generate report with:
    - Summary stats (total revenue, units, top product)
    - Breakdown by region
    - Breakdown by product
    - Top 5 days by revenue
    """
    pass
```

**Problem 16: Nested JSON Flattener/Unflattener**
Convert nested JSON to flat key-value pairs and back.

```python
def flatten_json(data, separator="."):
    """{"a": {"b": 1}} → {"a.b": 1}"""
    pass

def unflatten_json(flat_data, separator="."):
    """{"a.b": 1} → {"a": {"b": 1}}"""
    pass
```

**Problem 17: Multi-Format Exporter**
Write a class that exports a list of records to CSV, JSON, or plain text based on file extension.

```python
class Exporter:
    def export(self, records, filepath):
        # Auto-detect format from extension (.csv, .json, .txt)
        pass

exporter = Exporter()
exporter.export(records, "output.csv")   # CSV format
exporter.export(records, "output.json")  # JSON format
exporter.export(records, "output.txt")   # Human-readable text
```

**Problem 18: Config Validator**
Load a JSON config file and validate it against a schema.

```python
def validate_config(config, schema):
    """
    schema: {
      "database.host": {"type": str, "required": True},
      "database.port": {"type": int, "required": True, "min": 1, "max": 65535},
      "app.debug": {"type": bool, "required": False, "default": False}
    }
    Returns: (is_valid, list_of_errors)
    """
    pass
```

**Problem 19: Incremental CSV Processor**
Process a large CSV file in chunks to avoid loading everything into memory.

```python
def process_csv_in_chunks(filepath, chunk_size=1000, processor=None):
    """
    Read chunk_size rows at a time.
    Apply processor function to each chunk.
    Yield results.
    """
    pass

def summarize_chunk(chunk):
    return {
        "count": len(chunk),
        "total_salary": sum(float(r["salary"]) for r in chunk)
    }

for chunk_result in process_csv_in_chunks("large_employees.csv", 1000, summarize_chunk):
    print(chunk_result)
```

**Problem 20: Transactional File Writer**
Write to a file transactionally — write to a temp file first, then atomically replace.

```python
@contextmanager
def atomic_write(filepath):
    """
    Write to a temp file, then atomically replace the original.
    If an error occurs, original file is untouched.
    """
    pass

with atomic_write("critical_data.json") as f:
    json.dump(important_data, f)
# Only replaces critical_data.json if no errors occur
```

---

## Answer Keys (Selected Problems)

### Problem 1 Solution:
```python
def count_file_stats(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    lines = content.splitlines()
    words = content.split()
    return {
        "lines": len(lines),
        "words": len(words),
        "characters": len(content)
    }
```

### Problem 6 Solution:
```python
def merge_csv_files(input_files, output_file):
    all_rows = []
    headers = None
    
    for filepath in input_files:
        with open(filepath, "r", newline="") as f:
            reader = csv.DictReader(f)
            if headers is None:
                headers = reader.fieldnames
            for row in reader:
                all_rows.append(dict(row))
    
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_rows)
```

### Problem 16 Solution:
```python
def flatten_json(data, separator=".", prefix=""):
    result = {}
    for key, value in data.items():
        full_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_json(value, separator, full_key))
        else:
            result[full_key] = value
    return result

def unflatten_json(flat_data, separator="."):
    result = {}
    for key, value in flat_data.items():
        parts = key.split(separator)
        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return result
```

---

## Mini-Project: Personal Finance Tracker

### Project Goal
A complete command-line personal finance tracker that reads/writes CSV, exports JSON reports, handles all errors gracefully, and maintains an audit log.

```python
"""
finance_tracker.py — Personal finance tracker with file persistence
"""

import csv
import json
import os
from datetime import datetime
from contextlib import contextmanager

# ── FILE PATHS ─────────────────────────────────────────────────────────────
TRANSACTIONS_FILE = "transactions.csv"
BUDGET_FILE = "budget.json"
AUDIT_LOG = "audit.log"
REPORT_FILE = "monthly_report.json"

# ── CUSTOM EXCEPTIONS ──────────────────────────────────────────────────────
class FinanceError(Exception): pass
class InvalidAmountError(FinanceError): pass
class InvalidCategoryError(FinanceError): pass
class BudgetExceededError(FinanceError):
    def __init__(self, category, spent, limit):
        self.category = category
        self.spent = spent
        self.limit = limit
        super().__init__(
            f"Budget exceeded for '{category}': "
            f"spent ${spent:.2f} of ${limit:.2f} limit"
        )

# ── VALID CATEGORIES ───────────────────────────────────────────────────────
CATEGORIES = {"food", "transport", "housing", "entertainment",
              "healthcare", "shopping", "utilities", "income", "other"}

# ── AUDIT LOGGING ──────────────────────────────────────────────────────────
def audit_log(action, details):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(AUDIT_LOG, "a") as f:
        f.write(f"{timestamp} | {action:<20} | {details}\n")

# ── TRANSACTION FUNCTIONS ──────────────────────────────────────────────────
def load_transactions():
    """Load all transactions from CSV"""
    try:
        transactions = []
        with open(TRANSACTIONS_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["amount"] = float(row["amount"])
                transactions.append(row)
        return transactions
    except FileNotFoundError:
        return []
    except (csv.Error, ValueError) as e:
        print(f"Warning: Could not read transactions: {e}")
        return []

def save_transaction(amount, category, description, transaction_type="expense"):
    """Append a new transaction to the CSV file"""
    if amount <= 0:
        raise InvalidAmountError(f"Amount must be positive, got {amount}")
    
    category = category.lower().strip()
    if category not in CATEGORIES:
        raise InvalidCategoryError(
            f"Invalid category '{category}'. Choose from: {', '.join(sorted(CATEGORIES))}"
        )
    
    # Check budget
    budget = load_budget()
    if transaction_type == "expense" and category in budget:
        month = datetime.now().strftime("%Y-%m")
        transactions = load_transactions()
        monthly_spending = sum(
            t["amount"] for t in transactions
            if t["category"] == category
            and t["date"].startswith(month)
            and t["type"] == "expense"
        )
        if monthly_spending + amount > budget[category]:
            raise BudgetExceededError(
                category, monthly_spending + amount, budget[category]
            )
    
    # Write transaction
    file_exists = os.path.exists(TRANSACTIONS_FILE)
    with open(TRANSACTIONS_FILE, "a", newline="") as f:
        fieldnames = ["date", "type", "category", "amount", "description"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        row = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": transaction_type,
            "category": category,
            "amount": amount,
            "description": description
        }
        writer.writerow(row)
    
    audit_log("TRANSACTION", f"{transaction_type.upper()} ${amount:.2f} [{category}] {description}")
    return row

# ── BUDGET FUNCTIONS ───────────────────────────────────────────────────────
def load_budget():
    """Load budget limits from JSON"""
    try:
        with open(BUDGET_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print("Warning: Budget file corrupted, using empty budget")
        return {}

def set_budget(category, limit):
    """Set monthly budget limit for a category"""
    category = category.lower().strip()
    if category not in CATEGORIES:
        raise InvalidCategoryError(f"Invalid category: {category}")
    if limit <= 0:
        raise InvalidAmountError("Budget limit must be positive")
    
    budget = load_budget()
    budget[category] = limit
    
    with open(BUDGET_FILE, "w") as f:
        json.dump(budget, f, indent=2)
    
    audit_log("BUDGET_SET", f"{category}: ${limit:.2f}/month")
    return budget

# ── REPORTING ──────────────────────────────────────────────────────────────
def generate_monthly_report(year_month=None):
    """Generate a report for a given month (default: current month)"""
    if year_month is None:
        year_month = datetime.now().strftime("%Y-%m")
    
    transactions = load_transactions()
    budget = load_budget()
    
    monthly = [t for t in transactions if t["date"].startswith(year_month)]
    
    # Aggregate by category
    by_category = {}
    total_income = 0
    total_expenses = 0
    
    for t in monthly:
        cat = t["category"]
        amount = t["amount"]
        
        if t["type"] == "income":
            total_income += amount
        else:
            total_expenses += amount
            if cat not in by_category:
                by_category[cat] = {"spent": 0, "count": 0, "budget": budget.get(cat)}
            by_category[cat]["spent"] += amount
            by_category[cat]["count"] += 1
    
    report = {
        "month": year_month,
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "net": round(total_income - total_expenses, 2),
        "by_category": {
            cat: {
                "spent": round(data["spent"], 2),
                "transactions": data["count"],
                "budget": data["budget"],
                "remaining": round(data["budget"] - data["spent"], 2) if data["budget"] else None,
                "over_budget": data["budget"] is not None and data["spent"] > data["budget"]
            }
            for cat, data in by_category.items()
        }
    }
    
    # Save report
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    
    return report

def print_report(report):
    """Print a formatted monthly report to console"""
    print(f"\n{'═' * 55}")
    print(f"  FINANCIAL REPORT — {report['month']}")
    print(f"{'═' * 55}")
    print(f"  {'Total Income:':<20} ${report['total_income']:>10,.2f}")
    print(f"  {'Total Expenses:':<20} ${report['total_expenses']:>10,.2f}")
    print(f"  {'Net:':<20} ${report['net']:>10,.2f}")
    print(f"\n  {'CATEGORY':<15} {'SPENT':>10} {'BUDGET':>10} {'STATUS':>12}")
    print(f"  {'-' * 50}")
    
    for cat, data in sorted(report["by_category"].items()):
        budget_str = f"${data['budget']:,.2f}" if data["budget"] else "No limit"
        status = "⚠ OVER" if data["over_budget"] else "✓ OK"
        print(f"  {cat:<15} ${data['spent']:>9,.2f} {budget_str:>10} {status:>12}")
    
    print(f"{'═' * 55}\n")

# ── MAIN DEMO ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Set up budgets
    set_budget("food", 500)
    set_budget("transport", 150)
    set_budget("entertainment", 100)
    
    # Record transactions
    transactions_to_add = [
        (3500, "income", "Monthly salary", "income"),
        (45.50, "food", "Grocery shopping", "expense"),
        (12.00, "transport", "Metro pass top-up", "expense"),
        (89.99, "entertainment", "Netflix + Spotify", "expense"),
        (23.00, "food", "Lunch with colleagues", "expense"),
        (8.50, "transport", "Cab ride", "expense"),
    ]
    
    for amount, category, description, t_type in transactions_to_add:
        try:
            save_transaction(amount, category, description, t_type)
            print(f"✓ Recorded: {description} (${amount:.2f})")
        except BudgetExceededError as e:
            print(f"⚠ Budget alert: {e}")
        except (InvalidAmountError, InvalidCategoryError) as e:
            print(f"✗ Error: {e}")
    
    # Generate and display report
    report = generate_monthly_report()
    print_report(report)
    
    print(f"Full report saved to: {REPORT_FILE}")
    print(f"Audit log saved to:   {AUDIT_LOG}")
```

---

## Chapter Summary

You can now build programs that interact with the real world!

✅ **File Reading**: `open()`, context managers, modes, `read()`, `readlines()`, line-by-line iteration
✅ **File Writing**: `write()`, `writelines()`, append vs overwrite
✅ **CSV**: `csv.DictReader` and `csv.DictWriter` for tabular data
✅ **JSON**: `json.load/dump` for structured/nested data
✅ **Exception Handling**: `try/except/else/finally`, catching specific exceptions
✅ **Custom Exceptions**: Creating your own exception hierarchy
✅ **Context Managers**: `with` statement, `@contextmanager`, building your own
✅ **Robustness**: Logging, retrying, checkpointing, safe type conversion

**Key Takeaways:**
- Always use `with open()` — never raw `open()` + `close()`
- Catch specific exceptions, not bare `except:`
- Custom exceptions make error handling precise and readable
- JSON for nested data, CSV for flat tabular data
- Logging and checkpoints make long-running tasks resilient

**Next Chapter Preview:**
Chapter 8 covers **Object-Oriented Programming (OOP)** — how to model the real world with classes and objects. You'll learn why classes exist, how inheritance works, and how Python's magic methods (`__str__`, `__len__`, `__add__`) let your objects behave like built-in types!
