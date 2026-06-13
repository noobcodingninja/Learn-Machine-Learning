# Chapter 5: Strings and Text Processing

## Part 1: Why Does Text Processing Matter?

### The Problem: The World Runs on Text

Think about how much of your daily digital life is text:
- Every tweet, email, and WhatsApp message
- Product reviews on Amazon
- Log files on servers tracking millions of events
- Code itself — Python files are just text!
- News articles scraped from websites
- User inputs in forms and apps

**A real-world moment:** A company gets 50,000 customer reviews. They want to know:
- Which products are mentioned most?
- What words do unhappy customers use?
- Are there phone numbers or emails hidden in reviews?
- Which reviews are suspiciously similar (fake reviews)?

All of this requires text processing — and Python's string tools make it elegant.

### What Is a String, Really?

```python
# A string is a SEQUENCE of characters
name = "Python"

# Each character has a position (index)
# P  y  t  h  o  n
# 0  1  2  3  4  5   ← forward index
# -6 -5 -4 -3 -2 -1  ← backward index

# So strings are like lists — but immutable!
print(name[0])   # P
print(name[-1])  # n
print(name[1])   # y

# Strings are immutable — you can READ by index, not WRITE
# name[0] = "J"  # TypeError: 'str' object does not support item assignment

# To "change" a string, you create a new one
name = "J" + name[1:]  # Jython — new string
```

---

## Part 2: Creating Strings

### Four Ways to Create Strings

```python
# 1. Single quotes
greeting = 'Hello, World!'

# 2. Double quotes (same as single — preference)
greeting = "Hello, World!"

# 3. Triple quotes — multi-line strings
poem = """
Roses are red,
Violets are blue,
Python is awesome,
And so are you.
"""

# 4. Triple single quotes — also multi-line
sql_query = '''
    SELECT name, email
    FROM users
    WHERE active = 1
    ORDER BY created_at DESC
'''

# When to use which?
# Single inside double: "It's a great day"   (no need to escape apostrophe)
# Double inside single: 'He said "hello"'    (no need to escape quote)
# Escape when needed:   "He said \"hello\""

message = "It's a great day"  # No issue
message = 'He said "hello"'   # No issue
message = "She said \"hi\""   # Escaped double quote
message = 'It\'s raining'     # Escaped apostrophe
```

### Special Characters (Escape Sequences)

```python
# \n — newline
print("Line 1\nLine 2")
# Line 1
# Line 2

# \t — tab
print("Name:\tAlice")
# Name:    Alice

# \\ — literal backslash
print("C:\\Users\\Alice")
# C:\Users\Alice

# \' and \" — literal quotes
print("She said \"hello\"")
# She said "hello"

# Raw strings — treat backslashes literally (great for file paths, regex)
path = r"C:\Users\Alice\Documents"
print(path)  # C:\Users\Alice\Documents  (no escaping needed)

# Without r prefix, \U would be interpreted as a Unicode escape
# path = "C:\Users\Alice"  # Could cause issues!
```

---

## Part 3: String Indexing and Slicing

### The Problem: Extracting Parts of Text

Imagine you have a date string "2025-01-15" and need just the year, or a month. You need slicing.

```python
date = "2025-01-15"

# Slicing syntax: string[start:stop:step]
# start = inclusive, stop = exclusive

# Extract year
year = date[0:4]
print(year)  # 2025

# Extract month
month = date[5:7]
print(month)  # 01

# Extract day
day = date[8:]   # No stop = go to end
print(day)  # 15

# From the end
print(date[-5:])  # 01-15 (last 5 characters)

# Step
alphabet = "abcdefghij"
print(alphabet[0:10:2])  # acegi (every second character)
print(alphabet[::2])     # acegi (same, from start to end)
print(alphabet[::-1])    # jihgfedcba (REVERSE a string!)

# Real-world: extract parts of a URL
url = "https://www.example.com/products/laptop"
# Check if it's HTTPS
is_secure = url[:5] == "https"
print(f"Secure: {is_secure}")  # True

# Get domain
# We'll learn a better way later, but slicing works too!
```

### Worked Example 1: Parsing Log Lines

```python
# Server log format: "2025-01-15 14:23:01 ERROR Database connection failed"
log_line = "2025-01-15 14:23:01 ERROR Database connection failed"

# Extract components using slicing
date_str = log_line[0:10]     # "2025-01-15"
time_str = log_line[11:19]    # "14:23:01"
level = log_line[20:25].strip()  # "ERROR"
message = log_line[26:]       # "Database connection failed"

print(f"Date: {date_str}")
print(f"Time: {time_str}")
print(f"Level: {level}")
print(f"Message: {message}")

# Better approach (we'll refine with split() shortly)
parts = log_line.split(" ", 3)  # Split on space, max 3 splits → 4 parts
date_str, time_str, level, message = parts

print(f"\nParsed: {date_str} | {time_str} | {level} | {message}")
```

---

## Part 4: String Methods

### The Problem: Text Is Messy

Real-world text comes with:
- Extra spaces around words
- Mixed UPPER and lower case
- Inconsistent punctuation
- Line breaks in unexpected places

Python's string methods are your toolkit.

### Category 1: Case Methods

```python
text = "hello, World! PYTHON is GREAT."

# Convert case
print(text.upper())       # HELLO, WORLD! PYTHON IS GREAT.
print(text.lower())       # hello, world! python is great.
print(text.title())       # Hello, World! Python Is Great.
print(text.capitalize())  # Hello, world! python is great.
print(text.swapcase())    # HELLO, wORLD! python IS great.

# Why this matters: Comparing strings case-insensitively
user_input = "PYTHON"
expected = "python"

# ❌ WRONG — case-sensitive comparison fails
print(user_input == expected)   # False

# ✓ CORRECT — normalize before comparing
print(user_input.lower() == expected.lower())   # True

# Real-world: username validation
username = "  Alice123  "
username_clean = username.strip().lower()
print(username_clean)  # "alice123"
```

### Category 2: Whitespace and Stripping Methods

```python
messy = "   hello world   "

print(repr(messy.strip()))    # 'hello world'   — removes both sides
print(repr(messy.lstrip()))   # 'hello world   ' — removes left side
print(repr(messy.rstrip()))   # '   hello world' — removes right side

# repr() shows the string with quotes so you can see spaces clearly

# Strip specific characters
price_str = "$$99.99$$"
print(price_str.strip("$"))  # 99.99

# Remove unwanted characters from user input
email_input = "   alice@example.com   \n"
clean_email = email_input.strip()  # "alice@example.com"

# Real-world: reading a CSV file
csv_line = "  Alice , 28 , Engineer "
fields = [field.strip() for field in csv_line.split(",")]
print(fields)  # ['Alice', '28', 'Engineer']
```

### Category 3: Search and Find Methods

```python
text = "Python is easy to learn. Python is powerful."

# Check if substring exists
print("Python" in text)            # True  (preferred for simple checks)
print(text.startswith("Python"))   # True
print(text.endswith("powerful."))  # True
print(text.startswith("Java"))     # False

# Find position of substring
print(text.find("is"))      # 7   (index of first occurrence)
print(text.find("Java"))    # -1  (not found — returns -1, no error)
print(text.index("is"))     # 7   (index of first occurrence)
# print(text.index("Java")) # ValueError! (raises error if not found)

# When to use find() vs index():
# find() → when not finding is okay (check the return value)
# index() → when not finding means something is wrong (let it crash loudly)

# Find last occurrence
print(text.rfind("Python"))  # 26  (last occurrence)

# Count occurrences
print(text.count("Python"))  # 2
print(text.count("is"))      # 2

# Real-world: validate email format (basic)
email = "alice@example.com"
has_at = "@" in email
at_position = email.find("@")
has_dot_after_at = "." in email[at_position:]
print(f"Basic email valid: {has_at and has_dot_after_at}")  # True
```

### Category 4: Split and Join Methods

```python
# split() — break string into a list
sentence = "Python is easy to learn"

words = sentence.split()       # Split on whitespace (default)
print(words)  # ['Python', 'is', 'easy', 'to', 'learn']

csv = "Alice,28,Engineer,New York"
fields = csv.split(",")
print(fields)  # ['Alice', '28', 'Engineer', 'New York']

# Split with limit
text = "one:two:three:four"
parts = text.split(":", 2)  # Max 2 splits → 3 parts
print(parts)  # ['one', 'two', 'three:four']

# splitlines() — split on line breaks
multi_line = "Line 1\nLine 2\nLine 3"
lines = multi_line.splitlines()
print(lines)  # ['Line 1', 'Line 2', 'Line 3']

# join() — OPPOSITE of split (combine list into string)
words = ["Python", "is", "awesome"]

sentence = " ".join(words)       # "Python is awesome"
csv_row = ",".join(words)        # "Python,is,awesome"
slugified = "-".join(words)      # "Python-is-awesome"
no_space = "".join(words)        # "Pythonisawesome"

print(sentence)
print(csv_row)
print(slugified)

# Common pattern: clean up → split → process → rejoin
messy_tags = "  python , data-science , machine learning  "
tags = [tag.strip() for tag in messy_tags.split(",")]
print(tags)  # ['python', 'data-science', 'machine learning']

clean_tags = ", ".join(sorted(tags))  # Sort and rejoin
print(clean_tags)  # 'data-science, machine learning, python'
```

### Category 5: Replace Methods

```python
text = "Hello World! Hello Python!"

# replace(old, new, count=-1)
new_text = text.replace("Hello", "Hi")
print(new_text)  # Hi World! Hi Python!

# Replace only first occurrence
new_text = text.replace("Hello", "Hi", 1)
print(new_text)  # Hi World! Hello Python!

# Chain replacements
text = "I love cats. Cats are amazing. All cats are great."
text = text.replace("cats", "dogs").replace("Cats", "Dogs")
print(text)  # I love dogs. Dogs are amazing. All dogs are great.

# Real-world: clean up user input
phone = "(555) 123-4567"
clean_phone = phone.replace("(", "").replace(")", "").replace(" ", "").replace("-", "")
print(clean_phone)  # 5551234567

# Real-world: template substitution (basic version)
template = "Dear [NAME], your order [ORDER_ID] is ready."
message = template.replace("[NAME]", "Alice").replace("[ORDER_ID]", "ORD-001")
print(message)  # Dear Alice, your order ORD-001 is ready.
```

### Category 6: Check Methods (Boolean)

```python
# isalpha() — only letters
print("Python".isalpha())    # True
print("Python3".isalpha())   # False (has a number)
print("".isalpha())          # False (empty string)

# isdigit() — only digits
print("12345".isdigit())     # True
print("123.45".isdigit())    # False (has a dot)
print("123abc".isdigit())    # False

# isalnum() — letters or digits (no spaces or symbols)
print("Python3".isalnum())   # True
print("Hello World".isalnum()) # False (has space)

# isspace() — only whitespace
print("   ".isspace())       # True
print("  a  ".isspace())     # False

# isupper() / islower()
print("PYTHON".isupper())    # True
print("python".islower())    # True
print("Python".isupper())    # False

# Real-world: form validation
def validate_username(username):
    """Username: 3-20 chars, only letters/numbers, no spaces"""
    if not (3 <= len(username) <= 20):
        return False, "Username must be 3-20 characters"
    if not username.isalnum():
        return False, "Username can only contain letters and numbers"
    return True, "Valid username"

tests = ["Alice123", "al", "Alice 123", "a" * 25, "ValidUser"]
for test in tests:
    valid, msg = validate_username(test)
    print(f"'{test}': {msg}")
```

---

## Part 5: String Formatting

### The Problem: Combining Variables with Text

You have a user's name, score, and rank. You want to print:
`"Alice scored 9850 points and ranked #3 globally."`

There are 3 main ways.

### Method 1: % Formatting (Old Style — Avoid)

```python
name = "Alice"
score = 9850
rank = 3

# Old style — don't use this
message = "Hello, %s! You scored %d points." % (name, score)
print(message)

# Why avoid? Less readable, error-prone with multiple variables
```

### Method 2: .format() (Better)

```python
name = "Alice"
score = 9850
rank = 3

# Basic positional
message = "Hello, {}! You scored {} points.".format(name, score)

# Named placeholders (clearer)
message = "Hello, {name}! You scored {score} points.".format(name=name, score=score)

# Number formatting inside format
price = 1234567.89
print("Price: ${:,.2f}".format(price))  # Price: $1,234,567.89
# {:,.2f} means: comma separator, 2 decimal places, float

# Padding and alignment
print("{:<10} | {:>10} | {:^10}".format("left", "right", "center"))
# left       |      right |   center
# < = left align, > = right align, ^ = center, 10 = width
```

### Method 3: f-strings (Modern — Prefer This!)

```python
name = "Alice"
score = 9850
rank = 3

# Basic f-string
message = f"Hello, {name}! You scored {score} points and ranked #{rank}."
print(message)

# Expressions inside f-strings
a, b = 10, 3
print(f"{a} + {b} = {a + b}")    # 10 + 3 = 13
print(f"{a} / {b} = {a / b:.2f}")  # 10 / 3 = 3.33

# Calling methods inside f-strings
name = "alice"
print(f"Welcome, {name.title()}!")  # Welcome, Alice!

# Formatting numbers
price = 9999.9
discount = 0.15
final = price * (1 - discount)
print(f"Original: ${price:,.2f}")   # $9,999.90
print(f"Discount: {discount:.0%}")   # 15%
print(f"Final:    ${final:,.2f}")   # $8,499.92

# Padding and alignment
items = [("Apple", 1.50, 10), ("Banana", 0.75, 25), ("Cherry", 3.00, 5)]
print(f"{'Item':<10} {'Price':>8} {'Qty':>6}")
print("-" * 26)
for item, price, qty in items:
    print(f"{item:<10} ${price:>7.2f} {qty:>6}")

# Output:
# Item          Price    Qty
# --------------------------
# Apple         $ 1.50     10
# Banana        $ 0.75     25
# Cherry        $ 3.00      5

# Debugging with f-strings (Python 3.8+)
x = 42
print(f"{x = }")   # x = 42  (prints name AND value — great for debugging!)
```

### Worked Example 2: Generating Reports

```python
def generate_sales_report(sales_data):
    """Generate a formatted sales report"""
    
    report = []
    report.append("=" * 50)
    report.append(f"{'MONTHLY SALES REPORT':^50}")
    report.append("=" * 50)
    
    total_sales = 0
    report.append(f"\n{'Product':<20} {'Units':>6} {'Revenue':>12}")
    report.append("-" * 40)
    
    for product, units, price_per_unit in sales_data:
        revenue = units * price_per_unit
        total_sales += revenue
        report.append(f"{product:<20} {units:>6} ${revenue:>10,.2f}")
    
    report.append("-" * 40)
    report.append(f"{'TOTAL':.<20} {'':>6} ${total_sales:>10,.2f}")
    report.append("=" * 50)
    
    return "\n".join(report)

# Sample data
sales_data = [
    ("MacBook Pro", 45, 1299.99),
    ("iPhone 15", 120, 799.99),
    ("AirPods Pro", 200, 249.99),
    ("iPad Air", 60, 599.99),
]

print(generate_sales_report(sales_data))

# Output:
# ==================================================
#              MONTHLY SALES REPORT
# ==================================================
#
# Product               Units      Revenue
# ----------------------------------------
# MacBook Pro              45   $58,499.55
# iPhone 15               120   $95,998.80
# AirPods Pro             200   $49,998.00
# iPad Air                 60   $35,999.40
# ----------------------------------------
# TOTAL...............         $240,495.75
# ==================================================
```

---

## Part 6: Common Text Processing Tasks

### Task 1: Cleaning User Input

```python
def clean_user_input(raw_input):
    """
    Clean user input:
    - Strip whitespace
    - Remove extra spaces between words
    - Title case for names
    """
    # Strip outer whitespace
    cleaned = raw_input.strip()
    
    # Remove extra spaces between words
    # split() with no args splits on ANY whitespace and removes extras
    cleaned = " ".join(cleaned.split())
    
    return cleaned

# Test
inputs = [
    "  alice   johnson  ",
    "\tBOB   SMITH\n",
    "carol    davis",
]

for raw in inputs:
    clean = clean_user_input(raw)
    print(f"'{raw}' → '{clean}'")

# Output:
# '  alice   johnson  ' → 'alice   johnson'
# ... etc

def clean_name(raw_name):
    cleaned = " ".join(raw_name.strip().split())
    return cleaned.title()

print(clean_name("  alice   johnson  "))  # Alice Johnson
print(clean_name("\tBOB   SMITH\n"))      # Bob Smith
```

### Task 2: Parsing Structured Text

```python
# Parsing a configuration file (INI style)
config_text = """
# Database settings
host = localhost
port = 5432
database = myapp_db
username = admin

# App settings
debug = true
log_level = INFO
max_connections = 10
"""

def parse_config(config_text):
    """Parse simple key=value config file"""
    config = {}
    
    for line in config_text.splitlines():
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue
        
        # Split on first = only
        if "=" in line:
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    
    return config

settings = parse_config(config_text)
for key, value in settings.items():
    print(f"{key}: {value}")

# Output:
# host: localhost
# port: 5432
# database: myapp_db
# username: admin
# debug: true
# log_level: INFO
# max_connections: 10
```

### Task 3: Text Analysis

```python
def analyze_text(text):
    """
    Analyze a piece of text:
    - Word count
    - Sentence count
    - Average word length
    - Most common words
    """
    import string
    
    # Clean and tokenize
    # Remove punctuation and convert to lowercase
    clean_text = text.lower()
    for punct in string.punctuation:
        clean_text = clean_text.replace(punct, " ")
    
    # Split into words, filter empty strings
    words = [w for w in clean_text.split() if w]
    
    # Count sentences (roughly — split on . ! ?)
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    
    # Word frequency
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Top 5 words (exclude common "stop words")
    stop_words = {"the", "a", "an", "is", "it", "in", "on", "at", "to", "and", "of", "for"}
    top_words = sorted(
        [(word, count) for word, count in word_freq.items() if word not in stop_words],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    return {
        "total_words": len(words),
        "unique_words": len(word_freq),
        "total_sentences": len(sentences),
        "avg_word_length": sum(len(w) for w in words) / len(words),
        "top_words": top_words
    }

sample = """
Python is a high-level programming language. Python was created by Guido van Rossum.
Python is known for its simple syntax. Many programmers love Python for data science
and machine learning. Python is one of the most popular programming languages today.
"""

results = analyze_text(sample)
print(f"Total words: {results['total_words']}")
print(f"Unique words: {results['unique_words']}")
print(f"Sentences: {results['total_sentences']}")
print(f"Avg word length: {results['avg_word_length']:.1f}")
print(f"Top words: {results['top_words']}")
```

### Task 4: Slug Generation (URL Friendly Text)

```python
def create_slug(title):
    """
    Convert blog title to URL-friendly slug
    "My Python Tutorial! #1" → "my-python-tutorial-1"
    """
    import string
    
    # Convert to lowercase
    slug = title.lower()
    
    # Replace special characters with space
    slug = slug.replace("&", "and")  # Handle common cases
    
    # Remove all characters except letters, numbers, spaces, hyphens
    allowed = set(string.ascii_lowercase + string.digits + " -")
    slug = "".join(char for char in slug if char in allowed)
    
    # Replace spaces with hyphens
    slug = "-".join(slug.split())  # split() handles multiple spaces
    
    # Remove leading/trailing hyphens
    slug = slug.strip("-")
    
    return slug

# Test cases
titles = [
    "My Python Tutorial! #1",
    "Data Science & Machine Learning",
    "  Hello World  ",
    "10 Tips for Better Code",
    "Python's Best Practices"
]

for title in titles:
    print(f"'{title}' → '{create_slug(title)}'")

# Output:
# 'My Python Tutorial! #1' → 'my-python-tutorial-1'
# 'Data Science & Machine Learning' → 'data-science-and-machine-learning'
# '  Hello World  ' → 'hello-world'
# '10 Tips for Better Code' → '10-tips-for-better-code'
# "Python's Best Practices" → 'pythons-best-practices'
```

### Worked Example 3: CSV Parser (Without csv Module)

```python
def parse_csv(csv_text):
    """Parse simple CSV text into list of dictionaries"""
    lines = csv_text.strip().splitlines()
    
    if not lines:
        return []
    
    # First line is header
    headers = [h.strip() for h in lines[0].split(",")]
    
    records = []
    for line in lines[1:]:
        if not line.strip():
            continue
        
        values = [v.strip() for v in line.split(",")]
        
        # Zip headers with values into a dictionary
        record = dict(zip(headers, values))
        records.append(record)
    
    return records

# Test
csv_data = """
name, age, city, salary
Alice Johnson, 28, New York, 95000
Bob Smith, 35, San Francisco, 120000
Carol Davis, 42, Chicago, 85000
Dave Wilson, 29, Austin, 78000
"""

records = parse_csv(csv_data)
for record in records:
    print(record)

# Output:
# {'name': 'Alice Johnson', 'age': '28', 'city': 'New York', 'salary': '95000'}
# {'name': 'Bob Smith', 'age': '35', 'city': 'San Francisco', 'salary': '120000'}
# ...

# Now we can filter!
high_earners = [r for r in records if int(r["salary"]) > 90000]
print(f"\nHigh earners:")
for r in high_earners:
    print(f"  {r['name']} - ${int(r['salary']):,}")
```

### Worked Example 4: Password Strength Checker

```python
def check_password_strength(password):
    """
    Check password strength based on:
    - Length (8+ chars)
    - Has uppercase
    - Has lowercase
    - Has digit
    - Has special character
    """
    import string
    
    checks = {
        "length": len(password) >= 8,
        "has_uppercase": any(c.isupper() for c in password),
        "has_lowercase": any(c.islower() for c in password),
        "has_digit": any(c.isdigit() for c in password),
        "has_special": any(c in string.punctuation for c in password)
    }
    
    passed = sum(checks.values())
    
    if passed == 5:
        strength = "Strong 💪"
    elif passed >= 3:
        strength = "Medium 😐"
    else:
        strength = "Weak ⚠️"
    
    failed_checks = [key for key, passed_check in checks.items() if not passed_check]
    
    return {
        "strength": strength,
        "score": f"{passed}/5",
        "recommendations": failed_checks
    }

# Test passwords
passwords = ["abc", "password123", "P@ssw0rd!", "Secure#2025Pass"]
for pwd in passwords:
    result = check_password_strength(pwd)
    print(f"\n'{pwd}':")
    print(f"  Strength: {result['strength']} ({result['score']})")
    if result["recommendations"]:
        print(f"  Missing: {', '.join(result['recommendations'])}")
```

### Worked Example 5: Name Formatter

```python
def format_name(raw_name, format_type="full"):
    """
    Format names in various styles.
    format_type: 'full', 'last_first', 'initials', 'abbreviated'
    """
    # Clean the name
    parts = raw_name.strip().split()
    parts = [part.strip().title() for part in parts if part.strip()]
    
    if not parts:
        return ""
    
    if format_type == "full":
        return " ".join(parts)  # "John Michael Smith"
    
    elif format_type == "last_first":
        if len(parts) == 1:
            return parts[0]
        return f"{parts[-1]}, {' '.join(parts[:-1])}"  # "Smith, John Michael"
    
    elif format_type == "initials":
        return ".".join(part[0] for part in parts) + "."  # "J.M.S."
    
    elif format_type == "abbreviated":
        if len(parts) == 1:
            return parts[0]
        # First name + last initial: "John S."
        return f"{parts[0]} {parts[-1][0]}."
    
    return " ".join(parts)

# Test
names = ["john michael smith", "ALICE JOHNSON", "  bob  "]
for name in names:
    print(f"Original: '{name}'")
    for fmt in ["full", "last_first", "initials", "abbreviated"]:
        print(f"  {fmt}: {format_name(name, fmt)}")
    print()
```

### Worked Example 6: Simple Template Engine

```python
def render_template(template, context):
    """
    Render a template string with variable substitution.
    Variables are written as {{variable_name}}
    """
    result = template
    
    for key, value in context.items():
        placeholder = "{{" + key + "}}"
        result = result.replace(placeholder, str(value))
    
    return result

# Email templates
welcome_template = """
Dear {{name}},

Welcome to {{platform}}! Your account has been created.

Account Details:
- Username: {{username}}
- Email: {{email}}
- Plan: {{plan}}

You have {{trial_days}} days of free trial.

Best regards,
The {{platform}} Team
"""

user_data = {
    "name": "Alice Johnson",
    "platform": "DataFlow Pro",
    "username": "alice_j",
    "email": "alice@example.com",
    "plan": "Professional",
    "trial_days": 14
}

print(render_template(welcome_template, user_data))

# Invoice template
invoice_template = "Invoice #{{invoice_id}}: {{product}} — ${{amount}}"
invoices = [
    {"invoice_id": "INV001", "product": "Laptop", "amount": "999.99"},
    {"invoice_id": "INV002", "product": "Mouse", "amount": "29.99"},
]

for inv in invoices:
    print(render_template(invoice_template, inv))
```

---

## Part 7: Common Mistakes and How to Avoid Them

### Mistake 1: Strings Are Immutable — Concatenating in a Loop Is Slow

```python
# ❌ WRONG — Slow for large data
words = ["Python", "is", "fast", "and", "fun"]
result = ""
for word in words:
    result += word + " "   # Creates a NEW string every iteration!
print(result)

# ✓ CORRECT — Use join
result = " ".join(words)
print(result)  # Python is fast and fun

# Why it matters: With 10,000 words, the loop creates 10,000 strings.
# join() creates exactly ONE string. Much faster!
```

### Mistake 2: Comparing Strings Without Normalizing Case

```python
# ❌ WRONG
user_city = "new york"
if user_city == "New York":
    print("Match!")  # Doesn't print!

# ✓ CORRECT
if user_city.lower() == "New York".lower():
    print("Match!")  # Match!

# Or normalize at input time:
user_city = input("Enter city: ").strip().lower()
# Now all comparisons use lowercase
```

### Mistake 3: Off-by-One Errors in Slicing

```python
date = "2025-01-15"

# Counting: 2 0 2 5 - 0 1 - 1 5
#           0 1 2 3 4 5 6 7 8 9

# ❌ WRONG — forgetting stop is exclusive
year = date[0:3]   # "202" (only 3 chars, not 4!)

# ✓ CORRECT
year = date[0:4]   # "2025" (indices 0,1,2,3)
month = date[5:7]  # "01"
day = date[8:10]   # "15"

# Better approach: name your positions
YEAR_START, YEAR_END = 0, 4
MONTH_START, MONTH_END = 5, 7
DAY_START, DAY_END = 8, 10

year = date[YEAR_START:YEAR_END]
```

### Mistake 4: Using + Instead of join() for Multiple Strings

```python
first = "John"
last = "Smith"

# ❌ WRONG — works but inefficient and less readable at scale
full_name = first + " " + last

# ✓ CORRECT — join is preferred when combining multiple
full_name = " ".join([first, last])

# For 2-3 strings, + is fine. For many strings, always use join.
parts = ["Hello", "World", "from", "Python"]
sentence = " ".join(parts)  # Much better than "Hello" + " " + "World" + ...
```

### Mistake 5: Forgetting strip() When Reading Input

```python
# ❌ WRONG — user input often has newline at end
user_input = "Alice\n"   # From input() or reading a file
if user_input == "Alice":
    print("Welcome!")    # Won't print!

# ✓ CORRECT — always strip user input
user_input = user_input.strip()
if user_input == "Alice":
    print("Welcome!")    # Welcome!
```

---

## Practice Problems

### Easy (1–5)

**Problem 1: Reverse a String**
Write a function that reverses any string.

```python
def reverse_string(s):
    # Your code here (one line using slicing!)
    pass

print(reverse_string("Python"))  # nohtyP
print(reverse_string("racecar")) # racecar (palindrome!)
```

**Problem 2: Count Vowels**
Write a function that counts vowels in a string.

```python
def count_vowels(text):
    # Your code here
    pass

print(count_vowels("Hello World"))  # 3
print(count_vowels("Python"))       # 1
```

**Problem 3: Title Formatter**
Convert a messy string to proper title case, stripping extra spaces.

```python
def format_title(title):
    pass

print(format_title("  the great   gatsby  "))  # "The Great Gatsby"
print(format_title("PYTHON programming"))       # "Python Programming"
```

**Problem 4: Currency Formatter**
Format a float as a currency string.

```python
def format_currency(amount, symbol="$"):
    pass

print(format_currency(1234567.89))        # $1,234,567.89
print(format_currency(99.9, "₹"))         # ₹99.90
print(format_currency(0.5))               # $0.50
```

**Problem 5: Initials Generator**
Generate initials from a full name.

```python
def get_initials(full_name):
    pass

print(get_initials("John Michael Smith"))  # J.M.S.
print(get_initials("Alice Johnson"))       # A.J.
```

### Medium (6–12)

**Problem 6: Palindrome Checker**
Check if a string is a palindrome (reads same forwards and backwards), ignoring spaces, punctuation, and case.

```python
def is_palindrome(text):
    pass

print(is_palindrome("racecar"))               # True
print(is_palindrome("A man a plan a canal Panama"))  # True
print(is_palindrome("hello"))                 # False
```

**Problem 7: Word Wrap**
Wrap a long string to a specified line width without breaking words.

```python
def word_wrap(text, width=40):
    pass

long_text = "Python is a versatile language used in web development data science machine learning and automation"
print(word_wrap(long_text, 30))
# Python is a versatile language
# used in web development data
# science machine learning and
# automation
```

**Problem 8: Email Extractor**
Extract all email-like strings from a block of text.

```python
def extract_emails(text):
    # Hint: split on whitespace, check each word for @ and .
    pass

text = """
Contact us at support@company.com or sales@company.com.
For billing, email billing@accounts.company.com.
The address is 123 Main St (not an email).
"""
print(extract_emails(text))
# ['support@company.com', 'sales@company.com', 'billing@accounts.company.com']
```

**Problem 9: Caesar Cipher**
Encrypt/decrypt text using Caesar cipher (shift each letter by n positions).

```python
def caesar_cipher(text, shift, mode="encrypt"):
    # Hint: Use ord() to get character code, chr() to convert back
    # ord('a') = 97, ord('z') = 122
    # Use modulo to wrap around: (ord(c) - 97 + shift) % 26 + 97
    pass

message = "Hello World"
encrypted = caesar_cipher(message, 3, "encrypt")
decrypted = caesar_cipher(encrypted, 3, "decrypt")

print(f"Original:  {message}")    # Hello World
print(f"Encrypted: {encrypted}")   # Khoor Zruog
print(f"Decrypted: {decrypted}")   # Hello World
```

**Problem 10: Log Parser**
Parse server logs and extract statistics.

```python
logs = """
2025-01-15 10:23:01 INFO User alice logged in
2025-01-15 10:24:15 ERROR Database connection failed
2025-01-15 10:24:16 INFO Retrying database connection
2025-01-15 10:24:17 INFO Database connected successfully
2025-01-15 10:25:30 WARNING High memory usage detected
2025-01-15 10:26:00 ERROR Timeout on request /api/users
"""

def parse_logs(log_text):
    """
    Return:
    - Count of each log level (INFO, ERROR, WARNING)
    - List of ERROR messages
    - All unique dates
    """
    pass

stats = parse_logs(logs)
print(stats)
```

**Problem 11: Text Truncator**
Truncate text to a maximum length, ending with "..." if truncated — but don't break words.

```python
def truncate(text, max_length=50):
    pass

text = "Python is a versatile and powerful programming language"
print(truncate(text, 30))  # "Python is a versatile and..."
print(truncate(text, 100)) # Full text (not truncated)
print(truncate(text, 10))  # "Python..."
```

**Problem 12: String Compression**
Implement basic run-length encoding: "aaabbbcccc" → "a3b3c4".

```python
def compress_string(s):
    pass

print(compress_string("aaabbbcccc"))   # a3b3c4
print(compress_string("abcd"))         # abcd (no compression if not helpful)
print(compress_string("aabbaaa"))      # a2b2a3
```

### Hard (13–20)

**Problem 13: Markdown to Plain Text**
Strip basic Markdown formatting from text.

```python
def strip_markdown(text):
    # Remove: **bold**, *italic*, # headers, [text](url), `code`
    pass

md_text = """
# Hello World
This is **bold** and *italic* text.
Visit [Python.org](https://python.org) for more.
Use `print()` to output text.
"""
print(strip_markdown(md_text))
# Hello World
# This is bold and italic text.
# Visit Python.org for more.
# Use print() to output text.
```

**Problem 14: Sentence Tokenizer**
Split text into sentences, handling edge cases like "Mr." and "Dr." abbreviations.

```python
def tokenize_sentences(text):
    # Handle abbreviations: Mr., Dr., Mrs., etc.
    # Don't split on abbreviation periods
    pass

text = "Dr. Smith works with Mr. Jones. They study Python 3.11. It's great!"
sentences = tokenize_sentences(text)
for i, s in enumerate(sentences, 1):
    print(f"{i}: {s}")
# 1: Dr. Smith works with Mr. Jones.
# 2: They study Python 3.11.
# 3: It's great!
```

**Problem 15: Fuzzy String Matching**
Calculate similarity between two strings (simple character-level).

```python
def string_similarity(s1, s2):
    """
    Calculate what % of characters match at same positions.
    Also handles length differences.
    """
    pass

print(string_similarity("python", "pytohn"))   # High similarity (typo)
print(string_similarity("hello", "world"))     # Low similarity
print(string_similarity("kitten", "sitting"))  # Medium
```

**Problem 16: Text Anonymizer**
Replace sensitive information with placeholders.

```python
def anonymize_text(text):
    """
    Replace:
    - Email addresses → [EMAIL]
    - Phone numbers → [PHONE]
    - Names after "Dear" → [NAME]
    """
    pass

text = """
Dear Alice Johnson,
Please call us at 555-123-4567 or email billing@company.com
for your account issues. Also CC support@help.com.
"""
print(anonymize_text(text))
# Dear [NAME],
# Please call us at [PHONE] or email [EMAIL]
# for your account issues. Also CC [EMAIL].
```

**Problem 17: Word Frequency Heatmap (Text)**
Display word frequency as a text-based bar chart.

```python
def word_heatmap(text, top_n=10, bar_width=40):
    pass

sample = """
Python is great Python is fast Python is easy
Python helps with data science Python helps with automation
data is important Python data analysis is powerful
"""
word_heatmap(sample, top_n=6)
# python ████████████████████████████████████████ 6
# is     ████████████████████████████████         4
# data   ████████████████████                     3
# helps  ██████████                               2
# great  █████                                    1
# fast   █████                                    1
```

**Problem 18: Multi-Column Formatter**
Format a list of items into multiple columns.

```python
def format_columns(items, num_cols=3, col_width=20):
    pass

cities = [
    "New York", "Los Angeles", "Chicago", "Houston",
    "Phoenix", "Philadelphia", "San Antonio", "San Diego",
    "Dallas", "San Jose", "Austin", "Jacksonville"
]

format_columns(cities, num_cols=3, col_width=18)
# New York           Los Angeles        Chicago
# Houston            Phoenix            Philadelphia
# San Antonio        San Diego          Dallas
# San Jose           Austin             Jacksonville
```

**Problem 19: Template Variable Validator**
Check a template string for undefined or unused variables.

```python
def validate_template(template, context):
    """
    Check:
    1. All {{variable}} in template are in context
    2. All keys in context are used in template
    Return: (is_valid, undefined_vars, unused_vars)
    """
    pass

template = "Dear {{name}}, your {{product}} order is {{status}}."
context = {"name": "Alice", "product": "Laptop", "shipping": "Express"}

is_valid, undefined, unused = validate_template(template, context)
print(f"Valid: {is_valid}")         # False
print(f"Undefined: {undefined}")     # {'status'}
print(f"Unused: {unused}")           # {'shipping'}
```

**Problem 20: Phone Number Normalizer**
Normalize different phone number formats to a standard.

```python
def normalize_phone(phone_str, format_type="dashes"):
    """
    Accept any of: (555) 123-4567, 555.123.4567, 5551234567, +1-555-123-4567
    Output: 555-123-4567 (dashes) or (555) 123-4567 (parens) or 5551234567 (plain)
    Handle country code +1 if present.
    """
    pass

phones = [
    "(555) 123-4567",
    "555.123.4567",
    "5551234567",
    "+1-555-123-4567",
    "+1 (555) 123-4567"
]

for phone in phones:
    print(f"{phone} → {normalize_phone(phone, 'dashes')}")
# All should output: 555-123-4567
```

---

## Answer Keys (Selected Problems)

### Problem 1 Solution:
```python
def reverse_string(s):
    return s[::-1]
```

### Problem 6 Solution:
```python
def is_palindrome(text):
    import string
    # Remove punctuation and spaces, lowercase
    cleaned = "".join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]

print(is_palindrome("A man a plan a canal Panama"))  # True
```

### Problem 9 Solution:
```python
def caesar_cipher(text, shift, mode="encrypt"):
    if mode == "decrypt":
        shift = -shift
    
    result = []
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            shifted = (ord(char) - base + shift) % 26 + base
            result.append(chr(shifted))
        else:
            result.append(char)  # Non-letters unchanged
    
    return "".join(result)
```

### Problem 12 Solution:
```python
def compress_string(s):
    if not s:
        return s
    
    result = []
    count = 1
    
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            count += 1
        else:
            result.append(s[i-1] + (str(count) if count > 1 else ""))
            count = 1
    
    result.append(s[-1] + (str(count) if count > 1 else ""))
    
    compressed = "".join(result)
    return compressed if len(compressed) < len(s) else s
```

### Problem 17 Solution:
```python
def word_heatmap(text, top_n=10, bar_width=40):
    import string
    clean = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    words = clean.split()
    
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    max_count = top[0][1] if top else 1
    max_word_len = max(len(word) for word, _ in top)
    
    for word, count in top:
        bar_len = int(count / max_count * bar_width)
        bar = "█" * bar_len
        print(f"{word:<{max_word_len}} {bar:<{bar_width}} {count}")
```

---

## Mini-Project: Text Analytics Dashboard

### Project Overview
Build a comprehensive text analyzer that processes any text and generates a full analytics report.

**Features:**
1. Basic statistics (word count, sentence count, reading time)
2. Word frequency analysis
3. Readability metrics
4. Text cleaning and normalization
5. Keyword extraction
6. Formatted console report

```python
def text_dashboard(text):
    """
    Complete text analytics dashboard.
    Takes raw text, returns detailed analytics report.
    """
    import string
    
    # ── 1. CLEAN TEXT ──────────────────────────────────────────────────
    clean_text = text.strip()
    
    # ── 2. BASIC STATS ─────────────────────────────────────────────────
    # Characters
    char_count = len(text)
    char_no_spaces = len(text.replace(" ", ""))
    
    # Words
    words = clean_text.split()
    word_count = len(words)
    
    # Sentences (split on . ! ?)
    sentences = [s.strip() for s in 
                 clean_text.replace("!", ".").replace("?", ".").split(".")
                 if s.strip()]
    sentence_count = len(sentences)
    
    # Paragraphs
    paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
    para_count = len(paragraphs)
    
    # Reading time (avg 200 words per minute)
    reading_time_min = word_count / 200
    
    # ── 3. WORD FREQUENCY ──────────────────────────────────────────────
    stop_words = {"the", "a", "an", "is", "it", "in", "on", "at", "to",
                  "and", "of", "for", "are", "was", "were", "be", "has",
                  "had", "that", "this", "with", "as", "by", "from", "or"}
    
    # Clean words
    clean_words = []
    for word in words:
        word = word.lower().strip(string.punctuation)
        if word and word not in stop_words and word.isalpha():
            clean_words.append(word)
    
    freq = {}
    for word in clean_words:
        freq[word] = freq.get(word, 0) + 1
    
    top_10 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # ── 4. READABILITY METRICS ─────────────────────────────────────────
    avg_word_length = sum(len(w) for w in words) / word_count if word_count else 0
    avg_sentence_length = word_count / sentence_count if sentence_count else 0
    
    # Flesch Reading Ease (simplified estimate)
    # Higher = easier to read (0-100 scale)
    # Simple estimate based on avg word and sentence length
    reading_ease = max(0, 100 - (avg_sentence_length * 1.5) - (avg_word_length * 5))
    
    if reading_ease > 70:
        readability = "Easy"
    elif reading_ease > 40:
        readability = "Medium"
    else:
        readability = "Difficult"
    
    # ── 5. UNIQUE WORDS ────────────────────────────────────────────────
    unique_words = set(w.lower().strip(string.punctuation) for w in words if w.isalpha())
    vocabulary_richness = len(unique_words) / word_count * 100 if word_count else 0
    
    # ── 6. FORMAT REPORT ───────────────────────────────────────────────
    bar_width = 30
    max_count = top_10[0][1] if top_10 else 1
    max_word_len = max(len(w) for w, _ in top_10) if top_10 else 10
    
    report = []
    report.append("\n" + "═" * 60)
    report.append(f"{'📊 TEXT ANALYTICS DASHBOARD':^60}")
    report.append("═" * 60)
    
    report.append("\n📝 BASIC STATISTICS")
    report.append("-" * 40)
    report.append(f"  {'Characters (total)':<25} {char_count:>8,}")
    report.append(f"  {'Characters (no spaces)':<25} {char_no_spaces:>8,}")
    report.append(f"  {'Words':<25} {word_count:>8,}")
    report.append(f"  {'Unique words':<25} {len(unique_words):>8,}")
    report.append(f"  {'Sentences':<25} {sentence_count:>8,}")
    report.append(f"  {'Paragraphs':<25} {para_count:>8,}")
    report.append(f"  {'Est. reading time':<25} {'~' + str(round(reading_time_min, 1)) + ' min':>8}")
    
    report.append("\n📐 READABILITY")
    report.append("-" * 40)
    report.append(f"  {'Avg word length':<25} {avg_word_length:>7.1f} chars")
    report.append(f"  {'Avg sentence length':<25} {avg_sentence_length:>7.1f} words")
    report.append(f"  {'Vocabulary richness':<25} {vocabulary_richness:>7.1f}%")
    report.append(f"  {'Readability level':<25} {readability:>8}")
    
    report.append("\n🔤 TOP KEYWORDS")
    report.append("-" * 40)
    for word, count in top_10:
        bar_len = int(count / max_count * bar_width)
        bar = "█" * bar_len
        report.append(f"  {word:<{max_word_len+2}} {bar:<{bar_width}} {count}")
    
    report.append("\n" + "═" * 60)
    
    return "\n".join(report)


# ── TEST THE DASHBOARD ────────────────────────────────────────────────────
sample_text = """
Python is a high-level, general-purpose programming language. Python's design 
philosophy emphasizes code readability and simplicity. Python was created by 
Guido van Rossum and first released in 1991.

Python is dynamically typed and garbage-collected. It supports multiple 
programming paradigms, including structured, object-oriented and functional 
programming. Python is often described as a batteries included language 
due to its comprehensive standard library.

Python is consistently ranked as one of the most popular programming languages.
Its versatility makes it suitable for web development, data science, artificial
intelligence, automation, and scientific computing. Many developers choose Python
as their first programming language because Python is easy to learn.
"""

print(text_dashboard(sample_text))
```

---

## Chapter Summary

You've mastered:

✅ **String Basics**: Immutability, indexing, slicing with `[start:stop:step]`
✅ **String Methods**: Case, strip, find, split, join, replace, and validation
✅ **f-strings**: Modern, readable, powerful string formatting
✅ **Text Processing Tasks**: Cleaning, parsing, analyzing, and generating text
✅ **Common Pitfalls**: Concatenation in loops, case sensitivity, off-by-one errors
✅ **Real-World Applications**: CSV parsing, log analysis, templates, password validation

**Key Takeaways:**
- Strings are immutable sequences — every "change" creates a new string
- Use `join()` for building strings from lists — never `+` in a loop
- Always `strip()` user input before processing
- f-strings are the modern way to format strings — prefer them
- `split()` + `join()` are the power duo of text processing

**Next Chapter Preview:**
Chapter 6 dives into **Functions** — writing reusable, clean, and powerful code blocks. You'll learn about `*args`, `**kwargs`, lambda functions, closures, and how professional developers structure their code!
