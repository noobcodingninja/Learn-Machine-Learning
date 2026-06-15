# Chapter 14: Regular Expressions

## Part 1: Why Regular Expressions Exist

### The Problem: Text Is Messy and Patterns Are Everywhere

You're building a data pipeline that receives 50,000 user-submitted records daily. You need to:
- Validate that emails look like real emails
- Extract phone numbers from freeform text (in any format users typed them)
- Find all dates in log files regardless of how they were written
- Mask credit card numbers before storing logs
- Parse server logs where every line has the same structure

Without regex, you'd need dozens of lines of string manipulation for each task. With regex, each is one line.

```python
# ── WITHOUT REGEX — validating a phone number ────────────────────────────
def is_valid_phone_slow(phone):
    """Check if string looks like a phone number. Fragile and incomplete."""
    cleaned = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    if len(cleaned) == 10 and cleaned.isdigit():
        return True
    if len(cleaned) == 11 and cleaned[0] == "1" and cleaned[1:].isdigit():
        return True
    # But what about +91? +1-800? (555).123.4567? ...runs out of steam
    return False

# ── WITH REGEX — same task in one line ───────────────────────────────────
import re
def is_valid_phone_fast(phone):
    return bool(re.match(r"^(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$", phone))
```

### What Is a Regular Expression?

A regular expression (regex) is a **mini-language for describing text patterns**. Think of it as a highly sophisticated "find" that doesn't just search for exact strings but for patterns — "a word that starts with a capital letter followed by digits" or "anything that looks like a date."

Regex is not unique to Python — it works the same in JavaScript, Java, grep, sed, SQL, and dozens of other tools. Learning it once pays off everywhere.

---

## Part 2: The `re` Module — Python's Regex Engine

### Core Functions

```python
import re

text = "Contact Alice at alice@example.com or call 555-123-4567 for help."

# ── re.search() — find FIRST match anywhere in string ────────────────────
match = re.search(r"\d{3}-\d{4}", text)
if match:
    print(match.group())   # 123-4567  (not 555 — that's part of 555-123)
    print(match.start())   # 41  (position where match starts)
    print(match.end())     # 49  (position where match ends)
    print(match.span())    # (41, 49)

# ── re.match() — match ONLY at the START of string ───────────────────────
m = re.match(r"Contact", text)   # Matches — text starts with "Contact"
print(m.group())                  # Contact

m2 = re.match(r"Alice", text)    # Returns None — "Alice" not at start
print(m2)                         # None

# KEY DIFFERENCE:
# re.search() → finds anywhere in string
# re.match()  → only matches at the BEGINNING

# ── re.fullmatch() — pattern must match ENTIRE string ────────────────────
re.fullmatch(r"\d{5}", "12345")       # Match
re.fullmatch(r"\d{5}", "12345-6789")  # None — extra chars
re.fullmatch(r"\d{5}", "123")         # None — too short

# ── re.findall() — find ALL non-overlapping matches ──────────────────────
all_numbers = re.findall(r"\d+", text)
print(all_numbers)   # ['555', '123', '4567']

all_words = re.findall(r"\b[A-Z][a-z]+\b", text)
print(all_words)     # ['Contact', 'Alice']

# ── re.finditer() — iterator of match objects (memory efficient) ──────────
for match in re.finditer(r"\d+", text):
    print(f"Found '{match.group()}' at position {match.start()}-{match.end()}")

# ── re.sub() — find and REPLACE ───────────────────────────────────────────
# Replace phone number with [PHONE]
masked = re.sub(r"\d{3}-\d{3}-\d{4}", "[PHONE]", text)
print(masked)
# "Contact Alice at alice@example.com or call [PHONE] for help."

# Replace with a function (dynamic replacement)
def mask_digits(m):
    return "*" * len(m.group())   # Replace each digit with *

masked2 = re.sub(r"\d", "*", text)
print(masked2)   # "Contact Alice at alice@example.com or call ***-***-**** for help."

# Count replacements with count= parameter
masked3 = re.sub(r"\d+", "#", text, count=2)  # Only replace first 2 matches

# ── re.split() — split on a pattern ──────────────────────────────────────
data = "Alice,28,,Engineer;Bob 35, Marketing  ;  Carol,42"

# Split on commas, semicolons, or whitespace
parts = re.split(r"[,;\s]+", data)
print(parts)   # ['Alice', '28', 'Engineer', 'Bob', '35', 'Marketing', 'Carol', '42']

# Capture the delimiter too (put pattern in a group)
parts_with_sep = re.split(r"([,;])", "a,b;c,d")
print(parts_with_sep)   # ['a', ',', 'b', ';', 'c', ',', 'd']

# ── re.compile() — pre-compile for repeated use ──────────────────────────
# When using the same pattern many times, compile it ONCE
email_pattern = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b")

# Now use on many strings without recompiling
for text_item in ["alice@co.com", "not-an-email", "bob@example.org"]:
    if email_pattern.search(text_item):
        print(f"Email found: {text_item}")
```

---

## Part 3: Pattern Syntax — The Building Blocks

### Literal Characters

```python
# Most characters match themselves
re.search(r"hello", "say hello world")   # Matches "hello"
re.search(r"42", "the answer is 42")     # Matches "42"

# Special characters that need escaping:
# . ^ $ * + ? { } [ ] \ | ( )
# To match them literally, prefix with backslash

re.search(r"\.", "price: $9.99")         # Matches the literal dot
re.search(r"\$", "price: $9.99")         # Matches the literal dollar sign
re.search(r"\(555\)", "(555) 123-4567")  # Matches "(555)"
```

### The Dot — Wildcard for Any Character

```python
# . matches ANY character EXCEPT newline
re.findall(r"c.t", "cat, cut, cot, c4t, c t")
# ['cat', 'cut', 'cot', 'c4t', 'c t']

re.findall(r"c.t", "act, can't, fact")
# [] — no match ("can't" has "c" then "a" not followed by "t" immediately)

# re.DOTALL flag makes . match newlines too
re.search(r"hello.world", "hello\nworld", re.DOTALL)
```

### Character Classes `[...]`

```python
# [abc] — match 'a', 'b', or 'c'
re.findall(r"[aeiou]", "hello world")    # ['e', 'o', 'o']

# [a-z] — range: any lowercase letter
re.findall(r"[a-z]+", "Hello World 123")  # ['ello', 'orld']

# [A-Z] — any uppercase letter
# [0-9] — any digit (same as \d)
# [a-zA-Z] — any letter (upper or lower)
# [a-zA-Z0-9] — any alphanumeric (same as \w but no underscore)

# [^...] — NEGATE: match anything NOT listed
re.findall(r"[^aeiou ]+", "hello world")  # ['h', 'll', 'w', 'rld'] — consonants

# Combining ranges
re.findall(r"[a-zA-Z0-9_]+", "user_name123 and some-text!")
# ['user_name123', 'and', 'some', 'text']

# Inside [], most special chars are literal (no escaping needed)
re.findall(r"[.!?]", "Hello! How are you?")  # ['!', '?']
# The . inside [] is a literal dot, not wildcard
```

### Shorthand Character Classes

```python
# \d — digit: [0-9]
# \D — NOT digit: [^0-9]
# \w — word char: [a-zA-Z0-9_]
# \W — NOT word char
# \s — whitespace: [ \t\n\r\f\v]
# \S — NOT whitespace
# \b — word boundary (between \w and \W)
# \B — NOT word boundary

text = "Phone: (555) 123-4567, ID: abc_123"

print(re.findall(r"\d+", text))      # ['555', '123', '4567', '123']
print(re.findall(r"\w+", text))      # ['Phone', '555', '123', '4567', 'ID', 'abc_123']
print(re.findall(r"\s+", text))      # [' ', ' ', ' ', ' ', ' ']

# \b is zero-width — it marks position, doesn't consume characters
print(re.findall(r"\bcat\b", "cat cats concatenate"))
# ['cat']  — matches whole word only
print(re.findall(r"cat", "cat cats concatenate"))
# ['cat', 'cat', 'cat']  — matches anywhere

# Real example: find whole words
print(re.findall(r"\bPython\b", "Python is pythonic, not PYTHON"))
# ['Python']  — case-sensitive, whole word
```

### Quantifiers — How Many?

```python
# * — 0 or more
# + — 1 or more
# ? — 0 or 1 (makes the preceding element optional)
# {n} — exactly n times
# {n,} — n or more times
# {n,m} — between n and m times (inclusive)

text = "colour color colouur"

re.findall(r"colou?r", text)    # ['colour', 'color']  — u is optional
re.findall(r"colou*r", text)    # ['colour', 'color', 'colouur']  — 0+ u's
re.findall(r"colou+r", text)    # ['colour', 'colouur']  — 1+ u's

# Exact count
re.findall(r"\d{4}", "2025-01-15 and 12 and 12345")
# ['2025', '1234']  — exactly 4 digits

# Range count
re.findall(r"\d{2,4}", "1 12 123 1234 12345")
# ['12', '123', '1234', '1234']  — 2 to 4 digits (greedy: takes as many as possible)

# ── GREEDY vs LAZY ────────────────────────────────────────────────────────
# By default quantifiers are GREEDY — match as much as possible
html = "<b>bold</b> and <i>italic</i>"

greedy = re.findall(r"<.+>", html)
print(greedy)   # ['<b>bold</b> and <i>italic</i>']  — matches TOO MUCH!

# Add ? after quantifier to make LAZY — match as little as possible
lazy = re.findall(r"<.+?>", html)
print(lazy)     # ['<b>', '</b>', '<i>', '</i>']  — each tag separately

# The question marks:
# ?  after a character = optional (0 or 1 of that character)
# *? after *           = lazy 0-or-more
# +? after +           = lazy 1-or-more
# {n,m}? after {}      = lazy range
```

### Anchors — Position in String

```python
# ^ — start of string (or start of line with re.MULTILINE)
# $ — end of string (or end of line with re.MULTILINE)
# \A — start of string (ignores re.MULTILINE)
# \Z — end of string (ignores re.MULTILINE)

# Match strings that START with a digit
re.findall(r"^\d+", "123 hello")   # ['123']
re.findall(r"^\d+", "hello 123")   # []  — doesn't start with digit

# Match strings that END with punctuation
re.findall(r"[.!?]$", "Hello!")    # ['!']

# ── MULTILINE mode — ^ and $ match each LINE ─────────────────────────────
text = """
ERROR: disk full
INFO: connection ok
ERROR: timeout
WARNING: high memory
"""

# Without MULTILINE: ^ matches only the very start of the entire string
re.findall(r"^ERROR", text)               # []  — first line is blank

# With MULTILINE: ^ matches start of each line
errors = re.findall(r"^ERROR.+", text, re.MULTILINE)
print(errors)
# ['ERROR: disk full', 'ERROR: timeout']
```

---

## Part 4: Groups — Capturing and Using Parts of Matches

### Capturing Groups `(...)`

```python
# Without groups: entire match returned
re.findall(r"\d{4}-\d{2}-\d{2}", "Born: 1995-06-15")
# ['1995-06-15']  — full match

# WITH groups: only group contents returned
re.findall(r"(\d{4})-(\d{2})-(\d{2})", "Born: 1995-06-15")
# [('1995', '06', '15')]  — tuple of groups

# When only ONE group:
re.findall(r"(\d{4})", "Born: 1995-06-15, Graduated: 2017-05-20")
# ['1995', '2017']  — just the group, not full match

# match.group(n) to access groups
m = re.search(r"(\d{4})-(\d{2})-(\d{2})", "Born: 1995-06-15")
print(m.group(0))   # 1995-06-15  — entire match
print(m.group(1))   # 1995        — first group
print(m.group(2))   # 06          — second group
print(m.group(3))   # 15          — third group
print(m.groups())   # ('1995', '06', '15')  — all groups as tuple

# Named groups — (?P<name>...)
m = re.search(
    r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})",
    "Born: 1995-06-15"
)
print(m.group("year"))   # 1995
print(m.group("month"))  # 06
print(m.group("day"))    # 15
print(m.groupdict())     # {'year': '1995', 'month': '06', 'day': '15'}

# ── BACK REFERENCES — reference a group WITHIN the pattern ───────────────
# \1 refers to what group 1 matched
# Useful for finding repeated words or matching opening/closing tags

# Find repeated words
text = "the the quick brown fox fox"
duplicates = re.findall(r"\b(\w+)\s+\1\b", text)
print(duplicates)   # ['the', 'fox']

# Match HTML tags
html = "<title>Hello</title>"
m = re.search(r"<(\w+)>(.+?)</\1>", html)
print(m.group(1))   # title
print(m.group(2))   # Hello
```

### Non-Capturing Groups and Lookarounds

```python
# ── NON-CAPTURING GROUPS (?:...) ─────────────────────────────────────────
# Group without capturing — use for alternation or repetition without capture

# ❌ Without (?:...) — captures group content
re.findall(r"(cat|dog)s?", "cats dogs parrot")
# ['cat', 'dog']  — returns the GROUP content, not full match!

# ✓ With (?:...) — groups for alternation but doesn't capture
re.findall(r"(?:cat|dog)s?", "cats dogs parrot")
# ['cats', 'dogs']  — returns full match

# ── LOOKAHEAD (?=...) and (?!...) ────────────────────────────────────────
# Lookahead: "followed by" — zero-width (doesn't consume characters)

# Positive lookahead (?=...)  — must be followed by...
prices = re.findall(r"\d+(?=\$)", "100$ 200 300$ 400EUR")
print(prices)   # ['100', '300']  — digits followed by $

# Negative lookahead (?!...)  — must NOT be followed by...
non_usd = re.findall(r"\d+(?!\$)(?!\d)", "100$ 200 300$ 400EUR")
print(non_usd)  # ['200']

# ── LOOKBEHIND (?<=...) and (?<!...) ─────────────────────────────────────
# Lookbehind: "preceded by"

# Positive lookbehind (?<=...)  — must be preceded by...
amounts = re.findall(r"(?<=\$)\d+", "Price: $100, Sale: $75, EUR: 50")
print(amounts)   # ['100', '75']  — digits preceded by $

# Negative lookbehind (?<!...)  — must NOT be preceded by...
no_dollar = re.findall(r"(?<!\$)\b\d+\b", "Price: $100, Count: 75, Code: $50")
print(no_dollar)  # ['75']  — digits NOT preceded by $

# ── REAL-WORLD: extract prices without the symbol ────────────────────────
text = "Items: $29.99, ₹1,299, €45.00, £32.50, HKD 250"
dollar_prices = re.findall(r"(?<=\$)[\d,.]+", text)
print(dollar_prices)  # ['29.99']

all_prices = re.findall(r"(?:[$₹€£])(\d[\d,.]*)", text)
print(all_prices)  # ['29.99', '1,299', '45.00', '32.50']
```

---

## Part 5: Flags — Modifying Behavior

```python
import re

# ── re.IGNORECASE (re.I) ──────────────────────────────────────────────────
text = "Python PYTHON python PyThOn"

print(re.findall(r"python", text))          # ['python']
print(re.findall(r"python", text, re.I))    # ['Python', 'PYTHON', 'python', 'PyThOn']

# ── re.MULTILINE (re.M) ───────────────────────────────────────────────────
log = """2025-01-15 ERROR Connection failed
2025-01-15 INFO Server started
2025-01-15 ERROR Disk full"""

# ^ matches start of each line with MULTILINE
errors = re.findall(r"^\d{4}-\d{2}-\d{2} ERROR .+", log, re.M)
print(errors)
# ['2025-01-15 ERROR Connection failed', '2025-01-15 ERROR Disk full']

# ── re.DOTALL (re.S) ─────────────────────────────────────────────────────
# Makes . match newlines too
html = "<p>This is\na paragraph</p>"

# Without DOTALL: . doesn't match \n
print(re.search(r"<p>.+</p>", html))           # None

# With DOTALL: . matches everything including \n
print(re.search(r"<p>.+</p>", html, re.S).group())
# <p>This is\na paragraph</p>

# ── re.VERBOSE (re.X) — write readable regex with comments ───────────────
# re.VERBOSE ignores whitespace and allows # comments in pattern

# Without VERBOSE — hard to read
phone_pattern = re.compile(r"^(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$")

# With VERBOSE — documented and readable!
phone_pattern = re.compile(r"""
    ^                       # Start of string
    (\+?\d{1,3}[-.\s]?)?   # Optional country code: +1, +91, etc.
    \(?                     # Optional opening paren
    \d{3}                   # Area code (3 digits)
    \)?                     # Optional closing paren
    [-.\s]?                 # Optional separator
    \d{3}                   # First 3 digits
    [-.\s]?                 # Optional separator
    \d{4}                   # Last 4 digits
    $                       # End of string
""", re.VERBOSE)

# Test
phones = ["+1-555-123-4567", "(555) 123-4567", "5551234567",
          "+91 99887 76655", "555-CALL-NOW"]
for p in phones:
    match = phone_pattern.match(p)
    print(f"{'✓' if match else '✗'} {p}")

# ── COMBINING FLAGS ───────────────────────────────────────────────────────
# Use | to combine multiple flags
result = re.findall(r"^python.+", text, re.I | re.M)
```

---

## Part 6: Essential Regex Patterns

### The Patterns You'll Use Every Day

```python
import re

# ── EMAIL ADDRESSES ───────────────────────────────────────────────────────
EMAIL = re.compile(r"""
    \b                      # Word boundary
    [\w.+-]+                # Local part (letters, digits, dots, plus, hyphen)
    @                       # @ symbol
    [\w-]+                  # Domain name
    (?:\.[\w-]+)*           # Optional subdomains (e.g., .co.uk)
    \.[a-zA-Z]{2,}          # TLD (2+ letters)
    \b                      # Word boundary
""", re.VERBOSE)

test_emails = [
    "alice@example.com",
    "bob.smith+tag@company.co.uk",
    "user@subdomain.example.org",
    "not-an-email",
    "@missing-local.com",
    "missing-at-sign.com"
]

for email in test_emails:
    m = EMAIL.search(email)
    print(f"{'✓' if m else '✗'} {email}")


# ── PHONE NUMBERS ──────────────────────────────────────────────────────────
PHONE = re.compile(r"""
    (?:\+?1[-.\s]?)?        # Optional US country code (+1)
    \(?                     # Optional (
    \d{3}                   # Area code
    \)?                     # Optional )
    [-.\s]?                 # Optional separator
    \d{3}                   # Exchange
    [-.\s]?                 # Optional separator
    \d{4}                   # Number
""", re.VERBOSE)

INDIAN_PHONE = re.compile(r"(?:\+91[-.\s]?)?[6-9]\d{9}")

test_phones = ["(555) 123-4567", "+1-800-555-1234", "555.123.4567",
               "+91 98765 43210", "9876543210"]
for p in test_phones:
    print(f"{'✓' if PHONE.search(p) or INDIAN_PHONE.search(p) else '✗'} {p}")


# ── DATES ────────────────────────────────────────────────────────────────
DATE_ISO    = re.compile(r"\b\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b")
DATE_US     = re.compile(r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:\d{2}|\d{4})\b")
DATE_LONG   = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}\b", re.I
)

text_with_dates = "Meeting on 2025-01-15, follow-up 01/20/2025, and January 25, 2025."
print(DATE_ISO.findall(text_with_dates))   # ['2025-01-15']
print(DATE_US.findall(text_with_dates))    # ['01/20/2025']
print(DATE_LONG.findall(text_with_dates))  # ['January 25, 2025']


# ── URLs ────────────────────────────────────────────────────────────────
URL = re.compile(r"""
    https?://               # http or https
    (?:[\w-]+\.)+           # Domain parts (e.g., www.sub.)
    [a-zA-Z]{2,}            # TLD
    (?:/[^\s]*)?            # Optional path
""", re.VERBOSE)

urls = URL.findall("Visit https://www.example.com/path?q=1 or http://sub.site.org")
print(urls)   # ['https://www.example.com/path?q=1', 'http://sub.site.org']


# ── IP ADDRESSES ─────────────────────────────────────────────────────────
IP = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")

ips = IP.findall("Server 192.168.1.1 and 10.0.0.255 connected. Not 999.0.0.0.")
print(ips)   # ['192.168.1.1', '10.0.0.255']


# ── CREDIT CARDS (for masking) ────────────────────────────────────────────
CREDIT_CARD = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

def mask_card(text):
    return CREDIT_CARD.sub(lambda m: "*" * 12 + m.group()[-4:], text)

log_text = "Payment with card 4111-1111-1111-1234 approved."
print(mask_card(log_text))   # Payment with card ************1234 approved.


# ── PASSWORDS ─────────────────────────────────────────────────────────────
PASSWORD = re.compile(r"""
    (?=.*[a-z])             # At least one lowercase
    (?=.*[A-Z])             # At least one uppercase
    (?=.*\d)                # At least one digit
    (?=.*[!@#$%^&*])        # At least one special char
    .{8,}                   # At least 8 characters total
""", re.VERBOSE)

passwords = ["weak", "Better1", "Str0ng!Pass", "NoSpecial1Char", "G00d@Pass!"]
for pwd in passwords:
    print(f"{'✓' if PASSWORD.fullmatch(pwd) else '✗'} {pwd}")


# ── HEXADECIMAL COLORS ────────────────────────────────────────────────────
HEX_COLOR = re.compile(r"#(?:[0-9a-fA-F]{3}){1,2}\b")

css = "color: #FF5733; background: #abc; border: 1px solid #FFFFFF;"
print(HEX_COLOR.findall(css))   # ['#FF5733', '#abc', '#FFFFFF']


# ── MARKDOWN LINKS ────────────────────────────────────────────────────────
MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

md = "See [Python docs](https://python.org) and [our guide](https://guide.com)."
for m in MD_LINK.finditer(md):
    print(f"Text: '{m.group(1)}', URL: '{m.group(2)}'")


# ── NUMBERS (integers and floats) ────────────────────────────────────────
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
print(NUMBER.findall("Values: -42, 3.14, 1.5e10, -2.7E-3, 100"))
# ['-42', '3.14', '1.5e10', '-2.7E-3', '100']
```

---

## Part 7: Practical Text Processing

### Worked Example 1: Log File Parser

```python
import re
from datetime import datetime
from collections import defaultdict, Counter

LOG_PATTERN = re.compile(r"""
    (?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})   # Timestamp
    \s+
    (?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)             # Log level
    \s+
    (?P<logger>[\w.]+)                                        # Logger name
    :\s+
    (?P<message>.+)                                           # Message
""", re.VERBOSE)

# IP and request pattern for web server logs
ACCESS_PATTERN = re.compile(r"""
    (?P<ip>\d+\.\d+\.\d+\.\d+)       # Client IP
    \s+-\s+-\s+
    \[(?P<time>[^\]]+)\]              # Request time
    \s+"
    (?P<method>GET|POST|PUT|DELETE|PATCH|HEAD)\s+
    (?P<path>/[^\s"]*)\s+
    HTTP/\d\.\d"
    \s+
    (?P<status>\d{3})                 # HTTP status code
    \s+
    (?P<size>\d+|-)                   # Response size
""", re.VERBOSE)

def parse_application_log(log_text):
    """Parse and analyze application log."""
    entries = []
    errors = []
    level_counts = Counter()
    
    for line_num, line in enumerate(log_text.strip().splitlines(), 1):
        m = LOG_PATTERN.match(line.strip())
        if m:
            entry = m.groupdict()
            entry["timestamp"] = datetime.strptime(
                entry["timestamp"], "%Y-%m-%d %H:%M:%S"
            )
            entry["line"] = line_num
            entries.append(entry)
            level_counts[entry["level"]] += 1
            
            if entry["level"] in ("ERROR", "CRITICAL"):
                errors.append(entry)
    
    return entries, errors, level_counts


def parse_access_log(log_text):
    """Parse and analyze web server access log."""
    requests = []
    for line in log_text.strip().splitlines():
        m = ACCESS_PATTERN.match(line.strip())
        if m:
            req = m.groupdict()
            req["status"] = int(req["status"])
            req["size"]   = int(req["size"]) if req["size"] != "-" else 0
            requests.append(req)
    
    # Analysis
    by_ip     = Counter(r["ip"] for r in requests)
    by_status = Counter(r["status"] for r in requests)
    by_path   = Counter(r["path"] for r in requests)
    errors_4xx = [r for r in requests if 400 <= r["status"] < 500]
    errors_5xx = [r for r in requests if r["status"] >= 500]
    
    return {
        "total_requests": len(requests),
        "top_ips":        by_ip.most_common(5),
        "status_codes":   dict(by_status),
        "top_paths":      by_path.most_common(5),
        "client_errors":  len(errors_4xx),
        "server_errors":  len(errors_5xx),
        "total_bytes":    sum(r["size"] for r in requests)
    }


# Test with sample log
sample_log = """
2025-01-15 10:23:01 INFO myapp.server: Server started on port 8000
2025-01-15 10:23:15 INFO myapp.auth: User alice authenticated
2025-01-15 10:24:02 WARNING myapp.cache: Cache miss rate above 20%
2025-01-15 10:24:45 ERROR myapp.database: Connection timeout after 30s
2025-01-15 10:24:46 INFO myapp.database: Reconnecting to database
2025-01-15 10:25:30 ERROR myapp.api: Rate limit exceeded for user bob
2025-01-15 10:26:00 CRITICAL myapp.disk: Disk space below 5% threshold
"""

entries, errors, counts = parse_application_log(sample_log)
print(f"Total entries: {len(entries)}")
print(f"Level counts: {dict(counts)}")
print(f"Errors ({len(errors)}):")
for err in errors:
    print(f"  [{err['timestamp'].strftime('%H:%M:%S')}] "
          f"{err['level']}: {err['message']}")
```

### Worked Example 2: Data Extraction and Cleaning

```python
import re

# ── EXTRACT STRUCTURED DATA FROM UNSTRUCTURED TEXT ────────────────────────

EXTRACTORS = {
    "emails":   re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
    "phones":   re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    "dates":    re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    "prices":   re.compile(r"[$₹€£][\d,]+(?:\.\d{1,2})?"),
    "urls":     re.compile(r"https?://[\w.-]+(?:/[^\s]*)?"),
    "hashtags": re.compile(r"#\w+"),
    "mentions": re.compile(r"@\w+"),
}

def extract_all(text):
    """Extract all structured data from unstructured text."""
    return {
        key: pattern.findall(text)
        for key, pattern in EXTRACTORS.items()
        if pattern.findall(text)
    }

messy_text = """
Hi @alice, follow up with @bob about the $1,299.99 invoice.
Contact alice@company.com or call (555) 123-4567 by 2025-01-20.
See https://company.com/invoice/INV-2025-001 for details.
Use #payment #urgent tags. Bob's email: bob.smith@example.co.uk
"""

extracted = extract_all(messy_text)
for key, values in extracted.items():
    print(f"{key}: {values}")


# ── ADDRESS PARSER ────────────────────────────────────────────────────────
ADDRESS_PATTERN = re.compile(r"""
    (?P<number>\d+[A-Za-z]?)        # Street number (optional letter suffix)
    \s+
    (?P<street>.+?)                 # Street name
    (?:,\s*                         # Optional comma before city
    (?P<city>[A-Za-z\s]+?)          # City
    (?:,\s*
    (?P<state>[A-Z]{2})             # 2-letter state code
    (?:\s+
    (?P<zip>\d{5}(?:-\d{4})?))?     # Zip code (optional +4)
    )?)?$
""", re.VERBOSE)

addresses = [
    "123 Main Street, Springfield, IL 62701",
    "456B Oak Avenue, Austin, TX",
    "789 Elm Drive",
]

for addr in addresses:
    m = ADDRESS_PATTERN.match(addr)
    if m:
        parts = {k: v for k, v in m.groupdict().items() if v}
        print(f"Parsed: {parts}")


# ── MARKDOWN TO PLAIN TEXT ────────────────────────────────────────────────
def strip_markdown(text):
    """Remove Markdown formatting, return plain text."""
    # Headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)
    # Bold/italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Links: [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Images: ![alt](url) → alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Blockquotes
    text = re.sub(r"^>\s+", "", text, flags=re.M)
    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.M)
    # List markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)
    # Clean up extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

md = """
# Hello World

This is **bold** and _italic_ text.
Visit [Python.org](https://python.org) for more.

- Item one
- Item two

```python
print("hello")
```

> A blockquote
"""

print(strip_markdown(md))
```

### Worked Example 3: Text Anonymizer

```python
import re

class TextAnonymizer:
    """
    Anonymize sensitive data in text by replacing with placeholders.
    Keeps a mapping so you can de-anonymize if needed.
    """
    
    PATTERNS = {
        "EMAIL":   re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
        "PHONE":   re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
        "IP":      re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "CC":      re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "SSN":     re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "NAME":    re.compile(r"\bDear\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"),
    }
    
    def __init__(self):
        self.registry = {}   # placeholder → original value
        self.counters = {}   # type → count
    
    def _make_placeholder(self, type_name, value):
        """Create or reuse a placeholder for a value."""
        # Same value → same placeholder
        for ph, val in self.registry.items():
            if val == value:
                return ph
        
        self.counters[type_name] = self.counters.get(type_name, 0) + 1
        placeholder = f"[{type_name}_{self.counters[type_name]}]"
        self.registry[placeholder] = value
        return placeholder
    
    def anonymize(self, text):
        """Replace all sensitive data with placeholders."""
        result = text
        
        # Process in order of specificity (most specific first)
        for type_name, pattern in self.PATTERNS.items():
            def replace_match(m, tn=type_name):
                original = m.group()
                return self._make_placeholder(tn, original)
            result = pattern.sub(replace_match, result)
        
        return result
    
    def deanonymize(self, text):
        """Restore original values from placeholders."""
        result = text
        for placeholder, original in self.registry.items():
            result = result.replace(placeholder, original)
        return result
    
    def get_summary(self):
        """Show what was anonymized."""
        by_type = {}
        for ph, val in self.registry.items():
            type_name = ph[1:ph.rindex("_")]
            by_type.setdefault(type_name, []).append((ph, val))
        return by_type


# Demo
anon = TextAnonymizer()

sensitive = """
Dear Alice Johnson,

Your account 4111-1111-1111-1234 was charged $299.00.
Contact us at support@company.com or call (555) 123-4567.
Your request came from IP 192.168.1.100.
SSN on file: 123-45-6789.
"""

anonymized = anon.anonymize(sensitive)
print("=== ANONYMIZED ===")
print(anonymized)

print("\n=== PLACEHOLDERS ===")
for type_name, items in anon.get_summary().items():
    for ph, val in items:
        print(f"  {ph} → {val}")

restored = anon.deanonymize(anonymized)
print("\n=== RESTORED ===")
print(restored)
```

### Worked Example 4: Config File Parser

```python
import re

def parse_config(config_text):
    """
    Parse a configuration file supporting:
    - Sections: [section_name]
    - Key-value pairs: key = value or key: value
    - Comments: # or ;
    - Continuation lines: value \
        continued here
    - List values: key = val1, val2, val3
    """
    SECTION  = re.compile(r"^\[(?P<name>[\w.-]+)\]\s*$")
    KEYVAL   = re.compile(r"^(?P<key>[\w.-]+)\s*[:=]\s*(?P<value>.*)$")
    COMMENT  = re.compile(r"^\s*[#;]")
    BLANK    = re.compile(r"^\s*$")
    CONTINUE = re.compile(r"^(\s+)(.+)$")  # Indented continuation
    
    config = {}
    current_section = None
    current_key = None
    current_value = None
    
    lines = config_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip comments and blank lines
        if COMMENT.match(line) or BLANK.match(line):
            # Save pending key-value
            if current_key and current_section:
                config[current_section][current_key] = current_value
                current_key = current_value = None
            i += 1
            continue
        
        # Section header
        m = SECTION.match(line)
        if m:
            if current_key and current_section:
                config[current_section][current_key] = current_value
                current_key = current_value = None
            current_section = m.group("name")
            config[current_section] = {}
            i += 1
            continue
        
        # Continuation line (starts with whitespace)
        m = CONTINUE.match(line)
        if m and current_key:
            current_value = current_value + " " + m.group(2).strip()
            i += 1
            continue
        
        # Key-value pair
        m = KEYVAL.match(line)
        if m and current_section is not None:
            # Save previous key-value if pending
            if current_key:
                config[current_section][current_key] = current_value
            
            current_key = m.group("key").strip()
            value = m.group("value").strip()
            
            # Strip inline comments
            value = re.sub(r"\s+[#;].*$", "", value).strip()
            
            # Detect list values (comma-separated)
            if "," in value:
                value = [v.strip() for v in re.split(r",\s*", value)]
            
            current_value = value
        
        i += 1
    
    # Save last pending key-value
    if current_key and current_section:
        config[current_section][current_key] = current_value
    
    return config


config_text = """
# Application Configuration File
; Semicolons also work as comments

[database]
host = localhost        # Main database
port = 5432
name = myapp_db
allowed_hosts = localhost, 127.0.0.1,
    ::1, 10.0.0.0/8

[api]
base_url = https://api.example.com
timeout = 30  ; seconds
retry_count = 3

[features]
dark_mode = true
max_upload_mb = 50
supported_formats = jpg, png, gif, webp
"""

config = parse_config(config_text)
import json
print(json.dumps(config, indent=2))
```

### Worked Example 5: Template Engine

```python
import re

class TemplateEngine:
    """
    Simple template engine supporting:
    - Variable substitution: {{ variable }}
    - Conditionals: {% if condition %}...{% endif %}
    - Loops: {% for item in list %}...{% endfor %}
    - Filters: {{ variable | upper }}
    """
    
    VAR_PATTERN  = re.compile(r"\{\{\s*(\w+)(?:\s*\|\s*(\w+))?\s*\}\}")
    IF_PATTERN   = re.compile(r"\{%\s*if\s+(\w+)\s*%\}(.*?)\{%\s*endif\s*%\}", re.S)
    FOR_PATTERN  = re.compile(r"\{%\s*for\s+(\w+)\s+in\s+(\w+)\s*%\}(.*?)\{%\s*endfor\s*%\}", re.S)
    
    FILTERS = {
        "upper":    str.upper,
        "lower":    str.lower,
        "title":    str.title,
        "strip":    str.strip,
        "len":      lambda x: str(len(x)),
        "reverse":  lambda x: x[::-1],
    }
    
    def render(self, template, context):
        """Render template with given context dictionary."""
        result = template
        
        # Process conditionals first
        def render_if(m):
            var_name, body = m.group(1), m.group(2)
            return body if context.get(var_name) else ""
        result = self.IF_PATTERN.sub(render_if, result)
        
        # Process loops
        def render_for(m):
            item_name, list_name, body = m.group(1), m.group(2), m.group(3)
            items = context.get(list_name, [])
            parts = []
            for item in items:
                # Create inner context with loop variable
                inner_ctx = {**context, item_name: item}
                parts.append(self.render(body, inner_ctx))
            return "".join(parts)
        result = self.FOR_PATTERN.sub(render_for, result)
        
        # Process variables
        def render_var(m):
            var_name, filter_name = m.group(1), m.group(2)
            value = str(context.get(var_name, ""))
            if filter_name and filter_name in self.FILTERS:
                value = self.FILTERS[filter_name](value)
            return value
        result = self.VAR_PATTERN.sub(render_var, result)
        
        return result


# Demo
engine = TemplateEngine()

email_template = """
Dear {{ name | title }},

{% if is_premium %}
Thank you for being a Premium member!
{% endif %}
Your recent orders:
{% for order in orders %}
  - Order {{ order }}: shipped
{% endfor %}

Total items: {{ count }} orders.

Best regards,
The Team
"""

context = {
    "name": "alice johnson",
    "is_premium": True,
    "orders": ["ORD-001", "ORD-002", "ORD-003"],
    "count": "3"
}

print(engine.render(email_template, context))
```

---

## Part 8: Common Mistakes and How to Avoid Them

```python
import re

# ── MISTAKE 1: Not using raw strings ──────────────────────────────────────
# ❌ WRONG — \b is backspace character in regular strings
pattern_bad = "\bword\b"   # \b is chr(8) — backspace! Not a word boundary.

# ✓ CORRECT — \b is word boundary in raw strings
pattern_good = r"\bword\b"

# Always use r"..." for regex patterns!

# ── MISTAKE 2: Forgetting greedy vs lazy ─────────────────────────────────
html = "<b>first</b> and <b>second</b>"

# ❌ WRONG — greedy, captures everything between first < and last >
wrong = re.findall(r"<b>.+</b>", html)
print(wrong)   # ['<b>first</b> and <b>second</b>']  — one big match!

# ✓ CORRECT — lazy, captures each tag separately
right = re.findall(r"<b>.+?</b>", html)
print(right)   # ['<b>first</b>', '<b>second</b>']

# ── MISTAKE 3: Using .* when you want "anything but X" ────────────────────
text = 'Name: "Alice", Age: "28"'

# ❌ WRONG — .* is greedy, matches from first " to last "
wrong = re.findall(r'".*"', text)
print(wrong)   # ['"Alice", Age: "28"']  — too much!

# ✓ CORRECT — [^"]* matches any character except "
right = re.findall(r'"[^"]*"', text)
print(right)   # ['"Alice"', '"28"']

# ── MISTAKE 4: Not anchoring validation patterns ──────────────────────────
# ❌ WRONG — search finds pattern ANYWHERE in string
def validate_email_wrong(email):
    return bool(re.search(r"[\w.]+@[\w.]+\.\w+", email))

print(validate_email_wrong("not!an!email!but@has.com!in!it"))  # True  ← Bug!

# ✓ CORRECT — fullmatch or ^ and $ anchors for full validation
def validate_email_right(email):
    return bool(re.fullmatch(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", email))

print(validate_email_right("not!an!email!but@has.com!in!it"))  # False ← Correct

# ── MISTAKE 5: Recompiling in loops ───────────────────────────────────────
data = ["alice@co.com", "bob@co.com"] * 10000

# ❌ WRONG — compiles pattern 20,000 times!
for email in data:
    if re.match(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", email):
        pass

# ✓ CORRECT — compile once, use many times
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
for email in data:
    if EMAIL_RE.match(email):
        pass

# ── MISTAKE 6: Catastrophic Backtracking ──────────────────────────────────
# Patterns like (a+)+ can cause exponential time on bad input
# "aaaaaaaaaaaaaaab" → might hang for minutes!

# ❌ DANGEROUS — nested quantifiers on overlapping patterns
bad_pattern = re.compile(r"(a+)+b")

# ✓ SAFER — use atomic groups or possessive quantifiers (not all engines support)
# Or restructure the pattern to avoid ambiguity

# ── MISTAKE 7: Matching across lines without DOTALL ──────────────────────
text = """First line
Second line"""

# ❌ WRONG — . doesn't match \n without DOTALL
m = re.search(r"First.+Second", text)
print(m)   # None

# ✓ CORRECT — use re.DOTALL or [\s\S]
m = re.search(r"First.+Second", text, re.DOTALL)
print(m.group())   # "First line\nSecond"

# Or: re.search(r"First[\s\S]+Second", text)  — works without flag
```

---

## Practice Problems

### Easy (1–5)

**Problem 1: Basic Pattern Matching**
Write regex patterns and test them.

```python
import re

# Write patterns to match:
# a) Any word that starts with a capital letter
# b) Any sequence of digits (one or more)
# c) Any word ending in "ing"
# d) A string that contains ONLY letters (no spaces, digits, punctuation)
# e) A date in DD/MM/YYYY format

text = "Running quickly through 42 streets on 15/01/2025 while Thinking deeply"

# a)
capital_words = re.findall(r"???", text)     # ['Running', 'Thinking']
# b)
numbers = re.findall(r"???", text)           # ['42', '15', '01', '2025']
# c)
ing_words = re.findall(r"???", text)         # ['Running', 'Thinking']
# d)
letters_only = re.findall(r"???", text)      # Words with only letters
# e)
dates = re.findall(r"???", text)             # ['15/01/2025']
```

**Problem 2: Email Validator**
Build a robust email validator.

```python
def is_valid_email(email):
    """
    Valid email requirements:
    - Has exactly one @
    - Local part: letters, digits, dots, plus, hyphens, underscores
    - Domain: letters, digits, hyphens (no consecutive dots)
    - TLD: 2-6 letters
    - No leading/trailing dots in local part
    """
    pass

tests = [
    ("alice@example.com", True),
    ("bob.smith+tag@company.co.uk", True),
    ("user@sub.domain.org", True),
    ("@example.com", False),           # No local part
    ("user@.example.com", False),      # Dot after @
    ("user@example.", False),          # No TLD
    ("user@@example.com", False),      # Double @
    (".user@example.com", False),      # Leading dot
    ("user@exam_ple.com", False),      # Underscore in domain
]
for email, expected in tests:
    result = is_valid_email(email)
    status = "✓" if result == expected else "✗"
    print(f"{status} '{email}': {result} (expected {expected})")
```

**Problem 3: Phone Number Normalizer**
Extract and normalize phone numbers to (XXX) XXX-XXXX format.

```python
def normalize_phones(text):
    """
    Find all phone numbers in text (in any format).
    Return list of normalized (XXX) XXX-XXXX strings.
    """
    pass

text = """
Call us at 555-123-4567 or (800) 555.9876.
International: +1 415 555 2671. Quick: 8005559999.
Not a phone: 123 or 12345678901234.
"""
print(normalize_phones(text))
# ['(555) 123-4567', '(800) 555-9876', '(415) 555-2671', '(800) 555-9999']
```

**Problem 4: HTML Tag Stripper**
Remove all HTML tags from text.

```python
def strip_html(html_text):
    """Remove all HTML tags, return plain text. Handle nested tags."""
    pass

html = """
<html><body>
<h1>Title</h1>
<p>This is <strong>bold</strong> and <em>italic</em> text.</p>
<a href="https://example.com">Click here</a>
<!-- This is a comment -->
<br/>
</body></html>
"""
print(strip_html(html))
# Title
# This is bold and italic text.
# Click here
```

**Problem 5: Password Strength Checker**
Validate passwords using multiple regex checks.

```python
def check_password(password):
    """
    Return dict with:
    - is_valid: bool
    - strength: "Weak" / "Medium" / "Strong" / "Very Strong"
    - failures: list of unmet requirements
    - score: 0-5
    """
    requirements = {
        "min_8_chars":     (r".{8,}", "At least 8 characters"),
        "has_uppercase":   (r"[A-Z]", "At least one uppercase letter"),
        "has_lowercase":   (r"[a-z]", "At least one lowercase letter"),
        "has_digit":       (r"\d", "At least one digit"),
        "has_special":     (r"[!@#$%^&*(),.?\":{}|<>]", "At least one special character"),
    }
    pass
```

### Medium (6–12)

**Problem 6: CSV Row Parser**
Parse a CSV line that may contain quoted fields with commas inside.

```python
def parse_csv_line(line):
    """
    Parse a single CSV line, handling:
    - Quoted fields: "hello, world" → one field
    - Escaped quotes inside fields: "say ""hello""" → say "hello"
    - Mixed quoted and unquoted fields
    """
    pass

tests = [
    ('Alice,28,"New York, NY",Engineer', ["Alice", "28", "New York, NY", "Engineer"]),
    ('"John ""JD"" Doe",42,Austin', ["John \"JD\" Doe", "42", "Austin"]),
    ('simple,plain,row', ["simple", "plain", "row"]),
]
for line, expected in tests:
    result = parse_csv_line(line)
    print(f"{'✓' if result == expected else '✗'} {result}")
```

**Problem 7: Git Log Parser**
Parse the output of `git log --oneline`.

```python
def parse_git_log(log_output):
    """
    Parse git log output like:
    'a1b2c3d Fix: handle null pointer in auth module'
    'e4f5g6h Feature: add dark mode support (#1234)'
    
    Return list of dicts with: hash, type, description, issue (if any)
    """
    pass

git_log = """
a1b2c3d Fix: handle null pointer in auth module
e4f5g6h Feature: add dark mode support (#1234)
i7j8k9l Refactor: extract payment logic to service
m1n2o3p Docs: update API documentation for v2
q4r5s6t Fix: correct date parsing bug (#5678)
"""
entries = parse_git_log(git_log)
for e in entries:
    print(e)
# {'hash': 'a1b2c3d', 'type': 'Fix', 'description': 'handle null pointer in auth module', 'issue': None}
# {'hash': 'e4f5g6h', 'type': 'Feature', 'description': 'add dark mode support', 'issue': '1234'}
```

**Problem 8: Markdown Link Extractor**
Extract and validate all links from Markdown text.

```python
def extract_markdown_links(md_text):
    """
    Extract all link types:
    - Inline: [text](url)
    - Reference: [text][label] and [label]: url
    - Auto: <https://url>
    - Images: ![alt](url)
    
    Return: {'inline': [...], 'reference': [...], 'auto': [...], 'images': [...]}
    """
    pass

md = """
See [Google](https://google.com) and [GitHub][gh].
Also check <https://python.org> directly.
![Logo](https://example.com/logo.png)

[gh]: https://github.com "GitHub"
"""
links = extract_markdown_links(md)
for link_type, urls in links.items():
    print(f"{link_type}: {urls}")
```

**Problem 9: Code Tokenizer**
Tokenize a simple expression into its components.

```python
def tokenize(expression):
    """
    Tokenize a mathematical/logical expression into:
    - NUMBER: integers and floats
    - IDENTIFIER: variable names
    - OPERATOR: +, -, *, /, **, ==, !=, <=, >=, <, >, and, or, not
    - LPAREN / RPAREN: ( )
    - STRING: 'text' or "text"
    - WHITESPACE: spaces (skip these)
    """
    pass

expr = "x + 2.5 * (y - 1) >= 10 and name == 'Alice'"
tokens = tokenize(expr)
for token_type, value in tokens:
    print(f"{token_type:12} {value!r}")
```

**Problem 10: SQL Query Parser**
Extract components from a simple SQL SELECT statement.

```python
def parse_select(sql):
    """
    Parse SQL like:
    SELECT col1, col2 FROM table WHERE condition ORDER BY col3 LIMIT n
    
    Return dict with: select, from, where, order_by, limit
    Handle: aliases (AS), multiple tables (FROM a, b), subqueries
    """
    pass

queries = [
    "SELECT name, age FROM users WHERE age > 18 ORDER BY name LIMIT 10",
    "SELECT u.name, o.total FROM users u, orders o WHERE u.id = o.user_id",
    "SELECT COUNT(*) as count FROM products WHERE category = 'Electronics'",
]
for q in queries:
    result = parse_select(q)
    print(result)
```

**Problem 11: Multi-format Date Parser**
Parse dates in any reasonable format.

```python
def parse_date_flexible(date_str):
    """
    Parse dates in any of these formats:
    - 2025-01-15
    - 15/01/2025 or 01/15/2025
    - Jan 15, 2025 or January 15, 2025
    - 15 Jan 2025 or 15 January 2025
    - 15-Jan-25 (2-digit year)
    
    Return: datetime.date object or None if unparseable
    """
    from datetime import date
    pass

test_dates = [
    "2025-01-15", "15/01/2025", "01/15/2025",
    "Jan 15, 2025", "January 15, 2025",
    "15 Jan 2025", "15-Jan-25", "garbage"
]
for d in test_dates:
    result = parse_date_flexible(d)
    print(f"'{d}' → {result}")
```

**Problem 12: Log Anomaly Detector**
Use regex to detect patterns in log files that signal problems.

```python
def detect_anomalies(log_text):
    """
    Scan log text for known anomaly patterns:
    1. Repeated failed login (same IP, >5 times in log)
    2. SQL injection attempts (UNION, DROP, --, etc. in requests)
    3. Path traversal attempts (../../ in URLs)
    4. Response time warnings (> 5000ms)
    5. Error rate spikes (>3 errors in same second)
    
    Return: list of anomaly dicts with type, details, line numbers
    """
    pass
```

### Hard (13–20)

**Problem 13: Recursive Template Expander**
Build a regex-based template system with nested variables.

```python
def expand_template(template, variables, max_depth=10):
    """
    Expand a template where variables can reference other variables.
    {{greeting}} = "Hello, {{name}}!"
    {{name}} = "World"
    → "Hello, World!"
    
    Detect circular references and raise an error.
    """
    pass

variables = {
    "greeting": "Hello, {{name | upper}}!",
    "name": "alice {{surname}}",
    "surname": "johnson",
    "farewell": "Goodbye, {{name}}. {{greeting}}"
}

print(expand_template("{{greeting}}", variables))
# Hello, ALICE JOHNSON!
```

**Problem 14: Diff Engine**
Implement a simple line-based diff using regex.

```python
def diff_text(original, modified):
    """
    Find differences between two texts.
    Return list of change objects:
    {type: 'added'|'removed'|'changed', line_no: n, content: '...'}
    
    Handle: whitespace normalization, moved blocks, similar lines
    """
    pass
```

**Problem 15: Natural Language Number Parser**
Convert written numbers to integers.

```python
def text_to_number(text):
    """
    Convert English number words to integers.
    "forty-two" → 42
    "one hundred and twenty-three" → 123
    "two thousand and twenty-five" → 2025
    "three million five hundred thousand" → 3500000
    
    Handle: hyphenated, "and", various orderings
    """
    pass

tests = [
    ("forty-two", 42),
    ("one hundred and twenty-three", 123),
    ("two thousand and twenty-five", 2025),
    ("three million five hundred thousand", 3500000),
]
for text, expected in tests:
    result = text_to_number(text)
    status = "✓" if result == expected else f"✗ (got {result})"
    print(f"{status} '{text}' → {expected}")
```

**Problem 16: Regex-based Lexer**
Build a complete lexer for a mini programming language.

```python
def make_lexer(token_specs):
    """
    Build a lexer from a list of (token_type, pattern) pairs.
    Return a function that tokenizes any input string.
    Uses re.compile with OR-joined named groups for efficiency.
    """
    pass

MINI_LANG_TOKENS = [
    ("NUMBER",   r"\d+(?:\.\d+)?"),
    ("STRING",   r'"[^"]*"'),
    ("IF",       r"\bif\b"),
    ("ELSE",     r"\belse\b"),
    ("WHILE",    r"\bwhile\b"),
    ("IDENT",    r"[a-zA-Z_]\w*"),
    ("EQ",       r"=="),
    ("ASSIGN",   r"="),
    ("LBRACE",   r"\{"),
    ("RBRACE",   r"\}"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("SEMI",     r";"),
    ("SKIP",     r"\s+"),   # Skip whitespace
    ("MISMATCH", r"."),     # Any other character
]

lexer = make_lexer(MINI_LANG_TOKENS)
code = 'if (x == 42) { result = "found"; }'
for token in lexer(code):
    print(token)
```

**Problem 17: Citation Extractor**
Extract academic citations from text.

```python
def extract_citations(text):
    """
    Extract citations in multiple formats:
    - APA: Author, A. (Year). Title. Journal, Vol(Issue), pages.
    - MLA: Author. "Title." Journal Vol.Issue (Year): pages.
    - IEEE: [1] A. Author, "Title," Journal, vol. 1, no. 1, pp. 1-10, Year.
    - In-text: (Author, Year) or (Author et al., Year)
    
    Return list of normalized citation dicts.
    """
    pass
```

**Problem 18: Pattern Frequency Analyzer**
Analyze which regex patterns are most common in a corpus.

```python
def find_recurring_patterns(texts, min_length=3, min_frequency=2):
    """
    Discover common patterns in a list of strings WITHOUT knowing patterns upfront.
    Use suffix arrays or similar to find repeated substrings.
    Return: [(pattern, frequency, example_context), ...]
    
    Useful for: discovering data format patterns in messy data
    """
    pass
```

**Problem 19: Regex Optimizer**
Analyze and suggest optimizations for regex patterns.

```python
def analyze_regex(pattern):
    """
    Analyze a regex pattern and report:
    - Is it anchored? (starts with ^ or \A)
    - Does it have catastrophic backtracking risk? (nested quantifiers)
    - Can any .* be replaced with [^x]* for better performance?
    - Are there redundant character classes?
    - Estimated complexity: O(n), O(n²), O(2^n)
    - Suggestions for improvement
    """
    pass

patterns_to_analyze = [
    r".*@.*",                    # Unanchored, uses .*
    r"(a+)+b",                   # Catastrophic backtracking!
    r"[a-zA-Z0-9][a-zA-Z0-9]+", # Redundant — [a-zA-Z0-9]{2,}
    r"^[\w.+-]+@[\w-]+\.[a-z]{2,}$",  # Well-written
]
```

**Problem 20: Mini Regex Engine**
Implement a simplified regex engine from scratch.

```python
class MiniRegex:
    """
    Simplified regex engine supporting:
    - Literal characters
    - . (any char)
    - * (zero or more)
    - + (one or more)
    - ? (zero or one)
    - [] character classes
    - ^ start anchor
    - $ end anchor
    
    No lookaheads, groups, or backreferences for simplicity.
    """
    
    def __init__(self, pattern):
        self.pattern = pattern
        self._compiled = self._compile(pattern)
    
    def _compile(self, pattern):
        """Parse pattern into list of (type, value) tokens."""
        pass
    
    def match(self, string):
        """Return True if pattern matches string from the start."""
        pass
    
    def search(self, string):
        """Return start position if pattern found anywhere, else -1."""
        pass
    
    def findall(self, string):
        """Return list of all non-overlapping matches."""
        pass

# Test
r = MiniRegex(r"\d+\.\d+")
print(r.search("price: 9.99"))    # Found at position 7
print(r.findall("3.14 and 2.71")) # ['3.14', '2.71']
```

---

## Answer Keys (Selected Problems)

### Problem 1 Solution:
```python
text = "Running quickly through 42 streets on 15/01/2025 while Thinking deeply"

capital_words = re.findall(r"\b[A-Z][a-zA-Z]+\b", text)   # ['Running', 'Thinking']
numbers       = re.findall(r"\d+", text)                    # ['42', '15', '01', '2025']
ing_words     = re.findall(r"\b\w+ing\b", text)             # ['Running', 'Thinking']
letters_only  = re.findall(r"\b[a-zA-Z]+\b", text)         # All words
dates         = re.findall(r"\b\d{2}/\d{2}/\d{4}\b", text) # ['15/01/2025']
```

### Problem 3 Solution:
```python
def normalize_phones(text):
    PHONE_RE = re.compile(r"""
        (?:\+?1[-.\s]?)?      # Optional country code
        \(?(\d{3})\)?         # Area code (captured)
        [-.\s]?               # Separator
        (\d{3})               # Exchange (captured)
        [-.\s]?               # Separator
        (\d{4})               # Number (captured)
    """, re.VERBOSE)
    
    results = []
    for m in PHONE_RE.finditer(text):
        area, exchange, number = m.group(1), m.group(2), m.group(3)
        results.append(f"({area}) {exchange}-{number}")
    return results
```

### Problem 7 Solution:
```python
def parse_git_log(log_output):
    COMMIT_RE = re.compile(
        r"^(?P<hash>[a-f0-9]{7,})\s+"
        r"(?P<type>\w+):\s+"
        r"(?P<description>.+?)"
        r"(?:\s+\(#(?P<issue>\d+)\))?$",
        re.M
    )
    return [m.groupdict() for m in COMMIT_RE.finditer(log_output.strip())]
```

### Problem 9 Solution:
```python
def tokenize(expression):
    TOKEN_RE = re.compile(r"""
        (?P<NUMBER>    \d+(?:\.\d+)?)           |
        (?P<STRING>    "[^"]*"|'[^']*')          |
        (?P<OPERATOR>  \*\*|==|!=|<=|>=|[-+*/=<>]|\band\b|\bor\b|\bnot\b) |
        (?P<LPAREN>    \()                       |
        (?P<RPAREN>    \))                       |
        (?P<IDENTIFIER>[a-zA-Z_]\w*)             |
        (?P<WHITESPACE>\s+)
    """, re.VERBOSE)
    
    tokens = []
    for m in TOKEN_RE.finditer(expression):
        if m.lastgroup != "WHITESPACE":
            tokens.append((m.lastgroup, m.group()))
    return tokens
```

---

## Mini-Project: Text Intelligence Suite

```python
"""
text_intelligence.py
A complete text processing suite using regular expressions.
Combines extraction, validation, cleaning, and analysis.
"""

import re
from collections import Counter, defaultdict
from datetime import datetime

class TextIntelligence:
    """
    Comprehensive text analysis and processing toolkit.
    All heavy lifting done with compiled regex patterns.
    """
    
    # ── COMPILED PATTERNS ──────────────────────────────────────────────────
    _PATTERNS = {
        "email":    re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
        "phone":    re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
        "url":      re.compile(r"https?://[\w.-]+(?:/[^\s]*)?"),
        "ip":       re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "date_iso": re.compile(r"\b\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"),
        "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "currency": re.compile(r"[$€£₹¥][\d,]+(?:\.\d{1,2})?"),
        "hashtag":  re.compile(r"#\w+"),
        "mention":  re.compile(r"@\w+"),
        "sentence": re.compile(r"(?<=[.!?])\s+"),
        "word":     re.compile(r"\b[a-zA-Z]+\b"),
        "number":   re.compile(r"-?\d+(?:\.\d+)?"),
        "punct":    re.compile(r"[^\w\s]"),
    }
    
    STOPWORDS = {
        "the", "a", "an", "is", "it", "in", "on", "at", "to",
        "and", "or", "but", "of", "for", "with", "as", "by",
        "from", "this", "that", "was", "are", "be", "been",
        "have", "has", "had", "will", "would", "could", "should"
    }
    
    # ── EXTRACTION ─────────────────────────────────────────────────────────
    
    def extract(self, text, *entity_types):
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            *entity_types: 'email', 'phone', 'url', 'ip', 'date_iso',
                           'currency', 'hashtag', 'mention'
        Returns:
            dict of {entity_type: [values]}
        """
        if not entity_types:
            entity_types = list(self._PATTERNS.keys())
        
        return {
            etype: self._PATTERNS[etype].findall(text)
            for etype in entity_types
            if etype in self._PATTERNS
        }
    
    def extract_all_pii(self, text):
        """Extract all Personally Identifiable Information."""
        pii_types = ["email", "phone", "ip", "credit_card"]
        results = self.extract(text, *pii_types)
        
        # Also look for SSN
        ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        ssns = ssn_pattern.findall(text)
        if ssns:
            results["ssn"] = ssns
        
        return {k: v for k, v in results.items() if v}
    
    # ── MASKING / ANONYMIZATION ────────────────────────────────────────────
    
    def mask_pii(self, text, replacement_style="placeholder"):
        """
        Mask PII in text.
        
        replacement_style:
            'placeholder' → [EMAIL], [PHONE], etc.
            'redact'      → ████████
            'hash'        → shows last 4 chars: ****1234
        """
        result = text
        
        masks = {
            "email":    (self._PATTERNS["email"],    "EMAIL"),
            "phone":    (self._PATTERNS["phone"],    "PHONE"),
            "ip":       (self._PATTERNS["ip"],       "IP"),
        }
        
        credit_card_re = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
        
        def make_replacement(type_name, value, style):
            if style == "placeholder":
                return f"[{type_name}]"
            elif style == "redact":
                return "█" * len(value)
            elif style == "hash":
                return "*" * (len(value) - 4) + value[-4:]
            return f"[{type_name}]"
        
        for type_name, (pattern, label) in masks.items():
            def replacer(m, tn=label):
                return make_replacement(tn, m.group(), replacement_style)
            result = pattern.sub(replacer, result)
        
        def cc_replacer(m):
            val = m.group()
            digits_only = re.sub(r"\D", "", val)
            if len(digits_only) == 16:
                return make_replacement("CARD", val, replacement_style)
            return val
        result = credit_card_re.sub(cc_replacer, result)
        
        return result
    
    # ── TEXT ANALYSIS ──────────────────────────────────────────────────────
    
    def analyze(self, text):
        """Full statistical analysis of text."""
        words = self._PATTERNS["word"].findall(text.lower())
        sentences = self._PATTERNS["sentence"].split(text.strip())
        sentences = [s for s in sentences if s.strip()]
        numbers = self._PATTERNS["number"].findall(text)
        
        # Word frequency (excluding stopwords)
        content_words = [w for w in words if w not in self.STOPWORDS and len(w) > 2]
        word_freq = Counter(content_words)
        
        # Sentence lengths
        sent_lengths = [len(s.split()) for s in sentences]
        
        return {
            "char_count":       len(text),
            "char_no_spaces":   len(text.replace(" ", "")),
            "word_count":       len(words),
            "unique_words":     len(set(words)),
            "sentence_count":   len(sentences),
            "avg_sent_length":  sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0,
            "vocabulary_richness": len(set(words)) / len(words) if words else 0,
            "top_10_words":     word_freq.most_common(10),
            "numbers_found":    numbers,
            "has_pii":          bool(self.extract_all_pii(text)),
        }
    
    # ── VALIDATION ─────────────────────────────────────────────────────────
    
    def validate(self, value, type_name):
        """
        Validate a value against a known pattern type.
        Returns (is_valid, error_message_or_None)
        """
        validators = {
            "email":    (
                re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}$"),
                "Must be in format: user@domain.tld"
            ),
            "phone":    (
                re.compile(r"^(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$"),
                "Must be a valid 10-digit phone number"
            ),
            "url":      (
                re.compile(r"^https?://[\w.-]+(?:\.[a-zA-Z]{2,})+(?:/[^\s]*)?$"),
                "Must start with http:// or https://"
            ),
            "ip":       (
                re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"),
                "Must be a valid IPv4 address (0-255 in each octet)"
            ),
            "date_iso": (
                re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$"),
                "Must be in YYYY-MM-DD format"
            ),
            "password": (
                re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*]).{8,}$"),
                "Must have uppercase, lowercase, digit, special char, min 8 chars"
            ),
        }
        
        if type_name not in validators:
            return False, f"Unknown type: {type_name}"
        
        pattern, error_msg = validators[type_name]
        is_valid = bool(pattern.match(value.strip()))
        return is_valid, None if is_valid else error_msg
    
    # ── CLEANING ───────────────────────────────────────────────────────────
    
    def clean(self, text, operations=None):
        """
        Apply a series of cleaning operations.
        
        operations: list of operation names (default: all)
            'normalize_whitespace'  — collapse multiple spaces/newlines
            'remove_html'           — strip HTML tags
            'remove_urls'           — remove URL strings
            'remove_punctuation'    — remove punctuation
            'fix_encoding'          — fix common encoding issues
            'normalize_quotes'      — standardize quote characters
        """
        if operations is None:
            operations = [
                "normalize_whitespace", "remove_html",
                "fix_encoding", "normalize_quotes"
            ]
        
        result = text
        
        if "remove_html" in operations:
            result = re.sub(r"<[^>]+>", "", result)
            result = re.sub(r"<!--.*?-->", "", result, flags=re.S)
        
        if "remove_urls" in operations:
            result = self._PATTERNS["url"].sub("", result)
        
        if "normalize_quotes" in operations:
            result = re.sub(r"[""„‟]", '"', result)
            result = re.sub(r"[''‛]", "'", result)
        
        if "fix_encoding" in operations:
            result = re.sub(r"&amp;",  "&", result)
            result = re.sub(r"&lt;",   "<", result)
            result = re.sub(r"&gt;",   ">", result)
            result = re.sub(r"&nbsp;", " ", result)
            result = re.sub(r"&quot;", '"', result)
        
        if "remove_punctuation" in operations:
            result = self._PATTERNS["punct"].sub(" ", result)
        
        if "normalize_whitespace" in operations:
            result = re.sub(r"[ \t]+", " ", result)
            result = re.sub(r"\n{3,}", "\n\n", result)
            result = result.strip()
        
        return result
    
    # ── SEARCH ─────────────────────────────────────────────────────────────
    
    def search(self, text, query, case_sensitive=False, whole_word=False):
        """
        Search text for query with highlighting.
        Returns (match_count, highlighted_text)
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern_str = re.escape(query)
        
        if whole_word:
            pattern_str = r"\b" + pattern_str + r"\b"
        
        pattern = re.compile(pattern_str, flags)
        matches = list(pattern.finditer(text))
        
        # Highlight matches
        highlighted = pattern.sub(lambda m: f">>>{m.group()}<<<", text)
        
        return len(matches), highlighted
    
    def replace_pattern(self, text, pattern_str, replacement,
                        case_insensitive=False, max_replacements=0):
        """
        Safe pattern replacement with error handling and stats.
        Returns (new_text, replacements_made, error)
        """
        try:
            flags = re.IGNORECASE if case_insensitive else 0
            pattern = re.compile(pattern_str, flags)
            
            count = [0]
            def count_replace(m):
                count[0] += 1
                if callable(replacement):
                    return replacement(m)
                return m.expand(replacement)
            
            new_text = pattern.sub(
                count_replace, text,
                count=max_replacements if max_replacements > 0 else 0
            )
            return new_text, count[0], None
        
        except re.error as e:
            return text, 0, str(e)


# ── DEMO ────────────────────────────────────────────────────────────────────

def main():
    ti = TextIntelligence()
    
    sample_text = """
    Dear Alice Johnson (@alice_j), 
    
    Thank you for your order placed on 2025-01-15.
    Your credit card 4111-1111-1111-2345 was charged $1,299.00.
    
    For support, contact support@techcorp.com or visit https://techcorp.com/help.
    Our server at 192.168.1.100 processed your request.
    Phone: (555) 123-4567. Reference: #ORDER2025 #URGENT.
    
    Python is an amazing language! Python is used for data science, 
    web development, and automation. Many developers love Python because
    Python is easy to learn and very powerful.
    """
    
    print("=" * 55)
    print("TEXT INTELLIGENCE SUITE DEMO")
    print("=" * 55)
    
    # ── PII Detection ────────────────────────────────────────────────────
    print("\n── PII Detection ────────────────────────────")
    pii = ti.extract_all_pii(sample_text)
    for pii_type, values in pii.items():
        print(f"  {pii_type:<15}: {values}")
    
    # ── PII Masking ───────────────────────────────────────────────────────
    print("\n── PII Masking (placeholder) ─────────────────")
    masked = ti.mask_pii(sample_text, replacement_style="placeholder")
    print(masked[:300] + "...")
    
    # ── Entity Extraction ─────────────────────────────────────────────────
    print("\n── Entity Extraction ─────────────────────────")
    entities = ti.extract(sample_text, "hashtag", "mention", "currency", "date_iso")
    for etype, vals in entities.items():
        if vals:
            print(f"  {etype:<15}: {vals}")
    
    # ── Text Analysis ─────────────────────────────────────────────────────
    print("\n── Text Analysis ─────────────────────────────")
    analysis = ti.analyze(sample_text)
    print(f"  Words:          {analysis['word_count']}")
    print(f"  Unique words:   {analysis['unique_words']}")
    print(f"  Sentences:      {analysis['sentence_count']}")
    print(f"  Avg sent. len:  {analysis['avg_sent_length']:.1f} words")
    print(f"  Vocab richness: {analysis['vocabulary_richness']:.1%}")
    print(f"  Top 5 words:    {analysis['top_10_words'][:5]}")
    print(f"  Contains PII:   {analysis['has_pii']}")
    
    # ── Validation ────────────────────────────────────────────────────────
    print("\n── Validation ────────────────────────────────")
    test_values = [
        ("email",    "alice@example.com",    True),
        ("email",    "not-an-email",         False),
        ("phone",    "(555) 123-4567",        True),
        ("ip",       "192.168.1.256",         False),
        ("date_iso", "2025-13-45",            False),
        ("password", "Str0ng!Pass",           True),
        ("password", "weakpass",              False),
    ]
    for type_name, value, expected in test_values:
        is_valid, error = ti.validate(value, type_name)
        status = "✓" if is_valid == expected else "✗"
        result = "Valid" if is_valid else f"Invalid: {error}"
        print(f"  {status} {type_name:<12} '{value}': {result}")
    
    # ── Search ────────────────────────────────────────────────────────────
    print("\n── Search ────────────────────────────────────")
    count, highlighted = ti.search(sample_text, "python", case_sensitive=False)
    print(f"  Found 'Python' {count} times")
    
    count2, _ = ti.search(sample_text, "python", whole_word=True)
    print(f"  Found 'Python' as whole word {count2} times")
    
    # ── Replace ───────────────────────────────────────────────────────────
    print("\n── Pattern Replace ───────────────────────────")
    new_text, n, err = ti.replace_pattern(
        sample_text,
        r"\b(python)\b",
        r"🐍\1",
        case_insensitive=True
    )
    print(f"  Made {n} replacements (Python → 🐍Python)")
    
    print("\n" + "=" * 55)

main()
```

---

## Chapter Summary

You've mastered one of the most powerful tools in any programmer's toolkit!

✅ **Core Functions**: `search`, `match`, `fullmatch`, `findall`, `finditer`, `sub`, `split`, `compile`
✅ **Pattern Syntax**: Literals, `.`, `[...]`, `\d\w\s`, `^$\b`, quantifiers `* + ? {n,m}`
✅ **Greedy vs Lazy**: `*` vs `*?` — when each matters and why
✅ **Groups**: Capturing `(...)`, named `(?P<name>...)`, non-capturing `(?:...)`, backreferences `\1`
✅ **Lookarounds**: Lookahead `(?=...)`, lookbehind `(?<=...)` and their negations
✅ **Flags**: `re.I`, `re.M`, `re.S`, `re.X` — and combining them
✅ **Essential Patterns**: Email, phone, date, URL, IP, credit card, password
✅ **Practical Tasks**: Log parsing, data extraction, text anonymization, config parsing, template engines

**Key Takeaways:**
- Always use raw strings `r"..."` for regex patterns — never regular strings
- Use `re.compile()` when reusing the same pattern — compiles once, runs fast
- Greedy is the default — add `?` after quantifiers for lazy matching
- `re.VERBOSE` with comments makes complex patterns maintainable
- Test your patterns at [regex101.com](https://regex101.com) before hardcoding
- For validation, use `fullmatch()` or anchor with `^...$` — never `search()` alone

**Next Chapter Preview:**
Chapter 15 covers **Testing, Debugging, and Best Practices** — writing reliable code with `unittest` and `pytest`, debugging strategies, PEP 8 style, performance profiling, and the habits that separate professional developers from hobbyists!
