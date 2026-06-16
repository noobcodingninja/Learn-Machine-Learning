# Chapter 15: Testing, Debugging, and Best Practices

## Part 1: Why Testing Matters

### The Problem: "It Works on My Machine"

Imagine you write this function for an e-commerce app:

```python
def calculate_discount(price, discount_percent):
    return price - (price * discount_percent / 100)

# You test it manually once:
print(calculate_discount(100, 10))   # 90.0 — looks right!
```

You ship it. Three weeks later, production breaks:

```python
calculate_discount(100, -10)    # 110.0  — negative discount INCREASES price?!
calculate_discount(-50, 10)     # -45.0  — negative price?!
calculate_discount(100, 150)    # -50.0  — discount more than 100%, price goes negative!
calculate_discount("100", 10)   # TypeError — someone passed a string
```

**The core problem:** Manual testing only checks the cases you thought of. Real-world input is far messier than your imagination on a Tuesday afternoon.

**What automated tests give you:**
1. **Confidence** — change code without fear of breaking something else
2. **Documentation** — tests show exactly how a function should behave
3. **Regression prevention** — a bug fixed once never silently comes back
4. **Faster development** — catch bugs in seconds, not after a customer complains

```python
# With tests, every edge case is checked AUTOMATICALLY, every time:
def test_calculate_discount():
    assert calculate_discount(100, 10) == 90.0
    assert calculate_discount(100, 0) == 100.0
    assert calculate_discount(100, 100) == 0.0
    # Now imagine running this after EVERY code change — instantly!
```

---

## Part 2: Writing Tests with `unittest`

### The Standard Library Testing Framework

```python
import unittest

# ── THE FUNCTION WE'RE TESTING ────────────────────────────────────────────
def calculate_discount(price, discount_percent):
    """Calculate price after applying a percentage discount."""
    if price < 0:
        raise ValueError("Price cannot be negative")
    if not (0 <= discount_percent <= 100):
        raise ValueError("Discount must be between 0 and 100")
    return round(price - (price * discount_percent / 100), 2)


# ── TEST CLASS — inherits from unittest.TestCase ─────────────────────────
class TestCalculateDiscount(unittest.TestCase):
    """
    Naming convention:
    - Test classes start with Test
    - Test methods start with test_
    - unittest automatically discovers and runs all of them
    """
    
    def test_normal_discount(self):
        """Test a standard discount calculation."""
        result = calculate_discount(100, 10)
        self.assertEqual(result, 90.0)
    
    def test_zero_discount(self):
        """0% discount should return original price."""
        result = calculate_discount(100, 0)
        self.assertEqual(result, 100.0)
    
    def test_full_discount(self):
        """100% discount should return 0."""
        result = calculate_discount(100, 100)
        self.assertEqual(result, 0.0)
    
    def test_negative_price_raises_error(self):
        """Negative price should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_discount(-50, 10)
    
    def test_discount_over_100_raises_error(self):
        """Discount over 100% should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_discount(100, 150)
    
    def test_negative_discount_raises_error(self):
        """Negative discount should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_discount(100, -10)
    
    def test_float_precision(self):
        """Result should be rounded to 2 decimal places."""
        result = calculate_discount(99.99, 15)
        self.assertAlmostEqual(result, 84.99, places=2)


if __name__ == "__main__":
    unittest.main()
    # Run with: python test_discount.py
    # Or:       python -m unittest test_discount.py -v
```

### Assertion Methods Reference

```python
import unittest

class AssertionExamples(unittest.TestCase):
    
    def test_equality_assertions(self):
        self.assertEqual(2 + 2, 4)                    # a == b
        self.assertNotEqual(2 + 2, 5)                  # a != b
        self.assertAlmostEqual(0.1 + 0.2, 0.3, places=7)  # for floats!
    
    def test_boolean_assertions(self):
        self.assertTrue(5 > 3)
        self.assertFalse(5 < 3)
        self.assertIsNone(None)
        self.assertIsNotNone("value")
    
    def test_container_assertions(self):
        self.assertIn(3, [1, 2, 3, 4])                 # item in container
        self.assertNotIn(10, [1, 2, 3, 4])
        self.assertListEqual([1, 2, 3], [1, 2, 3])
        self.assertDictEqual({"a": 1}, {"a": 1})
        self.assertSetEqual({1, 2, 3}, {3, 2, 1})       # order doesn't matter
    
    def test_type_assertions(self):
        self.assertIsInstance(5, int)
        self.assertNotIsInstance(5, str)
    
    def test_exception_assertions(self):
        with self.assertRaises(ValueError):
            int("not a number")
        
        # Check the exception MESSAGE too
        with self.assertRaisesRegex(ValueError, "invalid literal"):
            int("abc")
    
    def test_comparison_assertions(self):
        self.assertGreater(10, 5)
        self.assertGreaterEqual(10, 10)
        self.assertLess(5, 10)
        self.assertLessEqual(5, 5)
```

### setUp and tearDown — Test Fixtures

```python
import unittest

class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
    
    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit must be positive")
        self.balance += amount
        return self.balance
    
    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
        return self.balance


class TestBankAccount(unittest.TestCase):
    
    def setUp(self):
        """
        Runs BEFORE every test method.
        Use this to set up fresh test data for each test.
        """
        self.account = BankAccount("Alice", 1000)
        print(f"  Setting up fresh account for {self._testMethodName}")
    
    def tearDown(self):
        """
        Runs AFTER every test method (even if the test failed!).
        Use for cleanup: closing files, database connections, etc.
        """
        print(f"  Cleaning up after {self._testMethodName}")
    
    def test_deposit_increases_balance(self):
        self.account.deposit(500)
        self.assertEqual(self.account.balance, 1500)
    
    def test_withdraw_decreases_balance(self):
        self.account.withdraw(300)
        self.assertEqual(self.account.balance, 700)
    
    def test_withdraw_more_than_balance_fails(self):
        with self.assertRaises(ValueError):
            self.account.withdraw(2000)
    
    @classmethod
    def setUpClass(cls):
        """Runs ONCE before ALL tests in this class (not per-test)."""
        print("Starting test suite for BankAccount")
    
    @classmethod
    def tearDownClass(cls):
        """Runs ONCE after ALL tests in this class."""
        print("Finished test suite for BankAccount")
```

---

## Part 3: `pytest` — The Modern Standard

### Why Pytest Over unittest

```python
# pip install pytest

# ── unittest STYLE — verbose, class-based ─────────────────────────────────
import unittest

class TestMathUnittest(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(2 + 2, 4)

# ── pytest STYLE — simple functions, plain assert ─────────────────────────
def test_addition():
    assert 2 + 2 == 4

# pytest advantages:
# 1. No boilerplate classes needed (though they're supported too)
# 2. Plain `assert` statements — pytest shows DETAILED failure info
# 3. Powerful fixtures system
# 4. Parametrized tests (run same test with many inputs)
# 5. Huge plugin ecosystem (coverage, mocking, parallel runs)
# 6. Auto-discovers tests in files named test_*.py or *_test.py
```

### Basic Pytest Tests

```python
# ── FILE: shopping_cart.py ───────────────────────────────────────────────
class ShoppingCart:
    def __init__(self):
        self.items = {}
    
    def add_item(self, name, price, quantity=1):
        if price < 0:
            raise ValueError("Price cannot be negative")
        if name in self.items:
            self.items[name]["quantity"] += quantity
        else:
            self.items[name] = {"price": price, "quantity": quantity}
    
    def remove_item(self, name):
        if name not in self.items:
            raise KeyError(f"{name} not in cart")
        del self.items[name]
    
    @property
    def total(self):
        return sum(item["price"] * item["quantity"] for item in self.items.values())
    
    @property
    def item_count(self):
        return sum(item["quantity"] for item in self.items.values())


# ── FILE: test_shopping_cart.py ──────────────────────────────────────────
import pytest
from shopping_cart import ShoppingCart

def test_empty_cart_has_zero_total():
    cart = ShoppingCart()
    assert cart.total == 0

def test_add_single_item():
    cart = ShoppingCart()
    cart.add_item("Apple", 0.50, 3)
    assert cart.total == 1.50
    assert cart.item_count == 3

def test_add_multiple_different_items():
    cart = ShoppingCart()
    cart.add_item("Apple", 0.50, 2)
    cart.add_item("Bread", 3.00, 1)
    assert cart.total == 4.00

def test_adding_same_item_twice_increases_quantity():
    cart = ShoppingCart()
    cart.add_item("Apple", 0.50, 2)
    cart.add_item("Apple", 0.50, 3)
    assert cart.items["Apple"]["quantity"] == 5

def test_negative_price_raises_error():
    cart = ShoppingCart()
    with pytest.raises(ValueError):
        cart.add_item("Bad Item", -5.00)

def test_remove_nonexistent_item_raises_error():
    cart = ShoppingCart()
    with pytest.raises(KeyError):
        cart.remove_item("Nonexistent")

def test_remove_item_works():
    cart = ShoppingCart()
    cart.add_item("Apple", 0.50, 2)
    cart.remove_item("Apple")
    assert cart.total == 0

# Run with: pytest test_shopping_cart.py -v
# Output shows DETAILED diffs on failure, e.g.:
#   assert 1.5 == 2.0
#   +  where 1.5 = cart.total
```

### Fixtures — Reusable Test Setup

```python
import pytest
from shopping_cart import ShoppingCart

# ── BASIC FIXTURE ──────────────────────────────────────────────────────────
@pytest.fixture
def empty_cart():
    """Provides a fresh, empty cart for any test that needs it."""
    return ShoppingCart()

@pytest.fixture
def cart_with_items():
    """Provides a cart pre-populated with items."""
    cart = ShoppingCart()
    cart.add_item("Apple", 0.50, 3)
    cart.add_item("Bread", 3.00, 1)
    return cart

# Tests REQUEST fixtures by naming them as parameters
def test_empty_cart(empty_cart):
    assert empty_cart.total == 0

def test_cart_with_items_total(cart_with_items):
    assert cart_with_items.total == 4.50

def test_cart_with_items_count(cart_with_items):
    assert cart_with_items.item_count == 4


# ── FIXTURE WITH TEARDOWN (yield) ─────────────────────────────────────────
@pytest.fixture
def temp_database():
    """Setup AND teardown using yield."""
    print("\n  Setting up test database...")
    db = {"users": [], "connected": True}
    
    yield db   # ← This value is what the test receives
    
    # Code after yield runs AFTER the test, even if it failed!
    print("  Tearing down test database...")
    db["connected"] = False

def test_database_starts_connected(temp_database):
    assert temp_database["connected"] is True

def test_can_add_user(temp_database):
    temp_database["users"].append("Alice")
    assert len(temp_database["users"]) == 1


# ── FIXTURE SCOPES ──────────────────────────────────────────────────────────
@pytest.fixture(scope="function")  # Default: new instance per TEST FUNCTION
def per_test_fixture():
    return {"created_count": 1}

@pytest.fixture(scope="class")     # One instance per TEST CLASS
def per_class_fixture():
    return {"shared": "across class tests"}

@pytest.fixture(scope="module")    # One instance per FILE
def per_module_fixture():
    print("Expensive setup — only runs once for the whole file")
    return connect_to_expensive_resource()

@pytest.fixture(scope="session")   # One instance for the ENTIRE test run
def per_session_fixture():
    print("Runs ONCE no matter how many test files exist")
    return "shared_resource"


# ── AUTOUSE FIXTURES — applied automatically, no need to request ─────────
@pytest.fixture(autouse=True)
def reset_global_state():
    """Runs before AND after every single test automatically."""
    print("Resetting state before test")
    yield
    print("Resetting state after test")
```

### Parametrized Tests — One Test, Many Inputs

```python
import pytest

def calculate_discount(price, discount_percent):
    if price < 0:
        raise ValueError("Price cannot be negative")
    if not (0 <= discount_percent <= 100):
        raise ValueError("Discount must be 0-100")
    return round(price - (price * discount_percent / 100), 2)


# ── WITHOUT PARAMETRIZE — repetitive! ────────────────────────────────────
def test_discount_10_percent():
    assert calculate_discount(100, 10) == 90.0

def test_discount_25_percent():
    assert calculate_discount(100, 25) == 75.0

def test_discount_50_percent():
    assert calculate_discount(100, 50) == 50.0
# ...repeated for every case you want to check


# ── WITH PARAMETRIZE — one test, many cases! ──────────────────────────────
@pytest.mark.parametrize("price,discount,expected", [
    (100, 10, 90.0),
    (100, 25, 75.0),
    (100, 50, 50.0),
    (200, 0, 200.0),
    (50, 100, 0.0),
    (99.99, 15, 84.99),
])
def test_calculate_discount_parametrized(price, discount, expected):
    result = calculate_discount(price, discount)
    assert result == pytest.approx(expected, abs=0.01)
    # pytest.approx handles floating-point comparison issues

# Running this generates 6 SEPARATE test results:
# test_calculate_discount_parametrized[100-10-90.0] PASSED
# test_calculate_discount_parametrized[100-25-75.0] PASSED
# ... etc — each shown individually in output!


# ── PARAMETRIZE WITH EXPECTED EXCEPTIONS ──────────────────────────────────
@pytest.mark.parametrize("price,discount", [
    (-50, 10),    # Negative price
    (100, -10),   # Negative discount
    (100, 150),   # Discount over 100%
])
def test_calculate_discount_raises_on_invalid_input(price, discount):
    with pytest.raises(ValueError):
        calculate_discount(price, discount)


# ── COMBINING MULTIPLE PARAMETRIZE DECORATORS (cartesian product) ────────
@pytest.mark.parametrize("price", [50, 100, 200])
@pytest.mark.parametrize("discount", [0, 25, 50])
def test_discount_combinations(price, discount):
    """Tests ALL 9 combinations: 3 prices × 3 discounts."""
    result = calculate_discount(price, discount)
    assert result <= price       # Result never exceeds original price
    assert result >= 0           # Result never negative


# ── PARAMETRIZE WITH IDS — readable test names ───────────────────────────
@pytest.mark.parametrize("input_str,expected", [
    ("hello", "HELLO"),
    ("World", "WORLD"),
    ("", ""),
], ids=["lowercase", "mixed_case", "empty_string"])
def test_uppercase(input_str, expected):
    assert input_str.upper() == expected
```

### Marks — Organizing and Skipping Tests

```python
import pytest
import sys

# ── SKIP — unconditionally skip a test ────────────────────────────────────
@pytest.mark.skip(reason="Feature not implemented yet")
def test_future_feature():
    assert some_unimplemented_function() == 42

# ── SKIPIF — conditionally skip ───────────────────────────────────────────
@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
def test_match_statement():
    pass

# ── XFAIL — expected to fail (known bug, not yet fixed) ──────────────────
@pytest.mark.xfail(reason="Known bug: see issue #123")
def test_known_broken_feature():
    assert broken_function() == "expected but currently fails"

# ── CUSTOM MARKS — categorize tests ───────────────────────────────────────
@pytest.mark.slow
def test_large_dataset_processing():
    """Mark slow tests so they can be excluded in quick runs."""
    pass

@pytest.mark.integration
def test_database_connection():
    """Mark integration tests (need external resources)."""
    pass

# Run only fast tests:    pytest -m "not slow"
# Run only integration:   pytest -m integration
# (requires registering marks in pytest.ini or pyproject.toml)


# ── pytest.ini configuration (separate file) ──────────────────────────────
"""
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests requiring external services
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
"""
```

---

## Part 4: Mocking — Testing in Isolation

### The Problem: External Dependencies

```python
# This function depends on a real API call, real time, real randomness
import requests
import time
import random

def get_weather_alert(city):
    """Real implementation calls an external API."""
    response = requests.get(f"https://api.weather.com/v1/{city}")
    if response.status_code != 200:
        raise ConnectionError("Weather API unavailable")
    data = response.json()
    return "ALERT" if data["temp"] > 40 else "NORMAL"

# Problems testing this directly:
# 1. Requires internet connection
# 2. API might be down — test fails for the WRONG reason
# 3. API might cost money per call
# 4. Real weather changes — test isn't repeatable!
# 5. Slow — network calls take time

# SOLUTION: Mock the external dependency
```

### `unittest.mock` — Replacing Real Objects

```python
from unittest.mock import Mock, MagicMock, patch
import pytest

# ── BASIC MOCK OBJECTS ────────────────────────────────────────────────────
mock_obj = Mock()

mock_obj.some_method()              # Doesn't error — returns another Mock
mock_obj.some_method.return_value = 42
print(mock_obj.some_method())       # 42

# Mocks track HOW they were called
mock_obj.process(1, 2, key="value")
print(mock_obj.process.called)              # True
print(mock_obj.process.call_count)          # 1
print(mock_obj.process.call_args)           # call(1, 2, key='value')

mock_obj.process.assert_called_once()
mock_obj.process.assert_called_with(1, 2, key="value")


# ── PATCH — replace a function/method for the duration of a test ─────────
def get_weather_alert(city, fetch_func):
    """Refactored to accept the fetch function — easier to test (DI)."""
    data = fetch_func(city)
    return "ALERT" if data["temp"] > 40 else "NORMAL"

def test_weather_alert_triggers_on_high_temp():
    # Create a fake fetch function
    fake_fetch = Mock(return_value={"temp": 45})
    result = get_weather_alert("Delhi", fake_fetch)
    assert result == "ALERT"
    fake_fetch.assert_called_once_with("Delhi")

def test_weather_alert_normal_on_low_temp():
    fake_fetch = Mock(return_value={"temp": 25})
    result = get_weather_alert("Mumbai", fake_fetch)
    assert result == "NORMAL"


# ── @patch DECORATOR — mock something at IMPORT location ─────────────────
import requests

def get_weather_alert_v2(city):
    """Original version that imports requests directly."""
    response = requests.get(f"https://api.weather.com/v1/{city}")
    data = response.json()
    return "ALERT" if data["temp"] > 40 else "NORMAL"

@patch("requests.get")    # Patches requests.get wherever it's used
def test_weather_alert_v2_high_temp(mock_get):
    # Configure what the mock returns
    mock_response = Mock()
    mock_response.json.return_value = {"temp": 45}
    mock_get.return_value = mock_response
    
    result = get_weather_alert_v2("Delhi")
    
    assert result == "ALERT"
    mock_get.assert_called_once_with("https://api.weather.com/v1/Delhi")


# ── PATCHING WITH CONTEXT MANAGER ─────────────────────────────────────────
def test_weather_alert_v2_with_context_manager():
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"temp": 50}
        result = get_weather_alert_v2("Chennai")
        assert result == "ALERT"


# ── MOCKING SIDE EFFECTS (raising exceptions, multiple return values) ────
@patch("requests.get")
def test_weather_api_failure(mock_get):
    mock_get.side_effect = ConnectionError("Network unreachable")
    
    with pytest.raises(ConnectionError):
        get_weather_alert_v2("Mumbai")

@patch("requests.get")
def test_weather_api_called_multiple_times(mock_get):
    """side_effect as a list — different return value each call."""
    mock_response_1 = Mock()
    mock_response_1.json.return_value = {"temp": 20}
    mock_response_2 = Mock()
    mock_response_2.json.return_value = {"temp": 45}
    
    mock_get.side_effect = [mock_response_1, mock_response_2]
    
    result1 = get_weather_alert_v2("CityA")
    result2 = get_weather_alert_v2("CityB")
    
    assert result1 == "NORMAL"
    assert result2 == "ALERT"


# ── MOCKING TIME AND RANDOMNESS ───────────────────────────────────────────
import datetime

def is_business_hours():
    now = datetime.datetime.now()
    return 9 <= now.hour < 17

@patch("datetime.datetime")
def test_is_business_hours_during_day(mock_datetime):
    mock_datetime.now.return_value = datetime.datetime(2025, 1, 15, 14, 30)
    assert is_business_hours() is True

@patch("datetime.datetime")
def test_is_business_hours_at_night(mock_datetime):
    mock_datetime.now.return_value = datetime.datetime(2025, 1, 15, 23, 0)
    assert is_business_hours() is False
```

### Pytest's `monkeypatch` Fixture — Built-in Mocking

```python
import os

def get_api_key():
    key = os.environ.get("API_KEY")
    if not key:
        raise ValueError("API_KEY not set")
    return key

def test_get_api_key_success(monkeypatch):
    """monkeypatch is a built-in pytest fixture — auto-cleanup after test."""
    monkeypatch.setenv("API_KEY", "test-key-123")
    assert get_api_key() == "test-key-123"

def test_get_api_key_missing(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(ValueError):
        get_api_key()

def test_replace_function_with_monkeypatch(monkeypatch):
    """Replace a function entirely."""
    def fake_input(prompt):
        return "mocked answer"
    
    monkeypatch.setattr("builtins.input", fake_input)
    
    user_response = input("Are you sure?")
    assert user_response == "mocked answer"
```

---

## Part 5: Debugging Techniques

### Level 1: Print Debugging (Quick but Limited)

```python
def process_order(order):
    print(f"DEBUG: order = {order}")            # Quick and dirty
    total = sum(item["price"] for item in order["items"])
    print(f"DEBUG: total = {total}")
    discount = total * 0.1 if order.get("vip") else 0
    print(f"DEBUG: discount = {discount}")
    return total - discount

# Problems with print debugging:
# - Have to remove/comment out before production
# - No way to inspect state interactively
# - Clutters output, hard to find in large logs
# - Can't easily inspect complex nested objects
```

### Level 2: The `logging` Module (Production-Ready)

```python
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def process_order(order):
    logger.debug(f"Processing order: {order}")
    total = sum(item["price"] for item in order["items"])
    logger.info(f"Order total calculated: ${total:.2f}")
    
    if order.get("vip"):
        discount = total * 0.1
        logger.debug(f"VIP discount applied: ${discount:.2f}")
    else:
        discount = 0
    
    final = total - discount
    if final < 0:
        logger.warning(f"Final total is negative: ${final:.2f}")
    
    return final

# Advantages over print:
# - Can turn off/filter by level WITHOUT removing code
# - Goes to files, not just console
# - Includes timestamps, severity, module name automatically
# - Production-safe — leave it in!
```

### Level 3: The Python Debugger (`pdb`)

```python
import pdb

def calculate_total(items):
    total = 0
    for item in items:
        pdb.set_trace()    # ← EXECUTION PAUSES HERE
        total += item["price"] * item["quantity"]
    return total

# When execution hits pdb.set_trace(), you get an interactive prompt:
# (Pdb) 
# 
# Commands you can use:
# n (next)       — execute current line, move to next
# s (step)       — step INTO function calls
# c (continue)   — resume normal execution
# l (list)       — show code around current line
# p variable     — print a variable's value
# pp variable    — pretty-print a variable
# w (where)      — show call stack
# q (quit)       — exit debugger
# 
# Example session:
# (Pdb) p item
# {'price': 10, 'quantity': 3}
# (Pdb) p total
# 0
# (Pdb) n
# (Pdb) p total
# 30


# ── MODERN ALTERNATIVE: breakpoint() (Python 3.7+) ────────────────────────
def calculate_total_v2(items):
    total = 0
    for item in items:
        breakpoint()    # Same as pdb.set_trace(), but more discoverable
        total += item["price"] * item["quantity"]
    return total

# breakpoint() respects the PYTHONBREAKPOINT environment variable
# Set PYTHONBREAKPOINT=0 to disable ALL breakpoints without removing code!
```

### Level 4: Debugging in VS Code / PyCharm (Visual Debuggers)

```python
"""
Visual debuggers (built into VS Code, PyCharm) give you:
- Click to set breakpoints (no code changes needed!)
- Step through line by line with buttons
- Hover over any variable to see its value
- Watch expressions that update live
- Call stack navigation
- Conditional breakpoints (only stop if x > 100)

This is the MOST PRODUCTIVE way to debug for most developers.
Learn your IDE's debugger — it will save you hundreds of hours.

Key concepts that transfer across all debuggers:
- Breakpoint: a line where execution pauses
- Step Over: execute current line, don't go into function calls
- Step Into: execute current line, DO go into function calls
- Step Out: finish current function, return to caller
- Continue: resume until next breakpoint
- Watch: expression evaluated and shown live as you step
"""
```

### Debugging Strategy: The Scientific Method

```python
"""
A systematic approach to debugging (works regardless of tools used):

1. REPRODUCE — Can you make the bug happen reliably?
   - If it's intermittent, what conditions trigger it?
   
2. ISOLATE — Narrow down WHERE the problem is
   - Binary search: comment out half the code, does bug persist?
   - Add print/log statements at decreasing intervals
   
3. HYPOTHESIZE — What do you THINK is wrong?
   - Form a specific, testable theory
   - "I think the loop is running one extra time"
   
4. TEST — Verify or refute your hypothesis
   - Add a breakpoint/print exactly where you expect the issue
   - Check if reality matches your theory
   
5. FIX — Make the smallest change that solves the root cause
   - Don't just patch the symptom!
   
6. VERIFY — Confirm the fix actually works
   - Re-run the original failing case
   - Run the FULL test suite (your fix might break something else!)
   
7. PREVENT — Add a test so this bug can never silently return
"""

# Example: debugging a real bug step by step
def find_average(numbers):
    """BUG: crashes on empty list."""
    total = sum(numbers)
    return total / len(numbers)

# Step 1: Reproduce
# find_average([])  →  ZeroDivisionError: division by zero

# Step 2: Isolate — the bug is clearly in division, when len == 0

# Step 3: Hypothesize — "We need to handle the empty list case"

# Step 4: Test the hypothesis
def find_average_debug(numbers):
    print(f"DEBUG: numbers = {numbers}, len = {len(numbers)}")
    total = sum(numbers)
    print(f"DEBUG: total = {total}")
    return total / len(numbers)
# Confirmed: len(numbers) == 0 causes the crash

# Step 5: Fix the ROOT CAUSE
def find_average_fixed(numbers):
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
        # or: return 0  / return None  (depending on business requirements)
    return sum(numbers) / len(numbers)

# Step 6: Verify
assert find_average_fixed([1, 2, 3]) == 2.0
try:
    find_average_fixed([])
    print("ERROR: Should have raised!")
except ValueError:
    print("✓ Correctly raises on empty list")

# Step 7: Prevent — add a permanent test
def test_average_of_empty_list_raises():
    with pytest.raises(ValueError):
        find_average_fixed([])
```

### Using `traceback` to Understand Errors

```python
import traceback

def level_3():
    return 1 / 0   # The actual error

def level_2():
    return level_3()

def level_1():
    return level_2()

try:
    level_1()
except ZeroDivisionError:
    # print_exc() shows the FULL call stack — invaluable for debugging
    traceback.print_exc()
    
    # Output shows EXACTLY which function called which:
    # Traceback (most recent call last):
    #   File "...", line 10, in <module>
    #     level_1()
    #   File "...", line 8, in level_1
    #     return level_2()
    #   File "...", line 5, in level_2
    #     return level_3()
    #   File "...", line 2, in level_3
    #     return 1 / 0
    # ZeroDivisionError: division by zero

# Getting traceback as a STRING (for logging)
try:
    level_1()
except ZeroDivisionError:
    error_details = traceback.format_exc()
    logger.error(f"Calculation failed:\n{error_details}")
```

---

## Part 6: Code Style — PEP 8

### Why Style Consistency Matters

```python
# ❌ Inconsistent style — technically works, but painful to read/maintain
def CalculateTotal(Items,discountRate = 0.1,tax_Rate=0.08):
    Total=0
    for i in Items :
        Total+=i['price']*i['qty']
    discounted=Total-(Total*discountRate)
    Final =discounted+(discounted*tax_Rate)
    return(Final)

# ✓ PEP 8 compliant — consistent, predictable, readable
def calculate_total(items, discount_rate=0.1, tax_rate=0.08):
    total = 0
    for item in items:
        total += item["price"] * item["qty"]
    discounted = total - (total * discount_rate)
    final = discounted + (discounted * tax_rate)
    return final
```

### Key PEP 8 Rules

```python
# ── NAMING CONVENTIONS ────────────────────────────────────────────────────
variable_name = 10              # snake_case for variables
function_name = lambda: None    # snake_case for functions
CONSTANT_VALUE = 100             # UPPER_SNAKE_CASE for constants
class ClassName:                 # PascalCase for classes
    _internal_attribute = 1     # leading underscore = "internal use"
    __private_attribute = 2     # double underscore = name-mangled private

def function_with_many_params(a, b, c, *args, d=1, e=2, **kwargs):
    pass


# ── WHITESPACE RULES ───────────────────────────────────────────────────────
# 4 spaces per indentation level (never tabs!)
def example():
    if True:
        return 1

# Spaces around operators
x = 1 + 2          # not: x=1+2
y = a if b else c  # spaces around 'if' and 'else'

# NO space before function call parens or indexing
my_function(arg)    # not: my_function (arg)
my_list[0]          # not: my_list [0]

# One space after commas, none before
func(a, b, c)        # not: func(a,b , c)

# Two blank lines before top-level functions/classes
def first_function():
    pass


def second_function():
    pass


class MyClass:
    # One blank line between methods
    def method_one(self):
        pass

    def method_two(self):
        pass


# ── LINE LENGTH ───────────────────────────────────────────────────────────
# Max 79 characters (some teams use 88 or 100 — be consistent!)

# ❌ Too long
result = some_function(argument_one, argument_two, argument_three, argument_four, argument_five)

# ✓ Break across lines
result = some_function(
    argument_one, argument_two, argument_three,
    argument_four, argument_five
)


# ── IMPORTS ───────────────────────────────────────────────────────────────
# Order: standard library, then third-party, then local — separated by blank lines

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from myapp.models import User
from myapp.utils import format_date

# One import per line (for "import X" style)
import os
import sys
# NOT: import os, sys

# Multiple names OK for "from X import" style
from collections import defaultdict, Counter, OrderedDict


# ── COMPARISONS ───────────────────────────────────────────────────────────
# Use is / is not for None, True, False
if value is None:        # not: if value == None
    pass
if flag is not True:     # not: if flag != True  (also just use: if not flag)
    pass

# Use isinstance() for type checking
if isinstance(x, int):   # not: if type(x) == int
    pass
```

### Docstrings — PEP 257

```python
def calculate_compound_interest(principal, rate, years, compounds_per_year=12):
    """
    Calculate compound interest on an investment.
    
    Uses the standard compound interest formula:
    A = P(1 + r/n)^(nt)
    
    Args:
        principal (float): Initial investment amount.
        rate (float): Annual interest rate as a decimal (e.g., 0.05 for 5%).
        years (int): Number of years to invest.
        compounds_per_year (int, optional): Compounding frequency. Defaults to 12.
    
    Returns:
        float: Final amount after compound interest.
    
    Raises:
        ValueError: If principal or rate is negative.
    
    Examples:
        >>> calculate_compound_interest(1000, 0.05, 10)
        1647.01
    """
    if principal < 0 or rate < 0:
        raise ValueError("Principal and rate must be non-negative")
    
    n = compounds_per_year
    return round(principal * (1 + rate / n) ** (n * years), 2)


class BankAccount:
    """
    Represents a simple bank account with deposit/withdraw capability.
    
    Attributes:
        owner (str): Name of the account owner.
        balance (float): Current account balance.
    """
    
    def __init__(self, owner, balance=0):
        """
        Initialize a new bank account.
        
        Args:
            owner (str): Name of the account owner.
            balance (float, optional): Starting balance. Defaults to 0.
        """
        self.owner = owner
        self.balance = balance
```

### Type Hints — Modern Python Best Practice

```python
from typing import Optional, Union

# ── BASIC TYPE HINTS ───────────────────────────────────────────────────────
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

def calculate_average(numbers: list[float]) -> float:   # Python 3.9+
    return sum(numbers) / len(numbers)

# ── OPTIONAL AND UNION TYPES ──────────────────────────────────────────────
def find_user(user_id: int) -> Optional[dict]:
    """Returns a user dict, or None if not found."""
    pass

def process_input(value: Union[int, str]) -> str:
    """Accept int OR str."""
    return str(value)

# Python 3.10+ syntax (cleaner)
def process_input_modern(value: int | str) -> str:
    return str(value)

def find_user_modern(user_id: int) -> dict | None:
    pass

# ── COMPLEX TYPE HINTS ────────────────────────────────────────────────────
from typing import Callable

def apply_discount(
    items: list[dict[str, float]],
    discount_func: Callable[[float], float]
) -> list[dict[str, float]]:
    """Apply a discount function to each item's price."""
    return [
        {**item, "price": discount_func(item["price"])}
        for item in items
    ]

# ── TYPE HINTS FOR CLASSES ────────────────────────────────────────────────
class ShoppingCart:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, float]] = {}
    
    def add_item(self, name: str, price: float, quantity: int = 1) -> None:
        self.items[name] = {"price": price, "quantity": quantity}
    
    @property
    def total(self) -> float:
        return sum(item["price"] * item["quantity"] for item in self.items.values())

# ── WHY TYPE HINTS MATTER ─────────────────────────────────────────────────
# 1. Self-documenting — function signature tells you what it expects
# 2. IDE autocomplete becomes much smarter
# 3. Tools like mypy can catch type errors WITHOUT running the code
# 4. Easier onboarding for new team members
# Note: Python does NOT enforce these at runtime by default — they're hints!
```

---

## Part 7: Performance Optimization and Profiling

### Measuring Before Optimizing — "Don't Guess, Measure"

```python
import time
import timeit

# ── METHOD 1: Simple timing ───────────────────────────────────────────────
def slow_function():
    return sum(i ** 2 for i in range(1_000_000))

start = time.perf_counter()
result = slow_function()
elapsed = time.perf_counter() - start
print(f"Took {elapsed:.4f} seconds")


# ── METHOD 2: timeit — more accurate, runs multiple times ─────────────────
# Good for comparing small code snippets
time_taken = timeit.timeit(
    "sum(i**2 for i in range(1000))",
    number=10000   # Run 10,000 times, return TOTAL time
)
print(f"Average per run: {time_taken / 10000 * 1000:.4f} ms")

# Comparing two approaches
list_comp_time = timeit.timeit(
    "[i**2 for i in range(1000)]",
    number=10000
)
gen_exp_time = timeit.timeit(
    "sum(i**2 for i in range(1000))",
    number=10000
)
print(f"List comprehension: {list_comp_time:.4f}s")
print(f"Generator + sum:    {gen_exp_time:.4f}s")


# ── METHOD 3: cProfile — full function-by-function breakdown ─────────────
import cProfile
import pstats

def slow_data_pipeline():
    data = [i for i in range(100_000)]
    squared = [x ** 2 for x in data]
    filtered = [x for x in squared if x % 3 == 0]
    total = sum(filtered)
    return total

profiler = cProfile.Profile()
profiler.enable()
slow_data_pipeline()
profiler.disable()

stats = pstats.Stats(profiler).sort_stats("cumulative")
stats.print_stats(10)   # Top 10 most time-consuming function calls

# Or from command line:
# python -m cProfile -s cumulative my_script.py
```

### Common Performance Pitfalls and Fixes

```python
import time

def benchmark(name, func, *args, repeat=5):
    """Run func multiple times, report best time."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - start)
    print(f"{name:<40} best: {min(times)*1000:.2f}ms")
    return result

N = 1_000_000

# ── PITFALL 1: String concatenation in a loop ─────────────────────────────
def build_string_slow(n):
    """❌ SLOW — creates a new string object every iteration."""
    result = ""
    for i in range(n):
        result += str(i)
    return result

def build_string_fast(n):
    """✓ FAST — join creates the string ONCE."""
    return "".join(str(i) for i in range(n))

benchmark("String concat (slow)", build_string_slow, 100_000)
benchmark("String join (fast)",   build_string_fast, 100_000)
# Typical result: join is 10-50x faster for large strings


# ── PITFALL 2: Repeated lookups in a loop ─────────────────────────────────
class Config:
    def __init__(self):
        self.settings = {"key": "value"}

def access_slow(obj, n):
    """❌ SLOW — re-evaluates obj.settings every iteration."""
    total = 0
    for i in range(n):
        if "key" in obj.settings:   # Re-accesses obj.settings every time
            total += 1
    return total

def access_fast(obj, n):
    """✓ FAST — cache the lookup outside the loop."""
    settings = obj.settings   # Cached ONCE
    total = 0
    for i in range(n):
        if "key" in settings:
            total += 1
    return total

cfg = Config()
benchmark("Attribute lookup in loop (slow)", access_slow, cfg, N)
benchmark("Cached attribute (fast)",          access_fast, cfg, N)


# ── PITFALL 3: Using list when set/dict membership is needed ─────────────
def check_membership_list(items, lookups):
    """❌ SLOW — list membership is O(n) per check."""
    return sum(1 for x in lookups if x in items)   # items is a LIST

def check_membership_set(items, lookups):
    """✓ FAST — set membership is O(1) per check."""
    items_set = set(items)   # Convert ONCE
    return sum(1 for x in lookups if x in items_set)

big_list = list(range(100_000))
lookups = list(range(0, 100_000, 100))   # 1000 lookups

benchmark("List membership (slow)", check_membership_list, big_list, lookups)
benchmark("Set membership (fast)",  check_membership_set,  big_list, lookups)
# Typical result: set is 100-1000x faster for large collections


# ── PITFALL 4: Not using built-in functions ───────────────────────────────
def sum_manual(numbers):
    """❌ SLOWER — Python-level loop."""
    total = 0
    for n in numbers:
        total += n
    return total

def sum_builtin(numbers):
    """✓ FASTER — sum() is implemented in C."""
    return sum(numbers)

numbers = list(range(1_000_000))
benchmark("Manual sum loop (slow)", sum_manual, numbers)
benchmark("Built-in sum() (fast)",  sum_builtin, numbers)


# ── PITFALL 5: Creating unnecessary intermediate lists ───────────────────
def process_chain_slow(data):
    """❌ Creates 3 full intermediate lists."""
    step1 = [x * 2 for x in data]
    step2 = [x for x in step1 if x % 3 == 0]
    step3 = [x ** 2 for x in step2]
    return sum(step3)

def process_chain_fast(data):
    """✓ Generator pipeline — no intermediate lists."""
    step1 = (x * 2 for x in data)
    step2 = (x for x in step1 if x % 3 == 0)
    step3 = (x ** 2 for x in step2)
    return sum(step3)

data = list(range(500_000))
benchmark("List chain (more memory)",  process_chain_slow, data)
benchmark("Generator chain (less memory)", process_chain_fast, data)
# Speed is similar, but generator version uses FAR less memory


# ── PITFALL 6: Using + for many small mutations on a list ────────────────
def grow_list_slow(n):
    """❌ Creates a new list every += !"""
    result = []
    for i in range(n):
        result = result + [i]    # NEW list created each time — O(n²) total!
    return result

def grow_list_fast(n):
    """✓ append() mutates in place — O(n) total."""
    result = []
    for i in range(n):
        result.append(i)
    return result

benchmark("List + (very slow)", grow_list_slow, 10_000)
benchmark("List.append (fast)", grow_list_fast, 10_000)
# At larger N, the slow version becomes catastrophically slower (quadratic!)
```

### When NOT to Optimize

```python
"""
THE GOLDEN RULE OF OPTIMIZATION:
"Premature optimization is the root of all evil." — Donald Knuth

A practical checklist before optimizing:
1. Is this code ACTUALLY slow? (measure, don't guess!)
2. Is this the BOTTLENECK? (profile to find the real hot path)
3. Does the optimization make the code significantly harder to read?
4. Is the performance gain WORTH the readability cost?

The 80/20 rule almost always applies:
- 80% of runtime is usually in 20% of the code
- Optimizing the OTHER 80% of code wastes your time and adds risk

WORKFLOW:
1. Write clear, correct code first
2. Write tests so you can refactor safely
3. Profile to find ACTUAL bottlenecks  
4. Optimize ONLY the bottleneck
5. Re-profile to confirm the improvement
6. Re-run tests to confirm correctness wasn't broken
"""

import cProfile

def find_bottleneck_example():
    """A realistic mixed workload — where's the actual bottleneck?"""
    # Fast part — runs once
    config = {"multiplier": 2}
    
    # SLOW part — this is the actual bottleneck (string ops in a loop)
    results = []
    for i in range(50_000):
        label = "item_" + str(i) + "_processed"   # Slow-ish but not the worst
        results.append(label)
    
    # FAST part — vectorizable-style operation
    total = sum(range(50_000)) * config["multiplier"]
    
    return results, total

cProfile.run("find_bottleneck_example()", sort="cumulative")
# The profile output tells you EXACTLY where time goes —
# don't optimize the parts that are already fast!
```

---

## Part 8: Worked Examples

### Worked Example 1: Test-Driven Development (TDD) Walkthrough

```python
"""
TDD Cycle: Red → Green → Refactor
1. RED:      Write a failing test for behavior that doesn't exist yet
2. GREEN:    Write the MINIMUM code to make it pass
3. REFACTOR: Clean up the code while keeping tests green
"""

# ── STEP 1: RED — write the test FIRST ────────────────────────────────────
import pytest

def test_is_palindrome_simple_word():
    assert is_palindrome("racecar") is True

# Run pytest now → FAILS (NameError: is_palindrome doesn't exist)


# ── STEP 2: GREEN — simplest implementation that passes ──────────────────
def is_palindrome(text):
    return text == text[::-1]

# Run pytest now → PASSES!


# ── STEP 3: Add MORE test cases (RED again) ───────────────────────────────
def test_is_palindrome_not_palindrome():
    assert is_palindrome("hello") is False

def test_is_palindrome_with_spaces():
    assert is_palindrome("race car") is True   # FAILS with current implementation!

# This new test FAILS — current implementation doesn't handle spaces


# ── STEP 4: GREEN again — fix to handle the new case ──────────────────────
def is_palindrome(text):
    cleaned = text.replace(" ", "")
    return cleaned == cleaned[::-1]

# Now both tests pass


# ── STEP 5: Add MORE edge cases ────────────────────────────────────────────
def test_is_palindrome_with_punctuation_and_case():
    assert is_palindrome("A man, a plan, a canal: Panama") is True   # FAILS!

def test_is_palindrome_empty_string():
    assert is_palindrome("") is True

def test_is_palindrome_single_char():
    assert is_palindrome("a") is True


# ── STEP 6: GREEN — handle punctuation and case ───────────────────────────
def is_palindrome(text):
    cleaned = "".join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]


# ── STEP 7: REFACTOR — clean up, still passing all tests ─────────────────
def is_palindrome(text: str) -> bool:
    """
    Check if a string is a palindrome, ignoring case, spaces, and punctuation.
    
    Args:
        text: The string to check.
    
    Returns:
        True if text reads the same forwards and backwards.
    """
    cleaned = "".join(char.lower() for char in text if char.isalnum())
    return cleaned == cleaned[::-1]

# Final test suite — ALL must pass
def test_all_palindrome_cases():
    assert is_palindrome("racecar") is True
    assert is_palindrome("hello") is False
    assert is_palindrome("race car") is True
    assert is_palindrome("A man, a plan, a canal: Panama") is True
    assert is_palindrome("") is True
    assert is_palindrome("a") is True
```

### Worked Example 2: Complete Test Suite for a Real Module

```python
"""
Module: inventory.py — testing a more complex, realistic module
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Product:
    sku: str
    name: str
    price: float
    stock: int = 0
    
    def __post_init__(self):
        if self.price < 0:
            raise ValueError(f"Price cannot be negative: {self.price}")
        if self.stock < 0:
            raise ValueError(f"Stock cannot be negative: {self.stock}")


class InventoryError(Exception):
    """Base exception for inventory operations."""
    pass

class InsufficientStockError(InventoryError):
    def __init__(self, sku, requested, available):
        self.sku = sku
        self.requested = requested
        self.available = available
        super().__init__(
            f"Insufficient stock for {sku}: requested {requested}, have {available}"
        )

class ProductNotFoundError(InventoryError):
    def __init__(self, sku):
        self.sku = sku
        super().__init__(f"Product not found: {sku}")


class Inventory:
    def __init__(self):
        self._products: dict[str, Product] = {}
    
    def add_product(self, product: Product) -> None:
        if product.sku in self._products:
            raise ValueError(f"Product {product.sku} already exists")
        self._products[product.sku] = product
    
    def get_product(self, sku: str) -> Product:
        if sku not in self._products:
            raise ProductNotFoundError(sku)
        return self._products[sku]
    
    def restock(self, sku: str, quantity: int) -> int:
        if quantity <= 0:
            raise ValueError("Restock quantity must be positive")
        product = self.get_product(sku)
        product.stock += quantity
        return product.stock
    
    def reserve(self, sku: str, quantity: int) -> int:
        if quantity <= 0:
            raise ValueError("Reserve quantity must be positive")
        product = self.get_product(sku)
        if product.stock < quantity:
            raise InsufficientStockError(sku, quantity, product.stock)
        product.stock -= quantity
        return product.stock
    
    def total_value(self) -> float:
        return sum(p.price * p.stock for p in self._products.values())
    
    def low_stock_products(self, threshold: int = 10) -> list[Product]:
        return [p for p in self._products.values() if p.stock < threshold]


# ═══════════════════════════════════════════════════════════════════════
# TEST SUITE
# ═══════════════════════════════════════════════════════════════════════

import pytest

# ── FIXTURES ────────────────────────────────────────────────────────────
@pytest.fixture
def empty_inventory():
    return Inventory()

@pytest.fixture
def sample_inventory():
    inv = Inventory()
    inv.add_product(Product("SKU001", "Laptop", 999.99, stock=10))
    inv.add_product(Product("SKU002", "Mouse",   29.99, stock=50))
    inv.add_product(Product("SKU003", "Monitor", 299.99, stock=5))
    return inv


# ── PRODUCT DATACLASS TESTS ────────────────────────────────────────────
class TestProduct:
    def test_create_valid_product(self):
        p = Product("SKU001", "Widget", 19.99, stock=100)
        assert p.sku == "SKU001"
        assert p.price == 19.99
    
    def test_negative_price_raises(self):
        with pytest.raises(ValueError, match="Price cannot be negative"):
            Product("SKU001", "Widget", -10.0)
    
    def test_negative_stock_raises(self):
        with pytest.raises(ValueError, match="Stock cannot be negative"):
            Product("SKU001", "Widget", 10.0, stock=-5)
    
    def test_default_stock_is_zero(self):
        p = Product("SKU001", "Widget", 10.0)
        assert p.stock == 0


# ── INVENTORY: ADD/GET TESTS ───────────────────────────────────────────
class TestInventoryAddGet:
    def test_add_product_succeeds(self, empty_inventory):
        empty_inventory.add_product(Product("SKU001", "Widget", 10.0))
        product = empty_inventory.get_product("SKU001")
        assert product.name == "Widget"
    
    def test_add_duplicate_sku_raises(self, sample_inventory):
        with pytest.raises(ValueError, match="already exists"):
            sample_inventory.add_product(Product("SKU001", "Duplicate", 5.0))
    
    def test_get_nonexistent_product_raises(self, empty_inventory):
        with pytest.raises(ProductNotFoundError) as exc_info:
            empty_inventory.get_product("MISSING")
        assert exc_info.value.sku == "MISSING"


# ── INVENTORY: RESTOCK TESTS ───────────────────────────────────────────
class TestInventoryRestock:
    def test_restock_increases_stock(self, sample_inventory):
        new_stock = sample_inventory.restock("SKU001", 5)
        assert new_stock == 15
    
    def test_restock_zero_raises(self, sample_inventory):
        with pytest.raises(ValueError, match="must be positive"):
            sample_inventory.restock("SKU001", 0)
    
    def test_restock_negative_raises(self, sample_inventory):
        with pytest.raises(ValueError):
            sample_inventory.restock("SKU001", -5)
    
    def test_restock_nonexistent_product_raises(self, sample_inventory):
        with pytest.raises(ProductNotFoundError):
            sample_inventory.restock("MISSING", 10)


# ── INVENTORY: RESERVE TESTS ───────────────────────────────────────────
class TestInventoryReserve:
    def test_reserve_decreases_stock(self, sample_inventory):
        new_stock = sample_inventory.reserve("SKU001", 3)
        assert new_stock == 7
    
    def test_reserve_exact_stock_succeeds(self, sample_inventory):
        """Edge case: reserving EXACTLY the available amount."""
        new_stock = sample_inventory.reserve("SKU003", 5)
        assert new_stock == 0
    
    def test_reserve_more_than_available_raises(self, sample_inventory):
        with pytest.raises(InsufficientStockError) as exc_info:
            sample_inventory.reserve("SKU003", 100)
        assert exc_info.value.requested == 100
        assert exc_info.value.available == 5
    
    @pytest.mark.parametrize("quantity", [0, -1, -100])
    def test_reserve_non_positive_raises(self, sample_inventory, quantity):
        with pytest.raises(ValueError, match="must be positive"):
            sample_inventory.reserve("SKU001", quantity)


# ── INVENTORY: ANALYTICS TESTS ─────────────────────────────────────────
class TestInventoryAnalytics:
    def test_total_value_calculation(self, sample_inventory):
        # 999.99*10 + 29.99*50 + 299.99*5 = 9999.90 + 1499.50 + 1499.95
        expected = 9999.90 + 1499.50 + 1499.95
        assert sample_inventory.total_value() == pytest.approx(expected, abs=0.01)
    
    def test_empty_inventory_total_value_is_zero(self, empty_inventory):
        assert empty_inventory.total_value() == 0
    
    def test_low_stock_products_default_threshold(self, sample_inventory):
        low_stock = sample_inventory.low_stock_products()
        skus = {p.sku for p in low_stock}
        assert skus == {"SKU003"}   # Only Monitor has stock < 10
    
    def test_low_stock_products_custom_threshold(self, sample_inventory):
        low_stock = sample_inventory.low_stock_products(threshold=60)
        skus = {p.sku for p in low_stock}
        assert skus == {"SKU001", "SKU002", "SKU003"}   # All under 60


# ── INTEGRATION-STYLE TEST: realistic workflow ─────────────────────────
class TestInventoryWorkflow:
    def test_complete_order_fulfillment_workflow(self, sample_inventory):
        """Test a realistic sequence of operations."""
        # Customer orders 3 laptops
        sample_inventory.reserve("SKU001", 3)
        assert sample_inventory.get_product("SKU001").stock == 7
        
        # Warehouse restocks
        sample_inventory.restock("SKU001", 20)
        assert sample_inventory.get_product("SKU001").stock == 27
        
        # Large order comes in
        sample_inventory.reserve("SKU001", 25)
        assert sample_inventory.get_product("SKU001").stock == 2
        
        # Now it's low stock
        low = sample_inventory.low_stock_products(threshold=5)
        assert any(p.sku == "SKU001" for p in low)
```

### Worked Example 3: Debugging a Real Bug Step-by-Step

```python
"""
Scenario: A reporting function is producing wrong totals in production.
Let's debug it systematically.
"""

# ── THE BUGGY CODE ─────────────────────────────────────────────────────────
def calculate_monthly_revenue(transactions):
    """
    BUG REPORT: "Monthly totals are higher than they should be"
    """
    monthly_totals = {}
    for transaction in transactions:
        month = transaction["date"][:7]   # "2025-01"
        if month not in monthly_totals:
            monthly_totals[month] = 0
        monthly_totals[month] += transaction["amount"]
    return monthly_totals


# ── STEP 1: REPRODUCE with a minimal test case ────────────────────────────
test_transactions = [
    {"date": "2025-01-15", "amount": 100, "type": "sale"},
    {"date": "2025-01-16", "amount": -20, "type": "refund"},
    {"date": "2025-01-17", "amount": 50, "type": "sale"},
]

result = calculate_monthly_revenue(test_transactions)
print(result)   # {'2025-01': 130}

# Wait — is this actually wrong? Let's verify the EXPECTED behavior.
# 100 (sale) - 20 (refund) + 50 (sale) = 130
# This looks... correct? Let's get more realistic data.


# ── STEP 2: Get the ACTUAL problematic data and isolate ──────────────────
production_like_data = [
    {"date": "2025-01-15", "amount": 100, "type": "sale"},
    {"date": "2025-01-15", "amount": 100, "type": "sale"},  # Duplicate?!
    {"date": "2025-01-16", "amount": -20, "type": "refund"},
    {"date": "2025-02-01", "amount": 50, "type": "sale"},   # Different month!
]

result = calculate_monthly_revenue(production_like_data)
print(result)   # {'2025-01': 180, '2025-02': 50}

# Hypothesis 1: Are there DUPLICATE transactions being double-counted?
# Let's check for duplicates
def find_duplicates(transactions):
    seen = set()
    duplicates = []
    for t in transactions:
        key = (t["date"], t["amount"], t["type"])
        if key in seen:
            duplicates.append(t)
        seen.add(key)
    return duplicates

dupes = find_duplicates(production_like_data)
print(f"Found {len(dupes)} duplicate transactions: {dupes}")
# Found 1 duplicate transactions — THIS is likely the real bug source,
# but it's an UPSTREAM data quality issue, not a code bug in this function!


# ── STEP 3: Hypothesize the REAL root cause ───────────────────────────────
# Maybe the data pipeline is fetching transactions twice (e.g., pagination bug)
# That's outside this function's scope — but we should defend against it!


# ── STEP 4: Fix — make the function ROBUST to bad input ───────────────────
def calculate_monthly_revenue_fixed(transactions):
    """
    Calculate monthly revenue, deduplicating identical transactions
    (same date + amount + type combination) which may occur due to
    upstream double-counting from pagination retries.
    """
    seen_transactions = set()
    monthly_totals = {}
    
    for transaction in transactions:
        # Create a unique key — if your data has transaction IDs, use those instead!
        key = (transaction["date"], transaction["amount"], transaction["type"])
        if key in seen_transactions:
            continue   # Skip duplicate
        seen_transactions.add(key)
        
        month = transaction["date"][:7]
        monthly_totals[month] = monthly_totals.get(month, 0) + transaction["amount"]
    
    return monthly_totals


# ── STEP 5: Verify the fix ─────────────────────────────────────────────────
result = calculate_monthly_revenue_fixed(production_like_data)
print(result)   # {'2025-01': 80, '2025-02': 50}
# 100 (sale, deduplicated) - 20 (refund) = 80  ✓ Correct now!


# ── STEP 6: Add permanent regression tests ────────────────────────────────
def test_duplicate_transactions_not_double_counted():
    transactions = [
        {"date": "2025-01-15", "amount": 100, "type": "sale"},
        {"date": "2025-01-15", "amount": 100, "type": "sale"},  # exact duplicate
    ]
    result = calculate_monthly_revenue_fixed(transactions)
    assert result == {"2025-01": 100}   # Counted ONCE, not twice

def test_legitimate_same_day_different_amounts_both_counted():
    """Make sure we don't OVER-deduplicate legitimate transactions!"""
    transactions = [
        {"date": "2025-01-15", "amount": 100, "type": "sale"},
        {"date": "2025-01-15", "amount": 150, "type": "sale"},  # different amount
    ]
    result = calculate_monthly_revenue_fixed(transactions)
    assert result == {"2025-01": 250}   # Both counted — they're different!

def test_normal_transactions_across_months():
    transactions = [
        {"date": "2025-01-15", "amount": 100, "type": "sale"},
        {"date": "2025-02-01", "amount": 50, "type": "sale"},
    ]
    result = calculate_monthly_revenue_fixed(transactions)
    assert result == {"2025-01": 100, "2025-02": 50}
```

---

## Part 9: Common Mistakes and How to Avoid Them

```python
import pytest

# ── MISTAKE 1: Testing implementation instead of behavior ─────────────────
class Calculator:
    def __init__(self):
        self._history = []
    def add(self, a, b):
        result = a + b
        self._history.append(result)
        return result

# ❌ WRONG — tests internal implementation detail (_history list)
def test_add_bad():
    calc = Calculator()
    calc.add(2, 3)
    assert calc._history == [5]   # Brittle! Breaks if internals change

# ✓ CORRECT — tests observable BEHAVIOR, not internals
def test_add_good():
    calc = Calculator()
    result = calc.add(2, 3)
    assert result == 5


# ── MISTAKE 2: One test checking too many things ──────────────────────────
# ❌ WRONG — if this fails, which assertion failed? Hard to tell quickly.
def test_everything_at_once():
    cart = ShoppingCart()
    cart.add_item("Apple", 1.0, 2)
    assert cart.total == 2.0
    cart.add_item("Bread", 3.0, 1)
    assert cart.total == 5.0
    cart.remove_item("Apple")
    assert cart.total == 3.0
    assert cart.item_count == 1

# ✓ CORRECT — separate, focused tests with clear names
def test_add_single_item_updates_total():
    cart = ShoppingCart()
    cart.add_item("Apple", 1.0, 2)
    assert cart.total == 2.0

def test_add_multiple_items_sums_total():
    cart = ShoppingCart()
    cart.add_item("Apple", 1.0, 2)
    cart.add_item("Bread", 3.0, 1)
    assert cart.total == 5.0

def test_remove_item_updates_total():
    cart = ShoppingCart()
    cart.add_item("Apple", 1.0, 2)
    cart.add_item("Bread", 3.0, 1)
    cart.remove_item("Apple")
    assert cart.total == 3.0


# ── MISTAKE 3: Tests that depend on each other (order matters) ────────────
# ❌ WRONG — test_2 depends on test_1 having run first!
class TestBadOrder:
    shared_list = []   # Module/class-level state — DANGEROUS
    
    def test_1_add_item(self):
        self.shared_list.append(1)
        assert len(self.shared_list) == 1
    
    def test_2_check_item(self):
        # Breaks if run alone, or if test order changes!
        assert self.shared_list == [1]

# ✓ CORRECT — each test is completely independent
class TestGoodIndependence:
    def test_add_item(self):
        my_list = []
        my_list.append(1)
        assert len(my_list) == 1
    
    def test_check_item(self):
        my_list = [1]   # Set up its OWN state
        assert my_list == [1]


# ── MISTAKE 4: Not testing edge cases ──────────────────────────────────────
def divide(a, b):
    return a / b

# ❌ INCOMPLETE — only tests the "happy path"
def test_divide_incomplete():
    assert divide(10, 2) == 5

# ✓ COMPLETE — tests happy path AND edge cases
def test_divide_normal_case():
    assert divide(10, 2) == 5

def test_divide_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

def test_divide_negative_numbers():
    assert divide(-10, 2) == -5

def test_divide_resulting_in_float():
    assert divide(1, 3) == pytest.approx(0.333, abs=0.01)


# ── MISTAKE 5: Over-mocking (mocking everything, testing nothing real) ───
# ❌ WRONG — mocked so much that the test doesn't verify real behavior
def test_overcooked_mock():
    mock_calc = Mock()
    mock_calc.add.return_value = 5
    assert mock_calc.add(2, 3) == 5
    # This ONLY tests that Mock works — tells you NOTHING about your code!

# ✓ CORRECT — mock only EXTERNAL dependencies, test real logic
def test_real_logic_with_mocked_dependency(monkeypatch):
    def fake_fetch_exchange_rate():
        return 83.5   # Mock the EXTERNAL API call only
    
    monkeypatch.setattr("currency.fetch_exchange_rate", fake_fetch_exchange_rate)
    
    # The REAL conversion logic still runs and is tested!
    result = convert_usd_to_inr(100)
    assert result == 8350.0


# ── MISTAKE 6: Catching the debugger's own breakpoint accidentally ───────
# ❌ WRONG — leaving a breakpoint() in committed code halts CI/CD forever!
def process_data(data):
    breakpoint()   # Oops — forgot to remove before committing!
    return data * 2

# ✓ Always grep for breakpoint() and pdb.set_trace() before committing
# Many teams add a pre-commit hook to catch this automatically
```

---

## Practice Problems

### Easy (1–5)

**Problem 1: Write Tests for a Validator**
Write a comprehensive test suite for this function.

```python
def is_valid_age(age):
    """Returns True if age is a reasonable human age (0-150)."""
    return isinstance(age, int) and 0 <= age <= 150

# Write tests covering:
# - Valid ages (0, 50, 150)
# - Invalid ages (negative, over 150)
# - Wrong types (string, float, None)
# - Boundary values (exactly 0, exactly 150)
```

**Problem 2: Fix the Failing Test**
This test is failing. Find and fix the bug — in the FUNCTION, not the test.

```python
def average(numbers):
    return sum(numbers) // len(numbers)   # Bug is here!

def test_average():
    assert average([1, 2, 3, 4]) == 2.5
```

**Problem 3: Add Type Hints**
Add appropriate type hints to this function.

```python
def process_orders(orders, discount=0, free_shipping_threshold=50):
    total = 0
    for order in orders:
        total += order["price"]
    if total > free_shipping_threshold:
        shipping = 0
    else:
        shipping = 5.99
    return total * (1 - discount) + shipping
```

**Problem 4: Write a Fixture**
Create a pytest fixture for testing a `TodoList` class.

```python
class TodoList:
    def __init__(self):
        self.items = []
    def add(self, text):
        self.items.append({"text": text, "done": False})
    def complete(self, index):
        self.items[index]["done"] = True

# Write a fixture called `todo_list_with_items` that provides
# a TodoList with 3 pre-added items, then write 2 tests using it.
```

**Problem 5: Debug This Code**
Use print debugging or pdb to find why this function returns wrong results.

```python
def find_max_difference(numbers):
    """Should find the maximum difference between any two numbers."""
    max_diff = 0
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            diff = numbers[i] - numbers[j]
            if diff > max_diff:
                max_diff = diff
    return max_diff

# Bug: find_max_difference([1, 5, 3, 9, 2]) — does it give the right answer?
# What if all numbers are negative? Debug and fix.
```

### Medium (6–12)

**Problem 6: Parametrized Test Suite**
Write a parametrized test suite for a `fizzbuzz` function.

```python
def fizzbuzz(n):
    if n % 15 == 0: return "FizzBuzz"
    if n % 3 == 0: return "Fizz"
    if n % 5 == 0: return "Buzz"
    return str(n)

# Write parametrized tests covering:
# - Multiples of 3 only
# - Multiples of 5 only
# - Multiples of both (15, 30, 45)
# - Numbers that are neither
# - At least 10 total test cases via parametrize
```

**Problem 7: Mock an External API**
Write tests for this function using mocking — no real network calls allowed.

```python
import requests

def get_user_country(user_id):
    response = requests.get(f"https://api.example.com/users/{user_id}")
    if response.status_code == 404:
        return None
    data = response.json()
    return data.get("country")

# Test cases to cover:
# - Successful response with country
# - 404 response (user not found)
# - Response missing 'country' field
# - Network error (ConnectionError)
```

**Problem 8: Profile and Optimize**
This function is slow. Profile it, find the bottleneck, and optimize it.

```python
def find_common_elements(list1, list2):
    """Find elements that appear in both lists."""
    common = []
    for item in list1:
        if item in list2:        # O(n) lookup — the likely bottleneck
            common.append(item)
    return common

# Test with: list1 = list(range(10000)), list2 = list(range(5000, 15000))
# Profile it, then write an optimized version. Benchmark both.
```

**Problem 9: Test a Class with Dependencies**
Write a complete test suite for this class using fixtures and mocks.

```python
class EmailService:
    def __init__(self, smtp_client):
        self.smtp_client = smtp_client
    
    def send_welcome_email(self, user_email, user_name):
        if "@" not in user_email:
            raise ValueError("Invalid email address")
        subject = f"Welcome, {user_name}!"
        body = f"Hi {user_name}, thanks for joining!"
        return self.smtp_client.send(user_email, subject, body)

# Write tests using a MOCK smtp_client (don't send real emails!)
# Cover: successful send, invalid email raises, smtp_client called correctly
```

**Problem 10: Write a Custom pytest Fixture with Teardown**
Create a fixture that simulates a temporary file-based database.

```python
import json
import os

# Write a fixture `temp_json_db` that:
# 1. Creates a temp JSON file with some seed data
# 2. Yields the file path to the test
# 3. Deletes the file after the test completes (even on failure!)

def test_read_from_temp_db(temp_json_db):
    with open(temp_json_db) as f:
        data = json.load(f)
    assert "users" in data
```

**Problem 11: Refactor for Testability**
This function is hard to test because it mixes I/O with logic. Refactor it.

```python
import datetime

def generate_report():
    """Hard to test: depends on current time and prints directly."""
    now = datetime.datetime.now()
    report = f"Report generated at {now}\n"
    report += f"Day of week: {now.strftime('%A')}\n"
    if now.weekday() >= 5:
        report += "Note: Generated on a weekend\n"
    print(report)
    return report

# Refactor so the TIME can be injected (dependency injection)
# Then write tests for weekday vs weekend generation, without
# depending on what day it actually is when tests run!
```

**Problem 12: PEP 8 Cleanup**
Rewrite this code to be fully PEP 8 compliant.

```python
import os,sys
def CalculatePrice( items,TaxRate = 0.08 ,Discount=0):
    Total=0
    for I in items:
        Total+=I['price']*I['qty']
    Total=Total-(Total*Discount)
    Final=Total+(Total*TaxRate)
    return( Final )
class shoppingCart :
  def __init__(self):
    self.Items=[]
  def AddItem(self,Name,Price):
      self.Items.append({'name':Name,'price':Price})
```

### Hard (13–20)

**Problem 13: Property-Based Testing**
Use the `hypothesis` library to write property-based tests.

```python
# pip install hypothesis
from hypothesis import given, strategies as st

def reverse_string(s):
    return s[::-1]

# Write property-based tests verifying:
# 1. Reversing twice returns the original string
# 2. Reversed string has the same length as original
# 3. Reversing preserves character frequency (Counter equality)
# Use @given(st.text()) to generate random test strings
```

**Problem 14: Integration Test with Test Database**
Design and implement an integration test for a simple data layer.

```python
import sqlite3

class UserRepository:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL
            )
        """)
    
    def add_user(self, name, email):
        cursor = self.conn.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)", (name, email)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def get_user(self, user_id):
        row = self.conn.execute(
            "SELECT id, name, email FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return {"id": row[0], "name": row[1], "email": row[2]} if row else None

# Write integration tests using an IN-MEMORY database (":memory:")
# Test: add + retrieve, duplicate email constraint, get nonexistent user
```

**Problem 15: Benchmark Suite**
Build a comprehensive benchmark comparing 4 different ways to deduplicate a list while preserving order.

```python
def dedupe_v1(items):
    """Using a list and 'in' check."""
    pass

def dedupe_v2(items):
    """Using dict.fromkeys (Python 3.7+ preserves order)."""
    pass

def dedupe_v3(items):
    """Using a set to track seen items."""
    pass

def dedupe_v4(items):
    """Using itertools and a generator."""
    pass

# Write a benchmark script that:
# 1. Tests correctness of all 4 (same output for same input)
# 2. Times all 4 on lists of size 100, 10_000, 1_000_000
# 3. Reports which is fastest at each scale, and WHY
```

**Problem 16: Mutation Testing Concept**
Implement a simple "mutation tester" to evaluate test quality.

```python
def mutate_and_test(source_function_code, test_function, mutations):
    """
    Apply each mutation to source code, re-run tests.
    If tests STILL pass after a mutation, that's a "surviving mutant" —
    it means your tests aren't strong enough to catch that bug!
    
    mutations: list of (old_str, new_str) pairs representing small code changes
       e.g., ("a + b", "a - b")  — flip an operator
             (">", ">=")          — flip a comparison
    
    Return: mutation score (% of mutations that were CAUGHT/killed by tests)
    """
    pass

# Test this concept on a simple function + its test suite
# A good test suite should catch (kill) most mutations!
```

**Problem 17: Test Coverage Analysis**
Write code and tests, then analyze and improve coverage.

```python
# pip install pytest-cov

def categorize_transaction(amount, category, is_recurring=False):
    if amount < 0:
        if is_recurring:
            return "recurring_expense"
        return "expense"
    else:
        if is_recurring:
            return "recurring_income"
        if category == "investment":
            return "investment_income"
        return "income"

# 1. Write an INCOMPLETE test suite (deliberately miss some branches)
# 2. Run: pytest --cov=. --cov-report=term-missing
# 3. Identify which lines/branches are NOT covered
# 4. Add tests until you reach 100% branch coverage
```

**Problem 18: Custom Pytest Plugin/Marker**
Create a custom pytest marker that retries flaky tests automatically.

```python
import pytest
import random

# Implement a pytest marker @pytest.mark.flaky(retries=3)
# that automatically re-runs a test up to `retries` times
# if it fails, before reporting a final failure.
# (Hint: use a pytest hook like pytest_runtest_protocol,
#  or a simpler decorator-based approach)

@pytest.mark.flaky(retries=3)
def test_sometimes_fails():
    assert random.random() > 0.3   # Fails ~30% of the time
```

**Problem 19: Performance Regression Test**
Write a test that fails if a function becomes too slow (performance regression).

```python
import time

def search_algorithm(data, target):
    """Should be O(log n) — binary search on sorted data."""
    pass

def test_search_performance_does_not_regress():
    """
    Generate a large sorted dataset.
    Assert that search completes within a reasonable time bound.
    This catches accidental O(n) or O(n²) regressions in what
    should be an O(log n) algorithm.
    """
    pass
```

**Problem 20: Full CI-Ready Test Suite**
Design a complete testing setup for a small project, ready for CI/CD.

```python
"""
Project: A simple URL shortener service.

class URLShortener:
    def shorten(self, long_url: str) -> str: ...
    def expand(self, short_code: str) -> str: ...
    def get_click_count(self, short_code: str) -> int: ...
    def record_click(self, short_code: str) -> None: ...

Design and implement:
1. The URLShortener class with reasonable validation
2. A complete pytest test suite (unit + integration style)
3. Fixtures for common setup
4. Parametrized tests for URL validation edge cases
5. A pytest.ini or pyproject.toml config with markers
6. A simple GitHub Actions YAML snippet that would run: pytest --cov
"""
```

---

## Answer Keys (Selected Problems)

### Problem 2 Solution:
```python
def average(numbers):
    return sum(numbers) / len(numbers)   # Fixed: / instead of //

def test_average():
    assert average([1, 2, 3, 4]) == 2.5   # Now passes
```

### Problem 3 Solution:
```python
def process_orders(
    orders: list[dict[str, float]],
    discount: float = 0,
    free_shipping_threshold: float = 50
) -> float:
    total = 0.0
    for order in orders:
        total += order["price"]
    if total > free_shipping_threshold:
        shipping = 0.0
    else:
        shipping = 5.99
    return total * (1 - discount) + shipping
```

### Problem 6 Solution:
```python
import pytest

def fizzbuzz(n):
    if n % 15 == 0: return "FizzBuzz"
    if n % 3 == 0: return "Fizz"
    if n % 5 == 0: return "Buzz"
    return str(n)

@pytest.mark.parametrize("n,expected", [
    (3, "Fizz"), (6, "Fizz"), (9, "Fizz"),
    (5, "Buzz"), (10, "Buzz"), (20, "Buzz"),
    (15, "FizzBuzz"), (30, "FizzBuzz"), (45, "FizzBuzz"),
    (1, "1"), (2, "2"), (7, "7"), (11, "11"),
])
def test_fizzbuzz(n, expected):
    assert fizzbuzz(n) == expected
```

### Problem 11 Solution:
```python
import datetime

def generate_report(now: datetime.datetime) -> str:
    """Now accepts time as a parameter — testable without mocking!"""
    report = f"Report generated at {now}\n"
    report += f"Day of week: {now.strftime('%A')}\n"
    if now.weekday() >= 5:
        report += "Note: Generated on a weekend\n"
    return report

def test_generate_report_on_weekday():
    monday = datetime.datetime(2025, 1, 13)  # A Monday
    report = generate_report(monday)
    assert "weekend" not in report

def test_generate_report_on_weekend():
    saturday = datetime.datetime(2025, 1, 18)  # A Saturday
    report = generate_report(saturday)
    assert "weekend" in report

# Caller decides the real time:
# generate_report(datetime.datetime.now())
```

---

## Mini-Project: Complete Testing & Quality Toolkit

```python
"""
quality_toolkit.py
A complete library + test suite demonstrating professional testing,
debugging utilities, and code quality practices — all in one project.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from functools import wraps
import time
import logging

# ── LOGGING SETUP ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── CUSTOM EXCEPTIONS ────────────────────────────────────────────────────
class ValidationError(Exception):
    """Raised when input data fails validation."""
    pass


class TaskNotFoundError(Exception):
    """Raised when a requested task doesn't exist."""
    def __init__(self, task_id: int):
        self.task_id = task_id
        super().__init__(f"Task {task_id} not found")


# ── DEBUGGING UTILITIES ──────────────────────────────────────────────────
def timed(func: Callable) -> Callable:
    """Decorator: log execution time of any function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed*1000:.2f}ms")
        return result
    return wrapper


def validate_inputs(**validators):
    """
    Decorator factory for input validation.
    Usage: @validate_inputs(age=lambda x: x >= 0)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValidationError(
                            f"Validation failed for '{param_name}': {value!r}"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ── MAIN APPLICATION CODE: Task Manager ───────────────────────────────────
@dataclass
class Task:
    id: int
    title: str
    priority: int = 2          # 1=high, 2=medium, 3=low
    completed: bool = False
    tags: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.title.strip():
            raise ValidationError("Task title cannot be empty")
        if self.priority not in (1, 2, 3):
            raise ValidationError(f"Priority must be 1, 2, or 3 — got {self.priority}")


class TaskManager:
    """
    A task management system with full validation,
    logging, and clean error handling — designed to be testable.
    """
    
    def __init__(self):
        self._tasks: dict[int, Task] = {}
        self._next_id = 1
    
    @timed
    def add_task(self, title: str, priority: int = 2, tags: Optional[list[str]] = None) -> Task:
        """Add a new task and return it."""
        task = Task(
            id=self._next_id,
            title=title,
            priority=priority,
            tags=tags or []
        )
        self._tasks[task.id] = task
        self._next_id += 1
        logger.info(f"Added task #{task.id}: {title}")
        return task
    
    def get_task(self, task_id: int) -> Task:
        if task_id not in self._tasks:
            raise TaskNotFoundError(task_id)
        return self._tasks[task_id]
    
    def complete_task(self, task_id: int) -> Task:
        task = self.get_task(task_id)
        task.completed = True
        logger.info(f"Completed task #{task_id}")
        return task
    
    def delete_task(self, task_id: int) -> None:
        if task_id not in self._tasks:
            raise TaskNotFoundError(task_id)
        del self._tasks[task_id]
    
    def list_tasks(self, completed: Optional[bool] = None,
                    priority: Optional[int] = None) -> list[Task]:
        """List tasks with optional filtering."""
        results = list(self._tasks.values())
        if completed is not None:
            results = [t for t in results if t.completed == completed]
        if priority is not None:
            results = [t for t in results if t.priority == priority]
        return sorted(results, key=lambda t: (t.priority, t.id))
    
    def tasks_by_tag(self, tag: str) -> list[Task]:
        return [t for t in self._tasks.values() if tag in t.tags]
    
    @property
    def completion_rate(self) -> float:
        if not self._tasks:
            return 0.0
        completed = sum(1 for t in self._tasks.values() if t.completed)
        return completed / len(self._tasks)
    
    def bulk_complete(self, task_ids: list[int]) -> dict[str, list[int]]:
        """
        Complete multiple tasks. Returns which succeeded/failed —
        doesn't stop on first error (partial success pattern).
        """
        succeeded, failed = [], []
        for task_id in task_ids:
            try:
                self.complete_task(task_id)
                succeeded.append(task_id)
            except TaskNotFoundError:
                failed.append(task_id)
        return {"succeeded": succeeded, "failed": failed}


# ═══════════════════════════════════════════════════════════════════════
# COMPLETE TEST SUITE
# ═══════════════════════════════════════════════════════════════════════

import pytest


# ── FIXTURES ────────────────────────────────────────────────────────────
@pytest.fixture
def manager():
    """Fresh TaskManager for each test."""
    return TaskManager()

@pytest.fixture
def manager_with_tasks():
    """TaskManager pre-populated with a realistic mix of tasks."""
    mgr = TaskManager()
    mgr.add_task("Write report", priority=1, tags=["work", "urgent"])
    mgr.add_task("Buy groceries", priority=3, tags=["home"])
    mgr.add_task("Review PR", priority=2, tags=["work"])
    mgr.add_task("Call dentist", priority=2, tags=["health"])
    return mgr


# ── TASK CREATION & VALIDATION TESTS ────────────────────────────────────
class TestTaskCreation:
    def test_create_task_with_valid_data(self):
        task = Task(id=1, title="Test task")
        assert task.title == "Test task"
        assert task.completed is False
    
    def test_empty_title_raises_validation_error(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            Task(id=1, title="")
    
    def test_whitespace_only_title_raises(self):
        with pytest.raises(ValidationError):
            Task(id=1, title="   ")
    
    @pytest.mark.parametrize("invalid_priority", [0, 4, -1, 100])
    def test_invalid_priority_raises(self, invalid_priority):
        with pytest.raises(ValidationError, match="Priority must be"):
            Task(id=1, title="Test", priority=invalid_priority)
    
    def test_default_tags_is_empty_list_not_shared(self):
        """Regression test for mutable default argument bug."""
        task1 = Task(id=1, title="Task 1")
        task2 = Task(id=2, title="Task 2")
        task1.tags.append("important")
        assert task2.tags == []   # task2 should NOT be affected!


# ── TASK MANAGER: ADD/GET TESTS ──────────────────────────────────────────
class TestTaskManagerAddGet:
    def test_add_task_returns_task_with_incrementing_id(self, manager):
        task1 = manager.add_task("First")
        task2 = manager.add_task("Second")
        assert task1.id == 1
        assert task2.id == 2
    
    def test_get_existing_task(self, manager):
        added = manager.add_task("Find me")
        found = manager.get_task(added.id)
        assert found.title == "Find me"
    
    def test_get_nonexistent_task_raises(self, manager):
        with pytest.raises(TaskNotFoundError) as exc_info:
            manager.get_task(999)
        assert exc_info.value.task_id == 999


# ── TASK MANAGER: COMPLETE/DELETE TESTS ──────────────────────────────────
class TestTaskManagerCompleteDelete:
    def test_complete_task_marks_as_done(self, manager):
        task = manager.add_task("Do something")
        manager.complete_task(task.id)
        assert manager.get_task(task.id).completed is True
    
    def test_complete_nonexistent_task_raises(self, manager):
        with pytest.raises(TaskNotFoundError):
            manager.complete_task(999)
    
    def test_delete_task_removes_it(self, manager):
        task = manager.add_task("Delete me")
        manager.delete_task(task.id)
        with pytest.raises(TaskNotFoundError):
            manager.get_task(task.id)
    
    def test_delete_nonexistent_task_raises(self, manager):
        with pytest.raises(TaskNotFoundError):
            manager.delete_task(999)


# ── TASK MANAGER: LISTING & FILTERING TESTS ──────────────────────────────
class TestTaskManagerListing:
    def test_list_all_tasks_sorted_by_priority(self, manager_with_tasks):
        tasks = manager_with_tasks.list_tasks()
        priorities = [t.priority for t in tasks]
        assert priorities == sorted(priorities)   # Should be sorted!
    
    def test_filter_by_completed_status(self, manager_with_tasks):
        manager_with_tasks.complete_task(1)
        completed = manager_with_tasks.list_tasks(completed=True)
        pending = manager_with_tasks.list_tasks(completed=False)
        assert len(completed) == 1
        assert len(pending) == 3
    
    def test_filter_by_priority(self, manager_with_tasks):
        high_priority = manager_with_tasks.list_tasks(priority=1)
        assert len(high_priority) == 1
        assert high_priority[0].title == "Write report"
    
    def test_combined_filters(self, manager_with_tasks):
        manager_with_tasks.complete_task(3)   # Review PR (priority 2)
        result = manager_with_tasks.list_tasks(completed=True, priority=2)
        assert len(result) == 1
        assert result[0].title == "Review PR"
    
    def test_tasks_by_tag(self, manager_with_tasks):
        work_tasks = manager_with_tasks.tasks_by_tag("work")
        titles = {t.title for t in work_tasks}
        assert titles == {"Write report", "Review PR"}
    
    def test_tasks_by_nonexistent_tag_returns_empty(self, manager_with_tasks):
        result = manager_with_tasks.tasks_by_tag("nonexistent")
        assert result == []


# ── TASK MANAGER: ANALYTICS TESTS ────────────────────────────────────────
class TestTaskManagerAnalytics:
    def test_completion_rate_empty_manager(self, manager):
        assert manager.completion_rate == 0.0
    
    def test_completion_rate_no_completed(self, manager_with_tasks):
        assert manager_with_tasks.completion_rate == 0.0
    
    def test_completion_rate_partial(self, manager_with_tasks):
        manager_with_tasks.complete_task(1)
        manager_with_tasks.complete_task(2)
        assert manager_with_tasks.completion_rate == pytest.approx(0.5)
    
    def test_completion_rate_all_done(self, manager_with_tasks):
        for task_id in [1, 2, 3, 4]:
            manager_with_tasks.complete_task(task_id)
        assert manager_with_tasks.completion_rate == 1.0


# ── TASK MANAGER: BULK OPERATIONS (PARTIAL SUCCESS PATTERN) ──────────────
class TestBulkComplete:
    def test_bulk_complete_all_succeed(self, manager_with_tasks):
        result = manager_with_tasks.bulk_complete([1, 2, 3])
        assert result["succeeded"] == [1, 2, 3]
        assert result["failed"] == []
    
    def test_bulk_complete_some_fail(self, manager_with_tasks):
        result = manager_with_tasks.bulk_complete([1, 999, 2, 888])
        assert result["succeeded"] == [1, 2]
        assert result["failed"] == [999, 888]
    
    def test_bulk_complete_partial_failure_still_completes_valid_ones(
        self, manager_with_tasks
    ):
        """Critical: one bad ID shouldn't prevent OTHER valid completions."""
        manager_with_tasks.bulk_complete([1, 999, 2])
        assert manager_with_tasks.get_task(1).completed is True
        assert manager_with_tasks.get_task(2).completed is True


# ── PERFORMANCE TEST ──────────────────────────────────────────────────────
class TestPerformance:
    def test_add_many_tasks_completes_quickly(self, manager):
        """Regression guard: adding 10,000 tasks shouldn't take > 1 second."""
        start = time.perf_counter()
        for i in range(10_000):
            manager.add_task(f"Task {i}")
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s for 10,000 tasks"
    
    def test_list_tasks_with_filter_is_fast_at_scale(self, manager):
        for i in range(10_000):
            manager.add_task(f"Task {i}", priority=(i % 3) + 1)
        
        start = time.perf_counter()
        result = manager.list_tasks(priority=1)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 0.1
        assert len(result) > 0


# ── RUNNING THE SUITE ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Run with: pytest quality_toolkit.py -v")
    print("With coverage: pytest quality_toolkit.py --cov=. --cov-report=term-missing")
    print("Only fast tests: pytest quality_toolkit.py -v -m 'not slow'")
    
    # Quick manual smoke test
    mgr = TaskManager()
    mgr.add_task("Learn pytest", priority=1, tags=["learning"])
    mgr.add_task("Write tests", priority=1, tags=["learning", "practice"])
    mgr.complete_task(1)
    
    print(f"\nCompletion rate: {mgr.completion_rate:.0%}")
    print(f"Learning tasks: {[t.title for t in mgr.tasks_by_tag('learning')]}")
```

---

## Chapter Summary — And the Journey So Far

You've completed the full Python learning path! This final chapter gave you the skills that separate hobbyist code from professional, production-ready software.

✅ **Testing**: `unittest` and `pytest`, assertions, fixtures, parametrization, marks
✅ **Mocking**: Isolating code from external dependencies (`Mock`, `patch`, `monkeypatch`)
✅ **Debugging**: print → logging → `pdb`/`breakpoint()` → visual debuggers, systematic methodology
✅ **Code Style**: PEP 8 naming, whitespace, imports; PEP 257 docstrings; type hints
✅ **Performance**: Measuring with `timeit`/`cProfile`, common pitfalls, when NOT to optimize
✅ **Best Practices**: TDD cycle, testing behavior not implementation, one assertion focus per test

**Key Takeaways:**
- Untested code is a liability — every bug found in production was preventable in a test
- Mock external dependencies (APIs, databases, time) — never the logic you're actually testing
- Debugging is a systematic process: reproduce, isolate, hypothesize, test, fix, verify, prevent
- Measure before optimizing — `cProfile` tells you where time ACTUALLY goes
- Readable code beats clever code; PEP 8 isn't bureaucracy, it's a shared language for your team

**You've now completed all 15 chapters** — from variables and loops, through data structures, functions, OOP, file handling, and into the professional-grade tools of NumPy, Pandas, visualization, regex, and testing. You have the full toolkit of a working Python developer.

**What's next?** The best way to solidify everything is to build something real: a personal project that combines several chapters — a data analysis tool, a small web scraper with tests, an automation script with proper logging and error handling. Pick something you actually want to exist, and build it with the practices from this guide.
