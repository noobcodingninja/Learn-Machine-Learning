# Chapter 8: Object-Oriented Programming (OOP)

## Part 1: Why Does OOP Exist?

### The Problem: Code Without Structure Falls Apart

Imagine you're building a banking app. Without OOP:

```python
# Account 1
account1_owner = "Alice"
account1_balance = 5000.0
account1_number = "ACC001"

# Account 2
account2_owner = "Bob"
account2_balance = 3200.0
account2_number = "ACC002"

# Now write a deposit function
def deposit(owner, balance, amount):
    if amount <= 0:
        raise ValueError("Amount must be positive")
    return balance + amount   # Return new balance, hope caller remembers to update!

account1_balance = deposit(account1_owner, account1_balance, 500)

# What if we have 1000 accounts? 
# What if accounts have more attributes (interest rate, account type, history)?
# What if two developers work on the same codebase — who tracks what?
```

**The problems:**
- Data (owner, balance) and behavior (deposit, withdraw) live separately
- Nothing stops someone from doing `account1_balance = -999999` directly
- Adding a new attribute means changing every function signature
- Doesn't scale — managing 100 variables for 10 accounts is chaos

### The OOP Solution: Bundle Data + Behavior Together

```python
class BankAccount:
    def __init__(self, owner, initial_balance=0):
        self.owner = owner
        self.balance = initial_balance
        self.account_number = "ACC" + str(id(self))[-6:]
    
    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self.balance += amount
        return self.balance
    
    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
        return self.balance

# Now data and behavior are inseparable!
alice_account = BankAccount("Alice", 5000)
bob_account = BankAccount("Bob", 3200)

alice_account.deposit(500)
bob_account.withdraw(100)

# Scale to 1000 accounts? Easy — same interface for all of them.
```

### What Is OOP Really?

OOP is a way of modeling the world using **objects** — things that have:
- **State** (attributes/properties): What the object *is*
- **Behavior** (methods): What the object *does*

Real-world analogy — a Car:
- **State**: color, brand, speed, fuel_level, is_running
- **Behavior**: start(), accelerate(), brake(), refuel()

A `class` is the **blueprint** (the design). An `object` is the **instance** (the actual thing built from that blueprint). Like the difference between a house floor plan and an actual house.

---

## Part 2: Classes and Objects

### Defining a Class

```python
class Dog:
    """
    A class representing a dog.
    
    The class definition is the BLUEPRINT.
    Each Dog() call creates a new INSTANCE (object).
    """
    
    # Class variable — shared by ALL instances
    species = "Canis lupus familiaris"
    total_dogs = 0
    
    # __init__ — the constructor (called when creating a new instance)
    # self refers to the specific instance being created
    def __init__(self, name, breed, age):
        # Instance variables — unique to each object
        self.name = name      # self.X creates an attribute on this instance
        self.breed = breed
        self.age = age
        self.tricks = []      # Each dog gets its OWN empty list
        Dog.total_dogs += 1   # Update class variable
    
    # Instance method — operates on a specific instance
    def bark(self):
        return f"{self.name} says: Woof!"
    
    def learn_trick(self, trick):
        self.tricks.append(trick)
        return f"{self.name} learned '{trick}'!"
    
    def show_tricks(self):
        if not self.tricks:
            return f"{self.name} doesn't know any tricks yet."
        return f"{self.name} can: {', '.join(self.tricks)}"
    
    def birthday(self):
        self.age += 1
        return f"Happy birthday {self.name}! Now {self.age} years old."


# Creating instances (objects)
rex = Dog("Rex", "German Shepherd", 3)
luna = Dog("Luna", "Labrador", 5)

# Accessing attributes
print(rex.name)    # Rex
print(luna.breed)  # Labrador
print(rex.age)     # 3

# Calling methods
print(rex.bark())              # Rex says: Woof!
print(luna.learn_trick("sit")) # Luna learned 'sit'!
print(luna.learn_trick("paw")) # Luna learned 'paw'!
print(luna.show_tricks())      # Luna can: sit, paw
print(rex.show_tricks())       # Rex doesn't know any tricks yet.

# rex and luna have SEPARATE tricks lists — not shared!
rex.learn_trick("roll over")
print(rex.show_tricks())   # Rex can: roll over
print(luna.show_tricks())  # Luna can: sit, paw  (unchanged)

# Class variables are shared
print(Dog.total_dogs)   # 2
print(rex.total_dogs)   # 2 (accessible via instance too)
print(luna.total_dogs)  # 2

# Class variable vs instance variable
print(rex.species)      # Canis lupus familiaris (from class)
rex.species = "Canis"   # Creates INSTANCE variable, shadows class variable!
print(rex.species)      # Canis (instance variable)
print(luna.species)     # Canis lupus familiaris (class variable, unchanged)
```

### `self` — What Is It Really?

```python
class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1

c = Counter()

# These two are IDENTICAL:
c.increment()           # Python rewrites this as...
Counter.increment(c)    # ...this! self is just the instance passed in.

# So self is NOT magic — it's just the first argument,
# a reference to the object the method is called on.
# "self" is a convention, not a keyword — but always use it!
```

---

## Part 3: The Four Pillars of OOP

### Pillar 1: Encapsulation — Protecting Data

**Problem:** Anyone can accidentally break your object's state.

```python
# Without encapsulation
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius

t = Temperature(20)
t.celsius = -500    # Physically impossible! No protection.
```

```python
# With encapsulation — control access to internal state
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius   # _ prefix = "private by convention"
    
    # Property — accessed like attribute, but runs code
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError(f"Temperature below absolute zero: {value}")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9   # Validates via celsius setter
    
    @property
    def kelvin(self):
        return self._celsius + 273.15

t = Temperature(20)
print(t.celsius)      # 20 (calls getter)
print(t.fahrenheit)   # 68.0
print(t.kelvin)       # 293.15

t.celsius = 100       # Calls setter — validates first
print(t.fahrenheit)   # 212.0

t.fahrenheit = 32     # Calls fahrenheit setter → celsius setter
print(t.celsius)      # 0.0

# t.celsius = -500    # ValueError! Protected!

# Name mangling — double underscore = truly private
class Secret:
    def __init__(self):
        self.__password = "secret123"   # __X becomes _ClassName__X
    
    def verify(self, attempt):
        return attempt == self.__password

s = Secret()
# print(s.__password)   # AttributeError — can't access directly
print(s.verify("secret123"))     # True
print(s._Secret__password)       # secret123 — accessible but discouraged!
```

### Pillar 2: Inheritance — Reusing and Extending

**Problem:** You have several classes that share common behavior, but also have unique behavior.

```python
# Without inheritance — massive code duplication
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def eat(self): return f"{self.name} is eating"
    def sleep(self): return f"{self.name} is sleeping"
    def speak(self): return f"{self.name}: Woof!"

class Cat:
    def __init__(self, name, age):   # SAME as Dog!
        self.name = name
        self.age = age
    def eat(self): return f"{self.name} is eating"   # SAME!
    def sleep(self): return f"{self.name} is sleeping"  # SAME!
    def speak(self): return f"{self.name}: Meow!"    # Different

# Every new animal type means copy-pasting shared code!
```

```python
# With inheritance — DRY (Don't Repeat Yourself)

class Animal:
    """Base class — shared behavior for all animals"""
    
    def __init__(self, name, age, sound):
        self.name = name
        self.age = age
        self.sound = sound
        self.energy = 100
    
    def eat(self, food="food"):
        self.energy = min(100, self.energy + 20)
        return f"{self.name} eats {food}. Energy: {self.energy}"
    
    def sleep(self):
        self.energy = min(100, self.energy + 40)
        return f"{self.name} sleeps. Energy: {self.energy}"
    
    def speak(self):
        return f"{self.name}: {self.sound}!"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', age={self.age})"


class Dog(Animal):
    """Dog inherits ALL of Animal's attributes and methods"""
    
    def __init__(self, name, age, breed):
        super().__init__(name, age, "Woof")   # Call parent's __init__
        self.breed = breed
        self.tricks = []
    
    # Override parent method
    def speak(self):
        base = super().speak()     # Get parent's version
        return f"{base} {base}"   # Dogs bark twice!
    
    # New method — only Dogs have this
    def fetch(self, item="ball"):
        self.energy -= 10
        return f"{self.name} fetches the {item}! Energy: {self.energy}"
    
    def learn_trick(self, trick):
        self.tricks.append(trick)


class Cat(Animal):
    def __init__(self, name, age, indoor=True):
        super().__init__(name, age, "Meow")
        self.indoor = indoor
    
    # Override eat — cats are picky!
    def eat(self, food="tuna"):
        if food in ["tuna", "salmon", "chicken"]:
            return super().eat(food)
        return f"{self.name} sniffs {food} and walks away."
    
    def purr(self):
        return f"{self.name} purrs contentedly..."


class ServiceDog(Dog):
    """ServiceDog inherits from Dog (which inherits from Animal) — multi-level"""
    
    def __init__(self, name, age, breed, specialty):
        super().__init__(name, age, breed)
        self.specialty = specialty
        self.certified = True
    
    def perform_duty(self):
        self.energy -= 20
        return f"{self.name} performs {self.specialty} duty. Energy: {self.energy}"


# Demonstration
rex = Dog("Rex", 3, "German Shepherd")
luna = Cat("Luna", 5)
buddy = ServiceDog("Buddy", 4, "Labrador", "Guide")

print(rex.eat("kibble"))      # Rex eats kibble. Energy: 120... (capped at 100)
print(rex.speak())            # Rex: Woof! Rex: Woof!
print(rex.fetch())            # Rex fetches the ball!
print(luna.eat("tuna"))       # Luna eats tuna.
print(luna.eat("broccoli"))   # Luna sniffs broccoli and walks away.
print(buddy.perform_duty())   # Buddy performs Guide duty.
print(buddy.fetch("stick"))   # buddy also has fetch — inherited from Dog!

# isinstance — check inheritance chain
print(isinstance(buddy, ServiceDog))  # True
print(isinstance(buddy, Dog))         # True — ServiceDog IS a Dog
print(isinstance(buddy, Animal))      # True — ServiceDog IS an Animal
print(isinstance(buddy, Cat))         # False

# issubclass — check class relationships
print(issubclass(ServiceDog, Dog))    # True
print(issubclass(Dog, Animal))        # True
print(issubclass(ServiceDog, Animal)) # True
```

### Pillar 3: Polymorphism — One Interface, Many Forms

**Problem:** You want to treat different object types uniformly when they share behavior.

```python
# Polymorphism — same method name, different behavior per class

class Shape:
    def area(self): raise NotImplementedError
    def perimeter(self): raise NotImplementedError
    def describe(self):
        return f"{self.__class__.__name__}: area={self.area():.2f}, perimeter={self.perimeter():.2f}"

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        import math
        return math.pi * self.radius ** 2
    def perimeter(self):
        import math
        return 2 * math.pi * self.radius

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def area(self):
        return self.width * self.height
    def perimeter(self):
        return 2 * (self.width + self.height)

class Triangle(Shape):
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c
    def area(self):
        s = self.perimeter() / 2   # Semi-perimeter
        return (s*(s-self.a)*(s-self.b)*(s-self.c)) ** 0.5  # Heron's formula
    def perimeter(self):
        return self.a + self.b + self.c

# POLYMORPHISM IN ACTION
shapes = [Circle(5), Rectangle(4, 6), Triangle(3, 4, 5)]

# Same code works for ALL shape types — this is polymorphism!
total_area = 0
for shape in shapes:
    print(shape.describe())    # Each calls its OWN area() and perimeter()
    total_area += shape.area() # .area() means different things for each!

print(f"\nTotal area: {total_area:.2f}")

# Even Python's built-in functions use polymorphism:
print(len("hello"))   # 5 (string's __len__)
print(len([1,2,3]))   # 3 (list's __len__)
print(len({1,2,3,4})) # 4 (set's __len__)
# Same function name, different behavior for each type!
```

### Pillar 4: Abstraction — Hiding Complexity

**Problem:** Users of your class shouldn't need to understand its internal workings.

```python
from abc import ABC, abstractmethod

class PaymentProcessor(ABC):
    """
    Abstract base class — defines the INTERFACE
    without specifying HOW things are done.
    
    Can't be instantiated directly — must be subclassed.
    """
    
    @abstractmethod
    def charge(self, amount, customer_id):
        """Charge a customer. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def refund(self, transaction_id, amount):
        """Issue a refund. Must be implemented by subclasses."""
        pass
    
    # Non-abstract method — shared by all processors
    def validate_amount(self, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > 1_000_000:
            raise ValueError("Amount exceeds maximum transaction limit")
        return True


class StripeProcessor(PaymentProcessor):
    def __init__(self, api_key):
        self.api_key = api_key
    
    def charge(self, amount, customer_id):
        self.validate_amount(amount)
        # (In reality, calls Stripe API here)
        return {"status": "success", "processor": "Stripe",
                "amount": amount, "transaction_id": f"stripe_{customer_id}_{amount}"}
    
    def refund(self, transaction_id, amount):
        return {"status": "refunded", "transaction_id": transaction_id, "amount": amount}


class PayPalProcessor(PaymentProcessor):
    def __init__(self, client_id, secret):
        self.client_id = client_id
        self.secret = secret
    
    def charge(self, amount, customer_id):
        self.validate_amount(amount)
        return {"status": "success", "processor": "PayPal",
                "amount": amount, "transaction_id": f"pp_{customer_id}_{amount}"}
    
    def refund(self, transaction_id, amount):
        return {"status": "refunded", "processor": "PayPal",
                "transaction_id": transaction_id}


# The ORDER CLASS doesn't care WHICH processor — just uses the interface!
class Order:
    def __init__(self, order_id, customer_id, amount, processor: PaymentProcessor):
        self.order_id = order_id
        self.customer_id = customer_id
        self.amount = amount
        self.processor = processor       # Any PaymentProcessor works!
    
    def checkout(self):
        result = self.processor.charge(self.amount, self.customer_id)
        return {"order_id": self.order_id, "payment": result}

stripe = StripeProcessor("sk_live_xxx")
paypal = PayPalProcessor("client_id", "secret")

order1 = Order("ORD001", "CUST001", 99.99, stripe)
order2 = Order("ORD002", "CUST002", 149.99, paypal)

print(order1.checkout())
print(order2.checkout())
# SAME checkout code works for BOTH payment processors!

# PaymentProcessor()  # TypeError! Can't instantiate abstract class
```

---

## Part 4: Magic Methods (Dunder Methods)

### Making Your Objects Behave Like Built-in Types

Magic methods (double underscore = "dunder") let your objects respond to Python's built-in operators and functions.

```python
class Vector:
    """2D vector with full operator support via magic methods"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # ── STRING REPRESENTATION ──────────────────────────────────────────
    def __str__(self):
        """Called by print() and str() — human-readable"""
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        """Called in REPL and repr() — developer-friendly, unambiguous"""
        return f"Vector(x={self.x!r}, y={self.y!r})"
    
    # ── ARITHMETIC OPERATORS ──────────────────────────────────────────
    def __add__(self, other):
        """v1 + v2"""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """v1 - v2"""
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """v * 3  (vector * scalar)"""
        return Vector(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar):
        """3 * v  (scalar * vector) — right-hand multiply"""
        return self.__mul__(scalar)
    
    def __neg__(self):
        """-v"""
        return Vector(-self.x, -self.y)
    
    def __abs__(self):
        """abs(v) — magnitude/length of vector"""
        return (self.x ** 2 + self.y ** 2) ** 0.5
    
    # ── COMPARISON OPERATORS ──────────────────────────────────────────
    def __eq__(self, other):
        """v1 == v2"""
        if not isinstance(other, Vector):
            return NotImplemented
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other):
        """v1 < v2 — compare by magnitude"""
        return abs(self) < abs(other)
    
    def __le__(self, other):
        return abs(self) <= abs(other)
    
    # ── CONTAINER-LIKE BEHAVIOR ───────────────────────────────────────
    def __len__(self):
        """len(v) — number of dimensions"""
        return 2
    
    def __getitem__(self, index):
        """v[0], v[1]"""
        if index == 0: return self.x
        if index == 1: return self.y
        raise IndexError("Vector index out of range")
    
    def __iter__(self):
        """for component in v:"""
        yield self.x
        yield self.y
    
    def __contains__(self, value):
        """3 in v"""
        return value in (self.x, self.y)
    
    # ── HASH (to use in sets/dict keys) ──────────────────────────────
    def __hash__(self):
        return hash((self.x, self.y))
    
    # ── BOOL CONVERSION ───────────────────────────────────────────────
    def __bool__(self):
        """bool(v) — False only for zero vector"""
        return self.x != 0 or self.y != 0


# Using all the magic methods
v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(str(v1))        # Vector(3, 4)          — __str__
print(repr(v1))       # Vector(x=3, y=4)      — __repr__
print(v1 + v2)        # Vector(4, 6)           — __add__
print(v1 - v2)        # Vector(2, 2)           — __sub__
print(v1 * 3)         # Vector(9, 12)          — __mul__
print(2 * v1)         # Vector(6, 8)           — __rmul__
print(-v1)            # Vector(-3, -4)         — __neg__
print(abs(v1))        # 5.0                    — __abs__ (3-4-5 triangle!)
print(v1 == Vector(3, 4))  # True             — __eq__
print(v1 == v2)            # False
print(v1 > v2)        # True (magnitude 5 > ~2.24) — __lt__ via >
print(len(v1))        # 2                      — __len__
print(v1[0], v1[1])   # 3 4                   — __getitem__
print(list(v1))       # [3, 4]                 — __iter__
print(3 in v1)        # True                   — __contains__
print(bool(Vector(0, 0)))  # False             — __bool__
print(bool(v1))            # True

# Can sort vectors! (uses __lt__)
vectors = [Vector(5,5), Vector(1,1), Vector(3,4)]
print(sorted(vectors))     # [Vector(1,1), Vector(3,4), Vector(5,5)]

# Can use in sets! (uses __hash__ and __eq__)
vector_set = {v1, v2, Vector(3, 4)}
print(len(vector_set))  # 2 — v1 and Vector(3,4) are the same!
```

### The Most Important Magic Methods at a Glance

```python
class MyClass:
    # ── LIFECYCLE ──────────────────────────────────────────────────────
    def __init__(self):      pass   # Constructor
    def __del__(self):       pass   # Destructor (called when object is garbage collected)
    
    # ── REPRESENTATION ────────────────────────────────────────────────
    def __str__(self):       pass   # str(obj), print(obj)
    def __repr__(self):      pass   # repr(obj), REPL display
    def __format__(self, s): pass   # f"{obj:format_spec}"
    
    # ── COMPARISON ────────────────────────────────────────────────────
    def __eq__(self, other): pass   # obj == other
    def __lt__(self, other): pass   # obj < other
    def __le__(self, other): pass   # obj <= other
    def __gt__(self, other): pass   # obj > other
    def __ge__(self, other): pass   # obj >= other
    def __ne__(self, other): pass   # obj != other
    
    # ── ARITHMETIC ────────────────────────────────────────────────────
    def __add__(self, other):  pass  # obj + other
    def __sub__(self, other):  pass  # obj - other
    def __mul__(self, other):  pass  # obj * other
    def __truediv__(self, o):  pass  # obj / other
    def __floordiv__(self, o): pass  # obj // other
    def __mod__(self, other):  pass  # obj % other
    def __pow__(self, other):  pass  # obj ** other
    
    # ── CONTAINER ─────────────────────────────────────────────────────
    def __len__(self):           pass  # len(obj)
    def __getitem__(self, key):  pass  # obj[key]
    def __setitem__(self, k, v): pass  # obj[key] = value
    def __delitem__(self, key):  pass  # del obj[key]
    def __contains__(self, item):pass  # item in obj
    def __iter__(self):          pass  # for x in obj
    def __next__(self):          pass  # next(obj)
    
    # ── TYPE CONVERSION ───────────────────────────────────────────────
    def __bool__(self):  pass   # bool(obj)
    def __int__(self):   pass   # int(obj)
    def __float__(self): pass   # float(obj)
    def __len__(self):   pass   # Also used for bool if __bool__ missing
    
    # ── CONTEXT MANAGER ───────────────────────────────────────────────
    def __enter__(self): pass   # with obj as x
    def __exit__(self, *args): pass  # end of with block
    
    # ── CALLABLE ──────────────────────────────────────────────────────
    def __call__(self, *args): pass  # obj()  — makes instance callable!
    
    # ── ATTRIBUTE ACCESS ──────────────────────────────────────────────
    def __getattr__(self, name):     pass  # Called when attr NOT found normally
    def __setattr__(self, name, v):  pass  # Called on EVERY attribute set
    def __hasattr__(self, name):     pass  # hasattr(obj, name)
```

### The `__call__` Method — Callable Objects

```python
class Multiplier:
    """An object that behaves like a function!"""
    
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, value):
        return value * self.factor

double = Multiplier(2)
triple = Multiplier(3)

print(double(5))    # 10 — calling the OBJECT like a function!
print(triple(5))    # 15
print(callable(double))  # True

# Real-world use: Stateful function
class RateLimiter:
    """Callable that tracks how often it's invoked"""
    
    def __init__(self, func, max_calls_per_minute=60):
        self.func = func
        self.max_calls = max_calls_per_minute
        self.calls = []
        import time
        self.time = time
    
    def __call__(self, *args, **kwargs):
        now = self.time.time()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]
        if len(self.calls) >= self.max_calls:
            raise RuntimeError(f"Rate limit exceeded: {self.max_calls} calls/minute")
        self.calls.append(now)
        return self.func(*args, **kwargs)

def fetch_data(url):
    return f"Data from {url}"

limited_fetch = RateLimiter(fetch_data, max_calls_per_minute=3)
print(limited_fetch("https://api.example.com"))
print(limited_fetch("https://api.example.com"))
print(limited_fetch("https://api.example.com"))
# limited_fetch("...")  # RuntimeError: Rate limit exceeded
```

---

## Part 5: Class Methods and Static Methods

```python
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    
    # ── REGULAR INSTANCE METHOD ────────────────────────────────────────
    # Receives the instance as first argument (self)
    def is_leap_year(self):
        return (self.year % 4 == 0 and self.year % 100 != 0) or (self.year % 400 == 0)
    
    def format(self, fmt="ISO"):
        if fmt == "ISO":
            return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"
        elif fmt == "US":
            return f"{self.month:02d}/{self.day:02d}/{self.year:04d}"
    
    # ── CLASS METHOD ──────────────────────────────────────────────────
    # Receives the CLASS as first argument (cls), not the instance
    # Often used as alternative constructors
    @classmethod
    def from_string(cls, date_string):
        """Create Date from 'YYYY-MM-DD' string"""
        year, month, day = map(int, date_string.split("-"))
        return cls(year, month, day)    # cls() creates an instance
    
    @classmethod
    def today(cls):
        """Create a Date for today"""
        import datetime
        d = datetime.date.today()
        return cls(d.year, d.month, d.day)
    
    @classmethod
    def from_timestamp(cls, timestamp):
        """Create Date from Unix timestamp"""
        import datetime
        d = datetime.date.fromtimestamp(timestamp)
        return cls(d.year, d.month, d.day)
    
    # ── STATIC METHOD ──────────────────────────────────────────────────
    # No self or cls — just a regular function, organized under the class
    # Use when logic is related to the class but doesn't need instance or class
    @staticmethod
    def is_valid_date(year, month, day):
        """Check if a date is valid (no self or cls needed)"""
        if not (1 <= month <= 12):
            return False
        max_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        # Check leap year for February
        if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
            max_days[2] = 29
        return 1 <= day <= max_days[month]
    
    @staticmethod
    def days_in_month(month, year=2025):
        """Return number of days in a given month"""
        import calendar
        return calendar.monthrange(year, month)[1]
    
    def __repr__(self):
        return f"Date({self.year}, {self.month}, {self.day})"
    
    def __str__(self):
        return self.format("ISO")


# Using instance methods
d1 = Date(2024, 2, 29)
print(d1.is_leap_year())    # True
print(d1.format("US"))      # 02/29/2024

# Using class methods (alternative constructors)
d2 = Date.from_string("2025-01-15")
print(d2)                   # 2025-01-15

d3 = Date.today()
print(d3)                   # Today's date

# Using static methods (utility functions)
print(Date.is_valid_date(2025, 2, 30))  # False — Feb only has 28/29 days
print(Date.is_valid_date(2025, 6, 15))  # True
print(Date.days_in_month(2, 2024))      # 29 (leap year)
```

---

## Part 6: Composition vs Inheritance

### The Problem: Inheritance Isn't Always the Right Tool

```python
# Inheritance gone wrong — the "is-a" violation

# Sounds natural: "A car IS A vehicle"
class Vehicle:
    def __init__(self, speed, fuel):
        self.speed = speed
        self.fuel = fuel
    def refuel(self): pass

class Car(Vehicle):
    def drive(self): pass

# Now: "An electric car IS A car" — but it doesn't use fuel!
class ElectricCar(Car):
    def __init__(self, speed, battery):
        super().__init__(speed, fuel=0)  # Fuel doesn't make sense!
        self.battery = battery
    
    def refuel(self):
        # This is called "refueling" but electric cars CHARGE
        # The interface doesn't fit!
        pass
```

```python
# Composition — "has-a" instead of "is-a"
# Build complex objects from simpler components

class Engine:
    def __init__(self, horsepower, fuel_type):
        self.horsepower = horsepower
        self.fuel_type = fuel_type
    
    def start(self): return f"{self.fuel_type} engine started"
    def stop(self): return "Engine stopped"

class ElectricMotor:
    def __init__(self, kilowatts, battery_capacity):
        self.kilowatts = kilowatts
        self.battery_capacity = battery_capacity
        self.charge_level = 100
    
    def start(self): return "Electric motor running silently"
    def stop(self): return "Motor stopped"
    def charge(self, amount): 
        self.charge_level = min(100, self.charge_level + amount)

class GPS:
    def navigate(self, destination):
        return f"Navigating to {destination}"

class AudioSystem:
    def __init__(self, brand):
        self.brand = brand
    def play(self, song): return f"Playing '{song}' on {self.brand}"

# Now build cars from components — flexible!
class GasCar:
    def __init__(self, make, model):
        self.make = make
        self.model = model
        self.engine = Engine(200, "petrol")   # HAS an engine
        self.gps = GPS()                       # HAS a GPS
        self.audio = AudioSystem("Bose")       # HAS an audio system
    
    def start(self): return self.engine.start()
    def navigate(self, dest): return self.gps.navigate(dest)
    def play_music(self, song): return self.audio.play(song)

class ElectricCar:
    def __init__(self, make, model):
        self.make = make
        self.model = model
        self.motor = ElectricMotor(300, 100)   # HAS an electric motor
        self.gps = GPS()
        self.audio = AudioSystem("Harman")
    
    def start(self): return self.motor.start()
    def charge(self, amount): return self.motor.charge(amount)
    def navigate(self, dest): return self.gps.navigate(dest)

tesla = ElectricCar("Tesla", "Model 3")
bmw = GasCar("BMW", "3 Series")

print(tesla.start())                   # Electric motor running silently
print(bmw.start())                     # petrol engine started
print(bmw.navigate("Airport"))         # Navigating to Airport
print(bmw.play_music("Bohemian Rhapsody"))  # Playing on Bose

# When to use each:
# Inheritance: "IS-A" relationship, share interface, override behavior
# Composition: "HAS-A" relationship, combine components, more flexible
```

---

## Part 7: Worked Examples

### Worked Example 1: Full Stack Item — Playing Card Deck

```python
import random

class Card:
    SUITS = ["♠", "♥", "♦", "♣"]
    RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    RANK_VALUES = {r: i+2 for i, r in enumerate(RANKS)}
    
    def __init__(self, rank, suit):
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}")
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}")
        self.rank = rank
        self.suit = suit
        self.value = self.RANK_VALUES[rank]
    
    def __str__(self):  return f"{self.rank}{self.suit}"
    def __repr__(self): return f"Card('{self.rank}', '{self.suit}')"
    def __eq__(self, other): return self.value == other.value and self.suit == other.suit
    def __lt__(self, other): return self.value < other.value
    def __hash__(self): return hash((self.rank, self.suit))


class Deck:
    def __init__(self):
        self.cards = [Card(r, s) for s in Card.SUITS for r in Card.RANKS]
    
    def __len__(self):      return len(self.cards)
    def __str__(self):      return f"Deck({len(self)} cards)"
    def __contains__(self, card): return card in self.cards
    def __iter__(self):     return iter(self.cards)
    
    def shuffle(self):
        random.shuffle(self.cards)
        return self
    
    def deal(self, n=1):
        if n > len(self.cards):
            raise ValueError(f"Not enough cards: {len(self.cards)} remaining")
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt
    
    def deal_hand(self, players, cards_each):
        hands = {f"Player {i+1}": [] for i in range(players)}
        for _ in range(cards_each):
            for player in hands:
                hands[player].extend(self.deal(1))
        return hands


class Hand:
    def __init__(self, cards=None):
        self.cards = cards or []
    
    def __len__(self):  return len(self.cards)
    def __str__(self):  return " ".join(str(c) for c in sorted(self.cards, reverse=True))
    def __iter__(self): return iter(self.cards)
    
    def add(self, card): self.cards.append(card)
    
    @property
    def total_value(self): return sum(c.value for c in self.cards)
    
    @property
    def highest_card(self): return max(self.cards) if self.cards else None


# Demo
deck = Deck()
print(deck)            # Deck(52 cards)
deck.shuffle()
print(f"After shuffle, first 5: {[str(c) for c in deck.cards[:5]]}")

hands = deck.deal_hand(players=4, cards_each=5)
for player, cards in hands.items():
    hand = Hand(cards)
    print(f"{player}: {hand} (total: {hand.total_value})")

print(f"\nCards remaining: {len(deck)}")
```

### Worked Example 2: E-Commerce System

```python
from datetime import datetime

class Product:
    def __init__(self, product_id, name, price, stock=0, category="General"):
        self.product_id = product_id
        self.name = name
        self._price = price
        self.stock = stock
        self.category = category
    
    @property
    def price(self): return self._price
    
    @price.setter
    def price(self, value):
        if value < 0: raise ValueError("Price cannot be negative")
        self._price = round(value, 2)
    
    def is_in_stock(self): return self.stock > 0
    def reserve(self, qty=1):
        if qty > self.stock:
            raise ValueError(f"Only {self.stock} units available")
        self.stock -= qty
    def restock(self, qty): self.stock += qty
    def __str__(self): return f"{self.name} (${self._price:.2f})"
    def __repr__(self): return f"Product('{self.product_id}', '{self.name}', {self._price})"


class CartItem:
    def __init__(self, product, quantity=1):
        self.product = product
        self.quantity = quantity
    
    @property
    def subtotal(self): return self.product.price * self.quantity
    def __str__(self): return f"{self.product.name} x{self.quantity} = ${self.subtotal:.2f}"


class ShoppingCart:
    def __init__(self, customer_id):
        self.customer_id = customer_id
        self._items = {}       # product_id → CartItem
        self.coupon = None
    
    def add(self, product, quantity=1):
        if not product.is_in_stock():
            raise ValueError(f"{product.name} is out of stock")
        if product.stock < quantity:
            raise ValueError(f"Only {product.stock} units of {product.name} available")
        
        if product.product_id in self._items:
            self._items[product.product_id].quantity += quantity
        else:
            self._items[product.product_id] = CartItem(product, quantity)
    
    def remove(self, product_id, quantity=None):
        if product_id not in self._items:
            raise KeyError(f"Product {product_id} not in cart")
        if quantity is None or quantity >= self._items[product_id].quantity:
            del self._items[product_id]
        else:
            self._items[product_id].quantity -= quantity
    
    def apply_coupon(self, code, discount_pct):
        self.coupon = {"code": code, "discount": discount_pct}
    
    @property
    def subtotal(self): return sum(item.subtotal for item in self._items.values())
    
    @property
    def discount_amount(self):
        if self.coupon:
            return self.subtotal * (self.coupon["discount"] / 100)
        return 0
    
    @property
    def total(self): return self.subtotal - self.discount_amount
    
    def __len__(self): return sum(item.quantity for item in self._items.values())
    def __bool__(self): return len(self._items) > 0
    def __iter__(self): return iter(self._items.values())
    def __contains__(self, product_id): return product_id in self._items
    
    def checkout(self):
        """Reserve stock and return order summary"""
        for item in self._items.values():
            item.product.reserve(item.quantity)
        
        order = Order(
            order_id=f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            customer_id=self.customer_id,
            items=list(self._items.values()),
            total=self.total,
            coupon=self.coupon
        )
        self._items.clear()
        return order


class Order:
    STATUS_FLOW = ["pending", "confirmed", "shipped", "delivered", "cancelled"]
    
    def __init__(self, order_id, customer_id, items, total, coupon=None):
        self.order_id = order_id
        self.customer_id = customer_id
        self.items = items
        self.total = total
        self.coupon = coupon
        self._status = "pending"
        self.created_at = datetime.now()
        self.history = [("pending", self.created_at)]
    
    @property
    def status(self): return self._status
    
    def advance_status(self):
        idx = self.STATUS_FLOW.index(self._status)
        if idx < len(self.STATUS_FLOW) - 2:  # Can't advance past "delivered"
            self._status = self.STATUS_FLOW[idx + 1]
            self.history.append((self._status, datetime.now()))
        return self._status
    
    def cancel(self):
        if self._status in ("shipped", "delivered"):
            raise ValueError(f"Cannot cancel order that is already {self._status}")
        # Restore stock
        for item in self.items:
            item.product.restock(item.quantity)
        self._status = "cancelled"
        self.history.append(("cancelled", datetime.now()))
    
    def __str__(self):
        lines = [f"Order {self.order_id} [{self._status.upper()}]"]
        for item in self.items:
            lines.append(f"  {item}")
        if self.coupon:
            lines.append(f"  Coupon ({self.coupon['code']}): -{self.coupon['discount']}%")
        lines.append(f"  TOTAL: ${self.total:.2f}")
        return "\n".join(lines)


# Demo
laptop = Product("P001", "MacBook Pro", 1299.99, stock=10, category="Electronics")
mouse  = Product("P002", "MX Master 3",   89.99, stock=50, category="Peripherals")
case   = Product("P003", "Laptop Case",   29.99, stock=25, category="Accessories")

cart = ShoppingCart("CUST001")
cart.add(laptop, 1)
cart.add(mouse, 2)
cart.add(case, 1)
cart.apply_coupon("SAVE10", 10)

print(f"Items in cart: {len(cart)}")
print(f"Subtotal: ${cart.subtotal:.2f}")
print(f"Discount: -${cart.discount_amount:.2f}")
print(f"Total:    ${cart.total:.2f}")

order = cart.checkout()
print("\n" + str(order))
print(f"\nLaptop stock after checkout: {laptop.stock}")  # 9

order.advance_status()  # pending → confirmed
order.advance_status()  # confirmed → shipped
print(f"Order status: {order.status}")

print("\nOrder history:")
for status, timestamp in order.history:
    print(f"  {timestamp.strftime('%H:%M:%S')} → {status}")
```

### Worked Example 3: Plugin System Using OOP

```python
from abc import ABC, abstractmethod

class Plugin(ABC):
    """Base class for all plugins"""
    name = "unnamed"
    version = "1.0.0"
    
    @abstractmethod
    def process(self, data):
        pass
    
    def validate(self, data):
        return data is not None
    
    def __str__(self):
        return f"Plugin: {self.name} v{self.version}"


class UpperCasePlugin(Plugin):
    name = "uppercase"
    def process(self, data): return data.upper()

class StripPlugin(Plugin):
    name = "strip"
    def process(self, data): return data.strip()

class ReversePlugin(Plugin):
    name = "reverse"
    def process(self, data): return data[::-1]

class WordCountPlugin(Plugin):
    name = "wordcount"
    def process(self, data):
        words = data.split()
        return {"text": data, "word_count": len(words), "char_count": len(data)}


class PluginManager:
    def __init__(self):
        self._plugins = {}
    
    def register(self, plugin):
        self._plugins[plugin.name] = plugin
        print(f"Registered: {plugin}")
    
    def run(self, plugin_name, data):
        if plugin_name not in self._plugins:
            raise KeyError(f"No plugin '{plugin_name}' registered")
        plugin = self._plugins[plugin_name]
        if not plugin.validate(data):
            raise ValueError("Invalid input data")
        return plugin.process(data)
    
    def run_pipeline(self, plugin_names, data):
        """Run data through multiple plugins in sequence"""
        result = data
        for name in plugin_names:
            result = self.run(name, result)
        return result
    
    def list_plugins(self):
        return list(self._plugins.keys())


pm = PluginManager()
pm.register(StripPlugin())
pm.register(UpperCasePlugin())
pm.register(ReversePlugin())
pm.register(WordCountPlugin())

raw_text = "  hello world  "

# Run individual plugins
print(pm.run("strip", raw_text))     # "hello world"
print(pm.run("uppercase", raw_text)) # "  HELLO WORLD  "

# Run a pipeline
result = pm.run_pipeline(["strip", "uppercase", "reverse"], raw_text)
print(result)  # "DLROW OLLEH"

print(f"Available plugins: {pm.list_plugins()}")
```

---

## Practice Problems

### Easy (1–5)

**Problem 1: Rectangle Class**
Create a Rectangle class with width and height. Add methods for area, perimeter, and a check if it's a square.

```python
class Rectangle:
    def __init__(self, width, height): pass
    def area(self): pass
    def perimeter(self): pass
    def is_square(self): pass
    def __str__(self): pass

r = Rectangle(4, 6)
print(r.area())       # 24
print(r.perimeter())  # 20
print(r.is_square())  # False
print(Rectangle(5, 5).is_square())  # True
```

**Problem 2: Stack Class**
Implement a Stack data structure (Last-In, First-Out) using a class.

```python
class Stack:
    def __init__(self): pass
    def push(self, item): pass   # Add to top
    def pop(self): pass          # Remove and return top
    def peek(self): pass         # View top without removing
    def is_empty(self): pass
    def __len__(self): pass
    def __str__(self): pass

s = Stack()
s.push(1)
s.push(2)
s.push(3)
print(s.peek())  # 3
print(s.pop())   # 3
print(len(s))    # 2
```

**Problem 3: Student Grade Book**
Create a Student class that tracks grades and calculates GPA.

```python
class Student:
    def __init__(self, name, student_id): pass
    def add_grade(self, course, grade): pass  # grade is 0-100
    def gpa(self): pass        # Average of all grades
    def highest_grade(self): pass
    def lowest_grade(self): pass
    def __str__(self): pass

alice = Student("Alice", "S001")
alice.add_grade("Math", 92)
alice.add_grade("Science", 88)
alice.add_grade("English", 95)
print(alice.gpa())             # 91.67
print(alice.highest_grade())   # ('English', 95)
```

**Problem 4: Timer Class**
Create a Timer class that can be used as a context manager and also has start/stop/reset methods.

```python
class Timer:
    def __init__(self): pass
    def start(self): pass
    def stop(self): pass
    def reset(self): pass
    @property
    def elapsed(self): pass   # Seconds elapsed
    def __enter__(self): pass
    def __exit__(self, *args): pass

# Works both ways:
t = Timer()
t.start()
# ... do stuff ...
t.stop()
print(t.elapsed)

with Timer() as t:
    # ... do stuff ...
    pass
print(t.elapsed)
```

**Problem 5: Temperature Converter Class**
Build on Chapter 5's temperature concepts into a proper class.

```python
class Temperature:
    def __init__(self, value, unit="C"): pass
    def to_celsius(self): pass
    def to_fahrenheit(self): pass
    def to_kelvin(self): pass
    def __str__(self): pass
    def __eq__(self, other): pass
    def __lt__(self, other): pass   # Compare by Celsius value

t1 = Temperature(100, "C")
t2 = Temperature(212, "F")
print(t1 == t2)           # True (both are 100°C)
print(t1.to_kelvin())     # 373.15
print(str(t1))            # "100°C"
```

### Medium (6–12)

**Problem 6: Linked List**
Implement a singly linked list with all basic operations.

```python
class Node:
    def __init__(self, data): pass

class LinkedList:
    def __init__(self): pass
    def append(self, data): pass    # Add to end
    def prepend(self, data): pass   # Add to front
    def delete(self, data): pass    # Remove first occurrence
    def find(self, data): pass      # Returns True/False
    def __len__(self): pass
    def __str__(self): pass         # "1 → 2 → 3 → None"
    def __iter__(self): pass
```

**Problem 7: Observable Property**
Implement a class whose attribute changes trigger callbacks.

```python
class Observable:
    def __init__(self, initial_value):
        self._value = initial_value
        self._listeners = []
    
    def subscribe(self, callback): pass
    def unsubscribe(self, callback): pass
    
    @property
    def value(self): pass
    
    @value.setter
    def value(self, new_val): pass  # Notify all listeners on change

price = Observable(100)
price.subscribe(lambda old, new: print(f"Price: ${old} → ${new}"))
price.value = 120  # Triggers callback
price.value = 95   # Triggers callback
```

**Problem 8: Matrix Class**
Build a 2D Matrix class with arithmetic operations.

```python
class Matrix:
    def __init__(self, data): pass       # data is list of lists
    def __add__(self, other): pass       # Matrix addition
    def __mul__(self, other): pass       # Matrix * scalar OR Matrix * Matrix
    def __str__(self): pass
    def transpose(self): pass
    def __eq__(self, other): pass
    @property
    def shape(self): pass                # Returns (rows, cols)

m1 = Matrix([[1, 2], [3, 4]])
m2 = Matrix([[5, 6], [7, 8]])
print(m1 + m2)          # [[6, 8], [10, 12]]
print(m1 * 2)           # [[2, 4], [6, 8]]
print(m1.transpose())   # [[1, 3], [2, 4]]
```

**Problem 9: Job Queue**
Implement a priority job queue where higher-priority jobs run first.

```python
class Job:
    def __init__(self, name, priority, func): pass
    def __lt__(self, other): pass   # Higher priority = "less than" for min-heap

class JobQueue:
    def __init__(self): pass
    def add_job(self, job): pass
    def run_next(self): pass        # Run highest priority job
    def run_all(self): pass
    def __len__(self): pass

queue = JobQueue()
queue.add_job(Job("email", priority=1, func=lambda: print("Sending email")))
queue.add_job(Job("backup", priority=5, func=lambda: print("Running backup")))
queue.add_job(Job("report", priority=3, func=lambda: print("Generating report")))
queue.run_all()
# Running backup  (priority 5 first)
# Generating report
# Sending email
```

**Problem 10: Inventory System with Inheritance**
Build an inventory system for different product types.

```python
class Product:
    def __init__(self, name, price, quantity): pass
    def total_value(self): pass

class PhysicalProduct(Product):
    def __init__(self, name, price, quantity, weight_kg): pass
    def shipping_cost(self, rate_per_kg=5): pass

class DigitalProduct(Product):
    def __init__(self, name, price, quantity, file_size_mb): pass
    def download_time(self, speed_mbps=10): pass

class SubscriptionProduct(Product):
    def __init__(self, name, monthly_price, subscribers): pass
    def annual_revenue(self): pass
```

**Problem 11: Expression Builder**
Build a class hierarchy for mathematical expressions that can be evaluated and printed.

```python
class Expr:
    def evaluate(self): pass
    def __str__(self): pass

class Number(Expr):
    def __init__(self, value): pass

class Add(Expr):
    def __init__(self, left, right): pass

class Multiply(Expr):
    def __init__(self, left, right): pass

# Usage should work like this:
expr = Add(Multiply(Number(3), Number(4)), Number(5))
print(expr)            # (3 * 4) + 5
print(expr.evaluate()) # 17
```

**Problem 12: Mixin Classes**
Use mixins to add serialization and validation to multiple classes.

```python
class SerializableMixin:
    def to_dict(self): pass
    def to_json(self): pass
    @classmethod
    def from_dict(cls, data): pass

class ValidatableMixin:
    _required_fields = []
    def validate(self): pass   # Returns (is_valid, list_of_errors)

class User(SerializableMixin, ValidatableMixin):
    _required_fields = ["name", "email"]
    def __init__(self, name, email, age=None): pass

class Product(SerializableMixin, ValidatableMixin):
    _required_fields = ["name", "price"]
    def __init__(self, name, price, stock=0): pass
```

### Hard (13–20)

**Problem 13: Observer Pattern**
Implement the Observer design pattern from scratch.

```python
class EventEmitter:
    def on(self, event, handler): pass
    def off(self, event, handler): pass
    def emit(self, event, *args, **kwargs): pass
    def once(self, event, handler): pass  # Handler fires only once!

class StockMarket(EventEmitter):
    def __init__(self):
        super().__init__()
        self.prices = {}
    
    def update_price(self, ticker, price):
        old = self.prices.get(ticker)
        self.prices[ticker] = price
        self.emit("price_change", ticker, old, price)
        if old and abs(price - old) / old > 0.05:
            self.emit("large_move", ticker, old, price)
```

**Problem 14: Descriptor Protocol**
Implement custom descriptors for validated attributes.

```python
class PositiveNumber:
    """Descriptor: ensures attribute is always a positive number"""
    def __set_name__(self, owner, name): pass
    def __get__(self, obj, type=None): pass
    def __set__(self, obj, value): pass

class NonEmptyString:
    """Descriptor: ensures attribute is always a non-empty string"""
    def __set_name__(self, owner, name): pass
    def __get__(self, obj, type=None): pass
    def __set__(self, obj, value): pass

class Product:
    name = NonEmptyString()
    price = PositiveNumber()
    stock = PositiveNumber()
    
    def __init__(self, name, price, stock):
        self.name = name     # Triggers NonEmptyString.__set__
        self.price = price   # Triggers PositiveNumber.__set__
        self.stock = stock
```

**Problem 15: Undo/Redo System**
Implement an undo/redo system using the Command pattern.

```python
class Command(ABC):
    @abstractmethod
    def execute(self): pass
    @abstractmethod
    def undo(self): pass

class TextEditor:
    def __init__(self):
        self.text = ""
        self._history = []
        self._redo_stack = []
    
    def execute(self, command): pass
    def undo(self): pass
    def redo(self): pass

class TypeCommand(Command):
    def __init__(self, editor, text): pass
    def execute(self): pass
    def undo(self): pass

class DeleteCommand(Command):
    def __init__(self, editor, n_chars): pass
    def execute(self): pass
    def undo(self): pass

editor = TextEditor()
editor.execute(TypeCommand(editor, "Hello"))
editor.execute(TypeCommand(editor, " World"))
print(editor.text)   # Hello World
editor.undo()
print(editor.text)   # Hello
editor.redo()
print(editor.text)   # Hello World
```

**Problem 16: Lazy Property**
Implement a @lazy_property decorator that computes a value once and caches it.

```python
class lazy_property:
    """
    Computed once on first access, then cached directly on the instance.
    Subsequent accesses return cached value without recomputation.
    """
    def __init__(self, func): pass
    def __get__(self, obj, type=None): pass

class DataAnalyzer:
    def __init__(self, data):
        self.data = data
    
    @lazy_property
    def mean(self):
        print("Computing mean...")   # Should print only ONCE
        return sum(self.data) / len(self.data)
    
    @lazy_property
    def sorted_data(self):
        print("Sorting...")          # Should print only ONCE
        return sorted(self.data)

analyzer = DataAnalyzer([5, 3, 8, 1, 9, 2, 7])
print(analyzer.mean)   # Computing mean... → 5.0
print(analyzer.mean)   # 5.0 (no recomputation)
```

**Problem 17: Fluent Interface / Method Chaining**
Build a query builder with a fluent (chainable) interface.

```python
class QueryBuilder:
    def __init__(self, table): pass
    def select(self, *fields): return self       # Chainable
    def where(self, **conditions): return self   # Chainable
    def order_by(self, field, desc=False): return self
    def limit(self, n): return self
    def offset(self, n): return self
    def build(self): pass   # Return final SQL string

query = (QueryBuilder("users")
    .select("name", "email", "age")
    .where(active=True, role="admin")
    .order_by("name")
    .limit(10)
    .offset(20)
    .build())

print(query)
# SELECT name, email, age FROM users
# WHERE active = True AND role = 'admin'
# ORDER BY name ASC
# LIMIT 10 OFFSET 20
```

**Problem 18: Generic Container with Type Safety**
Build a typed list that only allows items of a specific type.

```python
class TypedList:
    def __init__(self, item_type):
        self.item_type = item_type
        self._items = []
    
    def append(self, item): pass   # Validates type
    def extend(self, items): pass
    def __getitem__(self, idx): pass
    def __len__(self): pass
    def __iter__(self): pass
    def __str__(self): pass

int_list = TypedList(int)
int_list.append(1)
int_list.append(2)
# int_list.append("hello")  # TypeError!

str_list = TypedList(str)
str_list.append("hello")
```

**Problem 19: State Machine**
Implement a generic state machine.

```python
class StateMachine:
    def __init__(self, initial_state):
        self.state = initial_state
        self._transitions = {}   # (from_state, event) → to_state
        self._callbacks = {}     # (from_state, event) → callback function
    
    def add_transition(self, from_state, event, to_state, callback=None): pass
    def trigger(self, event): pass   # Fire an event, transition state
    def can_trigger(self, event): pass

# Traffic light state machine
light = StateMachine("red")
light.add_transition("red", "go", "green", lambda: print("Go!"))
light.add_transition("green", "slow", "yellow", lambda: print("Slow down..."))
light.add_transition("yellow", "stop", "red", lambda: print("Stop!"))

light.trigger("go")    # Go! → state is now "green"
light.trigger("slow")  # Slow down... → "yellow"
light.trigger("stop")  # Stop! → "red"
# light.trigger("go")  # Would work again — cycle continues
```

**Problem 20: OOP Design Challenge — Library System**
Design and implement a complete library management system.

```python
"""
Design requirements:
- Books can be physical or digital
- Members can borrow up to 3 physical books at a time
- Digital books can be borrowed by unlimited members simultaneously
- Track borrow/return history
- Members have different tiers (standard, premium) with different limits
- Late fees calculated automatically
- Search books by title, author, genre
"""

class Book(ABC): pass
class PhysicalBook(Book): pass
class DigitalBook(Book): pass

class Member: pass
class StandardMember(Member): pass
class PremiumMember(Member): pass

class Library: pass
class BorrowRecord: pass

# Implement the full system!
```

---

## Answer Keys (Selected Problems)

### Problem 1 Solution:
```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self): return self.width * self.height
    def perimeter(self): return 2 * (self.width + self.height)
    def is_square(self): return self.width == self.height
    def __str__(self): return f"Rectangle({self.width}×{self.height})"
    def __repr__(self): return f"Rectangle({self.width}, {self.height})"
```

### Problem 2 Solution:
```python
class Stack:
    def __init__(self):
        self._items = []
    
    def push(self, item): self._items.append(item)
    def pop(self):
        if self.is_empty(): raise IndexError("Stack is empty")
        return self._items.pop()
    def peek(self):
        if self.is_empty(): raise IndexError("Stack is empty")
        return self._items[-1]
    def is_empty(self): return len(self._items) == 0
    def __len__(self): return len(self._items)
    def __str__(self): return f"Stack({self._items})"
```

### Problem 11 Solution:
```python
class Expr:
    def evaluate(self): raise NotImplementedError
    def __str__(self): raise NotImplementedError

class Number(Expr):
    def __init__(self, value): self.value = value
    def evaluate(self): return self.value
    def __str__(self): return str(self.value)

class Add(Expr):
    def __init__(self, left, right): self.left = left; self.right = right
    def evaluate(self): return self.left.evaluate() + self.right.evaluate()
    def __str__(self): return f"({self.left} + {self.right})"

class Multiply(Expr):
    def __init__(self, left, right): self.left = left; self.right = right
    def evaluate(self): return self.left.evaluate() * self.right.evaluate()
    def __str__(self): return f"({self.left} * {self.right})"
```

---

## Mini-Project: RPG Character System

```python
"""
Role-Playing Game Character System — complete OOP design
"""

from abc import ABC, abstractmethod
import random

# ── BASE CHARACTER ──────────────────────────────────────────────────────────

class Character(ABC):
    """Abstract base for all characters (players and enemies)"""
    
    def __init__(self, name, hp, attack, defense, level=1):
        self.name = name
        self._max_hp = hp
        self._hp = hp
        self.attack = attack
        self.defense = defense
        self.level = level
        self.status_effects = set()
        self.combat_log = []
    
    @property
    def hp(self): return self._hp
    
    @property
    def max_hp(self): return self._max_hp
    
    @property
    def is_alive(self): return self._hp > 0
    
    def take_damage(self, damage):
        reduced = max(0, damage - self.defense)
        self._hp = max(0, self._hp - reduced)
        msg = f"{self.name} takes {reduced} damage (HP: {self._hp}/{self._max_hp})"
        self.combat_log.append(msg)
        return reduced
    
    def heal(self, amount):
        before = self._hp
        self._hp = min(self._max_hp, self._hp + amount)
        healed = self._hp - before
        msg = f"{self.name} heals {healed} HP (HP: {self._hp}/{self._max_hp})"
        self.combat_log.append(msg)
        return healed
    
    @abstractmethod
    def special_ability(self, target): pass
    
    def basic_attack(self, target):
        damage = self.attack + random.randint(-2, 5)
        actual = target.take_damage(damage)
        return actual
    
    @property
    def hp_bar(self):
        ratio = self._hp / self._max_hp
        filled = int(ratio * 20)
        bar = "█" * filled + "░" * (20 - filled)
        return f"[{bar}] {self._hp}/{self._max_hp}"
    
    def __str__(self):
        return f"{self.name} (Lvl {self.level}) {self.hp_bar}"
    
    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}', hp={self._hp})"


# ── PLAYER CLASSES ──────────────────────────────────────────────────────────

class Warrior(Character):
    def __init__(self, name):
        super().__init__(name, hp=120, attack=18, defense=10)
        self.rage = 0
    
    def take_damage(self, damage):
        actual = super().take_damage(damage)
        self.rage = min(100, self.rage + actual // 2)
        return actual
    
    def special_ability(self, target):
        if self.rage < 30:
            return f"{self.name} lacks rage (need 30, have {self.rage})"
        bonus = self.rage // 10
        damage = self.attack * 2 + bonus
        actual = target.take_damage(damage)
        self.rage -= 30
        return f"{self.name} unleashes Rage Strike for {actual} damage!"


class Mage(Character):
    def __init__(self, name):
        super().__init__(name, hp=70, attack=25, defense=3)
        self.mana = 100
    
    def special_ability(self, target):
        if self.mana < 40:
            return f"{self.name} is out of mana!"
        damage = self.attack * 3 + random.randint(10, 25)
        actual = target.take_damage(damage)
        self.mana -= 40
        return f"{self.name} casts Fireball for {actual} damage! (Mana: {self.mana})"
    
    def basic_attack(self, target):
        damage = self.attack - 5 + random.randint(0, 10)
        return target.take_damage(damage)


class Rogue(Character):
    def __init__(self, name):
        super().__init__(name, hp=85, attack=15, defense=6)
        self.stealth = False
    
    def go_stealth(self):
        self.stealth = True
        return f"{self.name} enters stealth..."
    
    def special_ability(self, target):
        multiplier = 4 if self.stealth else 2
        damage = self.attack * multiplier + random.randint(5, 15)
        actual = target.take_damage(damage)
        self.stealth = False
        return f"{self.name} strikes from {'stealth ' if multiplier == 4 else ''}for {actual} damage!"


# ── ENEMIES ──────────────────────────────────────────────────────────────────

class Goblin(Character):
    def __init__(self):
        super().__init__("Goblin", hp=40, attack=8, defense=2)
    
    def special_ability(self, target):
        damage = self.attack + random.randint(3, 8)
        actual = target.take_damage(damage)
        return f"Goblin scratches for {actual} damage!"


class Dragon(Character):
    def __init__(self, name="Ancient Dragon"):
        super().__init__(name, hp=300, attack=30, defense=15, level=10)
        self.breath_cooldown = 0
    
    def special_ability(self, target):
        if self.breath_cooldown > 0:
            self.breath_cooldown -= 1
            return self.basic_attack(target)
        damage = 50 + random.randint(10, 30)
        actual = target.take_damage(damage)
        self.breath_cooldown = 3
        return f"Dragon breathes fire for {actual} damage! (3-turn cooldown)"


# ── ITEMS ────────────────────────────────────────────────────────────────────

class Item(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    @abstractmethod
    def use(self, character): pass
    def __str__(self): return f"{self.name}: {self.description}"


class HealthPotion(Item):
    def __init__(self, size="medium"):
        heal_amounts = {"small": 25, "medium": 50, "large": 100}
        self.heal_amount = heal_amounts.get(size, 50)
        super().__init__(f"{size.title()} Health Potion", f"Restores {self.heal_amount} HP")
    
    def use(self, character):
        healed = character.heal(self.heal_amount)
        return f"Used {self.name}: restored {healed} HP"


class AttackScroll(Item):
    def __init__(self, duration=3):
        self.duration = duration
        self.original_attack = None
        super().__init__("Attack Scroll", f"Doubles attack for {duration} turns")
    
    def use(self, character):
        self.original_attack = character.attack
        character.attack *= 2
        return f"Attack boosted to {character.attack} for {self.duration} turns!"


# ── INVENTORY ────────────────────────────────────────────────────────────────

class Inventory:
    def __init__(self, max_size=10):
        self._items = []
        self.max_size = max_size
    
    def add(self, item):
        if len(self._items) >= self.max_size:
            raise OverflowError("Inventory is full!")
        self._items.append(item)
    
    def use_item(self, index, character):
        if not (0 <= index < len(self._items)):
            raise IndexError("Invalid item index")
        item = self._items.pop(index)
        return item.use(character)
    
    def __len__(self): return len(self._items)
    def __iter__(self): return iter(self._items)
    def __str__(self):
        if not self._items:
            return "Inventory: (empty)"
        items = "\n".join(f"  [{i}] {item}" for i, item in enumerate(self._items))
        return f"Inventory ({len(self)}/{self.max_size}):\n{items}"


# ── COMBAT SYSTEM ─────────────────────────────────────────────────────────────

class Battle:
    def __init__(self, hero, enemy):
        self.hero = hero
        self.enemy = enemy
        self.turn = 1
        self.log = []
    
    def _log(self, msg):
        entry = f"Turn {self.turn}: {msg}"
        self.log.append(entry)
        print(entry)
    
    def hero_turn(self, action="attack", target=None):
        if action == "attack":
            dmg = self.hero.basic_attack(self.enemy)
            self._log(f"{self.hero.name} attacks for {dmg} damage")
        elif action == "special":
            result = self.hero.special_ability(self.enemy)
            self._log(result)
        elif action == "item" and target is not None:
            result = target
            self._log(result)
    
    def enemy_turn(self):
        if random.random() < 0.3:
            result = self.enemy.special_ability(self.hero)
            self._log(result)
        else:
            dmg = self.enemy.basic_attack(self.hero)
            self._log(f"{self.enemy.name} attacks for {dmg} damage")
    
    def is_over(self):
        return not self.hero.is_alive or not self.enemy.is_alive
    
    def winner(self):
        if self.hero.is_alive: return self.hero
        if self.enemy.is_alive: return self.enemy
        return None
    
    def auto_battle(self, max_turns=20):
        print(f"\n⚔  Battle: {self.hero.name} vs {self.enemy.name}")
        print("=" * 50)
        
        while not self.is_over() and self.turn <= max_turns:
            print(f"\n--- Turn {self.turn} ---")
            print(f"  {self.hero}")
            print(f"  {self.enemy}")
            
            # Hero acts
            if random.random() < 0.3:
                self.hero_turn("special")
            else:
                self.hero_turn("attack")
            
            if self.is_over(): break
            
            # Enemy acts
            self.enemy_turn()
            self.turn += 1
        
        print("\n" + "=" * 50)
        w = self.winner()
        if w:
            print(f"🏆 {w.name} wins after {self.turn} turns!")
        else:
            print("⚔  Draw — maximum turns reached!")
        
        return w


# ── MAIN DEMO ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create hero
    hero = Warrior("Aragorn")
    inventory = Inventory()
    inventory.add(HealthPotion("large"))
    inventory.add(HealthPotion("medium"))
    inventory.add(AttackScroll())
    
    print(hero)
    print(inventory)
    
    # Quick battle
    goblin = Goblin()
    battle1 = Battle(hero, goblin)
    battle1.auto_battle()
    
    # Heal up for boss fight
    result = inventory.use_item(0, hero)
    print(f"\n{result}")
    
    # Boss fight
    dragon = Dragon()
    battle2 = Battle(hero, dragon)
    winner = battle2.auto_battle()
```

---

## Chapter Summary

You've mastered the most powerful paradigm in software engineering!

✅ **Why OOP**: Bundle data + behavior, prevent bugs, scale with complexity
✅ **Classes & Objects**: Blueprint vs instance, `__init__`, `self`
✅ **4 Pillars**: Encapsulation (protect state), Inheritance (reuse), Polymorphism (one interface), Abstraction (hide complexity)
✅ **Magic Methods**: Make objects behave like built-in types (`__str__`, `__add__`, `__len__`, `__call__`, and more)
✅ **Properties**: `@property`, getters, setters — controlled attribute access
✅ **Class vs Static Methods**: `@classmethod` for alternative constructors, `@staticmethod` for utilities
✅ **Composition vs Inheritance**: "has-a" vs "is-a" — when to use each
✅ **Abstract Base Classes**: Define interfaces that subclasses must implement

**Key Takeaways:**
- A class is a blueprint; an object is an instance of that blueprint
- Encapsulation protects integrity; use properties instead of raw attributes
- Prefer composition over inheritance when possible — more flexible
- Magic methods let your objects integrate seamlessly with Python
- Abstract base classes enforce consistent interfaces across subclasses

**Next Chapter Preview:**
Chapter 9 covers **Modules and Packages** — how to organize large codebases, import standard library tools, create your own reusable modules, and manage dependencies with virtual environments!
