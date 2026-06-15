# Chapter 12: Pandas — Data Analysis in Python

## Part 1: Why Pandas Exists

### The Problem: NumPy Is Powerful But Labeless

NumPy is incredible for numerical operations, but it has a critical weakness for real-world data:

```python
import numpy as np

# You have sales data: 5 products, 4 metrics
data = np.array([
    [1200, 45, 26.67, "Electronics"],
    [890,  32, 27.81, "Clothing"],
    [2300, 78, 29.49, "Electronics"],
    [450,  15, 30.00, "Food"],
    [3100, 90, 34.44, "Electronics"],
])

# Questions:
# What's the average revenue?  → data[:, 0].mean()  — but dtype is object!
# Total units for Electronics? → data[data[:, 3] == "Electronics", 1].sum()
# Revenue per unit?            → you already have it in col 2 — but what is col 2?

# The problems:
# 1. Mixed types (int, float, string) → everything becomes object dtype
# 2. Columns have no names — you must remember what col 0, 1, 2 means
# 3. No index — rows have no identity
# 4. String filtering is clunky and brittle
```

**Pandas solves all of this:**

```python
import pandas as pd

# Same data, but now labelled, typed, and queryable
df = pd.DataFrame({
    "revenue": [1200, 890, 2300, 450, 3100],
    "units":   [45, 32, 78, 15, 90],
    "price":   [26.67, 27.81, 29.49, 30.00, 34.44],
    "category": ["Electronics", "Clothing", "Electronics", "Food", "Electronics"]
})

# Now questions become readable:
print(df["revenue"].mean())                                    # 1608.0
print(df[df["category"] == "Electronics"]["units"].sum())     # 213
print(df.groupby("category")["revenue"].mean())               # Average by category
```

### What Is Pandas?

Pandas provides two fundamental data structures:

- **Series**: A 1D labelled array (like a dictionary + array combined)
- **DataFrame**: A 2D labelled table (like a spreadsheet or SQL table)

Think of a DataFrame as a dictionary of Series — each column is a Series.

```
DataFrame:
         name    age    salary    dept
index
0        Alice   28     95000     Engineering
1        Bob     35     72000     Marketing
2        Carol   42     110000    Engineering
3        Dave    29     68000     Marketing
   ↑      ↑      ↑       ↑          ↑
   index  col    col     col        col
          (each column is a Series)
```

---

## Part 2: Series — The 1D Building Block

```python
import pandas as pd
import numpy as np

# ── CREATING A SERIES ─────────────────────────────────────────────────────

# From a list — index is 0, 1, 2... by default
s = pd.Series([10, 20, 30, 40, 50])
print(s)
# 0    10
# 1    20
# 2    30
# 3    40
# 4    50
# dtype: int64

# With a custom index — NOW it's like a labelled array
prices = pd.Series(
    [150.5, 203.2, 89.9, 445.0, 312.8],
    index=["AAPL", "GOOGL", "AMZN", "MSFT", "META"]
)
print(prices)
# AAPL     150.5
# GOOGL    203.2
# AMZN      89.9
# MSFT     445.0
# META     312.8
# dtype: float64

# From a dictionary — keys become the index
population = pd.Series({
    "Mumbai": 20_667_656,
    "Delhi":  32_941_308,
    "Bangalore": 12_765_000,
    "Chennai": 7_794_000
})

# ── ACCESSING DATA ────────────────────────────────────────────────────────

# By label (like dict)
print(prices["AAPL"])     # 150.5
print(prices[["AAPL", "MSFT"]])  # Multiple labels

# By position (like list)
print(prices.iloc[0])     # 150.5  — first element
print(prices.iloc[-1])    # 312.8  — last element

# Slicing by label (inclusive on both ends!)
print(prices["AAPL":"MSFT"])   # Includes MSFT

# Slicing by position (exclusive end — like Python)
print(prices.iloc[0:3])        # 0, 1, 2

# Boolean indexing
print(prices[prices > 200])
# GOOGL    203.2
# MSFT     445.0
# META     312.8

# ── PROPERTIES ────────────────────────────────────────────────────────────
print(prices.index)    # Index(['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'META'])
print(prices.values)   # array([150.5, 203.2,  89.9, 445. , 312.8])
print(prices.dtype)    # float64
print(prices.shape)    # (5,)
print(len(prices))     # 5
print(prices.name)     # None (can set: prices.name = "Stock Price")

# ── OPERATIONS ────────────────────────────────────────────────────────────
# All NumPy-style operations work on Series!
print(prices.mean())    # 240.28
print(prices.max())     # 445.0
print(prices.std())     # 134.8
print(prices.describe())
# count      5.000000
# mean     240.280000
# std      134.806988
# min       89.900000
# 25%      150.500000
# 50%      203.200000
# 75%      312.800000
# max      445.000000

# Vectorized operations (like NumPy)
print(prices * 1.1)       # 10% price increase on all
print(prices - prices.mean())  # Mean-centered

# Operations align on INDEX automatically!
series_a = pd.Series({"a": 1, "b": 2, "c": 3})
series_b = pd.Series({"b": 10, "c": 20, "d": 30})
print(series_a + series_b)
# a     NaN   ← 'a' not in series_b
# b    12.0
# c    23.0
# d     NaN   ← 'd' not in series_a
# (NaN = "Not a Number" = missing value)
```

---

## Part 3: DataFrame — The 2D Workhorse

### Creating DataFrames

```python
# ── FROM DICTIONARY OF LISTS ──────────────────────────────────────────────
df = pd.DataFrame({
    "name":    ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "age":     [28, 35, 42, 29, 38],
    "salary":  [95000, 72000, 110000, 68000, 88000],
    "dept":    ["Engineering", "Marketing", "Engineering", "Marketing", "Product"],
    "active":  [True, True, True, False, True]
})

print(df)
#     name  age  salary         dept  active
# 0  Alice   28   95000  Engineering    True
# 1    Bob   35   72000    Marketing    True
# 2  Carol   42  110000  Engineering    True
# 3   Dave   29   68000    Marketing   False
# 4    Eve   38   88000      Product    True

# ── FROM LIST OF DICTIONARIES (common from APIs/JSON) ─────────────────────
records = [
    {"name": "Alice", "age": 28, "salary": 95000},
    {"name": "Bob",   "age": 35, "salary": 72000},
    {"name": "Carol", "age": 42},  # Missing salary — becomes NaN
]
df2 = pd.DataFrame(records)
print(df2)
#     name  age    salary
# 0  Alice   28   95000.0
# 1    Bob   35   72000.0
# 2  Carol   42       NaN

# ── FROM NUMPY ARRAY ─────────────────────────────────────────────────────
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df3 = pd.DataFrame(arr, columns=["A", "B", "C"])

# ── WITH CUSTOM INDEX ─────────────────────────────────────────────────────
df4 = pd.DataFrame(
    {"price": [100, 200, 150], "volume": [1000, 500, 750]},
    index=["2025-01-01", "2025-01-02", "2025-01-03"]
)

# ── KEY PROPERTIES ────────────────────────────────────────────────────────
print(df.shape)        # (5, 5) — (rows, columns)
print(df.dtypes)       # dtype of each column
print(df.columns)      # Index(['name', 'age', 'salary', 'dept', 'active'])
print(df.index)        # RangeIndex(start=0, stop=5, step=1)
print(df.info())       # Summary: dtypes, non-null counts, memory
print(df.describe())   # Stats for numeric columns only
print(df.head(3))      # First 3 rows
print(df.tail(2))      # Last 2 rows
```

---

## Part 4: Reading and Writing Data

```python
# ── READING CSV ───────────────────────────────────────────────────────────
df = pd.read_csv("data.csv")

# With options:
df = pd.read_csv(
    "data.csv",
    index_col="employee_id",   # Use this column as the index
    parse_dates=["hire_date"],  # Convert to datetime
    dtype={"salary": float},    # Force dtype
    na_values=["N/A", "?", ""],# Treat these as NaN
    nrows=1000,                 # Read only first 1000 rows
    usecols=["name", "salary", "dept"],  # Read only these columns
    encoding="utf-8"
)

# ── WRITING CSV ───────────────────────────────────────────────────────────
df.to_csv("output.csv", index=False)   # index=False prevents writing row numbers

# ── READING EXCEL ─────────────────────────────────────────────────────────
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# Multiple sheets:
all_sheets = pd.read_excel("data.xlsx", sheet_name=None)  # Dict of DataFrames

# ── WRITING EXCEL ─────────────────────────────────────────────────────────
with pd.ExcelWriter("output.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Main Data", index=False)
    summary.to_excel(writer, sheet_name="Summary", index=False)

# ── JSON ─────────────────────────────────────────────────────────────────
df = pd.read_json("data.json")
df.to_json("output.json", orient="records", indent=2)

# ── SQL ──────────────────────────────────────────────────────────────────
import sqlite3
conn = sqlite3.connect("database.db")
df = pd.read_sql("SELECT * FROM employees WHERE salary > 80000", conn)
df.to_sql("employees_backup", conn, if_exists="replace", index=False)
conn.close()

# ── QUICK INSPECTION WORKFLOW ─────────────────────────────────────────────
def quick_inspect(df, name="DataFrame"):
    """Always run these after loading any new dataset."""
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn Types:")
    print(df.dtypes)
    print(f"\nNull Values:")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0])
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print(f"\nNumeric Summary:")
    print(df.describe())
```

---

## Part 5: Selection and Filtering

### Column Selection

```python
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "age":  [28, 35, 42, 29, 38],
    "salary": [95000, 72000, 110000, 68000, 88000],
    "dept": ["Engineering", "Marketing", "Engineering", "Marketing", "Product"]
})

# ── SINGLE COLUMN → returns Series ───────────────────────────────────────
print(df["name"])          # Series
print(df.name)             # Same, but dot notation (avoid: conflicts with methods)

# ── MULTIPLE COLUMNS → returns DataFrame ─────────────────────────────────
print(df[["name", "salary"]])
print(df[["age", "dept", "salary"]])   # Reorder too!
```

### Row Selection — `loc` vs `iloc`

```python
# ── loc — LABEL-based (use with string/named indices) ────────────────────
# iloc — INTEGER-based (use with position numbers)

# Set up a DataFrame with named index
df.index = ["emp_001", "emp_002", "emp_003", "emp_004", "emp_005"]

# loc: select by label
print(df.loc["emp_001"])                    # One row as Series
print(df.loc["emp_001":"emp_003"])          # Rows (INCLUSIVE end!)
print(df.loc["emp_001", "salary"])          # One cell
print(df.loc["emp_001":"emp_003", ["name", "salary"]])  # Rows + specific cols

# iloc: select by position
print(df.iloc[0])                           # First row
print(df.iloc[-1])                          # Last row
print(df.iloc[0:3])                         # First 3 rows (EXCLUSIVE end)
print(df.iloc[0, 2])                        # Row 0, column 2
print(df.iloc[0:3, [0, 2]])                 # First 3 rows, cols 0 and 2

# ── THE GOLDEN RULE ───────────────────────────────────────────────────────
# df[col]       → column selection (use column name)
# df.loc[row]   → row selection by LABEL
# df.iloc[row]  → row selection by POSITION
# df.loc[row, col] → specific cell by labels
# df.iloc[row, col] → specific cell by positions
```

### Boolean Filtering — The Most Common Operation

```python
df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "age":    [28, 35, 42, 29, 38],
    "salary": [95000, 72000, 110000, 68000, 88000],
    "dept":   ["Engineering", "Marketing", "Engineering", "Marketing", "Product"]
})

# ── SINGLE CONDITION ─────────────────────────────────────────────────────
high_earners = df[df["salary"] > 80000]
print(high_earners)
#     name  age  salary         dept
# 0  Alice   28   95000  Engineering
# 2  Carol   42  110000  Engineering
# 4    Eve   38   88000      Product

engineers = df[df["dept"] == "Engineering"]
print(engineers)

# ── MULTIPLE CONDITIONS (& | ~) ───────────────────────────────────────────
# Use & (and), | (or), ~ (not) — NOT Python's and/or/not!
# Always wrap each condition in parentheses!

senior_engineers = df[(df["dept"] == "Engineering") & (df["age"] > 30)]
print(senior_engineers)

well_paid_or_young = df[(df["salary"] > 90000) | (df["age"] < 30)]
print(well_paid_or_young)

not_marketing = df[~(df["dept"] == "Marketing")]
print(not_marketing)

# ── STRING METHODS ────────────────────────────────────────────────────────
# .str accessor gives you string operations on Series
names = pd.Series(["Alice Johnson", "Bob Smith", "Carol Davis"])

print(names.str.lower())           # ["alice johnson", "bob smith", "carol davis"]
print(names.str.upper())
print(names.str.len())             # [13, 9, 11]
print(names.str.contains("Smith")) # [False, True, False]
print(names.str.startswith("A"))   # [True, False, False]
print(names.str.replace("o", "0")) # ["Alice J0hns0n", ...]
print(names.str.split(" "))        # [["Alice", "Johnson"], ...]

# Filter rows with string methods
print(df[df["name"].str.startswith("A")])
print(df[df["dept"].str.contains("ing")])  # Engineering, Marketing

# ── isin — check membership ───────────────────────────────────────────────
tech_depts = ["Engineering", "Product"]
tech_employees = df[df["dept"].isin(tech_depts)]
print(tech_employees)

# ── between — range check ─────────────────────────────────────────────────
mid_range = df[df["salary"].between(70000, 100000)]
print(mid_range)

# ── query — SQL-like syntax (readable for complex conditions) ─────────────
result = df.query("salary > 80000 and dept == 'Engineering'")
print(result)

result = df.query("age in [28, 35, 42]")
result = df.query("name.str.startswith('A')", engine="python")
```

---

## Part 6: Data Manipulation

### Adding and Modifying Columns

```python
df = pd.DataFrame({
    "name":    ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "salary":  [95000.0, 72000.0, 110000.0, 68000.0, 88000.0],
    "dept":    ["Engineering", "Marketing", "Engineering", "Marketing", "Product"],
    "years":   [4, 2, 7, 1, 5]
})

# ── ADD NEW COLUMNS ───────────────────────────────────────────────────────
df["monthly_salary"] = df["salary"] / 12
df["bonus"] = df["salary"] * 0.10
df["total_comp"] = df["salary"] + df["bonus"]

# Conditional column
df["seniority"] = pd.cut(
    df["years"],
    bins=[0, 2, 5, 100],
    labels=["Junior", "Mid", "Senior"]
)

# Using np.where
df["is_senior_dept"] = np.where(df["dept"] == "Engineering", "Tech", "Non-Tech")

# Apply a function to each row
df["email"] = df["name"].str.lower().str.replace(" ", ".") + "@company.com"

# ── MODIFY EXISTING COLUMNS ───────────────────────────────────────────────
df["salary"] = df["salary"] * 1.05    # 5% raise for everyone
df["dept"] = df["dept"].str.upper()   # Normalize department names

# ── DROP COLUMNS / ROWS ───────────────────────────────────────────────────
df = df.drop(columns=["monthly_salary"])           # Drop one column
df = df.drop(columns=["bonus", "total_comp"])      # Drop multiple
df = df.drop(index=3)                              # Drop row with index 3
df = df.drop(index=[0, 2])                         # Drop multiple rows

# ── RENAME COLUMNS ────────────────────────────────────────────────────────
df = df.rename(columns={"years": "years_of_experience", "name": "full_name"})

# Or rename all at once
df.columns = ["full_name", "salary", "department", "years_exp", "seniority", "email", "track"]

# ── REORDER COLUMNS ───────────────────────────────────────────────────────
desired_order = ["full_name", "department", "seniority", "salary", "email"]
df = df[desired_order]  # Only these columns, in this order

# ── .assign() — chainable column addition ─────────────────────────────────
df = (pd.DataFrame({
    "salary": [95000, 72000, 110000],
    "bonus_pct": [0.10, 0.08, 0.12]
})
.assign(bonus=lambda x: x["salary"] * x["bonus_pct"])
.assign(total_comp=lambda x: x["salary"] + x["bonus"])
.assign(tax=lambda x: x["total_comp"] * 0.30)
)
print(df)
```

### Sorting

```python
# Sort by one column
df = df.sort_values("salary")              # Ascending (default)
df = df.sort_values("salary", ascending=False)  # Descending

# Sort by multiple columns
df = df.sort_values(["dept", "salary"], ascending=[True, False])
# Sort by dept alphabetically, then salary descending within each dept

# Sort by index
df = df.sort_index()

# Reset index after sorting (row numbers become 0,1,2... again)
df = df.sort_values("salary").reset_index(drop=True)
# drop=True prevents old index from becoming a column
```

### Applying Functions

```python
df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol"],
    "salary": [95000.0, 72000.0, 110000.0],
    "dept":   ["Engineering", "Marketing", "Engineering"]
})

# ── apply — apply function to each element, row, or column ───────────────

# To a single column (element-wise)
df["salary_fmt"] = df["salary"].apply(lambda x: f"${x:,.0f}")

# Custom function to column
def salary_band(salary):
    if salary < 70000: return "Low"
    if salary < 90000: return "Mid"
    return "High"

df["band"] = df["salary"].apply(salary_band)

# To each row (axis=1)
def create_profile(row):
    return f"{row['name']} ({row['dept']}): {row['salary_fmt']}"

df["profile"] = df.apply(create_profile, axis=1)

# To each column (axis=0, default)
numeric_cols = df.select_dtypes(include=np.number)
col_means = numeric_cols.apply("mean")

# ── map — element-wise for Series ────────────────────────────────────────
dept_codes = {"Engineering": "ENG", "Marketing": "MKT", "Product": "PRD"}
df["dept_code"] = df["dept"].map(dept_codes)

# ── applymap / map (for DataFrames) — apply to every cell ────────────────
numeric_df = df[["salary"]]
formatted = numeric_df.map(lambda x: f"${x:,.0f}")

# ── vectorized operations are MUCH faster than apply ─────────────────────
# ❌ Slow: uses Python loop internally
df["salary_2x"] = df["salary"].apply(lambda x: x * 2)

# ✓ Fast: vectorized
df["salary_2x"] = df["salary"] * 2
# Use apply only when vectorized alternatives don't exist!
```

---

## Part 7: Data Cleaning

### The Most Critical Skill

```python
# ── MISSING VALUES ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "name":   ["Alice", "Bob", None, "Dave", "Eve"],
    "age":    [28, None, 42, 29, 38],
    "salary": [95000, 72000, None, 68000, None],
    "dept":   ["Engineering", "Marketing", "Engineering", None, "Product"]
})

print(df)
#     name   age    salary         dept
# 0  Alice  28.0   95000.0  Engineering
# 1    Bob   NaN   72000.0    Marketing
# 2   None  42.0       NaN  Engineering
# 3   Dave  29.0   68000.0         None
# 4    Eve  38.0       NaN      Product

# ── DETECTING MISSING VALUES ──────────────────────────────────────────────
print(df.isnull())         # Boolean DataFrame: True where NaN
print(df.isnull().sum())   # Count nulls per column
print(df.isnull().sum().sum())  # Total null count
print(df.notnull())        # Opposite of isnull
print(df[df["salary"].isnull()])   # Rows with missing salary

# Percentage of nulls
print((df.isnull().sum() / len(df) * 100).round(1))

# ── DROPPING MISSING VALUES ───────────────────────────────────────────────
df.dropna()                     # Drop rows with ANY null
df.dropna(how="all")            # Drop rows where ALL values are null
df.dropna(subset=["salary"])    # Drop rows with null in salary only
df.dropna(thresh=3)             # Drop rows with fewer than 3 non-null values
df.dropna(axis=1)               # Drop COLUMNS with any null

# ── FILLING MISSING VALUES ────────────────────────────────────────────────
# Fill with a constant
df["dept"].fillna("Unknown")

# Fill with column statistics
df["age"].fillna(df["age"].mean())      # Fill with mean
df["age"].fillna(df["age"].median())    # Fill with median
df["age"].fillna(df["age"].mode()[0])   # Fill with mode

# Forward fill (use previous value)
df["salary"].ffill()   # or df["salary"].fillna(method="ffill")

# Backward fill (use next value)
df["salary"].bfill()

# Interpolation (for time series)
df["salary"].interpolate()

# Fill different columns differently
fill_values = {
    "name": "Unknown",
    "age": df["age"].median(),
    "salary": df["salary"].mean(),
    "dept": "Unassigned"
}
df.fillna(fill_values)

# ── DUPLICATES ────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Alice", "Carol", "Bob"],
    "dept": ["Eng", "Mkt", "Eng", "Prod", "Mkt"],
    "salary": [95000, 72000, 95000, 88000, 72000]
})

print(df.duplicated())                      # Boolean Series
print(df.duplicated().sum())               # Count duplicates
print(df[df.duplicated(keep=False)])       # Show ALL duplicate rows

df.drop_duplicates()                       # Remove duplicate rows
df.drop_duplicates(subset=["name"])        # Based on specific column
df.drop_duplicates(subset=["name"], keep="last")  # Keep last occurrence

# ── TYPE CONVERSION ───────────────────────────────────────────────────────
df = pd.DataFrame({
    "age":    ["28", "35", "42"],      # Should be int
    "salary": ["95,000", "72,000", "110,000"],  # Has commas!
    "date":   ["2025-01-15", "2025-02-20", "2025-03-10"],
    "is_active": ["True", "False", "True"]
})

# Convert column types
df["age"] = df["age"].astype(int)
df["salary"] = df["salary"].str.replace(",", "").astype(float)
df["date"] = pd.to_datetime(df["date"])
df["is_active"] = df["is_active"].map({"True": True, "False": False})

# Safe conversion with errors="coerce" (bad values become NaN)
df["age"] = pd.to_numeric(df["age"], errors="coerce")

# ── STRING CLEANING ───────────────────────────────────────────────────────
raw = pd.Series(["  Alice  ", "BOB", "carol johnson", "DAVE SMITH"])

cleaned = (raw
    .str.strip()           # Remove leading/trailing whitespace
    .str.title()           # Title case
)

# Remove special characters
df["phone"] = df["phone"].str.replace(r"[^\d]", "", regex=True)  # Keep only digits

# ── OUTLIER HANDLING ──────────────────────────────────────────────────────
salaries = pd.Series([95000, 72000, 110000, 68000, 88000, 1000000, 45000])

# IQR method
Q1 = salaries.quantile(0.25)
Q3 = salaries.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

is_outlier = (salaries < lower) | (salaries > upper)
print(f"Outliers: {salaries[is_outlier].tolist()}")  # [1000000]

# Cap outliers (winsorize)
salaries_capped = salaries.clip(lower=lower, upper=upper)
```

---

## Part 8: GroupBy — The Most Powerful Feature

### The Split-Apply-Combine Pattern

```python
df = pd.DataFrame({
    "name":    ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace"],
    "dept":    ["Engineering", "Marketing", "Engineering", "Marketing", "Product", "Engineering", "Product"],
    "salary":  [95000, 72000, 110000, 68000, 88000, 102000, 91000],
    "years":   [4, 2, 7, 1, 5, 6, 3],
    "gender":  ["F", "M", "F", "M", "F", "M", "F"]
})

# ── THE CONCEPT ───────────────────────────────────────────────────────────
# 1. SPLIT: Group the DataFrame by "dept"
# 2. APPLY: Apply a function to each group
# 3. COMBINE: Combine results into a new DataFrame

# ── BASIC GROUPBY + AGGREGATION ──────────────────────────────────────────
grouped = df.groupby("dept")

# Single aggregation
print(grouped["salary"].mean())
# dept
# Engineering    102333.33
# Marketing       70000.00
# Product         89500.00

print(grouped["salary"].sum())
print(grouped["salary"].max())
print(grouped["salary"].count())
print(grouped.size())   # Count rows in each group

# ── MULTIPLE AGGREGATIONS ─────────────────────────────────────────────────
# agg() lets you apply multiple functions
result = grouped["salary"].agg(["mean", "min", "max", "count", "std"])
print(result)

# Different aggregations for different columns
result = grouped.agg({
    "salary": ["mean", "max", "min"],
    "years": ["mean", "sum"]
})
print(result)

# Named aggregations (clean column names)
result = grouped.agg(
    avg_salary=("salary", "mean"),
    max_salary=("salary", "max"),
    headcount=("salary", "count"),
    avg_years=("years", "mean")
)
print(result)
#              avg_salary  max_salary  headcount  avg_years
# dept
# Engineering  102333.33      110000          3       5.67
# Marketing     70000.00       72000          2       1.50
# Product       89500.00       91000          2       4.00

# ── GROUPBY MULTIPLE COLUMNS ──────────────────────────────────────────────
result = df.groupby(["dept", "gender"])["salary"].mean()
print(result)
# dept         gender
# Engineering  F         102500.0
#              M         102000.0
# Marketing    M          70000.0
# Product      F          89500.0

# ── TRANSFORM — return same-size DataFrame ────────────────────────────────
# Unlike agg (which reduces), transform returns a value for each row

# Z-score within department
df["salary_zscore_in_dept"] = grouped["salary"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Rank within department
df["rank_in_dept"] = grouped["salary"].transform("rank", ascending=False)

# Department mean as new column
df["dept_avg_salary"] = grouped["salary"].transform("mean")
df["pct_above_dept_avg"] = (df["salary"] - df["dept_avg_salary"]) / df["dept_avg_salary"]

print(df[["name", "dept", "salary", "dept_avg_salary", "pct_above_dept_avg"]].round(2))

# ── FILTER — keep groups matching a condition ─────────────────────────────
# Keep only departments with average salary > 90000
high_pay_depts = grouped.filter(lambda x: x["salary"].mean() > 90000)
print(high_pay_depts)

# Keep departments with at least 3 employees
large_depts = grouped.filter(lambda x: len(x) >= 3)

# ── APPLY — most flexible (but slowest) ──────────────────────────────────
def top_earner_report(group):
    """For each department, return the top earner's info."""
    return group.nlargest(1, "salary")[["name", "salary"]]

top_earners = grouped.apply(top_earner_report)
print(top_earners)
```

---

## Part 9: Merging and Joining

```python
# ── DATAFRAMES TO MERGE ───────────────────────────────────────────────────
employees = pd.DataFrame({
    "emp_id": [1, 2, 3, 4, 5],
    "name":   ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "dept_id":[101, 102, 101, 103, 102]
})

departments = pd.DataFrame({
    "dept_id":   [101, 102, 103, 104],
    "dept_name": ["Engineering", "Marketing", "Product", "Finance"],
    "budget":    [500000, 200000, 300000, 400000]
})

salaries = pd.DataFrame({
    "emp_id": [1, 2, 3, 5, 6],   # Note: Dave (4) missing, Frank (6) extra
    "salary": [95000, 72000, 110000, 88000, 92000]
})

# ── MERGE (like SQL JOIN) ─────────────────────────────────────────────────

# INNER JOIN — only matching rows from both
inner = pd.merge(employees, departments, on="dept_id", how="inner")
print(inner)
#    emp_id   name  dept_id    dept_name   budget
# 0       1  Alice      101  Engineering  500000
# 1       3  Carol      101  Engineering  500000
# 2       2    Bob      102    Marketing  200000
# 3       5    Eve      102    Marketing  200000
# 4       4   Dave      103      Product  300000

# LEFT JOIN — all left rows, matching right rows (nulls if no match)
left = pd.merge(employees, salaries, on="emp_id", how="left")
print(left)
# Dave (emp_id=4) will have salary=NaN since not in salaries

# RIGHT JOIN — all right rows, matching left rows
right = pd.merge(employees, salaries, on="emp_id", how="right")
# Frank (emp_id=6) will have name=NaN

# OUTER JOIN — all rows from both, NaN where no match
outer = pd.merge(employees, salaries, on="emp_id", how="outer")

# Different column names
result = pd.merge(
    employees,
    departments.rename(columns={"dept_id": "department_id"}),
    left_on="dept_id",
    right_on="department_id"
)

# ── CONCAT — stack DataFrames ─────────────────────────────────────────────
# Stack VERTICALLY (add rows)
df1 = pd.DataFrame({"name": ["Alice"], "dept": ["Engineering"]})
df2 = pd.DataFrame({"name": ["Bob"],   "dept": ["Marketing"]})
combined = pd.concat([df1, df2], ignore_index=True)

# Stack multiple DataFrames
monthly_data = [pd.read_csv(f"sales_{month}.csv") for month in range(1, 13)]
annual = pd.concat(monthly_data, ignore_index=True)

# Add a label to track which DataFrame each row came from
all_data = pd.concat([df1, df2], keys=["group_a", "group_b"])

# Stack HORIZONTALLY (add columns)
combined_cols = pd.concat([df1, df2], axis=1)
```

---

## Part 10: Dates and Time Series

```python
# ── CREATING DATE RANGES ─────────────────────────────────────────────────
dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")   # Daily
biz_days = pd.date_range("2025-01-01", "2025-12-31", freq="B")  # Business days
months = pd.date_range("2025-01-01", "2025-12-31", freq="ME")   # Month end
quarters = pd.date_range("2025-01-01", "2025-12-31", freq="QE") # Quarter end

# ── DATETIME OPERATIONS ───────────────────────────────────────────────────
df = pd.DataFrame({
    "date":   pd.date_range("2025-01-01", periods=100, freq="D"),
    "value":  np.random.randn(100).cumsum() + 100
})

# Extract date parts
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"]   = df["date"].dt.day
df["weekday"] = df["date"].dt.day_name()
df["is_weekend"] = df["date"].dt.dayofweek >= 5
df["quarter"] = df["date"].dt.quarter

# ── TIME SERIES OPERATIONS ────────────────────────────────────────────────
# Set date as index (standard for time series)
df = df.set_index("date")

# Resample — like groupby for time
monthly_avg = df["value"].resample("ME").mean()   # Monthly average
weekly_sum  = df["value"].resample("W").sum()      # Weekly sum
quarterly   = df["value"].resample("QE").agg(["mean", "min", "max"])

# Rolling windows
df["rolling_7d_avg"]  = df["value"].rolling(window=7).mean()
df["rolling_30d_std"] = df["value"].rolling(window=30).std()
df["rolling_7d_max"]  = df["value"].rolling(window=7).max()

# Expanding window (cumulative since start)
df["cumulative_avg"] = df["value"].expanding().mean()

# Shift (lag/lead)
df["value_yesterday"] = df["value"].shift(1)   # Previous day
df["value_tomorrow"]  = df["value"].shift(-1)  # Next day
df["daily_change"]    = df["value"] - df["value"].shift(1)
df["pct_change"]      = df["value"].pct_change()   # % change

# Select by date range
jan_data = df["2025-01"]            # All of January
q1_data  = df["2025-01":"2025-03"]  # Q1
recent   = df["2025-03-01":]        # March onwards
```

---

## Part 11: Pivot Tables and Reshaping

```python
df = pd.DataFrame({
    "date":    pd.date_range("2025-01-01", periods=12, freq="ME"),
    "region":  ["North", "South", "East", "West"] * 3,
    "product": ["A", "A", "B", "B", "A", "A", "B", "B", "A", "A", "B", "B"],
    "revenue": np.random.randint(10000, 100000, 12),
    "units":   np.random.randint(100, 1000, 12)
})

# ── PIVOT TABLE ───────────────────────────────────────────────────────────
pivot = pd.pivot_table(
    df,
    values="revenue",           # What to aggregate
    index="region",             # Rows
    columns="product",          # Columns
    aggfunc="sum",              # How to aggregate
    margins=True,               # Add row/column totals
    fill_value=0                # Replace NaN with 0
)
print(pivot)
# product      A       B    All
# region
# East     xxxxx   xxxxx  xxxxx
# North    xxxxx   xxxxx  xxxxx
# South    xxxxx   xxxxx  xxxxx
# West     xxxxx   xxxxx  xxxxx
# All      xxxxx   xxxxx  xxxxx

# Multiple aggregations
pivot2 = pd.pivot_table(
    df,
    values=["revenue", "units"],
    index="region",
    columns="product",
    aggfunc={"revenue": "sum", "units": "mean"}
)

# ── MELT — wide to long format ────────────────────────────────────────────
wide = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "Jan": [1200, 900],
    "Feb": [1400, 1100],
    "Mar": [1300, 950]
})

# Wide format:        Long format:
# name Jan Feb Mar    name month value
# Alice 1200 1400 ...  Alice Jan  1200
# Bob    900 1100 ...  Alice Feb  1400
#                      Alice Mar  1300
#                      Bob   Jan   900

long = pd.melt(
    wide,
    id_vars=["name"],           # Columns to keep as-is
    value_vars=["Jan", "Feb", "Mar"],  # Columns to melt
    var_name="month",           # Name for the new "variable" column
    value_name="revenue"        # Name for the new "value" column
)
print(long)

# ── CROSSTAB ─ frequency table ────────────────────────────────────────────
ct = pd.crosstab(df["region"], df["product"])
ct_normalized = pd.crosstab(df["region"], df["product"], normalize="index")  # Row %
```

---

## Part 12: Worked Examples

### Worked Example 1: Full Data Cleaning Pipeline

```python
import pandas as pd
import numpy as np

def clean_employee_dataset(filepath):
    """
    Complete data cleaning pipeline for an employee dataset.
    Returns clean, ready-to-analyze DataFrame.
    """
    # ── 1. LOAD ───────────────────────────────────────────────────────────
    df = pd.read_csv(filepath)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Null values:\n{df.isnull().sum()}\n")
    
    # ── 2. RENAME COLUMNS ─────────────────────────────────────────────────
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # ── 3. FIX TYPES ──────────────────────────────────────────────────────
    # Salary has commas: "$95,000" → 95000.0
    df["salary"] = (df["salary"]
        .str.replace(r"[$,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce"))
    
    # Date parsing
    df["hire_date"] = pd.to_datetime(df["hire_date"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    
    # ── 4. HANDLE MISSING VALUES ──────────────────────────────────────────
    # Fill salary with department median (more accurate than overall median)
    df["salary"] = df.groupby("department")["salary"].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Fill categorical with mode
    df["department"] = df["department"].fillna(df["department"].mode()[0])
    
    # Fill age with median
    df["age"] = df["age"].fillna(df["age"].median()).astype(int)
    
    # Drop rows still missing critical fields
    df = df.dropna(subset=["name", "hire_date"])
    
    # ── 5. CLEAN STRINGS ──────────────────────────────────────────────────
    df["name"] = df["name"].str.strip().str.title()
    df["department"] = df["department"].str.strip().str.title()
    df["email"] = df["email"].str.strip().str.lower()
    
    # ── 6. REMOVE DUPLICATES ──────────────────────────────────────────────
    dupes_before = df.duplicated(subset=["email"]).sum()
    df = df.drop_duplicates(subset=["email"], keep="first")
    print(f"Removed {dupes_before} duplicate emails")
    
    # ── 7. HANDLE OUTLIERS ────────────────────────────────────────────────
    # Cap salary outliers at 3 std deviations
    mean, std = df["salary"].mean(), df["salary"].std()
    df["salary"] = df["salary"].clip(
        lower=mean - 3*std,
        upper=mean + 3*std
    )
    
    # ── 8. ADD DERIVED COLUMNS ────────────────────────────────────────────
    today = pd.Timestamp.today()
    df["years_of_service"] = ((today - df["hire_date"]).dt.days / 365).round(1)
    df["salary_band"] = pd.cut(
        df["salary"],
        bins=[0, 60000, 90000, 120000, float("inf")],
        labels=["Entry", "Mid", "Senior", "Executive"]
    )
    
    # ── 9. VALIDATE ───────────────────────────────────────────────────────
    assert df["salary"].isnull().sum() == 0, "Nulls remain in salary!"
    assert df["age"].between(18, 80).all(), "Invalid ages found!"
    assert df.duplicated(subset=["email"]).sum() == 0, "Duplicate emails!"
    
    print(f"\nClean dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


### Worked Example 2: Sales Analytics Report

def sales_analytics(df):
    """
    Comprehensive sales analysis using groupby, pivot tables, and time series.
    """
    # Assume df has: date, region, product, revenue, units, cost
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
    df["profit"] = df["revenue"] - df["cost"]
    df["margin"] = df["profit"] / df["revenue"]
    
    print("=" * 60)
    print("SALES ANALYTICS REPORT")
    print("=" * 60)
    
    # ── EXECUTIVE SUMMARY ─────────────────────────────────────────────────
    print("\n── Executive Summary ──")
    print(f"Total Revenue:  ${df['revenue'].sum():>12,.0f}")
    print(f"Total Profit:   ${df['profit'].sum():>12,.0f}")
    print(f"Overall Margin: {df['margin'].mean():>11.1%}")
    print(f"Total Units:    {df['units'].sum():>12,}")
    
    # ── BY REGION ─────────────────────────────────────────────────────────
    print("\n── Revenue by Region ──")
    region_summary = df.groupby("region").agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        margin=("margin", "mean"),
        transactions=("revenue", "count")
    ).sort_values("revenue", ascending=False)
    
    region_summary["revenue_pct"] = region_summary["revenue"] / region_summary["revenue"].sum()
    print(region_summary.to_string())
    
    # ── BY PRODUCT ────────────────────────────────────────────────────────
    print("\n── Top 5 Products by Revenue ──")
    product_summary = df.groupby("product").agg(
        revenue=("revenue", "sum"),
        units=("units", "sum"),
        avg_price=("revenue", lambda x: x.sum() / df.loc[x.index, "units"].sum()),
        margin=("margin", "mean")
    ).nlargest(5, "revenue")
    print(product_summary.to_string())
    
    # ── MONTHLY TREND ─────────────────────────────────────────────────────
    print("\n── Monthly Revenue Trend ──")
    monthly = df.groupby("month")["revenue"].sum()
    mom_growth = monthly.pct_change() * 100
    
    for month, revenue in monthly.items():
        growth = mom_growth[month]
        growth_str = f"{growth:+.1f}%" if not pd.isna(growth) else "  --"
        print(f"  {month}: ${revenue:>10,.0f}  {growth_str}")
    
    # ── PIVOT: Region × Product ────────────────────────────────────────────
    print("\n── Revenue by Region × Product ──")
    pivot = pd.pivot_table(
        df, values="revenue", index="region", columns="product", aggfunc="sum", fill_value=0
    )
    print(pivot.to_string())
    
    return {
        "by_region": region_summary,
        "by_product": product_summary,
        "monthly": monthly
    }
```

### Worked Example 3: Customer Cohort Analysis

```python
def cohort_analysis(orders_df):
    """
    Cohort analysis: track customer retention by signup month.
    orders_df: columns [customer_id, order_date, revenue]
    """
    orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])
    orders_df["order_month"] = orders_df["order_date"].dt.to_period("M")
    
    # Assign cohort: the month of each customer's FIRST order
    orders_df["cohort"] = orders_df.groupby("customer_id")["order_date"].transform("min").dt.to_period("M")
    
    # Calculate months since first order
    orders_df["months_since_first"] = (
        orders_df["order_month"] - orders_df["cohort"]
    ).apply(lambda x: x.n)
    
    # Count unique customers per cohort per period
    cohort_data = orders_df.groupby(["cohort", "months_since_first"])["customer_id"].nunique().reset_index()
    cohort_data.columns = ["cohort", "month_number", "customers"]
    
    # Pivot to cohort matrix
    cohort_matrix = cohort_data.pivot(index="cohort", columns="month_number", values="customers")
    
    # Calculate retention rates (as % of cohort size at month 0)
    cohort_sizes = cohort_matrix[0]  # Initial cohort size
    retention = cohort_matrix.divide(cohort_sizes, axis=0).round(3)
    
    print("Cohort Retention Rates:")
    print(retention.to_string())
    
    # Average retention by month
    avg_retention = retention.mean()
    print("\nAverage Retention by Month:")
    for month, rate in avg_retention.items():
        print(f"  Month {month}: {rate:.1%}")
    
    return retention
```

---

## Practice Problems

### Easy (1–5)

**Problem 1: DataFrame Creation and Inspection**
Create a DataFrame and answer basic questions.

```python
import pandas as pd

data = {
    "product": ["Laptop", "Phone", "Tablet", "Watch", "Earbuds", "Monitor"],
    "category": ["Electronics", "Electronics", "Electronics", "Wearables", "Audio", "Electronics"],
    "price": [999.99, 699.99, 499.99, 299.99, 199.99, 399.99],
    "stock": [50, 120, 80, 200, 300, 60],
    "rating": [4.5, 4.3, 4.1, 4.7, 4.2, 4.4]
}
df = pd.DataFrame(data)

# Answer without loops:
# 1. How many products are Electronics?
# 2. What is the most expensive product?
# 3. Average rating of products priced over $400?
# 4. Total inventory value (price × stock)?
# 5. Which product has the highest stock?
```

**Problem 2: Data Loading and Quick Inspection**
Write a function that loads any CSV and gives a structured report.

```python
def dataset_report(filepath):
    """
    Load CSV and print:
    - Shape
    - Column names and dtypes
    - % missing per column
    - Numeric column statistics
    - First 5 rows
    """
    pass
```

**Problem 3: Filtering Chains**
Using method chaining, answer these questions about a sales DataFrame.

```python
# Given sales DataFrame with: date, salesperson, region, product, amount, units
sales = pd.read_csv("sales.csv")

# Find using chained operations:
# 1. Top 3 salespersons by total revenue
# 2. Sales where amount > $5000 AND region == "North"
# 3. Monthly revenue totals for Q1 2025
# 4. Products that sold over 1000 units total
```

**Problem 4: GroupBy Basics**
Analyze employee data with groupby.

```python
employees = pd.DataFrame({
    "name": ["Alice","Bob","Carol","Dave","Eve","Frank","Grace","Henry"],
    "dept": ["Eng","Mkt","Eng","Mkt","Prod","Eng","Prod","Mkt"],
    "salary": [95000,72000,110000,68000,88000,102000,91000,75000],
    "years": [4, 2, 7, 1, 5, 6, 3, 4]
})

# Without loops:
# 1. Mean salary per department
# 2. Department with highest total salary budget
# 3. Number of employees per department
# 4. Employee with highest salary in each department
# 5. Percentage of company payroll each department represents
```

**Problem 5: Date Operations**
Work with a time series of daily website traffic.

```python
import pandas as pd
import numpy as np

dates = pd.date_range("2025-01-01", "2025-06-30", freq="D")
traffic = pd.Series(
    np.random.randint(1000, 10000, len(dates)) +
    (np.sin(np.linspace(0, 4*np.pi, len(dates))) * 1000).astype(int),
    index=dates,
    name="visitors"
)

# Answer:
# 1. Which month had the most total visitors?
# 2. What day of the week gets the most traffic on average?
# 3. Calculate 7-day rolling average
# 4. Which date had the highest single-day traffic?
# 5. What % of days had traffic above average?
```

### Medium (6–12)

**Problem 6: Data Cleaning Pipeline**
Clean a messy dataset.

```python
# This dataset has multiple issues — fix all of them:
messy = pd.DataFrame({
    "Name": ["  alice johnson", "BOB SMITH", "Carol ", None, "EVE DAVIS"],
    "Age": ["28", "abc", "42", "29", "38"],
    "Salary ($)": ["$95,000", "$72,000", "N/A", "$68,000", "$88000"],
    "Email": ["alice@co.com", "bob@co.com", "carol@co.com", "carol@co.com", "eve@co.com"],
    "Hire Date": ["2021-03-15", "2023/06/01", "2018.01.20", "2024-11-05", "2022-08-30"],
    "Score": [95, 87, None, 72, 91]
})

def clean_dataset(df):
    """Fix all issues and return clean DataFrame."""
    pass
```

**Problem 7: Advanced GroupBy**
Perform complex aggregations.

```python
sales = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=200, freq="D").repeat(3),
    "region": (["North", "South", "East"] * 200)[:600],
    "product": np.random.choice(["A", "B", "C", "D"], 600),
    "revenue": np.random.uniform(100, 5000, 600),
    "units": np.random.randint(1, 50, 600),
    "salesperson": np.random.choice(["Alice", "Bob", "Carol", "Dave"], 600)
})

# Solve with groupby + transform/filter/apply:
# 1. Add column: revenue as % of daily total
# 2. Keep only regions where average daily revenue > $2000
# 3. For each salesperson, find their best month (highest total revenue)
# 4. Rank products within each region by total revenue
# 5. Calculate market share per product per month
```

**Problem 8: Merging Multiple DataFrames**
Join four related tables.

```python
orders = pd.DataFrame({
    "order_id": [1, 2, 3, 4, 5],
    "customer_id": [101, 102, 101, 103, 102],
    "product_id": [10, 11, 12, 10, 13],
    "quantity": [2, 1, 3, 1, 2],
    "order_date": pd.date_range("2025-01-01", periods=5, freq="W")
})

customers = pd.DataFrame({
    "customer_id": [101, 102, 103],
    "name": ["Alice", "Bob", "Carol"],
    "city": ["Mumbai", "Delhi", "Bangalore"]
})

products = pd.DataFrame({
    "product_id": [10, 11, 12, 13],
    "name": ["Laptop", "Phone", "Tablet", "Watch"],
    "price": [999.99, 699.99, 499.99, 299.99]
})

# Build a full order report with customer names, product details, and total values
# Then: Which customer spent the most? Which product sold most units?
```

**Problem 9: Time Series Analysis**
Analyze stock price data.

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
dates = pd.date_range("2024-01-01", "2025-12-31", freq="B")
prices = pd.DataFrame({
    ticker: 100 * (1 + rng.normal(0.0003, 0.015, len(dates))).cumprod()
    for ticker in ["AAPL", "GOOGL", "MSFT", "AMZN"]
}, index=dates)

# Answer:
# 1. Daily % returns for each stock
# 2. 20-day rolling Sharpe ratio for each stock
# 3. Which stock had the best/worst year?
# 4. Correlation matrix between all stocks
# 5. Maximum drawdown for each stock
# 6. Which days were "crash days" (portfolio down > 2%)?
```

**Problem 10: Pivot Table Analysis**
Build a comprehensive pivot analysis.

```python
# Given: regional sales data with product categories
# Create pivot tables to answer business questions:
# 1. Revenue by region AND category (rows=region, cols=category)
# 2. Month-over-month growth by category
# 3. Market share by region for each category
# 4. Which region-category combination has best/worst margin?
```

**Problem 11: Fuzzy Deduplication**
Find near-duplicate records.

```python
# Problem: customer data from two systems with slightly different spellings
system_a = pd.DataFrame({
    "name": ["Alice Johnson", "Bob Smith", "Carol Davis", "Dave Wilson"],
    "email": ["alice@co.com", "bob@co.com", "carol@co.com", "dave@co.com"]
})

system_b = pd.DataFrame({
    "name": ["Alice Johnsen", "B. Smith", "Carol Davis", "David Wilson"],
    "phone": ["555-0001", "555-0002", "555-0003", "555-0004"]
})

# Match records between systems using fuzzy name matching
# Hint: use SequenceMatcher from difflib
from difflib import SequenceMatcher

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Find likely matches (similarity > 0.8)
```

**Problem 12: Cohort Retention Analysis**
Build a full cohort analysis from raw order data.

```python
# Generate synthetic order data
np.random.seed(42)
n_customers = 500
n_orders = 3000

orders = pd.DataFrame({
    "customer_id": np.random.randint(1, n_customers+1, n_orders),
    "order_date": pd.date_range("2024-01-01", periods=n_orders, freq="12H"),
    "revenue": np.random.uniform(20, 500, n_orders)
})

# Implement:
# 1. Assign customers to monthly cohorts (by first order date)
# 2. Build cohort × month retention matrix
# 3. Calculate average LTV (lifetime value) by cohort
# 4. Identify which cohorts have best/worst 3-month retention
```

### Hard (13–20)

**Problem 13: ETL Pipeline Class**
Build a reusable ETL pipeline using pandas.

```python
class DataPipeline:
    """
    Chainable data transformation pipeline.
    Each method returns self for chaining.
    """
    
    def __init__(self, df_or_path): pass
    def rename(self, mapping): return self
    def cast(self, **type_map): return self   # {"col": int, "col2": float}
    def fill_nulls(self, **strategies): return self  # {"col": "mean", "col2": "mode"}
    def remove_outliers(self, *cols, method="iqr", threshold=1.5): return self
    def add_column(self, name, formula): return self  # formula is a function of df
    def filter(self, query_str): return self
    def validate(self, **rules): return self  # {"age": lambda x: x.between(0, 120)}
    def to_csv(self, path): return self
    def execute(self): pass   # Return the final DataFrame

result = (DataPipeline("raw_data.csv")
    .rename({"Emp Name": "name", "Hire Dt": "hire_date"})
    .cast(salary=float, age=int)
    .fill_nulls(salary="median", department="mode")
    .remove_outliers("salary")
    .add_column("monthly_salary", lambda df: df["salary"] / 12)
    .filter("age >= 18 and salary > 0")
    .validate(age=lambda x: x.between(18, 80))
    .execute()
)
```

**Problem 14: Time Series Forecasting**
Implement a simple forecasting model using pandas.

```python
def decompose_and_forecast(series, periods=12):
    """
    Decompose a time series into trend + seasonality + residual.
    Forecast next `periods` using the decomposition.
    Use rolling means for trend extraction.
    """
    pass

# Monthly revenue for 3 years
dates = pd.date_range("2022-01-01", "2024-12-31", freq="ME")
# Add seasonal pattern + trend + noise
revenue = pd.Series(
    5000 +                              # Base
    np.linspace(0, 2000, len(dates)) +  # Upward trend
    1500 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12) +  # Seasonality
    np.random.normal(0, 200, len(dates)),  # Noise
    index=dates
)

trend, seasonal, residual, forecast = decompose_and_forecast(revenue, periods=6)
```

**Problem 15: Multi-Index Operations**
Work with hierarchical (multi-index) DataFrames.

```python
# Sales data with hierarchical index: (year, quarter, region)
idx = pd.MultiIndex.from_product(
    [[2024, 2025], [1, 2, 3, 4], ["North", "South", "East", "West"]],
    names=["year", "quarter", "region"]
)
sales = pd.Series(np.random.randint(100000, 1000000, len(idx)), index=idx)

# Answer:
# 1. Total sales by year
# 2. Q3 2025 sales for all regions
# 3. Best-performing region in each quarter of 2024
# 4. Year-over-year growth by region
# 5. Unstack region into columns, stack year into rows
```

**Problem 16: Custom Aggregation Functions**
Write complex custom aggregation functions.

```python
def custom_agg_analysis(df):
    """
    For each department, calculate:
    - Gini coefficient (inequality measure) of salaries
    - 80th percentile salary
    - % above company-wide median
    - Salary to years-of-experience ratio (efficiency)
    - Whether the department "passes" an equity test (all genders within 10% of dept mean)
    """
    pass
```

**Problem 17: Rolling Window Event Detection**
Find patterns in time series using rolling windows.

```python
def detect_events(prices, volume):
    """
    Detect trading events in stock data:
    1. Golden cross: 50-day MA crosses above 200-day MA
    2. Death cross: 50-day MA crosses below 200-day MA
    3. High volume spike: volume > 2x 20-day average volume
    4. Bollinger Band breakout: price exceeds mean ± 2*std over 20 days
    5. Support/resistance levels: price bounces from same level 3+ times
    """
    pass
```

**Problem 18: Memory Optimization**
Reduce memory usage of a large DataFrame.

```python
def optimize_dtypes(df):
    """
    Reduce memory footprint by:
    - Downcast int64 → int8/int16/int32 where values fit
    - Downcast float64 → float32 where precision allows
    - Convert low-cardinality string columns to category dtype
    - Convert boolean columns stored as int to bool
    
    Print memory usage before and after.
    Return optimized DataFrame.
    """
    pass

# A 1M-row DataFrame that starts at 500MB
# After optimization, should be ~150MB
```

**Problem 19: Vectorized Text Analytics**
Use pandas string methods for NLP preprocessing.

```python
def text_analytics(reviews_df):
    """
    reviews_df: columns [review_id, product, rating, text]
    
    Without loops, using vectorized operations:
    1. Clean: lowercase, remove punctuation, strip whitespace
    2. Word count per review
    3. Most common words per product (excluding stopwords)
    4. Identify negative reviews (rating < 3 AND text contains negative words)
    5. Review length vs rating correlation
    6. Flag reviews that might be spam (>90% same words as another review)
    """
    pass
```

**Problem 20: Full Business Intelligence Dashboard**
Build a complete BI analysis from raw transaction data.

```python
def build_bi_dashboard(transactions, customers, products):
    """
    transactions: [txn_id, customer_id, product_id, date, quantity, unit_price, discount]
    customers: [customer_id, name, city, signup_date, tier]
    products: [product_id, name, category, cost_price]
    
    Generate complete business report:
    
    1. REVENUE SECTION
       - Total revenue, profit, margin
       - Month-over-month growth
       - Revenue by customer tier
    
    2. CUSTOMER SECTION
       - Customer lifetime value by cohort
       - Churn rate by month
       - Average order value by tier
       - Top 10 customers by revenue
    
    3. PRODUCT SECTION
       - Best/worst margin products
       - Category performance
       - Cross-sell matrix (which products bought together)
    
    4. OPERATIONAL
       - Revenue by city
       - Discount effectiveness (do discounts increase units?)
       - Seasonality patterns
    """
    pass
```

---

## Answer Keys (Selected Problems)

### Problem 1 Solution:
```python
df = pd.DataFrame(data)

# 1. Electronics count
print((df["category"] == "Electronics").sum())   # 4

# 2. Most expensive
print(df.loc[df["price"].idxmax(), "product"])   # Laptop

# 3. Average rating over $400
print(df[df["price"] > 400]["rating"].mean())

# 4. Total inventory value
print((df["price"] * df["stock"]).sum())

# 5. Highest stock
print(df.loc[df["stock"].idxmax(), "product"])   # Earbuds
```

### Problem 4 Solution:
```python
# 1. Mean salary per dept
print(employees.groupby("dept")["salary"].mean())

# 2. Dept with highest total budget
print(employees.groupby("dept")["salary"].sum().idxmax())

# 3. Employee count per dept
print(employees.groupby("dept").size())

# 4. Top earner per dept
print(employees.loc[employees.groupby("dept")["salary"].idxmax()])

# 5. Payroll % per dept
dept_totals = employees.groupby("dept")["salary"].sum()
print((dept_totals / dept_totals.sum() * 100).round(1))
```

### Problem 9 Solution:
```python
# 1. Daily returns
returns = prices.pct_change().dropna()

# 2. 20-day rolling Sharpe
rolling_sharpe = returns.rolling(20).apply(
    lambda x: x.mean() / x.std() * np.sqrt(252)
)

# 3. Best/worst year
annual_return = returns.sum()
print(f"Best:  {annual_return.idxmax()} ({annual_return.max():.1%})")
print(f"Worst: {annual_return.idxmin()} ({annual_return.min():.1%})")

# 4. Correlation
print(returns.corr().round(2))

# 5. Max drawdown
def max_dd(s):
    peak = s.cummax()
    return ((s - peak) / peak).min()
print(prices.apply(max_dd))

# 6. Crash days
portfolio = returns.mean(axis=1)
crash_days = portfolio[portfolio < -0.02]
print(f"Crash days: {len(crash_days)}")
```

---

## Mini-Project: E-Commerce Analytics Platform

```python
"""
ecommerce_analytics.py
Complete analytics platform built with Pandas.
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ── DATA GENERATION ────────────────────────────────────────────────────────
def generate_ecommerce_data(n_customers=1000, n_orders=5000, seed=42):
    """Generate realistic e-commerce data."""
    rng = np.random.default_rng(seed)
    
    customers = pd.DataFrame({
        "customer_id": range(1, n_customers + 1),
        "name": [f"Customer_{i}" for i in range(1, n_customers + 1)],
        "city": rng.choice(["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"], n_customers),
        "tier": rng.choice(["Bronze", "Silver", "Gold", "Platinum"], n_customers,
                           p=[0.5, 0.3, 0.15, 0.05]),
        "signup_date": pd.date_range("2022-01-01", "2024-06-30",
                                      periods=n_customers)
    })
    
    categories = ["Electronics", "Clothing", "Food", "Home", "Beauty"]
    products = pd.DataFrame({
        "product_id": range(1, 51),
        "name": [f"Product_{i}" for i in range(1, 51)],
        "category": rng.choice(categories, 50),
        "base_price": rng.uniform(100, 5000, 50).round(2),
        "cost": rng.uniform(50, 2000, 50).round(2)
    })
    # Ensure cost < price
    products["cost"] = products[["cost", "base_price"]].min(axis=1) * 0.6
    
    orders = pd.DataFrame({
        "order_id": range(1, n_orders + 1),
        "customer_id": rng.integers(1, n_customers + 1, n_orders),
        "product_id": rng.integers(1, 51, n_orders),
        "order_date": pd.Timestamp("2024-01-01") +
                      pd.to_timedelta(rng.integers(0, 548, n_orders), unit="D"),
        "quantity": rng.integers(1, 6, n_orders),
        "discount_pct": rng.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20], n_orders)
    })
    
    return customers, products, orders


# ── ANALYTICS ENGINE ───────────────────────────────────────────────────────
class EcommerceAnalytics:
    def __init__(self, customers, products, orders):
        self.customers = customers
        self.products = products
        self.orders = orders
        self._build_enriched_orders()
    
    def _build_enriched_orders(self):
        """Join all tables and compute derived metrics."""
        self.enriched = (
            self.orders
            .merge(self.customers, on="customer_id", how="left")
            .merge(self.products, on="product_id", how="left")
            .assign(
                unit_price=lambda x: x["base_price"] * (1 - x["discount_pct"]),
                revenue=lambda x: x["unit_price"] * x["quantity"],
                profit=lambda x: (x["unit_price"] - x["cost"]) * x["quantity"],
                margin=lambda x: (x["unit_price"] - x["cost"]) / x["unit_price"],
                month=lambda x: x["order_date"].dt.to_period("M"),
                year=lambda x: x["order_date"].dt.year,
                weekday=lambda x: x["order_date"].dt.day_name()
            )
        )
    
    # ── REVENUE ANALYTICS ─────────────────────────────────────────────────
    def revenue_summary(self):
        df = self.enriched
        return {
            "total_revenue":    df["revenue"].sum(),
            "total_profit":     df["profit"].sum(),
            "overall_margin":   df["margin"].mean(),
            "total_orders":     len(df),
            "avg_order_value":  df.groupby("order_id")["revenue"].sum().mean(),
            "total_customers":  df["customer_id"].nunique()
        }
    
    def monthly_revenue(self):
        monthly = (self.enriched
            .groupby("month")
            .agg(revenue=("revenue", "sum"), orders=("order_id", "count"),
                 profit=("profit", "sum"))
            .assign(mom_growth=lambda x: x["revenue"].pct_change() * 100)
        )
        return monthly
    
    # ── CUSTOMER ANALYTICS ────────────────────────────────────────────────
    def customer_ltv(self):
        """Customer Lifetime Value analysis."""
        clv = (self.enriched
            .groupby("customer_id")
            .agg(
                total_revenue=("revenue", "sum"),
                total_orders=("order_id", "count"),
                avg_order_value=("revenue", "mean"),
                first_order=("order_date", "min"),
                last_order=("order_date", "max")
            )
        )
        clv["customer_age_days"] = (
            pd.Timestamp.today() - clv["first_order"]
        ).dt.days
        clv["days_since_last"] = (
            pd.Timestamp.today() - clv["last_order"]
        ).dt.days
        
        # RFM Segmentation
        clv["recency_score"]   = pd.qcut(clv["days_since_last"], 5, labels=[5,4,3,2,1])
        clv["frequency_score"] = pd.qcut(clv["total_orders"].rank(method="first"), 5, labels=[1,2,3,4,5])
        clv["monetary_score"]  = pd.qcut(clv["total_revenue"], 5, labels=[1,2,3,4,5])
        
        clv["rfm_score"] = (
            clv["recency_score"].astype(int) +
            clv["frequency_score"].astype(int) +
            clv["monetary_score"].astype(int)
        )
        
        clv["segment"] = pd.cut(
            clv["rfm_score"],
            bins=[0, 5, 8, 11, 15],
            labels=["At Risk", "Needs Attention", "Loyal", "Champions"]
        )
        return clv
    
    def cohort_retention(self):
        """Monthly cohort retention analysis."""
        df = self.enriched.copy()
        df["cohort"] = df.groupby("customer_id")["order_date"].transform("min").dt.to_period("M")
        df["period_number"] = (df["month"] - df["cohort"]).apply(lambda x: x.n)
        
        cohort_counts = (df
            .groupby(["cohort", "period_number"])["customer_id"].nunique()
            .unstack(fill_value=0)
        )
        
        retention = cohort_counts.divide(cohort_counts[0], axis=0).round(3)
        return retention
    
    # ── PRODUCT ANALYTICS ─────────────────────────────────────────────────
    def product_performance(self):
        return (self.enriched
            .groupby(["category", "name"])
            .agg(
                revenue=("revenue", "sum"),
                units=("quantity", "sum"),
                orders=("order_id", "count"),
                avg_margin=("margin", "mean")
            )
            .sort_values("revenue", ascending=False)
        )
    
    def category_trends(self):
        return (self.enriched
            .groupby(["month", "category"])["revenue"].sum()
            .unstack(fill_value=0)
        )
    
    # ── FULL REPORT ───────────────────────────────────────────────────────
    def print_report(self):
        print("=" * 65)
        print(f"{'E-COMMERCE ANALYTICS REPORT':^65}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 65)
        
        # Revenue Summary
        summary = self.revenue_summary()
        print("\n── KEY METRICS ────────────────────────────────────────────")
        print(f"  Total Revenue:    ${summary['total_revenue']:>12,.2f}")
        print(f"  Total Profit:     ${summary['total_profit']:>12,.2f}")
        print(f"  Avg Margin:       {summary['overall_margin']:>11.1%}")
        print(f"  Total Orders:     {summary['total_orders']:>12,}")
        print(f"  Avg Order Value:  ${summary['avg_order_value']:>12,.2f}")
        print(f"  Active Customers: {summary['total_customers']:>12,}")
        
        # Monthly Trend
        monthly = self.monthly_revenue()
        print("\n── MONTHLY REVENUE TREND ──────────────────────────────────")
        print(f"  {'Month':<10} {'Revenue':>12} {'MoM Growth':>12} {'Orders':>8}")
        print(f"  {'-'*46}")
        for month, row in monthly.iterrows():
            growth = f"{row['mom_growth']:+.1f}%" if not pd.isna(row['mom_growth']) else "  --"
            print(f"  {str(month):<10} ${row['revenue']:>11,.0f} {growth:>12} {row['orders']:>8,}")
        
        # Customer Segments
        clv = self.customer_ltv()
        segment_summary = (clv
            .groupby("segment", observed=True)
            .agg(count=("total_revenue", "count"),
                 avg_revenue=("total_revenue", "mean"))
        )
        print("\n── CUSTOMER SEGMENTS (RFM) ────────────────────────────────")
        print(f"  {'Segment':<20} {'Count':>8} {'Avg LTV':>12}")
        print(f"  {'-'*42}")
        for seg, row in segment_summary.iterrows():
            print(f"  {str(seg):<20} {row['count']:>8,} ${row['avg_revenue']:>11,.2f}")
        
        # Top Categories
        cat_rev = self.enriched.groupby("category")["revenue"].sum().sort_values(ascending=False)
        total = cat_rev.sum()
        print("\n── REVENUE BY CATEGORY ────────────────────────────────────")
        print(f"  {'Category':<15} {'Revenue':>12} {'Share':>8}")
        print(f"  {'-'*38}")
        for cat, rev in cat_rev.items():
            print(f"  {cat:<15} ${rev:>11,.0f} {rev/total:>7.1%}")
        
        # Cohort Retention
        retention = self.cohort_retention()
        if not retention.empty and 1 in retention.columns:
            avg_m1 = retention[1].mean()
            avg_m2 = retention[2].mean() if 2 in retention.columns else None
            print("\n── COHORT RETENTION ───────────────────────────────────────")
            print(f"  Average Month-1 Retention: {avg_m1:.1%}")
            if avg_m2:
                print(f"  Average Month-2 Retention: {avg_m2:.1%}")
        
        print("\n" + "=" * 65)


# ── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating e-commerce dataset...")
    customers, products, orders = generate_ecommerce_data(
        n_customers=500, n_orders=3000
    )
    
    print("Running analytics...")
    analytics = EcommerceAnalytics(customers, products, orders)
    analytics.print_report()
    
    # Individual analyses
    print("\nTop 5 Products by Revenue:")
    top_products = analytics.product_performance().head(5)
    print(top_products[["revenue", "units", "avg_margin"]].to_string())
    
    print("\nCategory Monthly Trends:")
    trends = analytics.category_trends()
    print(trends.tail(3).to_string())
```

---

## Chapter Summary

You've mastered Pandas — the backbone of data analysis in Python!

✅ **Series**: 1D labelled array, dict-like access, vectorized operations, index alignment
✅ **DataFrame**: 2D table with labelled rows and columns, the core data structure
✅ **Reading Data**: `read_csv`, `read_excel`, `read_json`, `read_sql` with all key options
✅ **Selection**: `df[col]`, `df.loc[row, col]`, `df.iloc[row, col]`, boolean masking
✅ **Manipulation**: Adding columns, `apply`, `map`, sorting, renaming
✅ **Data Cleaning**: Missing values (detect/fill/drop), duplicates, type conversion, outliers
✅ **GroupBy**: Split-apply-combine, `agg`, `transform`, `filter`, named aggregations
✅ **Merging**: `merge` (SQL joins), `concat` (stacking)
✅ **Time Series**: `resample`, `rolling`, `shift`, `pct_change`, date extraction
✅ **Pivot Tables**: `pivot_table`, `melt`, `crosstab`

**Key Takeaways:**
- Think in columns and vectorized operations — avoid row-level Python loops
- `loc` for label-based, `iloc` for position-based — never confuse them
- `groupby` + `agg` replaces 90% of your need for loops over grouped data
- Always inspect new data first: `shape`, `dtypes`, `isnull().sum()`, `describe()`
- `merge` is your SQL JOIN — learn the four types: inner, left, right, outer

**Next Chapter Preview:**
Chapter 13 covers **Data Visualization with Matplotlib and Seaborn** — turning your data into compelling charts, from simple line plots to complex multi-panel dashboards!
