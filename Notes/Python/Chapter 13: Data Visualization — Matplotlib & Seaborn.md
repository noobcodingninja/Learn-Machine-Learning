# Chapter 13: Data Visualization — Matplotlib & Seaborn

## Part 1: Why Visualization Matters

### The Problem: Numbers Alone Don't Tell Stories

Consider this dataset — four groups of 11 data points each (Anscombe's Quartet):

```python
import numpy as np
import pandas as pd

# All four datasets have IDENTICAL statistics:
# Mean of X:    9.0
# Mean of Y:    7.5
# Variance X:   11.0
# Variance Y:   4.12
# Correlation:  0.816
# Regression:   Y = 3 + 0.5X

quartet = {
    "I":   {"x": [10,8,13,9,11,14,6,4,12,7,5],  "y": [8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68]},
    "II":  {"x": [10,8,13,9,11,14,6,4,12,7,5],  "y": [9.14,8.14,8.74,8.77,9.26,8.10,6.13,3.10,9.13,7.26,4.74]},
    "III": {"x": [10,8,13,9,11,14,6,4,12,7,5],  "y": [7.46,6.77,12.74,7.11,7.81,8.84,6.08,5.39,8.15,6.42,5.73]},
    "IV":  {"x": [8,8,8,8,8,8,8,19,8,8,8],      "y": [6.58,5.76,7.71,8.84,8.47,7.04,5.25,12.50,5.56,7.91,6.89]},
}

# The statistics are identical — but the PATTERNS are completely different!
# You would NEVER know this without plotting.
# Dataset I:   Linear relationship
# Dataset II:  Clear curve — linear model is wrong!
# Dataset III: Linear except ONE outlier completely skewing the line
# Dataset IV:  Vertical cluster except ONE high-leverage point

# This is why we visualize FIRST, compute statistics SECOND.
```

**Three things visualization does that numbers cannot:**
1. **Reveals patterns** — trends, clusters, outliers, relationships
2. **Communicates clearly** — a chart explains in seconds what a table takes minutes to parse
3. **Catches problems** — data entry errors, unexpected distributions, wrong assumptions

---

## Part 2: The Matplotlib Architecture

### Understanding the Two Levels

```python
import matplotlib.pyplot as plt
import numpy as np

# ── LEVEL 1: PYPLOT (MATLAB-style, quick) ─────────────────────────────────
# plt.something() — implicit current figure/axes
# Great for: quick exploration, simple single plots

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.title("Simple Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# ── LEVEL 2: OBJECT-ORIENTED (explicit, recommended) ──────────────────────
# fig, ax = plt.subplots() — explicit Figure and Axes objects
# Great for: multiple plots, fine-grained control, production code

fig, ax = plt.subplots()          # Create Figure and Axes
ax.plot([1, 2, 3, 4], [1, 4, 9, 16])
ax.set_title("OO-Style Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.tight_layout()
plt.show()

# ALWAYS prefer the OO style — it's explicit and composable
# Rule: use plt only for plt.subplots(), plt.show(), plt.savefig()
```

### The Figure Architecture

```
Figure (the whole canvas)
├── Axes (one plot area, with its own coordinate system)
│   ├── Title
│   ├── X axis (XAxis)
│   │   ├── Label
│   │   ├── Ticks (major, minor)
│   │   └── Tick labels
│   ├── Y axis (YAxis)
│   │   └── (same as X)
│   ├── Lines (plot data)
│   ├── Patches (filled shapes)
│   ├── Artists (text, arrows, etc.)
│   └── Legend
└── (can have multiple Axes via subplots)
```

```python
# ── INSTALLATION ──────────────────────────────────────────────────────────
# pip install matplotlib seaborn

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd

print(f"Matplotlib: {mpl.__version__}")
print(f"Seaborn:    {sns.__version__}")

# ── GLOBAL STYLE SETTINGS ─────────────────────────────────────────────────
# Set once at the top of your notebook/script

# Seaborn themes (highly recommended — better defaults than matplotlib)
sns.set_theme(style="whitegrid")   # Clean, professional
# Options: "white", "dark", "whitegrid", "darkgrid", "ticks"

# Font sizes
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16
})
```

---

## Part 3: Line Plots — Showing Trends Over Time

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── BASIC LINE PLOT ───────────────────────────────────────────────────────
x = np.linspace(0, 2 * np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x, y_sin, label="sin(x)", color="steelblue", linewidth=2)
ax.plot(x, y_cos, label="cos(x)", color="coral",     linewidth=2, linestyle="--")

ax.set_title("Trigonometric Functions", fontsize=14, fontweight="bold")
ax.set_xlabel("x (radians)")
ax.set_ylabel("y")
ax.legend()
ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")  # Add y=0 reference line
ax.set_xlim(0, 2 * np.pi)

plt.tight_layout()
plt.savefig("trig_plot.png", dpi=150, bbox_inches="tight")
plt.show()


# ── MULTI-LINE TIME SERIES ────────────────────────────────────────────────
# Simulating stock prices
rng = np.random.default_rng(42)
dates = pd.date_range("2025-01-01", periods=252, freq="B")
tickers = ["AAPL", "GOOGL", "MSFT"]
colors  = ["#1f77b4", "#ff7f0e", "#2ca02c"]

prices = pd.DataFrame({
    t: 100 * (1 + rng.normal(0.0005, 0.015, 252)).cumprod()
    for t in tickers
}, index=dates)

fig, ax = plt.subplots(figsize=(12, 6))

for ticker, color in zip(tickers, colors):
    ax.plot(prices.index, prices[ticker], label=ticker,
            color=color, linewidth=1.5)
    # Annotate final value
    final = prices[ticker].iloc[-1]
    ax.annotate(f"${final:.0f}", xy=(prices.index[-1], final),
                xytext=(5, 0), textcoords="offset points",
                color=color, fontsize=9, va="center")

ax.set_title("Simulated Stock Prices — 2025", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.legend(loc="upper left")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))

# Format x-axis dates nicely
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ── LINE STYLES AND MARKERS REFERENCE ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

styles = [
    ("Solid",   "-",  "o", "steelblue"),
    ("Dashed",  "--", "s", "coral"),
    ("Dotted",  ":",  "^", "green"),
    ("DashDot", "-.", "D", "purple"),
]

x = np.arange(5)
for i, (label, ls, marker, color) in enumerate(styles):
    ax.plot(x, x + i * 0.3, linestyle=ls, marker=marker,
            color=color, label=label, linewidth=2, markersize=8)

ax.legend()
ax.set_title("Line Styles and Markers")
plt.tight_layout()
plt.show()
```

---

## Part 4: Bar Charts — Comparing Categories

```python
# ── VERTICAL BAR CHART ────────────────────────────────────────────────────
departments = ["Engineering", "Marketing", "Product", "Sales", "HR"]
headcounts  = [42, 18, 23, 35, 12]
colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(departments, headcounts, color=colors, edgecolor="white",
              linewidth=0.5, alpha=0.85, width=0.6)

# Add value labels ON TOP of bars
for bar, count in zip(bars, headcounts):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center", va="bottom", fontweight="bold", fontsize=11)

ax.set_title("Headcount by Department", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Number of Employees")
ax.set_ylim(0, max(headcounts) * 1.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


# ── HORIZONTAL BAR CHART (better for long category names) ────────────────
products = ["Smart TV 55\"", "Laptop Pro 16\"", "Wireless Earbuds",
            "Standing Desk", "Monitor Arm", "USB-C Hub"]
revenue  = [485000, 392000, 278000, 195000, 134000, 98000]

# Sort for readability (ascending so top bar is highest)
sorted_pairs = sorted(zip(products, revenue), key=lambda x: x[1])
products, revenue = zip(*sorted_pairs)

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.barh(products, revenue, color="#2196F3", alpha=0.8, edgecolor="white")

for bar, val in zip(bars, revenue):
    ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height() / 2,
            f"${val:,.0f}", va="center", fontsize=9)

ax.set_xlabel("Revenue ($)")
ax.set_title("Top Products by Revenue", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


# ── GROUPED BAR CHART ─────────────────────────────────────────────────────
quarters = ["Q1", "Q2", "Q3", "Q4"]
revenue_2024  = [1.2, 1.5, 1.8, 2.1]
revenue_2025  = [1.4, 1.7, 2.0, 2.4]

x = np.arange(len(quarters))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

bars1 = ax.bar(x - width/2, revenue_2024, width, label="2024",
               color="#90CAF9", edgecolor="white")
bars2 = ax.bar(x + width/2, revenue_2025, width, label="2025",
               color="#1565C0", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(quarters)
ax.set_ylabel("Revenue ($M)")
ax.set_title("Quarterly Revenue: 2024 vs 2025", fontsize=14, fontweight="bold")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.1f}M"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


# ── STACKED BAR CHART ─────────────────────────────────────────────────────
months = ["Jan", "Feb", "Mar", "Apr", "May"]
electronics = [42000, 38000, 45000, 51000, 48000]
clothing    = [18000, 22000, 19000, 24000, 26000]
food        = [12000, 14000, 13000, 15000, 16000]

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(months, electronics, label="Electronics", color="#1E88E5")
ax.bar(months, clothing,    label="Clothing",    color="#FDD835",
       bottom=electronics)
ax.bar(months, food,        label="Food",        color="#43A047",
       bottom=[e + c for e, c in zip(electronics, clothing)])

ax.set_ylabel("Revenue ($)")
ax.set_title("Monthly Revenue by Category (Stacked)", fontsize=14, fontweight="bold")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()
```

---

## Part 5: Scatter Plots — Showing Relationships

```python
# ── BASIC SCATTER PLOT ────────────────────────────────────────────────────
rng = np.random.default_rng(42)
n = 200

experience = rng.uniform(1, 15, n)
salary = 40000 + experience * 5000 + rng.normal(0, 8000, n)

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(experience, salary, alpha=0.6, color="steelblue",
           edgecolors="white", linewidth=0.5, s=60)

# Add trend line
m, b = np.polyfit(experience, salary, 1)
x_line = np.linspace(1, 15, 100)
ax.plot(x_line, m * x_line + b, color="coral", linewidth=2,
        label=f"Trend: ${m:,.0f}/year of exp")

ax.set_xlabel("Years of Experience")
ax.set_ylabel("Annual Salary ($)")
ax.set_title("Salary vs. Experience", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


# ── SCATTER WITH COLOR AND SIZE ENCODING ─────────────────────────────────
# Four variables in one chart: x, y, color, size
cities = pd.DataFrame({
    "city":        ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
                    "Pune", "Kolkata", "Ahmedabad", "Surat", "Jaipur"],
    "gdp_per_cap": [85000, 72000, 95000, 68000, 78000, 82000, 55000, 65000, 58000, 48000],
    "literacy":    [89, 87, 92, 84, 86, 88, 85, 83, 79, 77],
    "population":  [20.7, 32.9, 12.8, 7.8, 10.5, 6.5, 14.9, 8.5, 7.8, 4.0],
    "growth_rate": [4.2, 4.8, 6.1, 3.8, 5.5, 4.9, 2.8, 4.1, 5.2, 3.5]
})

fig, ax = plt.subplots(figsize=(10, 7))

scatter = ax.scatter(
    cities["gdp_per_cap"],
    cities["literacy"],
    c=cities["growth_rate"],           # Color = growth rate
    s=cities["population"] * 15,       # Size = population
    cmap="RdYlGn",                     # Color map
    alpha=0.8,
    edgecolors="white",
    linewidth=1
)

# Color bar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("GDP Growth Rate (%)")

# Add labels for each city
for _, row in cities.iterrows():
    ax.annotate(row["city"],
                xy=(row["gdp_per_cap"], row["literacy"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8)

# Legend for bubble size
for pop, label in [(5, "5M"), (10, "10M"), (20, "20M")]:
    ax.scatter([], [], s=pop*15, c="gray", alpha=0.5, label=label)
ax.legend(title="Population", loc="lower right")

ax.set_xlabel("GDP per Capita (₹)")
ax.set_ylabel("Literacy Rate (%)")
ax.set_title("Indian Cities: Economic & Social Indicators",
             fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()
```

---

## Part 6: Histograms and Distribution Plots

```python
# ── BASIC HISTOGRAM ───────────────────────────────────────────────────────
rng = np.random.default_rng(42)
response_times = np.abs(rng.normal(200, 50, 1000)) + 50   # in ms

fig, ax = plt.subplots(figsize=(9, 5))

n, bins, patches = ax.hist(response_times, bins=40, color="steelblue",
                            edgecolor="white", alpha=0.8)

# Highlight the SLA threshold
sla_threshold = 300
ax.axvline(sla_threshold, color="red", linestyle="--", linewidth=2,
           label=f"SLA Threshold ({sla_threshold}ms)")

# Shade region above SLA (slow requests)
ax.axvspan(sla_threshold, ax.get_xlim()[1], alpha=0.15, color="red")

# Statistics annotation
p95 = np.percentile(response_times, 95)
pct_over_sla = (response_times > sla_threshold).mean() * 100
stats_text = (f"n = {len(response_times):,}\n"
              f"Mean = {response_times.mean():.0f}ms\n"
              f"P95  = {p95:.0f}ms\n"
              f"SLA violation: {pct_over_sla:.1f}%")

ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
        va="top", ha="right", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

ax.set_xlabel("Response Time (ms)")
ax.set_ylabel("Frequency")
ax.set_title("API Response Time Distribution", fontsize=14, fontweight="bold")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


# ── OVERLAPPING DISTRIBUTIONS ─────────────────────────────────────────────
control   = rng.normal(50, 10, 500)
treatment = rng.normal(55, 10, 500)

fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(control, bins=30, alpha=0.6, color="steelblue", label="Control",
        edgecolor="white", density=True)
ax.hist(treatment, bins=30, alpha=0.6, color="coral", label="Treatment",
        edgecolor="white", density=True)

ax.axvline(control.mean(), color="steelblue", linestyle="--", linewidth=1.5,
           label=f"Control mean ({control.mean():.1f})")
ax.axvline(treatment.mean(), color="coral", linestyle="--", linewidth=1.5,
           label=f"Treatment mean ({treatment.mean():.1f})")

ax.set_xlabel("Metric Value")
ax.set_ylabel("Density")
ax.set_title("A/B Test: Metric Distribution Comparison",
             fontsize=14, fontweight="bold")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()
```

---

## Part 7: Seaborn — Statistical Visualization Made Easy

Seaborn is built on Matplotlib and provides:
- Better default aesthetics
- Statistical plot types (box, violin, regression, heatmap)
- Built-in integration with Pandas DataFrames
- Automatic handling of hue (color grouping)

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Generate sample data
rng = np.random.default_rng(42)
n = 300
df = pd.DataFrame({
    "salary":     rng.normal(80000, 25000, n).clip(30000, 200000),
    "experience": rng.uniform(0, 20, n),
    "dept":       rng.choice(["Engineering", "Marketing", "Product", "Sales"], n),
    "gender":     rng.choice(["Male", "Female"], n),
    "score":      rng.normal(75, 15, n).clip(0, 100)
})

# ── BOX PLOT ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

sns.boxplot(data=df, x="dept", y="salary", hue="gender",
            palette={"Male": "steelblue", "Female": "coral"},
            width=0.5, ax=ax)

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax.set_title("Salary Distribution by Department and Gender",
             fontsize=13, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Annual Salary ($)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


# ── VIOLIN PLOT — like box plot but shows full distribution shape ─────────
fig, ax = plt.subplots(figsize=(9, 5))

sns.violinplot(data=df, x="dept", y="salary",
               palette="Set2", inner="quart", ax=ax)

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax.set_title("Salary Distribution Shapes by Department",
             fontsize=13, fontweight="bold")
ax.set_xlabel("")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


# ── REGRESSION PLOT ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

sns.regplot(data=df, x="experience", y="salary",
            scatter_kws={"alpha": 0.4, "s": 30},
            line_kws={"color": "coral", "linewidth": 2},
            ax=ax)

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax.set_title("Salary vs. Experience (with Regression)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Annual Salary ($)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


# ── HEATMAP — correlations or pivot tables ───────────────────────────────
# Correlation heatmap
numeric_df = df[["salary", "experience", "score"]]
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(corr,
            annot=True,              # Show values in cells
            fmt=".2f",               # Format to 2 decimal places
            cmap="RdYlGn",           # Diverging color map
            center=0,                # Center color at 0
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax)

ax.set_title("Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()


# ── KDE PLOT — smooth density estimate ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

for dept, color in zip(df["dept"].unique(), ["steelblue", "coral", "green", "purple"]):
    subset = df[df["dept"] == dept]["salary"]
    sns.kdeplot(subset, label=dept, color=color, linewidth=2, fill=True, alpha=0.1, ax=ax)

ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax.set_xlabel("Salary ($)")
ax.set_ylabel("Density")
ax.set_title("Salary Density by Department (KDE)", fontsize=13, fontweight="bold")
ax.legend(title="Department")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


# ── COUNT PLOT ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(data=df, x="dept", hue="gender",
              palette={"Male": "steelblue", "Female": "coral"},
              edgecolor="white", ax=ax)

ax.set_title("Employee Count by Department and Gender",
             fontsize=13, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Count")
ax.legend(title="Gender")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


# ── PAIR PLOT — every variable against every other ───────────────────────
pair_data = df[["salary", "experience", "score", "dept"]].copy()

g = sns.pairplot(pair_data, hue="dept", palette="Set2",
                 plot_kws={"alpha": 0.5, "s": 20},
                 diag_kind="kde")

g.figure.suptitle("Pairwise Relationships", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()
```

---

## Part 8: Subplots and Layouts

```python
# ── BASIC SUBPLOTS ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Flatten for easy iteration
ax_flat = axes.flatten()

rng = np.random.default_rng(42)
data = rng.normal(0, 1, 500)

# Top-left: Histogram
ax_flat[0].hist(data, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
ax_flat[0].set_title("Histogram")

# Top-right: Box plot
ax_flat[1].boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.7))
ax_flat[1].set_title("Box Plot")

# Bottom-left: KDE via seaborn
import seaborn as sns
sns.kdeplot(data, ax=ax_flat[2], fill=True, color="steelblue")
ax_flat[2].set_title("KDE (Density Curve)")

# Bottom-right: Q-Q Plot (check normality)
from scipy import stats
(osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
ax_flat[3].scatter(osm, osr, alpha=0.5, s=10, color="steelblue")
ax_flat[3].plot(osm, slope * np.array(osm) + intercept, color="red",
                linewidth=2, label=f"R² = {r**2:.3f}")
ax_flat[3].set_title("Q-Q Plot (Normality Check)")
ax_flat[3].set_xlabel("Theoretical Quantiles")
ax_flat[3].set_ylabel("Sample Quantiles")
ax_flat[3].legend()

for ax in ax_flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Distribution Analysis Dashboard", fontsize=16,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# ── COMPLEX LAYOUTS WITH gridspec ────────────────────────────────────────
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# Large plot on left (spans 2 rows)
ax_main = fig.add_subplot(gs[:, 0])
# Three smaller plots on right
ax_tr = fig.add_subplot(gs[0, 1])
ax_mr = fig.add_subplot(gs[0, 2])
ax_br = fig.add_subplot(gs[1, 1])
ax_bl = fig.add_subplot(gs[1, 2])

# Main plot: time series
dates = pd.date_range("2025-01-01", periods=252, freq="B")
price = 100 * (1 + rng.normal(0.0003, 0.012, 252)).cumprod()
ax_main.plot(dates, price, color="steelblue", linewidth=1.5)
ax_main.fill_between(dates, price.min(), price, alpha=0.1, color="steelblue")
ax_main.set_title("Stock Price (Main)", fontweight="bold")
import matplotlib.dates as mdates
ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45)

# Returns histogram
returns = pd.Series(price).pct_change().dropna()
ax_tr.hist(returns, bins=25, color="coral", edgecolor="white", alpha=0.8)
ax_tr.set_title("Daily Returns", fontweight="bold", fontsize=10)

# Rolling volatility
vol_20 = returns.rolling(20).std() * np.sqrt(252)
ax_mr.plot(vol_20.values, color="green", linewidth=1.5)
ax_mr.set_title("Rolling 20D Volatility", fontweight="bold", fontsize=10)

# Drawdown
cummax = pd.Series(price).cummax()
drawdown = (price - cummax) / cummax * 100
ax_br.fill_between(range(len(drawdown)), drawdown, 0, color="red", alpha=0.4)
ax_br.set_title("Drawdown (%)", fontweight="bold", fontsize=10)

# Monthly returns heatmap
monthly_r = returns.values[:240].reshape(12, 20).mean(axis=1)
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
colors_bar = ["red" if r < 0 else "green" for r in monthly_r]
ax_bl.bar(months[:len(monthly_r)], monthly_r * 100, color=colors_bar, alpha=0.8)
ax_bl.set_title("Monthly Returns (%)", fontweight="bold", fontsize=10)
plt.setp(ax_bl.xaxis.get_majorticklabels(), rotation=45, fontsize=8)

for ax in [ax_main, ax_tr, ax_mr, ax_br, ax_bl]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Portfolio Analytics Dashboard", fontsize=16, fontweight="bold")
plt.savefig("dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Part 9: Advanced Customization

```python
# ── ANNOTATIONS ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-0.1 * x)

ax.plot(x, y, color="steelblue", linewidth=2)

# Find and annotate key points
max_idx = y.argmax()
min_idx = y.argmin()

ax.annotate("Maximum",
            xy=(x[max_idx], y[max_idx]),
            xytext=(x[max_idx] + 1, y[max_idx] + 0.15),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
            fontsize=11, color="green", fontweight="bold")

ax.annotate("Minimum",
            xy=(x[min_idx], y[min_idx]),
            xytext=(x[min_idx] + 1, y[min_idx] - 0.15),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            fontsize=11, color="red", fontweight="bold")

# Text box for statistics
stats_str = f"Max: {y.max():.3f}\nMin: {y.min():.3f}"
ax.text(0.02, 0.05, stats_str, transform=ax.transAxes,
        fontsize=10, verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_title("Damped Sine Wave with Annotations", fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


# ── CUSTOM COLOR PALETTES ─────────────────────────────────────────────────
# Seaborn palettes
palettes = ["deep", "muted", "bright", "pastel", "dark", "colorblind"]

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, palette in zip(axes.flatten(), palettes):
    colors = sns.color_palette(palette, 8)
    for i, color in enumerate(colors):
        ax.barh(i, 1, color=color)
    ax.set_title(palette)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Seaborn Color Palettes", fontsize=14)
plt.tight_layout()
plt.show()


# ── SAVING HIGH-QUALITY FIGURES ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_title("Publication-Ready Plot")

# For presentations / web (72-150 dpi)
plt.savefig("presentation.png", dpi=150, bbox_inches="tight")

# For print / publications (300+ dpi)
plt.savefig("publication.png", dpi=300, bbox_inches="tight")

# Vector format (infinitely scalable — best for papers)
plt.savefig("paper.pdf", bbox_inches="tight")
plt.savefig("paper.svg", bbox_inches="tight")

plt.show()
```

---

## Part 10: Seaborn FacetGrid — Conditioned Plots

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
n = 500
df = pd.DataFrame({
    "salary":     rng.normal(80000, 25000, n).clip(30000, 200000),
    "experience": rng.uniform(0, 20, n),
    "dept":       rng.choice(["Engineering", "Marketing", "Product", "Sales"], n),
    "gender":     rng.choice(["Male", "Female"], n),
    "year":       rng.choice([2023, 2024, 2025], n)
})

# ── FACETGRID — one subplot per category value ────────────────────────────
g = sns.FacetGrid(df, col="dept", col_wrap=2,
                  height=4, aspect=1.4, sharey=True)

g.map_dataframe(sns.histplot, x="salary", bins=20,
                color="steelblue", alpha=0.7, edgecolor="white")

g.set_axis_labels("Salary ($)", "Count")
g.set_titles(col_template="{col_name}")
g.figure.suptitle("Salary Distribution by Department", y=1.02,
                  fontsize=14, fontweight="bold")
g.tight_layout()
plt.show()


# ── FACETGRID WITH HUE ────────────────────────────────────────────────────
g = sns.FacetGrid(df, col="dept", col_wrap=2, hue="gender",
                  palette={"Male": "steelblue", "Female": "coral"},
                  height=4, aspect=1.3)

g.map_dataframe(sns.scatterplot, x="experience", y="salary", alpha=0.5, s=25)
g.add_legend(title="Gender")
g.set_axis_labels("Experience (yrs)", "Salary ($)")
g.set_titles("{col_name}")
g.figure.suptitle("Salary vs Experience by Department", y=1.02,
                  fontsize=14, fontweight="bold")
g.tight_layout()
plt.show()


# ── CATPLOT — categorical plots with FacetGrid ─────────────────────────────
g = sns.catplot(data=df, x="gender", y="salary", col="dept",
                kind="box", col_wrap=2,
                palette={"Male": "steelblue", "Female": "coral"},
                height=4, aspect=1.2)

g.set_axis_labels("", "Salary ($)")
g.set_titles("{col_name}")
g.figure.suptitle("Salary by Gender across Departments", y=1.02,
                  fontsize=14, fontweight="bold")
g.tight_layout()
plt.show()
```

---

## Part 11: Worked Examples

### Worked Example 1: Complete EDA Dashboard

```python
def eda_dashboard(df, numeric_col, cat_col, title="EDA Dashboard"):
    """
    Automated exploratory data analysis dashboard.
    Generates: histogram, box plots, bar chart, correlation heatmap.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = plt.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Plot 1: Histogram with KDE ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df[numeric_col].dropna(), bins=30, density=True,
             color="steelblue", edgecolor="white", alpha=0.7)
    from scipy import stats as scipy_stats
    kde_x = np.linspace(df[numeric_col].min(), df[numeric_col].max(), 200)
    kde = scipy_stats.gaussian_kde(df[numeric_col].dropna())
    ax1.plot(kde_x, kde(kde_x), color="coral", linewidth=2)
    ax1.axvline(df[numeric_col].mean(), color="red", linestyle="--",
                linewidth=1.5, label=f"Mean: {df[numeric_col].mean():.0f}")
    ax1.axvline(df[numeric_col].median(), color="green", linestyle="--",
                linewidth=1.5, label=f"Median: {df[numeric_col].median():.0f}")
    ax1.set_title(f"Distribution of {numeric_col}", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Plot 2: Box plot by category ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    categories = df[cat_col].unique()
    box_data = [df[df[cat_col] == cat][numeric_col].dropna() for cat in categories]
    bp = ax2.boxplot(box_data, patch_artist=True, labels=categories)
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_title(f"{numeric_col} by {cat_col}", fontweight="bold")
    ax2.tick_params(axis="x", rotation=30)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Plot 3: Category counts ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    counts = df[cat_col].value_counts()
    ax3.bar(counts.index, counts.values, color=colors[:len(counts)], alpha=0.8,
            edgecolor="white")
    for i, (cat, cnt) in enumerate(counts.items()):
        ax3.text(i, cnt + max(counts)*0.01, str(cnt), ha="center",
                 fontsize=9, fontweight="bold")
    ax3.set_title(f"{cat_col} Distribution", fontweight="bold")
    ax3.tick_params(axis="x", rotation=30)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # ── Plot 4: Numeric stats summary ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    stats_df = df.select_dtypes(include=np.number).describe().T
    ax4.axis("off")
    table = ax4.table(
        cellText=stats_df.round(1).values,
        rowLabels=stats_df.index,
        colLabels=stats_df.columns,
        cellLoc="center", rowLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax4.set_title("Numeric Summary", fontweight="bold", pad=15)

    # ── Plot 5: Correlation heatmap ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    num_df = df.select_dtypes(include=np.number)
    if len(num_df.columns) > 1:
        corr = num_df.corr()
        im = ax5.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax5.set_xticks(range(len(corr.columns)))
        ax5.set_yticks(range(len(corr.columns)))
        ax5.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax5.set_yticklabels(corr.columns, fontsize=8)
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                ax5.text(j, i, f"{corr.iloc[i, j]:.2f}",
                         ha="center", va="center", fontsize=8,
                         color="black")
        plt.colorbar(im, ax=ax5, shrink=0.8)
    ax5.set_title("Correlation Matrix", fontweight="bold")

    # ── Plot 6: Quantile plot ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    sorted_vals = np.sort(df[numeric_col].dropna())
    percentiles = np.linspace(0, 100, len(sorted_vals))
    ax6.plot(percentiles, sorted_vals, color="steelblue", linewidth=2)
    for p in [25, 50, 75, 90, 95]:
        val = np.percentile(sorted_vals, p)
        ax6.axhline(val, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax6.text(101, val, f"P{p}={val:.0f}", fontsize=7, va="center")
    ax6.set_xlabel("Percentile")
    ax6.set_ylabel(numeric_col)
    ax6.set_title(f"{numeric_col} Cumulative Distribution", fontweight="bold")
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    plt.savefig("eda_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()

# Run it
df = pd.DataFrame({
    "salary":     rng.normal(80000, 25000, 300).clip(30000, 200000),
    "experience": rng.uniform(0, 20, 300),
    "score":      rng.normal(75, 15, 300).clip(0, 100),
    "dept":       rng.choice(["Engineering", "Marketing", "Product", "Sales"], 300)
})
eda_dashboard(df, numeric_col="salary", cat_col="dept", title="Employee EDA")
```

### Worked Example 2: Sales Performance Dashboard

```python
def sales_dashboard(sales_df):
    """
    Full sales analytics dashboard:
    - Revenue trend with target line
    - Regional performance
    - Product category breakdown
    - YoY comparison
    """
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#F8F9FA")
    gs = plt.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"main": "#2196F3", "accent": "#FF9800", "good": "#4CAF50",
              "bad": "#F44336", "neutral": "#9E9E9E"}

    # ── Title ─────────────────────────────────────────────────────────────
    fig.text(0.5, 0.97, "Sales Performance Dashboard — 2025",
             ha="center", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.94, f"Generated: {pd.Timestamp.today().strftime('%B %d, %Y')}",
             ha="center", fontsize=10, color="gray")

    # ── KPIs as text ───────────────────────────────────────────────────────
    kpis = [
        ("Total Revenue", f"${sales_df['revenue'].sum():,.0f}", colors["main"]),
        ("Total Orders", f"{len(sales_df):,}", colors["accent"]),
        ("Avg Order Value", f"${sales_df['revenue'].mean():,.0f}", colors["good"]),
    ]
    for i, (label, value, color) in enumerate(kpis):
        ax_kpi = fig.add_subplot(gs[0, i])
        ax_kpi.set_facecolor(color)
        ax_kpi.text(0.5, 0.6, value, transform=ax_kpi.transAxes,
                    ha="center", va="center", fontsize=22, fontweight="bold",
                    color="white")
        ax_kpi.text(0.5, 0.2, label, transform=ax_kpi.transAxes,
                    ha="center", va="center", fontsize=11, color="white", alpha=0.9)
        ax_kpi.set_xticks([])
        ax_kpi.set_yticks([])
        for spine in ax_kpi.spines.values():
            spine.set_visible(False)

    # ── Revenue trend ─────────────────────────────────────────────────────
    ax_trend = fig.add_subplot(gs[1, :2])
    ax_trend.set_facecolor("#FFFFFF")
    monthly = sales_df.groupby("month")["revenue"].sum()

    ax_trend.plot(range(len(monthly)), monthly.values, color=colors["main"],
                  linewidth=2.5, marker="o", markersize=5, zorder=3)
    ax_trend.fill_between(range(len(monthly)), monthly.values, alpha=0.1,
                          color=colors["main"])

    # Target line
    target = monthly.mean() * 1.1
    ax_trend.axhline(target, color=colors["accent"], linestyle="--",
                     linewidth=1.5, label=f"Target: ${target:,.0f}")

    ax_trend.set_xticks(range(len(monthly)))
    ax_trend.set_xticklabels([str(m) for m in monthly.index],
                              rotation=45, fontsize=8)
    ax_trend.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax_trend.set_title("Monthly Revenue", fontweight="bold")
    ax_trend.legend()
    ax_trend.spines["top"].set_visible(False)
    ax_trend.spines["right"].set_visible(False)

    # ── Category breakdown ────────────────────────────────────────────────
    ax_cat = fig.add_subplot(gs[1, 2])
    ax_cat.set_facecolor("#FFFFFF")
    cat_rev = sales_df.groupby("category")["revenue"].sum().sort_values()
    bars = ax_cat.barh(cat_rev.index, cat_rev.values,
                       color=colors["main"], alpha=0.8, edgecolor="white")
    for bar, val in zip(bars, cat_rev.values):
        ax_cat.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"${val/1000:.0f}K", va="center", fontsize=8)
    ax_cat.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax_cat.set_title("Revenue by Category", fontweight="bold")
    ax_cat.spines["top"].set_visible(False)
    ax_cat.spines["right"].set_visible(False)

    plt.savefig("sales_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()


# Generate and run
rng = np.random.default_rng(42)
n = 500
sales = pd.DataFrame({
    "date":     pd.date_range("2025-01-01", periods=n, freq="12H"),
    "revenue":  rng.uniform(500, 15000, n),
    "category": rng.choice(["Electronics","Clothing","Food","Home","Beauty"], n),
    "region":   rng.choice(["North","South","East","West"], n)
})
sales["month"] = sales["date"].dt.to_period("M")
sales_dashboard(sales)
```

---

## Practice Problems

### Easy (1–5)

**Problem 1: Basic Line Plot**
Plot three different mathematical functions on the same axes.

```python
x = np.linspace(-2 * np.pi, 2 * np.pi, 500)

# Plot: sin(x), cos(x), tan(x) (clipped to [-3, 3])
# Requirements:
# - Different colors and line styles for each
# - Legend with proper labels
# - Horizontal line at y=0
# - Title and axis labels
# - Remove top and right spines
```

**Problem 2: Comparative Bar Chart**
Visualize survey results across multiple groups.

```python
categories = ["Work-Life Balance", "Salary", "Culture", "Growth", "Benefits"]
engineering = [7.2, 8.1, 8.5, 7.8, 7.0]
marketing   = [8.1, 7.4, 8.0, 6.5, 7.8]
product     = [7.8, 7.9, 8.3, 8.1, 7.5]

# Create grouped bar chart with:
# - Three groups side by side for each category
# - Value labels on each bar
# - Score range 0-10 on Y axis
# - Title: "Employee Satisfaction by Department"
# - Legend and clean styling
```

**Problem 3: Distribution Comparison**
Compare salary distributions between two years.

```python
rng = np.random.default_rng(42)
salaries_2024 = rng.normal(72000, 18000, 1000).clip(30000, 200000)
salaries_2025 = rng.normal(78000, 20000, 1000).clip(30000, 200000)

# Create overlapping histograms showing:
# - Both distributions (50% transparency)
# - Mean lines for each
# - Percentage change in mean annotated
# - Proper title, labels, legend
```

**Problem 4: Scatter Plot with Trend Line**
Visualize the relationship between ad spend and revenue.

```python
rng = np.random.default_rng(42)
ad_spend = rng.uniform(1000, 50000, 80)
revenue  = ad_spend * rng.uniform(3, 7, 80) + rng.normal(0, 20000, 80)

# Create scatter plot with:
# - Color coding by "ROI" (revenue/ad_spend)
# - Trend line
# - Colorbar showing ROI
# - Annotate the best and worst ROI points
```

**Problem 5: Time Series with Events**
Plot website traffic with annotated events.

```python
dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
rng = np.random.default_rng(42)
traffic = pd.Series(
    5000 + np.cumsum(rng.normal(10, 100, 365)),
    index=dates
)

events = {
    "2025-02-14": "Valentine's Campaign",
    "2025-07-04": "Independence Day Sale",
    "2025-11-29": "Black Friday",
    "2025-12-25": "Christmas"
}

# Plot traffic with:
# - 7-day rolling average overlay
# - Vertical lines for events
# - Event labels as annotations
# - Shaded weekends
```

### Medium (6–12)

**Problem 6: Heatmap Calendar**
Create a GitHub-style contribution heatmap.

```python
dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
activity = pd.Series(
    np.random.choice([0, 0, 0, 1, 2, 3, 5, 8], len(dates)),
    index=dates
)

# Reshape into a calendar grid (weeks × days)
# Color intensity = activity level
# X-axis = months, Y-axis = days of week
# Title: "Activity Heatmap 2025"
```

**Problem 7: Multi-Panel Time Series Dashboard**
Create a 4-panel financial analysis chart.

```python
dates = pd.date_range("2025-01-01", periods=252, freq="B")
rng = np.random.default_rng(42)
price = 100 * (1 + rng.normal(0.0005, 0.015, 252)).cumprod()
volume = rng.integers(1_000_000, 10_000_000, 252)

# Create 4-panel chart:
# Panel 1 (tall): Price with 20-day and 50-day moving averages
# Panel 2 (short): Volume as bar chart
# Panel 3 (short): Daily returns
# Panel 4 (short): RSI (Relative Strength Index = 14-day average gain/loss ratio)
# Share X axis between all panels
```

**Problem 8: Correlation Heatmap with Significance**
Create an annotated correlation matrix.

```python
rng = np.random.default_rng(42)
n = 200
df = pd.DataFrame({
    "revenue": rng.normal(0, 1, n),
    "ad_spend": rng.normal(0, 1, n),
    "customer_count": rng.normal(0, 1, n),
    "satisfaction": rng.normal(0, 1, n),
    "churn_rate": rng.normal(0, 1, n),
})
# Add some actual correlations
df["revenue"] = df["ad_spend"] * 0.7 + df["customer_count"] * 0.4 + rng.normal(0, 0.3, n)

# Plot:
# - Triangular correlation matrix (no duplicate values)
# - Color the cells by correlation strength
# - Mark statistically significant correlations (* p<0.05, ** p<0.01)
# - Circle size proportional to |correlation|
```

**Problem 9: Animated Plot**
Create an animation showing data changing over time.

```python
import matplotlib.animation as animation

# Animate a random walk:
# - Start at (0, 0)
# - Each frame adds one step (random direction)
# - Keep a trail of the last 50 steps
# - Show current position as a large dot
# - Title updates showing step number

fig, ax = plt.subplots(figsize=(8, 8))

def animate(frame):
    pass   # Your animation logic here

anim = animation.FuncAnimation(fig, animate, frames=200, interval=50)
plt.show()
```

**Problem 10: Interactive-Style Plot with Widgets**
Add a hover tooltip effect using event handling.

```python
# Create a scatter plot where hovering over a point shows:
# - The data point's values
# - Which group it belongs to
# - Use mplcursors or plt.connect() for interactivity

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)
x = rng.normal(0, 1, 50)
y = rng.normal(0, 1, 50)
labels = [f"Point_{i}" for i in range(50)]

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(x, y, picker=True)

# Add click event to show point info
def on_pick(event):
    pass   # Show annotation on click

fig.canvas.mpl_connect("pick_event", on_pick)
plt.show()
```

**Problem 11: Waterfall Chart**
Create a waterfall chart for financial statement visualization.

```python
items = ["Revenue", "COGS", "Gross Profit", "Marketing", "R&D",
         "G&A", "EBITDA", "Depreciation", "Net Income"]
values = [1000, -420, 580, -150, -80, -60, 290, -40, 250]
# Note: "Gross Profit" and "EBITDA" are running totals, rest are changes

# Create waterfall chart:
# - Green bars for increases (Gross Profit, EBITDA, Net Income)
# - Red bars for decreases
# - Gray connectors between bars
# - Running total line
# - Value labels on each bar
```

**Problem 12: Ridgeline / Joy Plot**
Show distribution of values across many categories.

```python
rng = np.random.default_rng(42)
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Create overlapping KDE plots for each month
# (simulating temperature distributions)
temps = {m: rng.normal(20 + 5 * np.sin(i * np.pi / 6), 3, 200)
         for i, m in enumerate(months)}

# Each month's distribution is offset vertically
# Color gradient from cold (blue) to warm (red)
```

### Hard (13–20)

**Problem 13: Network Graph Visualization**
Visualize a social network using networkx + matplotlib.

```python
# pip install networkx
import networkx as nx

# Create a social network and visualize:
# - Node size = number of connections
# - Node color = community (use community detection)
# - Edge thickness = connection strength
# - Label top-10 most connected nodes
# - Spring layout

G = nx.karate_club_graph()   # Built-in test graph
# Your visualization here
```

**Problem 14: Geographic Heatmap**
Plot data on a map using matplotlib basemap or cartopy.

```python
# City data: (city, lat, lon, value)
cities = [
    ("Mumbai", 19.07, 72.87, 485000),
    ("Delhi", 28.61, 77.20, 392000),
    ("Bangalore", 12.97, 77.59, 278000),
    ("Chennai", 13.08, 80.27, 195000),
    ("Hyderabad", 17.38, 78.47, 134000),
    ("Pune", 18.52, 73.85, 98000),
    ("Kolkata", 22.57, 88.36, 87000),
]

# Plot on India map:
# - Circle size = value
# - Color intensity = value
# - City labels
# - Title and colorbar
```

**Problem 15: 3D Surface Plot**
Visualize a mathematical function in 3D.

```python
from mpl_toolkits.mplot3d import Axes3D

# Plot the Rosenbrock function:
# f(x,y) = (1-x)^2 + 100(y-x^2)^2

# Requirements:
# - 3D surface with color gradient
# - Contour lines projected on the floor
# - Mark the global minimum at (1, 1)
# - Proper viewing angle
# - Colorbar
```

**Problem 16: Custom Legend and Annotations**
Build a fully annotated chart for a business presentation.

```python
# Create a "magic quadrant" scatter plot:
# X-axis: Market growth rate
# Y-axis: Market share
# Bubble size: Revenue
# Color: Profitability (green=profitable, red=loss)
# Quadrant labels: Leaders, Challengers, Visionaries, Niche Players
# Annotate each bubble with company name
# Add quadrant background colors (light shading)
```

**Problem 17: Gantt Chart**
Build a project timeline visualization.

```python
tasks = [
    ("Discovery", "2025-01-05", "2025-01-19", "Planning"),
    ("Design",    "2025-01-15", "2025-02-02", "Design"),
    ("Development","2025-02-01","2025-03-15", "Engineering"),
    ("Testing",   "2025-03-01", "2025-03-22", "QA"),
    ("UAT",       "2025-03-15", "2025-03-29", "Delivery"),
    ("Launch",    "2025-03-28", "2025-04-05", "Delivery"),
]

# Create Gantt chart:
# - Horizontal bars for each task
# - Color-coded by phase
# - Today's date as vertical red line
# - Task labels on each bar
# - Weekend shading
```

**Problem 18: Custom Statistical Plot**
Recreate a Tufte-style sparkline dashboard.

```python
# Sparklines: tiny inline charts showing trends
# For 10 different metrics, create a dashboard where:
# - Each row is a metric
# - Shows last 52 weeks of data as a tiny line
# - Highlights min and max points
# - Shows current value on the right
# - Color-coded based on trend (up/down)
# All in a single clean figure
```

**Problem 19: Word Cloud from Text**
Create a frequency-weighted word cloud.

```python
# pip install wordcloud
from wordcloud import WordCloud

text = """
Python data science machine learning artificial intelligence
deep learning neural networks natural language processing
computer vision pandas numpy matplotlib seaborn scikit-learn
tensorflow pytorch statistics visualization analytics
"""

# Create word cloud:
# - Word size = frequency
# - Custom color scheme
# - Mask image to shape (circle, heart, or custom)
# - Remove stopwords
# - Save as high-res PNG
```

**Problem 20: Full Automated Report Generator**
Build a function that creates a PDF report from data.

```python
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def generate_pdf_report(data_dict, output_path="report.pdf"):
    """
    Generate a multi-page PDF report.
    Each page has:
    - Page 1: Cover page with title, date, summary stats
    - Page 2: Revenue trend + category breakdown
    - Page 3: Customer analysis + cohort table
    - Page 4: Product performance heatmap
    - Page 5: Appendix with data tables
    
    data_dict: {"sales": df, "customers": df, "products": df}
    """
    with PdfPages(output_path) as pdf:
        # Page 1: Cover
        fig = plt.figure(figsize=(8.5, 11))
        # ... your content here
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()
        
        # Page 2: Revenue Analysis
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 11))
        # ... your content here
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()
        
        # Set PDF metadata
        d = pdf.infodict()
        d["Title"] = "Business Analytics Report"
        d["Author"] = "Analytics Team"
        d["Subject"] = "Monthly Performance Review"
```

---

## Answer Keys (Selected Problems)

### Problem 1 Solution:
```python
x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
fig, ax = plt.subplots(figsize=(10, 5))

tan_clipped = np.where(np.abs(np.tan(x)) < 3, np.tan(x), np.nan)

ax.plot(x, np.sin(x), color="steelblue", linewidth=2, label="sin(x)")
ax.plot(x, np.cos(x), color="coral", linewidth=2, linestyle="--", label="cos(x)")
ax.plot(x, tan_clipped, color="green", linewidth=2, linestyle=":", label="tan(x)")

ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("x (radians)")
ax.set_ylabel("y")
ax.set_title("Trigonometric Functions", fontsize=14, fontweight="bold")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(-2*np.pi, 2*np.pi)
ax.set_ylim(-3, 3)
ax.set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
ax.set_xticklabels(["-2π", "-π", "0", "π", "2π"])
plt.tight_layout()
plt.show()
```

### Problem 5 Solution:
```python
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(traffic.index, traffic.values, color="steelblue", linewidth=1, alpha=0.5)
rolling = traffic.rolling(7).mean()
ax.plot(rolling.index, rolling.values, color="steelblue", linewidth=2,
        label="7-Day Rolling Avg")

# Shade weekends
for date in traffic.index:
    if date.weekday() >= 5:
        ax.axvspan(date, date + pd.Timedelta(days=1), alpha=0.08, color="gray")

# Event annotations
for date_str, label in events.items():
    date = pd.Timestamp(date_str)
    val = traffic.get(date, traffic.rolling(3).mean().get(date))
    ax.axvline(date, color="coral", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.annotate(label, xy=(date, traffic.max()),
                rotation=90, fontsize=8, color="coral",
                xytext=(5, -5), textcoords="offset points", va="top")

ax.set_title("Website Traffic 2025", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Daily Visitors")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.tight_layout()
plt.show()
```

---

## Mini-Project: Automated EDA Report Generator

```python
"""
auto_eda.py
Generates a comprehensive visual EDA report from ANY DataFrame.
Saves a multi-page PDF with all standard analyses.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

sns.set_theme(style="whitegrid", font_scale=0.9)

def auto_eda_report(df, title="EDA Report", output_path="eda_report.pdf"):
    """
    Generate a complete EDA report as PDF.
    Automatically handles any DataFrame.
    
    Args:
        df: DataFrame to analyze
        title: Report title
        output_path: Where to save the PDF
    """
    numeric_cols  = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols      = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    
    print(f"Generating EDA report for {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Numeric: {numeric_cols}")
    print(f"Categorical: {cat_cols}")
    print(f"Datetime: {datetime_cols}")
    
    with PdfPages(output_path) as pdf:
        
        # ── PAGE 1: OVERVIEW ─────────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#1A1A2E")
        
        # Title
        fig.text(0.5, 0.82, title, ha="center", fontsize=28,
                 fontweight="bold", color="white")
        fig.text(0.5, 0.75, f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
                 ha="center", fontsize=12, color="#AAAAAA")
        
        # Dataset overview
        overview = [
            ("Rows",         f"{df.shape[0]:,}"),
            ("Columns",      f"{df.shape[1]}"),
            ("Numeric cols", f"{len(numeric_cols)}"),
            ("Cat. cols",    f"{len(cat_cols)}"),
            ("Total nulls",  f"{df.isnull().sum().sum():,}"),
            ("Null rate",    f"{df.isnull().mean().mean():.1%}"),
            ("Duplicates",   f"{df.duplicated().sum():,}"),
            ("Memory",       f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"),
        ]
        
        ax_info = fig.add_axes([0.1, 0.35, 0.8, 0.35])
        ax_info.set_facecolor("#16213E")
        ax_info.set_xticks([])
        ax_info.set_yticks([])
        for spine in ax_info.spines.values():
            spine.set_visible(False)
        
        cols_per_row = 4
        for i, (label, value) in enumerate(overview):
            row = i // cols_per_row
            col = i % cols_per_row
            x = 0.05 + col * 0.24
            y = 0.7 - row * 0.4
            ax_info.text(x, y, value, transform=ax_info.transAxes,
                         fontsize=20, fontweight="bold", color="#4FC3F7",
                         va="center")
            ax_info.text(x, y - 0.18, label, transform=ax_info.transAxes,
                         fontsize=9, color="#AAAAAA", va="center")
        
        # Column types summary
        ax_bar = fig.add_axes([0.1, 0.05, 0.8, 0.22])
        ax_bar.set_facecolor("#16213E")
        type_counts = df.dtypes.value_counts()
        colors_bar = ["#4FC3F7", "#FF7043", "#66BB6A", "#AB47BC"]
        bars = ax_bar.barh([str(t) for t in type_counts.index], type_counts.values,
                           color=colors_bar[:len(type_counts)], edgecolor="none")
        for bar, val in zip(bars, type_counts.values):
            ax_bar.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                        str(val), va="center", color="white", fontsize=10)
        ax_bar.set_title("Column Types", color="white", fontsize=11)
        ax_bar.tick_params(colors="white")
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        ax_bar.set_facecolor("#16213E")
        
        pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        
        # ── PAGE 2: NULL ANALYSIS ─────────────────────────────────────────
        if df.isnull().any().any():
            fig, axes = plt.subplots(1, 2, figsize=(11, 7))
            fig.suptitle("Missing Value Analysis", fontsize=16, fontweight="bold")
            
            # Null counts per column
            null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
            null_pct = null_pct[null_pct > 0]
            
            colors_null = ["#F44336" if p > 20 else "#FF9800" if p > 5 else "#4CAF50"
                          for p in null_pct.values]
            axes[0].barh(null_pct.index, null_pct.values, color=colors_null, alpha=0.8)
            axes[0].set_xlabel("% Missing")
            axes[0].set_title("Columns with Missing Values")
            for i, (col, pct) in enumerate(null_pct.items()):
                axes[0].text(pct + 0.3, i, f"{pct:.1f}%", va="center", fontsize=9)
            axes[0].spines["top"].set_visible(False)
            axes[0].spines["right"].set_visible(False)
            
            # Null heatmap
            null_sample = df.isnull().sample(min(100, len(df)), random_state=42)
            sns.heatmap(null_sample.T, cmap="RdYlGn_r", cbar=False,
                        xticklabels=False, ax=axes[1])
            axes[1].set_title("Null Pattern Heatmap (sample of 100 rows)")
            axes[1].set_xlabel("Rows")
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()
        
        # ── PAGE 3: NUMERIC DISTRIBUTIONS ────────────────────────────────
        if numeric_cols:
            n_cols_display = min(len(numeric_cols), 6)
            display_cols = numeric_cols[:n_cols_display]
            
            n_rows = (n_cols_display + 2) // 3
            fig, axes = plt.subplots(n_rows, 3, figsize=(11, 3.5 * n_rows))
            axes = np.array(axes).flatten()
            fig.suptitle("Numeric Column Distributions", fontsize=16, fontweight="bold")
            
            for i, col in enumerate(display_cols):
                ax = axes[i]
                data = df[col].dropna()
                ax.hist(data, bins=30, color="steelblue", edgecolor="white",
                        alpha=0.8, density=True)
                try:
                    from scipy import stats as scipy_stats
                    kde_x = np.linspace(data.min(), data.max(), 200)
                    kde = scipy_stats.gaussian_kde(data)
                    ax.plot(kde_x, kde(kde_x), color="coral", linewidth=2)
                    skew = scipy_stats.skew(data)
                    ax.text(0.95, 0.92, f"Skew: {skew:.2f}", transform=ax.transAxes,
                            ha="right", fontsize=8, color="purple")
                except Exception:
                    pass
                
                ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.5,
                           label=f"μ={data.mean():.1f}")
                ax.axvline(data.median(), color="green", linestyle="--", linewidth=1.5,
                           label=f"M={data.median():.1f}")
                ax.set_title(col, fontweight="bold", fontsize=10)
                ax.legend(fontsize=7)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            
            # Hide unused axes
            for j in range(len(display_cols), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()
        
        # ── PAGE 4: CORRELATION MATRIX ────────────────────────────────────
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(11, 8))
            fig.suptitle("Correlation Matrix", fontsize=16, fontweight="bold")
            
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr), k=1)   # Upper triangle mask
            
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                        square=True, linewidths=0.5, ax=ax,
                        annot_kws={"size": 9})
            
            ax.set_title("Pearson Correlation (lower triangle)")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()
        
        # ── PAGE 5: CATEGORICAL ANALYSIS ─────────────────────────────────
        if cat_cols:
            n_cats = min(len(cat_cols), 4)
            display_cats = cat_cols[:n_cats]
            
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            axes = axes.flatten()
            fig.suptitle("Categorical Column Analysis", fontsize=16, fontweight="bold")
            
            for i, col in enumerate(display_cats):
                ax = axes[i]
                vc = df[col].value_counts().head(15)
                palette = sns.color_palette("Set2", len(vc))
                ax.barh(vc.index[::-1], vc.values[::-1], color=palette[::-1], alpha=0.8)
                for j, (val, cnt) in enumerate(zip(vc.index[::-1], vc.values[::-1])):
                    ax.text(cnt + 0.1, j, f"{cnt:,} ({cnt/len(df):.1%})",
                            va="center", fontsize=8)
                ax.set_title(f"{col} (top {len(vc)} values)", fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            
            for j in range(len(display_cats), 4):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()
        
        # ── PAGE 6: DATA SAMPLE ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis("off")
        
        sample = df.head(20)
        col_labels = list(sample.columns)
        cell_data  = sample.values.tolist()
        
        table = ax.table(
            cellText=[[str(v)[:20] for v in row] for row in cell_data],
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width(col=list(range(len(col_labels))))
        
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#2196F3")
            table[0, j].set_text_props(color="white", fontweight="bold")
        
        for i in range(1, len(cell_data) + 1):
            for j in range(len(col_labels)):
                if i % 2 == 0:
                    table[i, j].set_facecolor("#F5F5F5")
        
        ax.set_title(f"Data Sample (first 20 rows of {len(df):,})",
                     fontsize=14, fontweight="bold", pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()
        
        # PDF metadata
        d = pdf.infodict()
        d["Title"] = title
        d["Author"] = "Auto-EDA Report Generator"
        d["Subject"] = f"EDA Report — {df.shape[0]} rows, {df.shape[1]} columns"
        d["CreationDate"] = datetime.now()
    
    print(f"\n✓ Report saved to: {output_path}")
    print(f"  Pages: 6")
    print(f"  Columns analyzed: {df.shape[1]}")


# ── DEMO ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1000
    
    df = pd.DataFrame({
        "age":         rng.integers(22, 65, n),
        "salary":      rng.normal(75000, 20000, n).clip(25000, 200000),
        "experience":  rng.uniform(0, 25, n),
        "score":       rng.beta(5, 2, n) * 100,
        "dept":        rng.choice(["Engineering","Marketing","Product","Sales","HR"], n),
        "gender":      rng.choice(["Male", "Female", "Other"], n, p=[0.52, 0.45, 0.03]),
        "city":        rng.choice(["Mumbai","Delhi","Bangalore","Chennai"], n),
        "hire_date":   pd.date_range("2018-01-01", periods=n, freq="8H").to_list(),
        "active":      rng.choice([True, False], n, p=[0.92, 0.08]),
    })
    
    # Introduce some nulls
    df.loc[rng.choice(n, 50, replace=False), "salary"] = np.nan
    df.loc[rng.choice(n, 30, replace=False), "dept"]   = np.nan
    
    auto_eda_report(df, title="Employee Dataset — EDA Report", output_path="eda_report.pdf")
```

---

## Chapter Summary

You can now turn raw data into clear, compelling visuals!

✅ **Why Visualize**: Anscombe's Quartet proves identical statistics can hide completely different patterns
✅ **Matplotlib Architecture**: Figure → Axes → Artists; always use OO style (`fig, ax = plt.subplots()`)
✅ **Line Plots**: Time series, multi-line, styles, markers, date formatting
✅ **Bar Charts**: Vertical, horizontal, grouped, stacked — with value labels
✅ **Scatter Plots**: Relationships, four-variable encoding (x, y, color, size), trend lines
✅ **Histograms**: Single distributions, overlapping, KDE, Q-Q plots
✅ **Seaborn**: Box, violin, regression, heatmap, KDE, count, pair plots
✅ **Subplots**: `plt.subplots()`, `GridSpec`, complex dashboards
✅ **Customization**: Annotations, spines, formatters, color palettes, saving
✅ **FacetGrid**: Conditioned plots across category values
✅ **PDF Reports**: `PdfPages` for multi-page automated reports

**Key Takeaways:**
- Always use `fig, ax = plt.subplots()` — avoid implicit pyplot state
- `sns.set_theme()` at the top dramatically improves aesthetics for free
- Remove top and right spines (`ax.spines["top"].set_visible(False)`) for cleaner charts
- `tight_layout()` and `bbox_inches="tight"` when saving prevents clipping
- For presentations: 150 dpi PNG. For print/papers: 300 dpi or PDF/SVG vector format

**Next Chapter Preview:**
Chapter 14 covers **Regular Expressions** — the Swiss Army knife for text processing. Pattern matching, extraction, validation, and replacement across any text data!
