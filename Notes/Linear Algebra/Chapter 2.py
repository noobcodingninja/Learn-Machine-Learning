"""
LINEAR ALGEBRA FOR MACHINE LEARNING - CHAPTER 2
Python Implementation: Linear Functions, Taylor Approximation, and Regression

This module demonstrates:
1. Linear functions (and affine functions)
2. Taylor approximation (1D and multivariate)
3. Linear regression (simple and multiple)
4. Gradient descent
5. Model evaluation metrics

Each concept is implemented in TWO ways:
- From scratch (using only basic Python)
- Using NumPy/Scikit-learn (standard ML libraries)
"""

import time

# ============================================================================
# PART 1: BASIC PYTHON IMPLEMENTATION (From Scratch)
# ============================================================================

print("="*70)
print("PART 1: BASIC PYTHON IMPLEMENTATION (From Scratch)")
print("="*70)
print()

# ----------------------------------------------------------------------------
# 1.1 Linear Functions
# ----------------------------------------------------------------------------

print("1.1 LINEAR FUNCTIONS")
print("-" * 50)

def linear_function(coefficients, x):
    """
    Evaluate linear function f(x) = a · x
    
    Args:
        coefficients: List of coefficients [a1, a2, ..., an]
        x: Input vector [x1, x2, ..., xn]
    
    Returns:
        Scalar result
    """
    if len(coefficients) != len(x):
        raise ValueError("Dimension mismatch")
    
    result = 0
    for i in range(len(coefficients)):
        result += coefficients[i] * x[i]
    
    return result

def affine_function(coefficients, x, bias):
    """
    Evaluate affine function f(x) = a · x + b
    
    Args:
        coefficients: List of coefficients
        x: Input vector
        bias: Scalar bias term
    
    Returns:
        Scalar result
    """
    return linear_function(coefficients, x) + bias

# Example: Smoothie shop pricing
prices = [1, 2, 3]  # (bananas, strawberries, protein)
order1 = [2, 1, 1]  # Order: 2 bananas, 1 strawberry, 1 protein

print("Smoothie Shop Example:")
print(f"Prices per item: {prices}")
print(f"Customer order: {order1}")

total = linear_function(prices, order1)
print(f"Total cost: ${total}")
print()

# Verify linearity properties
order2 = [1, 0, 1]
print("Verifying Linearity Properties:")
print(f"Order 1: {order1}, Cost: ${linear_function(prices, order1)}")
print(f"Order 2: {order2}, Cost: ${linear_function(prices, order2)}")

# Property 1: Scaling
scaled_order = [2*x for x in order1]
cost_original = linear_function(prices, order1)
cost_scaled = linear_function(prices, scaled_order)
print(f"\nScaling property:")
print(f"  f(2*order1) = {cost_scaled}")
print(f"  2*f(order1) = {2 * cost_original}")
print(f"  Equal? {cost_scaled == 2 * cost_original}")

# Property 2: Additivity
combined_order = [order1[i] + order2[i] for i in range(len(order1))]
cost_combined = linear_function(prices, combined_order)
cost_sum = linear_function(prices, order1) + linear_function(prices, order2)
print(f"\nAdditivity property:")
print(f"  f(order1 + order2) = {cost_combined}")
print(f"  f(order1) + f(order2) = {cost_sum}")
print(f"  Equal? {cost_combined == cost_sum}")
print()

# Affine function example: with service fee
service_fee = 5
print("With $5 service fee (affine function):")
total_with_fee = affine_function(prices, order1, service_fee)
print(f"Total cost: ${total_with_fee}")

# Check if still linear (it shouldn't be!)
scaled_cost_with_fee = affine_function(prices, scaled_order, service_fee)
print(f"\nIs affine function linear?")
print(f"  f(2*order) = {scaled_cost_with_fee}")
print(f"  2*f(order) = {2 * total_with_fee}")
print(f"  Equal? {scaled_cost_with_fee == 2 * total_with_fee} (NO! Not linear)")
print()

# ----------------------------------------------------------------------------
# 1.2 Application: House Price Prediction
# ----------------------------------------------------------------------------

print("1.2 HOUSE PRICE PREDICTION (Linear Function)")
print("-" * 50)

# Linear model: price = w · features + bias
weights = [100, 50000, -1000, 20000]  # Per: sqft, bedrooms, age, school_rating
bias = 50000

house1 = [2000, 3, 10, 8]  # 2000 sqft, 3 bed, 10 years, school 8/10
house2 = [1500, 2, 20, 6]
house3 = [2500, 4, 5, 9]

print("Model: price = 100*sqft + 50000*beds - 1000*age + 20000*school + 50000")
print()

def predict_price(features, weights, bias):
    """Predict house price using linear model"""
    return affine_function(weights, features, bias)

print(f"House 1 {house1}:")
price1 = predict_price(house1, weights, bias)
print(f"  Predicted price: ${price1:,}")

print(f"\nHouse 2 {house2}:")
price2 = predict_price(house2, weights, bias)
print(f"  Predicted price: ${price2:,}")

print(f"\nHouse 3 {house3}:")
price3 = predict_price(house3, weights, bias)
print(f"  Predicted price: ${price3:,}")
print()

# ----------------------------------------------------------------------------
# 1.3 Taylor Approximation (1D)
# ----------------------------------------------------------------------------

print("1.3 TAYLOR APPROXIMATION (1D)")
print("-" * 50)

def sqrt_taylor(x, a=1):
    """
    Approximate sqrt(x) using Taylor expansion around point a.
    f(x) ≈ f(a) + f'(a)(x - a)
    
    For f(x) = sqrt(x):
    f'(x) = 1/(2*sqrt(x))
    """
    # Calculate sqrt(a) using Newton's method
    def sqrt_newton(n, iterations=20):
        if n == 0:
            return 0
        x = n / 2
        for _ in range(iterations):
            x = (x + n / x) / 2
        return x
    
    f_a = sqrt_newton(a)
    f_prime_a = 1 / (2 * f_a)
    
    # Taylor approximation
    approx = f_a + f_prime_a * (x - a)
    
    return approx, f_a, f_prime_a

# Example: Approximate sqrt(10) using sqrt(9) = 3
print("Approximating sqrt(10) using Taylor expansion at a=9:")
print()

approx, f_a, f_prime = sqrt_taylor(10, a=9)
actual = 10 ** 0.5

print(f"Expansion point: a = 9")
print(f"f(9) = sqrt(9) = {f_a}")
print(f"f'(9) = 1/(2*sqrt(9)) = {f_prime:.4f}")
print(f"\nTaylor approximation: sqrt(10) ≈ {approx:.4f}")
print(f"Actual value: sqrt(10) = {actual:.4f}")
print(f"Error: {abs(approx - actual):.4f}")
print()

# Try different distances from expansion point
print("Testing approximation at different distances from a=9:")
test_points = [9.1, 9.5, 10, 11, 13, 16]
for x in test_points:
    approx, _, _ = sqrt_taylor(x, a=9)
    actual = x ** 0.5
    error = abs(approx - actual)
    print(f"  x={x:4.1f}: Approx={approx:.4f}, Actual={actual:.4f}, Error={error:.4f}")

print("\nNotice: Error increases as we move away from expansion point!")
print()

# ----------------------------------------------------------------------------
# 1.4 Taylor Approximation (Multivariate)
# ----------------------------------------------------------------------------

print("1.4 TAYLOR APPROXIMATION (Multivariate)")
print("-" * 50)

def compute_gradient(func, point, h=1e-5):
    """
    Compute gradient numerically using finite differences.
    
    Args:
        func: Function to differentiate
        point: List representing point [x1, x2, ...]
        h: Step size for finite difference
    
    Returns:
        Gradient as list
    """
    gradient = []
    for i in range(len(point)):
        # Create point with small perturbation in dimension i
        point_plus = point.copy()
        point_plus[i] += h
        
        point_minus = point.copy()
        point_minus[i] -= h
        
        # Finite difference approximation
        derivative = (func(point_plus) - func(point_minus)) / (2 * h)
        gradient.append(derivative)
    
    return gradient

def inner_product(u, v):
    """Inner product of two vectors"""
    return sum(u[i] * v[i] for i in range(len(u)))

def vector_subtract(u, v):
    """Subtract vectors: u - v"""
    return [u[i] - v[i] for i in range(len(u))]

def taylor_multivariate(func, point_a, point_x):
    """
    Multivariate Taylor approximation:
    f(x) ≈ f(a) + ∇f(a) · (x - a)
    
    Args:
        func: Function to approximate
        point_a: Expansion point
        point_x: Point to approximate at
    
    Returns:
        Approximation value
    """
    f_a = func(point_a)
    grad_a = compute_gradient(func, point_a)
    delta = vector_subtract(point_x, point_a)
    
    approx = f_a + inner_product(grad_a, delta)
    
    return approx, f_a, grad_a

# Example: Sales function
def sales_function(x):
    """
    Sales as function of (price, advertising, season)
    True function (unknown in practice): complex nonlinear
    """
    price, advertising, season = x
    # Simulated complex relationship
    return 20000 - 500*price + 2*advertising + 1000*season - 0.01*advertising**2

current_state = [10, 5000, 3]  # price=$10, ad=$5000, season=3
new_state = [11, 5200, 3]      # price increase, more advertising

print("Sales Prediction Example:")
print(f"Current state (price, ad, season): {current_state}")
print(f"Current sales: {sales_function(current_state):,.0f} units")
print()

approx, f_a, gradient = taylor_multivariate(sales_function, current_state, new_state)
actual = sales_function(new_state)

print(f"Gradient at current: {[f'{g:.1f}' for g in gradient]}")
print(f"  ∂sales/∂price ≈ {gradient[0]:.1f} (negative: price increase hurts)")
print(f"  ∂sales/∂ad ≈ {gradient[1]:.2f} (positive: advertising helps)")
print(f"  ∂sales/∂season ≈ {gradient[2]:.1f} (positive: better season helps)")
print()

print(f"New state: {new_state}")
print(f"Taylor approximation: {approx:,.0f} units")
print(f"Actual value: {actual:,.0f} units")
print(f"Error: {abs(approx - actual):.0f} units")
print()

# ----------------------------------------------------------------------------
# 1.5 Simple Linear Regression (One Feature)
# ----------------------------------------------------------------------------

print("1.5 SIMPLE LINEAR REGRESSION")
print("-" * 50)

def simple_linear_regression(x_data, y_data):
    """
    Fit simple linear regression: y = wx + b
    
    Using formulas:
    w = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
    b = ȳ - w*x̄
    
    Args:
        x_data: List of input values
        y_data: List of output values
    
    Returns:
        (w, b): slope and intercept
    """
    n = len(x_data)
    
    # Calculate means
    x_mean = sum(x_data) / n
    y_mean = sum(y_data) / n
    
    # Calculate w (slope)
    numerator = sum((x_data[i] - x_mean) * (y_data[i] - y_mean) 
                   for i in range(n))
    denominator = sum((x_data[i] - x_mean) ** 2 
                     for i in range(n))
    
    w = numerator / denominator
    
    # Calculate b (intercept)
    b = y_mean - w * x_mean
    
    return w, b

def predict(x, w, b):
    """Make prediction using linear model"""
    if isinstance(x, list):
        return [w * xi + b for xi in x]
    else:
        return w * x + b

# Example: House prices
sqft_data = [1500, 2000, 1200, 1800, 2500]
price_data = [300, 400, 250, 350, 480]

print("House Price Dataset:")
print("Square Feet:", sqft_data)
print("Prices ($1000s):", price_data)
print()

# Fit model
w, b = simple_linear_regression(sqft_data, price_data)

print(f"Fitted model: price = {w:.4f} * sqft + {b:.2f}")
print(f"Interpretation:")
print(f"  - Each square foot adds ${w*1000:.2f}")
print(f"  - Base price: ${b*1000:,.2f}")
print()

# Make predictions
predictions = predict(sqft_data, w, b)

print("Predictions on training data:")
print(f"{'Sqft':<10} {'True':<10} {'Predicted':<12} {'Error':<10}")
print("-" * 45)
for i in range(len(sqft_data)):
    error = price_data[i] - predictions[i]
    print(f"{sqft_data[i]:<10} {price_data[i]:<10} {predictions[i]:<12.2f} {error:<10.2f}")
print()

# Predict for new house
new_house_sqft = 2200
predicted_price = predict(new_house_sqft, w, b)
print(f"New house ({new_house_sqft} sqft): ${predicted_price:.2f}k = ${predicted_price*1000:,.2f}")
print()

# ----------------------------------------------------------------------------
# 1.6 Multiple Linear Regression
# ----------------------------------------------------------------------------

print("1.6 MULTIPLE LINEAR REGRESSION")
print("-" * 50)

def multiple_linear_regression(X, y):
    """
    Fit multiple linear regression using normal equations:
    w = (X^T X)^(-1) X^T y
    
    Args:
        X: List of feature vectors (each row is one example)
        y: List of target values
    
    Returns:
        weights: List of coefficients (including bias as last element)
    """
    # Add column of ones for bias term
    n = len(X)
    X_with_bias = [X[i] + [1] for i in range(n)]
    
    # X^T X
    d = len(X_with_bias[0])
    XTX = [[0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            for k in range(n):
                XTX[i][j] += X_with_bias[k][i] * X_with_bias[k][j]
    
    # X^T y
    XTy = [0] * d
    for i in range(d):
        for k in range(n):
            XTy[i] += X_with_bias[k][i] * y[k]
    
    # Solve (X^T X) w = X^T y using Gaussian elimination
    # This is simplified - in practice use more robust methods
    weights = gauss_solve(XTX, XTy)
    
    return weights

def gauss_solve(A, b):
    """
    Solve Ax = b using Gaussian elimination.
    Simplified implementation.
    """
    n = len(A)
    # Create augmented matrix
    M = [A[i] + [b[i]] for i in range(n)]
    
    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[max_row][i]):
                max_row = k
        M[i], M[max_row] = M[max_row], M[i]
        
        # Eliminate column
        for k in range(i + 1, n):
            if M[i][i] != 0:
                factor = M[k][i] / M[i][i]
                for j in range(i, n + 1):
                    M[k][j] -= factor * M[i][j]
    
    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        if M[i][i] != 0:
            x[i] /= M[i][i]
    
    return x

def predict_multiple(x, weights):
    """Predict using multiple linear regression"""
    # Last weight is bias
    prediction = weights[-1]  # bias
    for i in range(len(x)):
        prediction += weights[i] * x[i]
    return prediction

# Example: House prices with multiple features
print("Multiple Features Dataset:")
print("Features: [sqft, bedrooms, age]")

X_train = [
    [1500, 2, 20],
    [2000, 3, 15],
    [1200, 2, 30],
    [1800, 3, 10],
    [2500, 4, 5]
]

y_train = [300, 400, 250, 350, 480]

print("\nTraining data:")
for i, (features, price) in enumerate(zip(X_train, y_train)):
    print(f"  House {i+1}: {features} → ${price}k")
print()

# Fit model
weights = multiple_linear_regression(X_train, y_train)

print(f"Fitted weights: {[f'{w:.2f}' for w in weights]}")
print(f"Model: price = {weights[0]:.2f}*sqft + {weights[1]:.2f}*beds + {weights[2]:.2f}*age + {weights[3]:.2f}")
print()

print("Interpretation:")
print(f"  - Per sqft: ${weights[0]*1000:.2f}")
print(f"  - Per bedroom: ${weights[1]*1000:,.2f}")
print(f"  - Per year age: ${weights[2]*1000:.2f}")
print(f"  - Base price: ${weights[3]*1000:,.2f}")
print()

# Predictions
print("Predictions on training data:")
print(f"{'Features':<20} {'True':<10} {'Predicted':<12} {'Error':<10}")
print("-" * 55)
for i in range(len(X_train)):
    pred = predict_multiple(X_train[i], weights)
    error = y_train[i] - pred
    print(f"{str(X_train[i]):<20} {y_train[i]:<10} {pred:<12.2f} {error:<10.2f}")
print()

# New prediction
new_house = [2200, 3, 12]
pred_new = predict_multiple(new_house, weights)
print(f"New house {new_house}: ${pred_new:.2f}k = ${pred_new*1000:,.2f}")
print()

# ----------------------------------------------------------------------------
# 1.7 Gradient Descent for Linear Regression
# ----------------------------------------------------------------------------

print("1.7 GRADIENT DESCENT FOR LINEAR REGRESSION")
print("-" * 50)

def gradient_descent_regression(X, y, learning_rate=0.01, iterations=1000):
    """
    Fit linear regression using gradient descent.
    
    Loss: L(w) = (1/n) Σ (yi - w·xi)²
    Gradient: ∇L = -(2/n) Σ xi(yi - w·xi)
    
    Args:
        X: List of feature vectors
        y: List of targets
        learning_rate: Step size
        iterations: Number of iterations
    
    Returns:
        weights, loss_history
    """
    n = len(X)
    d = len(X[0]) + 1  # features + bias
    
    # Initialize weights randomly (small values)
    weights = [0.01] * d
    
    # Add bias feature (column of ones)
    X_with_bias = [X[i] + [1] for i in range(n)]
    
    loss_history = []
    
    for iteration in range(iterations):
        # Compute predictions
        predictions = [inner_product(weights, X_with_bias[i]) 
                      for i in range(n)]
        
        # Compute loss
        loss = sum((y[i] - predictions[i])**2 for i in range(n)) / n
        loss_history.append(loss)
        
        # Compute gradient
        gradient = [0] * d
        for j in range(d):
            for i in range(n):
                gradient[j] += -2 * X_with_bias[i][j] * (y[i] - predictions[i])
            gradient[j] /= n
        
        # Update weights
        for j in range(d):
            weights[j] -= learning_rate * gradient[j]
        
        # Print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Loss = {loss:.2f}")
    
    return weights, loss_history

# Train using gradient descent
print("Training with gradient descent...")
print(f"Learning rate: 0.0001, Iterations: 1000")
print()

# Use simpler data for faster convergence
X_simple = [[x] for x in sqft_data]  # Convert to list of lists
y_simple = price_data

weights_gd, loss_hist = gradient_descent_regression(
    X_simple, y_simple, 
    learning_rate=0.0001, 
    iterations=1000
)

print(f"\nFinal weights: {[f'{w:.4f}' for w in weights_gd]}")
print(f"Model: price = {weights_gd[0]:.4f} * sqft + {weights_gd[1]:.2f}")
print(f"Final loss: {loss_hist[-1]:.2f}")
print()

print("Compare with closed-form solution:")
print(f"  Gradient descent: w={weights_gd[0]:.4f}, b={weights_gd[1]:.2f}")
print(f"  Closed-form:     w={w:.4f}, b={b:.2f}")
print(f"  (Should be similar!)")
print()

# ----------------------------------------------------------------------------
# 1.8 Model Evaluation Metrics
# ----------------------------------------------------------------------------

print("1.8 MODEL EVALUATION METRICS")
print("-" * 50)

def mean_squared_error(y_true, y_pred):
    """Calculate MSE"""
    n = len(y_true)
    return sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n

def root_mean_squared_error(y_true, y_pred):
    """Calculate RMSE"""
    mse = mean_squared_error(y_true, y_pred)
    return mse ** 0.5

def mean_absolute_error(y_true, y_pred):
    """Calculate MAE"""
    n = len(y_true)
    return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n

def r_squared(y_true, y_pred):
    """Calculate R² score"""
    # Mean of true values
    y_mean = sum(y_true) / len(y_true)
    
    # Total sum of squares
    ss_tot = sum((y_true[i] - y_mean)**2 for i in range(len(y_true)))
    
    # Residual sum of squares
    ss_res = sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true)))
    
    # R²
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return r2

# Evaluate our model
predictions = predict(sqft_data, w, b)

mse = mean_squared_error(price_data, predictions)
rmse = root_mean_squared_error(price_data, predictions)
mae = mean_absolute_error(price_data, predictions)
r2 = r_squared(price_data, predictions)

print("Model Performance:")
print(f"MSE:  {mse:.2f} (thousand dollars squared)")
print(f"RMSE: {rmse:.2f} (thousand dollars)")
print(f"MAE:  {mae:.2f} (thousand dollars)")
print(f"R²:   {r2:.4f}")
print()

print("Interpretation:")
print(f"  - Average prediction error: ${rmse*1000:,.2f}")
print(f"  - Model explains {r2*100:.2f}% of variance")
print()

# ============================================================================
# PART 2: NUMPY AND SCIKIT-LEARN IMPLEMENTATION
# ============================================================================

print("="*70)
print("PART 2: NUMPY AND SCIKIT-LEARN IMPLEMENTATION")
print("="*70)
print()

try:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    print("Libraries imported successfully!")
    print(f"NumPy version: {np.__version__}")
    print()
    
    # ------------------------------------------------------------------------
    # 2.1 Linear Functions with NumPy
    # ------------------------------------------------------------------------
    
    print("2.1 LINEAR FUNCTIONS WITH NUMPY")
    print("-" * 50)
    
    # Coefficients and input
    coef_np = np.array([1, 2, 3])
    order_np = np.array([2, 1, 1])
    
    # Linear function (inner product)
    result = np.dot(coef_np, order_np)
    # Or: result = coef_np @ order_np
    
    print(f"Coefficients: {coef_np}")
    print(f"Input: {order_np}")
    print(f"f(x) = {result}")
    print()
    
    # Affine function
    bias = 5
    result_affine = np.dot(coef_np, order_np) + bias
    print(f"With bias {bias}: f(x) = {result_affine}")
    print()
    
    # ------------------------------------------------------------------------
    # 2.2 Gradient Computation
    # ------------------------------------------------------------------------
    
    print("2.2 NUMERICAL GRADIENT")
    print("-" * 50)
    
    def gradient_numerical(func, point, h=1e-5):
        """Compute gradient using NumPy"""
        point = np.array(point)
        grad = np.zeros_like(point, dtype=float)
        
        for i in range(len(point)):
            point_plus = point.copy()
            point_plus[i] += h
            
            point_minus = point.copy()
            point_minus[i] -= h
            
            grad[i] = (func(point_plus) - func(point_minus)) / (2 * h)
        
        return grad
    
    # Example function
    def f(x):
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1]
    
    point = np.array([1.0, 2.0])
    grad = gradient_numerical(f, point)
    
    print(f"Function: f(x1, x2) = x1² + 2x2² + x1*x2")
    print(f"Point: {point}")
    print(f"Gradient: {grad}")
    print(f"  ∂f/∂x1 = {grad[0]:.4f}")
    print(f"  ∂f/∂x2 = {grad[1]:.4f}")
    print()
    
    # ------------------------------------------------------------------------
    # 2.3 Simple Linear Regression with NumPy
    # ------------------------------------------------------------------------
    
    print("2.3 SIMPLE LINEAR REGRESSION WITH NUMPY")
    print("-" * 50)
    
    # Data
    x_np = np.array([1500, 2000, 1200, 1800, 2500])
    y_np = np.array([300, 400, 250, 350, 480])
    
    # Method 1: Manual calculation
    x_mean = np.mean(x_np)
    y_mean = np.mean(y_np)
    
    w_np = np.sum((x_np - x_mean) * (y_np - y_mean)) / np.sum((x_np - x_mean)**2)
    b_np = y_mean - w_np * x_mean
    
    print("Manual calculation:")
    print(f"  w = {w_np:.4f}")
    print(f"  b = {b_np:.2f}")
    print()
    
    # Method 2: Using polyfit (fits polynomial, degree 1 = line)
    coefficients = np.polyfit(x_np, y_np, deg=1)
    w_poly, b_poly = coefficients
    
    print("Using np.polyfit:")
    print(f"  w = {w_poly:.4f}")
    print(f"  b = {b_poly:.2f}")
    print()
    
    # Predictions
    y_pred_np = w_np * x_np + b_np
    
    # Metrics
    mse_np = np.mean((y_np - y_pred_np)**2)
    rmse_np = np.sqrt(mse_np)
    r2_np = r2_score(y_np, y_pred_np)
    
    print("Performance:")
    print(f"  MSE: {mse_np:.2f}")
    print(f"  RMSE: {rmse_np:.2f}")
    print(f"  R²: {r2_np:.4f}")
    print()
    
    # ------------------------------------------------------------------------
    # 2.4 Multiple Linear Regression with Scikit-learn
    # ------------------------------------------------------------------------
    
    print("2.4 MULTIPLE LINEAR REGRESSION WITH SCIKIT-LEARN")
    print("-" * 50)
    
    # Prepare data
    X_train_np = np.array([
        [1500, 2, 20],
        [2000, 3, 15],
        [1200, 2, 30],
        [1800, 3, 10],
        [2500, 4, 5]
    ])
    
    y_train_np = np.array([300, 400, 250, 350, 480])
    
    print("Training data shape:", X_train_np.shape)
    print("Features: [sqft, bedrooms, age]")
    print()
    
    # Create and train model
    model = LinearRegression()
    model.fit(X_train_np, y_train_np)
    
    # Extract parameters
    print("Fitted parameters:")
    print(f"  Coefficients: {model.coef_}")
    print(f"  Intercept: {model.intercept_:.2f}")
    print()
    
    print("Model interpretation:")
    feature_names = ['sqft', 'bedrooms', 'age']
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.2f}")
    print(f"  bias: {model.intercept_:.2f}")
    print()
    
    # Predictions
    y_pred_sklearn = model.predict(X_train_np)
    
    print("Predictions vs Actual:")
    print(f"{'Predicted':<12} {'Actual':<12} {'Error':<12}")
    print("-" * 36)
    for pred, actual in zip(y_pred_sklearn, y_train_np):
        error = actual - pred
        print(f"{pred:<12.2f} {actual:<12} {error:<12.2f}")
    print()
    
    # Evaluate
    mse_sklearn = mean_squared_error(y_train_np, y_pred_sklearn)
    rmse_sklearn = np.sqrt(mse_sklearn)
    mae_sklearn = mean_absolute_error(y_train_np, y_pred_sklearn)
    r2_sklearn = r2_score(y_train_np, y_pred_sklearn)
    
    print("Performance metrics:")
    print(f"  MSE:  {mse_sklearn:.2f}")
    print(f"  RMSE: {rmse_sklearn:.2f}")
    print(f"  MAE:  {mae_sklearn:.2f}")
    print(f"  R²:   {r2_sklearn:.4f}")
    print()
    
    # Predict new house
    new_house_np = np.array([[2200, 3, 12]])
    pred_new_sklearn = model.predict(new_house_np)
    print(f"New house [2200, 3, 12]: ${pred_new_sklearn[0]:.2f}k")
    print()
    
    # ------------------------------------------------------------------------
    # 2.5 Gradient Descent with NumPy
    # ------------------------------------------------------------------------
    
    print("2.5 GRADIENT DESCENT WITH NUMPY")
    print("-" * 50)
    
    def gradient_descent_numpy(X, y, learning_rate=0.01, iterations=1000, verbose=False):
        """Gradient descent using NumPy"""
        # Add bias column
        n, d = X.shape
        X_b = np.c_[X, np.ones(n)]  # Add column of ones
        
        # Initialize weights
        weights = np.random.randn(d + 1) * 0.01
        
        loss_history = []
        
        for i in range(iterations):
            # Predictions
            y_pred = X_b @ weights
            
            # Loss
            loss = np.mean((y - y_pred)**2)
            loss_history.append(loss)
            
            # Gradient
            gradient = -2 * X_b.T @ (y - y_pred) / n
            
            # Update
            weights -= learning_rate * gradient
            
            if verbose and i % 100 == 0:
                print(f"  Iteration {i}: Loss = {loss:.4f}")
        
        return weights, np.array(loss_history)
    
    # Reshape for sklearn format
    X_simple_np = x_np.reshape(-1, 1)
    
    print("Training with gradient descent...")
    weights_gd_np, loss_hist_np = gradient_descent_numpy(
        X_simple_np, y_np,
        learning_rate=0.0001,
        iterations=1000,
        verbose=True
    )
    
    print()
    print(f"Final weights: {weights_gd_np}")
    print(f"  w (slope): {weights_gd_np[0]:.4f}")
    print(f"  b (intercept): {weights_gd_np[1]:.2f}")
    print(f"Final loss: {loss_hist_np[-1]:.4f}")
    print()
    
    # Compare with sklearn
    model_sklearn = LinearRegression()
    model_sklearn.fit(X_simple_np, y_np)
    
    print("Comparison:")
    print(f"  Gradient descent: w={weights_gd_np[0]:.4f}, b={weights_gd_np[1]:.2f}")
    print(f"  Sklearn:         w={model_sklearn.coef_[0]:.4f}, b={model_sklearn.intercept_:.2f}")
    print()
    
    # ------------------------------------------------------------------------
    # 2.6 Polynomial Regression
    # ------------------------------------------------------------------------
    
    print("2.6 POLYNOMIAL REGRESSION")
    print("-" * 50)
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    
    # Generate nonlinear data
    np.random.seed(42)
    X_nonlinear = np.linspace(0, 3, 30).reshape(-1, 1)
    y_nonlinear = 2 + 3*X_nonlinear + 0.5*X_nonlinear**2 + np.random.randn(30, 1) * 0.5
    y_nonlinear = y_nonlinear.ravel()
    
    print("Generated nonlinear data (y = 2 + 3x + 0.5x²)")
    print()
    
    # Fit linear model
    model_linear = LinearRegression()
    model_linear.fit(X_nonlinear, y_nonlinear)
    y_pred_linear = model_linear.predict(X_nonlinear)
    r2_linear = r2_score(y_nonlinear, y_pred_linear)
    
    print(f"Linear model R²: {r2_linear:.4f}")
    
    # Fit polynomial model (degree 2)
    model_poly = make_pipeline(
        PolynomialFeatures(degree=2),
        LinearRegression()
    )
    model_poly.fit(X_nonlinear, y_nonlinear)
    y_pred_poly = model_poly.predict(X_nonlinear)
    r2_poly = r2_score(y_nonlinear, y_pred_poly)
    
    print(f"Polynomial model R²: {r2_poly:.4f}")
    print(f"Improvement: {(r2_poly - r2_linear)*100:.2f}%")
    print()
    
    print("Note: Polynomial regression is STILL linear regression!")
    print("We just created new features: x² as a feature")
    print("The model is linear in the PARAMETERS, not the input")
    print()
    
    # ------------------------------------------------------------------------
    # 2.7 Regularized Regression
    # ------------------------------------------------------------------------
    
    print("2.7 REGULARIZED REGRESSION (Ridge)")
    print("-" * 50)
    
    from sklearn.linear_model import Ridge, Lasso
    
    # Ridge regression (L2 regularization)
    model_ridge = Ridge(alpha=1.0)  # alpha = regularization strength
    model_ridge.fit(X_train_np, y_train_np)
    
    print("Ridge Regression (L2):")
    print(f"  Coefficients: {model_ridge.coef_}")
    print(f"  Intercept: {model_ridge.intercept_:.2f}")
    print()
    
    # Lasso regression (L1 regularization)
    model_lasso = Lasso(alpha=1.0)
    model_lasso.fit(X_train_np, y_train_np)
    
    print("Lasso Regression (L1):")
    print(f"  Coefficients: {model_lasso.coef_}")
    print(f"  Intercept: {model_lasso.intercept_:.2f}")
    print()
    
    print("Comparison:")
    print(f"{'Model':<15} {'sqft':<12} {'bedrooms':<12} {'age':<12}")
    print("-" * 51)
    print(f"{'Standard':<15} {model.coef_[0]:<12.2f} {model.coef_[1]:<12.2f} {model.coef_[2]:<12.2f}")
    print(f"{'Ridge':<15} {model_ridge.coef_[0]:<12.2f} {model_ridge.coef_[1]:<12.2f} {model_ridge.coef_[2]:<12.2f}")
    print(f"{'Lasso':<15} {model_lasso.coef_[0]:<12.2f} {model_lasso.coef_[1]:<12.2f} {model_lasso.coef_[2]:<12.2f}")
    print()
    
    print("Observation:")
    print("  - Ridge shrinks all coefficients")
    print("  - Lasso can shrink some coefficients to exactly zero")
    print("  - Both help prevent overfitting")
    print()
    
    # ------------------------------------------------------------------------
    # 2.8 Train-Test Split and Cross-Validation
    # ------------------------------------------------------------------------
    
    print("2.8 TRAIN-TEST SPLIT")
    print("-" * 50)
    
    from sklearn.model_selection import train_test_split, cross_val_score
    
    # Generate larger dataset
    np.random.seed(42)
    n_samples = 100
    X_large = np.random.rand(n_samples, 3) * 100  # 3 features
    true_weights = np.array([2, 5, -1])
    y_large = X_large @ true_weights + 50 + np.random.randn(n_samples) * 5
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_large, y_large, test_size=0.2, random_state=42
    )
    
    print(f"Total samples: {n_samples}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # Train model
    model_split = LinearRegression()
    model_split.fit(X_train, y_train)
    
    # Evaluate on both sets
    train_score = model_split.score(X_train, y_train)
    test_score = model_split.score(X_test, y_test)
    
    print("Model performance:")
    print(f"  Training R²: {train_score:.4f}")
    print(f"  Test R²: {test_score:.4f}")
    print(f"  Difference: {abs(train_score - test_score):.4f}")
    print()
    
    # Cross-validation
    cv_scores = cross_val_score(model_split, X_large, y_large, cv=5, 
                                 scoring='r2')
    
    print("5-Fold Cross-Validation:")
    print(f"  Scores: {cv_scores}")
    print(f"  Mean: {cv_scores.mean():.4f}")
    print(f"  Std: {cv_scores.std():.4f}")
    print()
    
    # ------------------------------------------------------------------------
    # 2.9 Complete Example: House Price Prediction Pipeline
    # ------------------------------------------------------------------------
    
    print("2.9 COMPLETE PIPELINE EXAMPLE")
    print("-" * 50)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Create larger house dataset
    np.random.seed(42)
    n_houses = 200
    
    sqft = np.random.uniform(1000, 3000, n_houses)
    bedrooms = np.random.randint(1, 6, n_houses)
    age = np.random.uniform(0, 50, n_houses)
    
    # True price model with some noise
    prices = (150 * sqft + 50000 * bedrooms - 1000 * age + 100000 + 
              np.random.randn(n_houses) * 20000)
    
    X_houses = np.column_stack([sqft, bedrooms, age])
    y_prices = prices
    
    # Split data
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_houses, y_prices, test_size=0.2, random_state=42
    )
    
    print(f"Dataset: {n_houses} houses")
    print(f"Features: square feet, bedrooms, age")
    print(f"Training: {len(X_train_h)}, Test: {len(X_test_h)}")
    print()
    
    # Create pipeline: standardize then linear regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    # Train
    pipeline.fit(X_train_h, y_train_h)
    
    # Predictions
    y_train_pred = pipeline.predict(X_train_h)
    y_test_pred = pipeline.predict(X_test_h)
    
    # Evaluate
    train_r2 = r2_score(y_train_h, y_train_pred)
    test_r2 = r2_score(y_test_h, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_h, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_h, y_test_pred))
    
    print("Pipeline Performance:")
    print(f"  Training R²: {train_r2:.4f}, RMSE: ${train_rmse:,.2f}")
    print(f"  Test R²: {test_r2:.4f}, RMSE: ${test_rmse:,.2f}")
    print()
    
    # Feature importance (from standardized coefficients)
    feature_names = ['sqft', 'bedrooms', 'age']
    coefficients = pipeline.named_steps['regressor'].coef_
    
    print("Feature importance (standardized coefficients):")
    for name, coef in zip(feature_names, coefficients):
        print(f"  {name}: {coef:,.2f}")
    print()
    
    # Predict new house
    new_houses = np.array([
        [2000, 3, 10],
        [1500, 2, 20],
        [2800, 4, 5]
    ])
    
    predictions = pipeline.predict(new_houses)
    
    print("Predictions for new houses:")
    for features, price in zip(new_houses, predictions):
        print(f"  {features} → ${price:,.2f}")
    print()
    
    # ------------------------------------------------------------------------
    # 2.10 Performance Comparison
    # ------------------------------------------------------------------------
    
    print("2.10 PERFORMANCE COMPARISON")
    print("-" * 50)
    
    # Generate large dataset
    n_large = 10000
    X_perf = np.random.randn(n_large, 10)
    y_perf = X_perf @ np.random.randn(10) + np.random.randn(n_large) * 0.1
    
    print(f"Dataset size: {n_large} samples, 10 features")
    print()
    
    # Time basic Python implementation
    X_perf_list = X_perf.tolist()
    y_perf_list = y_perf.tolist()
    
    print("Timing basic Python implementation...")
    start = time.time()
    # This would be very slow, so we skip it
    # weights_basic = multiple_linear_regression(X_perf_list, y_perf_list)
    time_basic = "Too slow (>10 seconds)"
    print(f"  Time: {time_basic}")
    print()
    
    # Time NumPy implementation
    print("Timing NumPy implementation...")
    start = time.time()
    X_b = np.c_[X_perf, np.ones(n_large)]
    weights_numpy = np.linalg.lstsq(X_b, y_perf, rcond=None)[0]
    time_numpy = time.time() - start
    print(f"  Time: {time_numpy:.4f} seconds")
    print()
    
    # Time Sklearn implementation
    print("Timing Scikit-learn implementation...")
    start = time.time()
    model_sklearn_perf = LinearRegression()
    model_sklearn_perf.fit(X_perf, y_perf)
    time_sklearn = time.time() - start
    print(f"  Time: {time_sklearn:.4f} seconds")
    print()
    
    print("Performance summary:")
    print("  - Basic Python: Very slow (not practical for real data)")
    print(f"  - NumPy: {time_numpy:.4f}s (fast!)")
    print(f"  - Scikit-learn: {time_sklearn:.4f}s (highly optimized!)")
    print()
    
    # ------------------------------------------------------------------------
    # 2.11 Visualization Helper
    # ------------------------------------------------------------------------
    
    print("2.11 RESIDUAL ANALYSIS")
    print("-" * 50)
    
    # Calculate residuals
    residuals = y_test_h - y_test_pred
    
    print("Residual statistics:")
    print(f"  Mean: {np.mean(residuals):.2f} (should be ~0)")
    print(f"  Std: {np.std(residuals):,.2f}")
    print(f"  Min: {np.min(residuals):,.2f}")
    print(f"  Max: {np.max(residuals):,.2f}")
    print()
    
    # Check for patterns
    print("Residual analysis:")
    print("  A good model has residuals that are:")
    print("  1. Centered around zero (mean ≈ 0)")
    print("  2. Randomly distributed (no patterns)")
    print("  3. Constant variance (homoscedasticity)")
    print()
    
    # Identify outliers (residuals > 2 std)
    outlier_threshold = 2 * np.std(residuals)
    outliers = np.abs(residuals) > outlier_threshold
    n_outliers = np.sum(outliers)
    
    print(f"Outliers (|residual| > 2σ): {n_outliers} ({n_outliers/len(residuals)*100:.1f}%)")
    if n_outliers > 0:
        print(f"  Expected ~5% for normal distribution")
    print()

except ImportError as e:
    print(f"Required library not installed: {e}")
    print("\nTo install required libraries:")
    print("  pip install numpy scikit-learn")
    print()

# ============================================================================
# SUMMARY AND COMPARISON
# ============================================================================

print("="*70)
print("SUMMARY AND COMPARISON")
print("="*70)
print()

print("IMPLEMENTATION COMPARISON")
print("-" * 50)
print()

print("Basic Python:")
print("✓ Full understanding and control")
print("✓ No external dependencies")
print("✓ Educational value")
print("✗ Slow for large datasets")
print("✗ More code to write")
print("✗ Prone to numerical errors")
print()

print("NumPy:")
print("✓ Much faster (10-100x)")
print("✓ Vectorized operations")
print("✓ Industry standard")
print("✓ Numerical stability")
print("✗ Requires library")
print("✓ Cleaner, more readable code")
print()

print("Scikit-learn:")
print("✓ Highest level abstraction")
print("✓ Highly optimized")
print("✓ Consistent API")
print("✓ Many built-in features")
print("✓ Production ready")
print("✗ May hide details")
print("✓ Best for real ML applications")
print()

print("="*70)
print("KEY CONCEPTS SUMMARY")
print("="*70)
print()

print("1. LINEAR FUNCTIONS")
print("   - Pure linear: f(x) = w·x")
print("   - Affine: f(x) = w·x + b")
print("   - Properties: homogeneity + additivity")
print("   - ML use: predictions, decision boundaries")
print()

print("2. TAYLOR APPROXIMATION")
print("   - Linearizes complex functions locally")
print("   - f(x) ≈ f(a) + ∇f(a)·(x-a)")
print("   - Gradient points in steepest direction")
print("   - ML use: gradient descent, sensitivity analysis")
print()

print("3. LINEAR REGRESSION")
print("   - Model: ŷ = w·x + b")
print("   - Loss: MSE = mean of (y - ŷ)²")
print("   - Solution: w = (X^T X)^(-1) X^T y")
print("   - Or: gradient descent for large data")
print()

print("4. EVALUATION METRICS")
print("   - MSE: Mean squared error")
print("   - RMSE: Root MSE (same units as target)")
print("   - MAE: Mean absolute error (robust)")
print("   - R²: Proportion of variance explained")
print()

print("="*70)
print("OPERATION COMPLEXITY")
print("="*70)
print()

print("For n samples, d features:")
print("-" * 50)
print(f"{'Operation':<30} {'Complexity':<15}")
print("-" * 50)
print(f"{'Single prediction':<30} {'O(d)':<15}")
print(f"{'All predictions':<30} {'O(nd)':<15}")
print(f"{'Compute gradient':<30} {'O(nd)':<15}")
print(f"{'Normal equations':<30} {'O(nd² + d³)':<15}")
print(f"{'Gradient descent (T steps)':<30} {'O(Tnd)':<15}")
print()

print("Note: For large n and small d, gradient descent is faster!")
print("      For small n, normal equations are faster!")
print()

print("="*70)
print("PRACTICAL TIPS")
print("="*70)
print()

print("1. ALWAYS split data into train/test sets")
print("2. Standardize features (especially for gradient descent)")
print("3. Check for multicollinearity (correlated features)")
print("4. Visualize residuals to check assumptions")
print("5. Start simple (linear) before going complex")
print("6. Use cross-validation for model selection")
print("7. Consider regularization (Ridge/Lasso) for many features")
print("8. Monitor both training and test performance")
print()

print("="*70)
print("NEXT STEPS")
print("="*70)
print()

print("After mastering this chapter, you can:")
print("✓ Understand how linear models work")
print("✓ Implement regression from scratch")
print("✓ Use NumPy for efficient computation")
print("✓ Apply Scikit-learn for production")
print("✓ Evaluate models properly")
print("✓ Debug common issues")
print()

print("Next: Chapter 3 - Norm, Distance, Standard Deviation, Angles")
print("       Learn how to measure similarity and structure in data!")
print()

print("="*70)
print("END OF CHAPTER 2 IMPLEMENTATION")
print("="*70)
