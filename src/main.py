import numpy as np
import matplotlib.pyplot as plt
import os
from decision_tree import DecisionTreeRegressor
from linear_regressor import LinearRegressor
import sys


# Add the current directory to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Load inputs and labels
X_train = np.arange(14).reshape(-1, 1)
y_train = np.array([1, 1.2, 1.4, 1.1, 1, 5.5, 6.1, 6.7, 6.4, 6, 6, 3, 3.2, 3.1])

# Plot training data
fig, ax = plt.subplots()
ax.plot(X_train, y_train, "x", label="y")
ax.set_xlabel("Input x")
ax.set_ylabel("Output y")
ax.grid()
ax.legend()
ax.set_title("Training data")
plt.savefig('figures/training_data.png')

# Train decision tree regressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)

# Generate a grid of values over the feature space for plotting the true model
X_eval = np.linspace(0, 15, 50).reshape(-1, 1)
y_hat_eval = tree_reg.predict(X_eval)

fig, ax = plt.subplots()
ax.plot(X_train, y_train, "x", label="y")
ax.plot(X_eval, y_hat_eval, label=r"$\hat{y}_{tree}$")
ax.set_xlabel("input x")
ax.set_ylabel("output y")
ax.grid()
ax.legend()
plt.savefig('figures/tree_regression.png')

# Visualize the tree
#print("Decision Tree Structure:")
#visualize_tree(tree_reg.tree)

# Train linear regression model
lin_reg = LinearRegressor("quad")
lin_reg.fit(X_train, y_train)
y_hat_lin_eval = lin_reg.predict(X_eval)

fig, ax = plt.subplots()
ax.plot(X_train, y_train, "x", label="y")
ax.plot(X_eval, y_hat_eval, label=r"$\hat{y}_{tree}$")
ax.plot(X_eval, y_hat_lin_eval, label=r"$\hat{y}_{reg}$")
ax.set_xlabel("input x")
ax.set_ylabel("output y")
ax.grid()
ax.legend()
plt.savefig('figures/linear_regression.png')

# Generate a synthetic dataset based on a parabola
n_samples = 100
X_train = np.random.rand(n_samples, 2) * 2 - 1  # Two features in range [-1, 1]
y = X_train[:, 0] ** 2 + X_train[:, 1] ** 2  # True model without noise

# Add noise to create the observed target variable
y_train = y + np.random.randn(n_samples) * 0.1

# Train a decision tree regressor
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)

# Train a linear regression model with quadratic features
lin_reg = LinearRegressor("quad")
lin_reg.fit(X_train, y_train)

# Generate a grid of values over the feature space for plotting the true model
x1_grid, x2_grid = np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)
x1, x2 = np.meshgrid(x1_grid, x2_grid)
X_test = np.c_[x1.ravel(), x2.ravel()]
y_test = x1**2 + x2**2

# Predict the target values for the grid using the trained regressors
y_hat_tree = tree.predict(X_test).reshape(x1.shape)
y_hat_lin_reg = lin_reg.predict(X_test).reshape(x1.shape)

fig = plt.figure(figsize=(10, 20))

# Plot the true model surface
ax = fig.add_subplot(311, projection="3d")
ax.plot_surface(x1, x2, y_test, color="blue", alpha=0.5)
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color="blue", label="Training data")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Target")
ax.set_title("True Model Surface")
ax.legend()
plt.savefig('figures/true_model_surface.png')

# Plot the decision tree prediction surface
ax = fig.add_subplot(312, projection="3d")
ax.plot_surface(x1, x2, y_hat_tree, color="red", alpha=0.5)
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color="blue", label="Training data")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Target")
ax.set_title("Decision Tree Prediction Surface")
ax.legend()
plt.savefig('figures/decision_tree_surface.png')

# Plot the linear regression with quadratic features prediction surface
ax = fig.add_subplot(313, projection="3d")
ax.plot_surface(x1, x2, y_hat_lin_reg, color="yellow", alpha=0.5)
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color="blue", label="Training data")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Target")
ax.set_title("Linear Regression with Quadratic Features Prediction Surface")
ax.legend()
plt.savefig('figures/linear_regression_surface.png')

plt.show()
