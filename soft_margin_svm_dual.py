import numpy as np
import cvxopt
import matplotlib.pyplot as plt

# Generate toy data for binary classification
np.random.seed(1)
X_pos = np.random.randn(20, 2) + 2  # Positive class centered at (2,2)
X_neg = np.random.randn(20, 2) - 2  # Negative class centered at (-2,-2)
X = np.vstack((X_pos, X_neg))  # Combine data
y = np.hstack((np.ones(20), -np.ones(20)))  # Labels (+1 and -1)

# Kernel function: Linear Kernel
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

# Prepare matrices for cvxopt solver
def solve_soft_svm(X, y, C):
    n_samples, n_features = X.shape
    K = np.array([[linear_kernel(X[i], X[j]) for j in range(n_samples)] for i in range(n_samples)])
    
    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-np.ones(n_samples))
    G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
    h = cvxopt.matrix(np.hstack((np.zeros(n_samples), C * np.ones(n_samples))))
    A = cvxopt.matrix(y, (1, n_samples), 'd')
    b = cvxopt.matrix(0.0)

    # Solve QP problem
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.ravel(solution['x'])

    # Get support vectors
    sv = alphas > 1e-5
    ind = np.arange(len(alphas))[sv]
    alphas_sv = alphas[sv]
    X_sv = X[sv]
    y_sv = y[sv]

    # Calculate weights
    w = np.sum(alphas_sv[:, None] * y_sv[:, None] * X_sv, axis=0)

    # Calculate bias
    b = np.mean([y_sv[i] - np.dot(w, X_sv[i]) for i in range(len(alphas_sv))])
    
    return w, b, X_sv, y_sv

# Solve Soft-SVM
C = 1.0  # Regularization parameter
w, b, X_sv, y_sv = solve_soft_svm(X, y, C)

# Plot decision boundary
def plot_decision_boundary(X, y, w, b, X_sv):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7, edgecolors='k')
    plt.scatter(X_sv[:, 0], X_sv[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    # Plot decision boundary
    x_min, x_max = plt.xlim(-5, 5)
    y_min, y_max = plt.ylim(-5, 5)
    xx = np.linspace(x_min, x_max, 50)
    yy = -(w[0] * xx + b) / w[1]
    plt.plot(xx, yy, 'k-', label='Decision Boundary')

    # Plot margins
    margin = 1 / np.linalg.norm(w)
    yy_up = yy + np.sqrt(1 + (w[0] / w[1]) ** 2) * margin
    yy_down = yy - np.sqrt(1 + (w[0] / w[1]) ** 2) * margin
    plt.plot(xx, yy_up, 'k--', label='Margin')
    plt.plot(xx, yy_down, 'k--')

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Soft-Margin SVM (Dual Form)")
    plt.legend()
    plt.show()

# Visualize the result
plot_decision_boundary(X, y, w, b, X_sv)
