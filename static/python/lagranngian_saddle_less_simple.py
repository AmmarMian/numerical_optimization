import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up the ENHANCED problem: minimize f(x) = -(x-3)² subject to x ≥ 1
# This creates compelling conflict: objective wants x → -∞, constraint forces x ≥ 1
# Results in active constraint at x* = 1 with meaningful Lagrange multiplier λ* = 4


def objective_function(x):
    """Objective function f(x) = -(x-3)²

    To minimize this, we want -(x-3)² as small as possible.
    Since -(x-3)² ≤ 0 always, we want (x-3)² as large as possible.
    This drives x away from 3, ideally toward x → -∞.
    But constraint x ≥ 1 prevents this, creating active constraint at x* = 1.
    """
    return -((x - 3) ** 2)


def lagrangian(x, lam):
    """Lagrangian L(x,λ) = -(x-3)² - λ(x-1)

    Key insight: This creates a true saddle point because:
    - Unconstrained min would be x → -∞ (f(x) → -∞)
    - Constraint x ≥ 1 forces solution to boundary x* = 1
    - Stationarity gives: ∂L/∂x = -2(x-3) - λ = 0
    - At x* = 1: -2(1-3) - λ = 0 ⟹ 4 - λ = 0 ⟹ λ* = 4
    """
    return -((x - 3) ** 2) - lam * (x - 1)


# Create the figure with subplots
fig = plt.figure(figsize=(16, 6))

# ============================================================================
# LEFT PLOT: 2D visualization showing the rich constraint conflict
# ============================================================================
ax1 = fig.add_subplot(121)

# Generate x values for plotting the objective function
x_vals = np.linspace(-1, 5, 400)
f_vals = objective_function(x_vals)

# Plot the objective function - an inverted parabola
ax1.plot(x_vals, f_vals, "b-", linewidth=3, label="$f(x) = -(x-3)^2$")

# Mark the unconstrained "maximum" at x = 3 (but we're minimizing!)
ax1.plot(3, 0, "go", markersize=12, label="Unconstrained max of $-(x-3)^2$: $(3, 0)$")

# Mark the constrained minimum at the boundary
ax1.plot(1, -4, "ro", markersize=12, label="Constrained minimum: $(1, -4)$")

# Draw the constraint boundary
ax1.axvline(
    x=1,
    color="red",
    linestyle="--",
    linewidth=3,
    alpha=0.8,
    label="Constraint boundary: $x = 1$",
)

# Shade the feasible region
ax1.axvspan(1, 5, alpha=0.3, color="lightgreen", label="Feasible region: $x \\geq 1$")

# Shade the infeasible region
ax1.axvspan(-1, 1, alpha=0.3, color="lightcoral", label="Infeasible region: $x < 1$")

# Add arrows showing the objective function's preference
ax1.arrow(
    -0.5,
    -6.25,
    -0.3,
    -1,
    head_width=0.15,
    head_length=0.3,
    fc="blue",
    ec="blue",
    linewidth=2,
)
ax1.text(
    -0.8,
    -5,
    "Objective wants\n$x \\to -\\infty$\n(to minimize $-(x-3)^2$)",
    fontsize=11,
    ha="center",
    color="blue",
    weight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
)

# Annotate the conflict zone
ax1.annotate(
    "MATHEMATICAL CONFLICT:\nObjective pulls left, constraint blocks!",
    xy=(0.5, -6.25),
    xytext=(2, -8),
    arrowprops=dict(arrowstyle="->", color="red", lw=2),
    fontsize=12,
    ha="center",
    weight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
)

# Annotate the forced solution
ax1.annotate(
    "Constraint forces\noptimum here\n$f(1) = -4$",
    xy=(1, -4),
    xytext=(3.5, -2),
    arrowprops=dict(arrowstyle="->", color="red", lw=2),
    fontsize=11,
    ha="center",
    weight="bold",
)

# Show what unconstrained solution would be
ax1.text(
    3,
    2,
    'Unconstrained "max"\nbut we\'re minimizing!\nSo want to go opposite direction',
    fontsize=10,
    ha="center",
    color="green",
    weight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
)

ax1.set_xlabel("$x$ (decision variable)", fontsize=13)
ax1.set_ylabel("$f(x) = -(x-3)^2$ (objective value)", fontsize=13)
ax1.set_title(
    "Rich Constraint Conflict in Optimization\n$\\min f(x) = -(x-3)^2$ subject to $x \\geq 1$",
    fontsize=14,
    weight="bold",
)
ax1.legend(fontsize=10, loc="lower right")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, 5)
ax1.set_ylim(-10, 1)

# ============================================================================
# RIGHT PLOT: 3D visualization of the sophisticated Lagrangian saddle point
# ============================================================================
ax2 = fig.add_subplot(122, projection="3d")

# Create meshgrid for the 3D surface
# Focus around the constrained optimum where rich behavior occurs
x_3d = np.linspace(0, 3, 60)  # Primal variable range
lambda_3d = np.linspace(0, 8, 60)  # Dual variable range (λ ≥ 0)
X, LAM = np.meshgrid(x_3d, lambda_3d)

# Calculate Lagrangian values: L(x,λ) = -(x-3)² - λ(x-1)
L_vals = -((X - 3) ** 2) - LAM * (X - 1)

# Plot the surface with colormap emphasizing the saddle structure
surface = ax2.plot_surface(
    X, LAM, L_vals, cmap="plasma", alpha=0.8, linewidth=0, antialiased=True
)

# Mark the saddle point: (x*, λ*) = (1, 4)
# At optimum: L(1,4) = -(1-3)² - 4(1-1) = -4 - 0 = -4
saddle_x, saddle_lambda = 1, 4
saddle_L = lagrangian(saddle_x, saddle_lambda)
ax2.scatter(
    [saddle_x],
    [saddle_lambda],
    [saddle_L],
    color="red",
    s=200,
    label=f"Saddle point: $(1, 4, {saddle_L})$",
)

# Show the minimization path over x (holding λ = 4 fixed)
# When λ = 4: L(x,4) = -(x-3)² - 4(x-1) = -(x-3)² - 4x + 4
x_min_path = np.linspace(0, 3, 100)
lambda_fixed = 4
L_min_path = lagrangian(x_min_path, lambda_fixed)
ax2.plot(
    x_min_path,
    lambda_fixed * np.ones_like(x_min_path),
    L_min_path,
    "b-",
    linewidth=6,
    label="Minimize over $x$ (λ=4)",
)

# Show the maximization path over λ (holding x = 1 fixed)
# When x = 1: L(1,λ) = -(1-3)² - λ(1-1) = -4 - 0 = -4 (constant!)
x_fixed = 1
lambda_max_path = np.linspace(0, 8, 100)
L_max_path = lagrangian(x_fixed, lambda_max_path)
ax2.plot(
    x_fixed * np.ones_like(lambda_max_path),
    lambda_max_path,
    L_max_path,
    "orange",
    linewidth=6,
    label="Maximize over λ (x=1): flat!",
)

# Add arrows showing optimization directions
# Show descent direction in x when away from optimum
x_arrow, lambda_arrow = 1.8, 4
L_arrow = lagrangian(x_arrow, lambda_arrow)
ax2.quiver(
    x_arrow,
    lambda_arrow,
    L_arrow,
    -0.4,
    0,
    1.2,  # Direction toward minimum in x
    color="blue",
    arrow_length_ratio=0.1,
    linewidth=4,
)

# Show ascent direction in λ when away from optimum
x_arrow2, lambda_arrow2 = 1, 2
L_arrow2 = lagrangian(x_arrow2, lambda_arrow2)
ax2.quiver(
    x_arrow2,
    lambda_arrow2,
    L_arrow2,
    0,
    1,
    0,  # Direction toward maximum in λ (though flat here)
    color="orange",
    arrow_length_ratio=0.1,
    linewidth=4,
)

# Set labels and title
ax2.set_xlabel("$x$ (primal variable)", fontsize=12)
ax2.set_ylabel("$\\lambda$ (dual variable)", fontsize=12)
ax2.set_zlabel("$\\mathcal{L}(x,\\lambda)$", fontsize=12)
ax2.set_title(
    "Sophisticated Lagrangian Saddle Point\n$\\mathcal{L}(x,\\lambda) = -(x-3)^2 - \\lambda(x-1)$",
    fontsize=14,
    weight="bold",
)

# Optimize viewing angle to show the curvature
ax2.view_init(elev=25, azim=35)

# Add legend
ax2.legend(loc="upper left", fontsize=10)

# Add colorbar
fig.colorbar(surface, ax=ax2, shrink=0.6, aspect=20, label="$\\mathcal{L}(x,\\lambda)$")

plt.tight_layout()
plt.show()

# ============================================================================
# Cross-sections revealing the mathematical beauty of the saddle structure
# ============================================================================
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Show how L varies with x for different values of λ
x_cross = np.linspace(0, 3, 200)

# Plot several λ values to show how the optimization landscape changes
lambda_values = [2, 4, 6]  # Include the optimal λ* = 4
colors = ["green", "red", "purple"]
for lam_val, color in zip(lambda_values, colors):
    L_x_cross = lagrangian(x_cross, lam_val)
    style = "-" if lam_val == 4 else "--"
    width = 3 if lam_val == 4 else 2
    ax3.plot(
        x_cross,
        L_x_cross,
        color=color,
        linewidth=width,
        linestyle=style,
        label=f"$\\mathcal{{L}}(x, \\lambda={lam_val})$",
    )

    # Mark the minimum of each curve
    min_idx = np.argmin(L_x_cross)
    ax3.plot(x_cross[min_idx], L_x_cross[min_idx], "o", color=color, markersize=8)

# Highlight the constraint boundary and optimal solution
ax3.axvline(x=1, color="red", linestyle=":", alpha=0.8, linewidth=2)
ax3.plot(1, lagrangian(1, 4), "ro", markersize=12, label="Constrained optimum")
ax3.set_xlabel("$x$", fontsize=12)
ax3.set_ylabel("$\\mathcal{L}(x, \\lambda)$", fontsize=12)
ax3.set_title(
    "Cross-sections: Minimization over $x$\nOnly λ=4 makes gradient zero at x=1",
    fontsize=12,
)
ax3.grid(True, alpha=0.3)
ax3.legend()

# Right: Show how L varies with λ for different values of x
lambda_cross = np.linspace(0, 8, 200)

# Plot several x values to show the dual landscape
x_values = [0.5, 1.0, 1.5]  # Include the optimal x* = 1
colors = ["green", "red", "purple"]
for x_val, color in zip(x_values, colors):
    L_lambda_cross = lagrangian(x_val, lambda_cross)
    style = "-" if x_val == 1.0 else "--"
    width = 3 if x_val == 1.0 else 2
    ax4.plot(
        lambda_cross,
        L_lambda_cross,
        color=color,
        linewidth=width,
        linestyle=style,
        label=f"$\\mathcal{{L}}(x={x_val}, \\lambda)$",
    )

    # Mark the point where λ = 4
    ax4.plot(4, lagrangian(x_val, 4), "o", color=color, markersize=8)

ax4.axvline(x=4, color="red", linestyle=":", alpha=0.8, linewidth=2)
ax4.plot(4, lagrangian(1, 4), "ro", markersize=12, label="Constrained optimum")
ax4.set_xlabel("$\\lambda$", fontsize=12)
ax4.set_ylabel("$\\mathcal{L}(x, \\lambda)$", fontsize=12)
ax4.set_title(
    "Cross-sections: Maximization over $λ$\nNote: x=1 gives flat line (constraint satisfied)",
    fontsize=12,
)
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.show()

# ============================================================================
# Educational summary with mathematical insights
# ============================================================================
print("=" * 75)
print("DEEP UNDERSTANDING: THE SOPHISTICATED SADDLE POINT")
print("=" * 75)
print("MATHEMATICAL FOUNDATION:")
print("   • Objective: minimize f(x) = -(x-3)² ")
print("   • Constraint: x ≥ 1")
print("   • Lagrangian: L(x,λ) = -(x-3)² - λ(x-1)")
print()
print("THE FUNDAMENTAL CONFLICT:")
print("   • To minimize -(x-3)², we need (x-3)² as large as possible")
print("   • This drives x away from 3, ideally toward x → -∞")
print("   • Constraint x ≥ 1 blocks this, forcing x* = 1")
print("   • At x* = 1: f(1) = -(1-3)² = -4")
print()
print("SADDLE POINT ANALYSIS:")
print("   • Stationarity: ∂L/∂x = -2(x-3) - λ = 0")
print("   • At x* = 1: -2(1-3) - λ = 0 ⟹ 4 - λ = 0 ⟹ λ* = 4")
print("   • Optimal point: (x*, λ*) = (1, 4) with L(1,4) = -4")
print()
print("WHY IT'S A SADDLE:")
print("   • Fix λ = 4: L(x,4) = -(x-3)² - 4(x-1) has unique minimum at x = 1")
print("   • Fix x = 1: L(1,λ) = -4 (constant in λ)")
print("   • Surface curves down in x-direction, flat in λ-direction at optimum")
print()
print("ECONOMIC INTERPRETATION:")
print("   • λ* = 4 is the shadow price of constraint x ≥ 1")
print("   • Relaxing to x ≥ 1-ε would improve objective by ≈ 4ε")
print("   • This measures how much the constraint 'costs' us")
print()
print("ALGORITHMIC INSIGHT:")
print("   • Projected gradient method alternates:")
print("     x^(k+1) = x^k - α∇_x L  (minimize over x)")
print("     λ^(k+1) = max(0, λ^k + β∇_λ L)  (maximize over λ)")
print("   • The saddle structure guarantees convergence")
print("=" * 75)
