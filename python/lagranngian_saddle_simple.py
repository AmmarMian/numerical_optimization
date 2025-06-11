import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up the CORRECTED problem: minimize f(x) = x subject to x ≥ 1
# This creates genuine conflict: objective wants x → -∞, constraint forces x ≥ 1
# Therefore: constraint is ACTIVE at x* = 1, creating meaningful saddle point structure


def objective_function(x):
    """Original objective function f(x) = x

    This function wants to minimize x, so it naturally wants x → -∞
    But the constraint x ≥ 1 prevents this, forcing x* = 1
    """
    return x


def lagrangian(x, lam):
    """Lagrangian L(x,λ) = x - λ(x-1) = x(1-λ) + λ

    Key insight: This formulation creates a true saddle point because:
    - Unconstrained min of f(x) = x would be x → -∞
    - Constraint x ≥ 1 forces the solution to the boundary x* = 1
    - This conflict generates λ* = 1 from stationarity condition
    """
    return x - lam * (x - 1)  # L(x,λ) = x - λ(x-1) = x(1-λ) + λ


# Create the figure with subplots
fig = plt.figure(figsize=(16, 6))

# ============================================================================
# LEFT PLOT: 2D visualization showing the REAL constraint conflict
# ============================================================================
ax1 = fig.add_subplot(121)

# Generate x values for plotting the objective function
x_vals = np.linspace(-1, 4, 300)
f_vals = objective_function(x_vals)

# Plot the objective function - a simple line with slope 1
ax1.plot(x_vals, f_vals, "b-", linewidth=3, label="$f(x) = x$ (wants to go left!)")

# Mark where the unconstrained minimum would try to go
ax1.arrow(
    -0.5,
    -0.5,
    -0.3,
    -0.3,
    head_width=0.1,
    head_length=0.1,
    fc="green",
    ec="green",
    linewidth=2,
)
ax1.text(
    -0.8,
    -0.2,
    "Unconstrained minimum\nwants $x \\to -\\infty$",
    fontsize=10,
    ha="center",
    color="green",
    weight="bold",
)

# Mark the constrained minimum (where the constraint becomes active)
ax1.plot(1, 1, "ro", markersize=12, label="Constrained optimum: $(1, 1)$")

# Draw the constraint boundary as a vertical line
ax1.axvline(
    x=1,
    color="red",
    linestyle="--",
    linewidth=3,
    alpha=0.8,
    label="Constraint boundary: $x = 1$",
)

# Shade the feasible region (everything to the right of x = 1)
ax1.axvspan(1, 4, alpha=0.3, color="lightgreen", label="Feasible region: $x \\geq 1$")

# Shade the infeasible region (everything to the left of x = 1)
ax1.axvspan(-1, 1, alpha=0.3, color="lightcoral", label="Infeasible region: $x < 1$")

# Add clear annotations showing the conflict
ax1.annotate(
    "CONFLICT ZONE:\nObjective wants to go here\nbut constraint forbids it!",
    xy=(0, 0),
    xytext=(-0.5, 2),
    arrowprops=dict(arrowstyle="->", color="red", lw=2),
    fontsize=11,
    ha="center",
    weight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
)

ax1.annotate(
    "Constraint forces\noptimum here",
    xy=(1, 1),
    xytext=(2.5, 1.5),
    arrowprops=dict(arrowstyle="->", color="red", lw=2),
    fontsize=11,
    ha="center",
    weight="bold",
)

ax1.set_xlabel("$x$ (decision variable)", fontsize=13)
ax1.set_ylabel("$f(x) = x$ (objective value)", fontsize=13)
ax1.set_title(
    "The Constraint Conflict Creates Optimization Challenge\n$\\min f(x) = x$ subject to $x \\geq 1$",
    fontsize=14,
    weight="bold",
)
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, 4)
ax1.set_ylim(-1, 4)

# ============================================================================
# RIGHT PLOT: 3D visualization of the Lagrangian saddle point
# ============================================================================
ax2 = fig.add_subplot(122, projection="3d")

# Create meshgrid for the 3D surface
# Focus around the constrained optimum where interesting behavior occurs
x_3d = np.linspace(-0.5, 3, 60)  # Primal variable range
lambda_3d = np.linspace(0, 3, 60)  # Dual variable range (λ ≥ 0 constraint)
X, LAM = np.meshgrid(x_3d, lambda_3d)

# Calculate Lagrangian values: L(x,λ) = x - λ(x-1) = x(1-λ) + λ
L_vals = X - LAM * (X - 1)  # This equals X*(1-LAM) + LAM

# Plot the surface with a colormap that emphasizes the saddle structure
surface = ax2.plot_surface(
    X, LAM, L_vals, cmap="coolwarm", alpha=0.8, linewidth=0, antialiased=True
)

# Mark the saddle point: (x*, λ*) = (1, 1) with L(1,1) = 1
saddle_x, saddle_lambda = 1, 1
saddle_L = lagrangian(saddle_x, saddle_lambda)
ax2.scatter(
    [saddle_x],
    [saddle_lambda],
    [saddle_L],
    color="red",
    s=150,
    label=f"Saddle point: $(1, 1, {saddle_L})$",
)

# Show the minimization path over x (holding λ = 1 fixed)
# When λ = 1: L(x,1) = x(1-1) + 1 = 1 (constant!)
x_min_path = np.linspace(-0.5, 3, 100)
lambda_fixed = 1
L_min_path = lagrangian(x_min_path, lambda_fixed)
ax2.plot(
    x_min_path,
    lambda_fixed * np.ones_like(x_min_path),
    L_min_path,
    "b-",
    linewidth=5,
    label="Minimize over $x$ (λ=1): flat!",
)

# Show the maximization path over λ (holding x = 1 fixed)
# When x = 1: L(1,λ) = 1(1-λ) + λ = 1 - λ + λ = 1 (constant!)
x_fixed = 1
lambda_max_path = np.linspace(0, 3, 100)
L_max_path = lagrangian(x_fixed, lambda_max_path)
ax2.plot(
    x_fixed * np.ones_like(lambda_max_path),
    lambda_max_path,
    L_max_path,
    "orange",
    linewidth=5,
    label="Maximize over λ (x=1): flat!",
)

# Add arrows to show optimization directions away from the saddle point
# Show what happens if you move away from x* = 1
x_arrow_start, lambda_arrow_start = 1.5, 1
L_arrow_start = lagrangian(x_arrow_start, lambda_arrow_start)
ax2.quiver(
    x_arrow_start,
    lambda_arrow_start,
    L_arrow_start,
    -0.3,
    0,
    -0.15,  # Direction toward minimum
    color="blue",
    arrow_length_ratio=0.15,
    linewidth=3,
)

# Show what happens if you move away from λ* = 1
x_arrow_start2, lambda_arrow_start2 = 1, 0.5
L_arrow_start2 = lagrangian(x_arrow_start2, lambda_arrow_start2)
ax2.quiver(
    x_arrow_start2,
    lambda_arrow_start2,
    L_arrow_start2,
    0,
    0.3,
    0,  # Direction toward maximum
    color="orange",
    arrow_length_ratio=0.15,
    linewidth=3,
)

# Set labels and title
ax2.set_xlabel("$x$ (primal variable)", fontsize=12)
ax2.set_ylabel("$\\lambda$ (dual variable)", fontsize=12)
ax2.set_zlabel("$\\mathcal{L}(x,\\lambda)$", fontsize=12)
ax2.set_title(
    "Lagrangian Saddle Point Structure\n$\\mathcal{L}(x,\\lambda) = x - \\lambda(x-1)$",
    fontsize=14,
    weight="bold",
)

# Adjust viewing angle to clearly show the saddle structure
ax2.view_init(elev=20, azim=45)

# Add legend
ax2.legend(loc="upper left", fontsize=10)

# Add colorbar to help interpret the surface height
fig.colorbar(surface, ax=ax2, shrink=0.6, aspect=20, label="$\\mathcal{L}(x,\\lambda)$")

plt.tight_layout()
plt.show()

# ============================================================================
# Educational cross-sections to show WHY it's a saddle point
# ============================================================================
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Show how L varies with x when λ is NOT at optimal value
x_cross = np.linspace(-0.5, 3, 200)

# Plot several different fixed values of λ to show the transition
lambda_values = [0.5, 1.0, 1.5]
colors = ["green", "red", "purple"]
for lam_val, color in zip(lambda_values, colors):
    L_x_cross = lagrangian(x_cross, lam_val)
    ax3.plot(
        x_cross,
        L_x_cross,
        color=color,
        linewidth=2,
        label=f"$\\mathcal{{L}}(x, \\lambda={lam_val})$",
    )

    # Mark the point where x = 1 for each curve
    ax3.plot(1, lagrangian(1, lam_val), "o", color=color, markersize=8)

ax3.axvline(x=1, color="red", linestyle="--", alpha=0.7, linewidth=2)
ax3.set_xlabel("$x$", fontsize=12)
ax3.set_ylabel("$\\mathcal{L}(x, \\lambda)$", fontsize=12)
ax3.set_title(
    "Cross-sections: How slope changes with $\\lambda$\nNote: only λ=1 gives flat slope at x=1",
    fontsize=12,
)
ax3.grid(True, alpha=0.3)
ax3.legend()

# Right: Show how L varies with λ when x is NOT at optimal value
lambda_cross = np.linspace(0, 3, 200)

# Plot several different fixed values of x to show the transition
x_values = [0.5, 1.0, 1.5]
colors = ["green", "red", "purple"]
for x_val, color in zip(x_values, colors):
    L_lambda_cross = lagrangian(x_val, lambda_cross)
    ax4.plot(
        lambda_cross,
        L_lambda_cross,
        color=color,
        linewidth=2,
        label=f"$\\mathcal{{L}}(x={x_val}, \\lambda)$",
    )

    # Mark the point where λ = 1 for each curve
    ax4.plot(1, lagrangian(x_val, 1), "o", color=color, markersize=8)

ax4.axvline(x=1, color="red", linestyle="--", alpha=0.7, linewidth=2)
ax4.set_xlabel("$\\lambda$", fontsize=12)
ax4.set_ylabel("$\\mathcal{L}(x, \\lambda)$", fontsize=12)
ax4.set_title(
    "Cross-sections: How slope changes with $x$\nNote: only x=1 gives flat slope at λ=1",
    fontsize=12,
)
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.show()

# ============================================================================
# Print key educational insights
# ============================================================================
print("=" * 70)
print("UNDERSTANDING THE CORRECTED SADDLE POINT STRUCTURE")
print("=" * 70)
print("STEP 1: The Fundamental Conflict")
print("   • Objective f(x) = x wants to minimize x → push toward -∞")
print("   • Constraint x ≥ 1 acts as a barrier, forcing x* = 1")
print("   • This conflict makes the constraint ACTIVE (binding)")
print()
print("STEP 2: Why We Get a Saddle Point")
print("   • At (x*, λ*) = (1, 1), the Lagrangian L(x,λ) = x(1-λ) + λ")
print("   • Stationarity condition: ∂L/∂x = 1 - λ = 0 ⟹ λ* = 1")
print("   • At optimum: L(1,1) = 1(1-1) + 1 = 1")
print()
print("STEP 3: The Saddle Structure Emerges")
print("   • Fix λ = λ* = 1: L(x,1) = 1 (flat in x direction)")
print("   • Fix x = x* = 1: L(1,λ) = λ (increasing in λ direction)")
print("   • This creates the classic saddle: minimize over x, maximize over λ")
print()
print("STEP 4: Algorithmic Implications")
print("   • Projected gradient alternates:")
print("     x^(k+1) = x^k - α∇_x L  (gradient descent)")
print("     λ^(k+1) = max(0, λ^k + β∇_λ L)  (projected gradient ascent)")
print()
print("STEP 5: Economic Interpretation")
print("   • λ* = 1 is the shadow price of the constraint")
print("   • If we could relax x ≥ 1 to x ≥ 1-ε, objective would improve by ε")
print("=" * 70)
