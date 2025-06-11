---
title: 5b. Constrained optimization - Projected Gradient Descent
weight: 6
math: true
chapter: 5
---

# Understanding Saddle Points in Constrained Optimization: From KKT Conditions to Projected Gradient Methods

When we first encounter the Karush-Kuhn-Tucker (KKT) conditions in constrained optimization, they often appear as a collection of mathematical requirements that characterize optimal solutions. However, these conditions actually emerge from a deeper geometric structure that reveals why constrained optimization problems possess fundamentally different mathematical properties than their unconstrained counterparts. This exploration will guide you through understanding how constraint conflicts create saddle point structures in the Lagrangian, and how this mathematical insight leads naturally to practical algorithms.

## The foundation: KKT conditions and the Lagrangian

Consider a general constrained optimization problem where we seek to minimize an objective function subject to both equality and inequality constraints. The mathematical framework begins with defining our constraint sets and constructing the Lagrangian function that will encode the relationship between our objective and constraints.

{{<definition "Constrained optimization problem" constrained_opt>}}
We seek to solve:
\begin{equation}
\begin{aligned}
\text{minimize} \quad & f(\mathbf{x}) \\\\
\text{subject to} \quad & c_i(\mathbf{x}) = 0, \quad i \in \mathcal{E} \\\\
& c_i(\mathbf{x}) \geq 0, \quad i \in \mathcal{I}
\end{aligned}
\label{eq:constrained_problem}
\end{equation}
where $\mathcal{E}$ represents the set of equality constraint indices and $\mathcal{I}$ represents the set of inequality constraint indices.
{{</definition>}}

The Lagrangian function serves as the mathematical bridge that connects our objective function with the constraint structure. Rather than treating constraints as separate mathematical entities, the Lagrangian weaves them together into a single function that encodes the fundamental trade-offs inherent in constrained optimization.

{{<definition "Lagrangian function" lagrangian_def>}}
For the constrained optimization problem in \eqref{eq:constrained_problem}, the **Lagrangian function** is defined as:
\begin{equation}
\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) - \sum_{i \in \mathcal{E} \cup \mathcal{I}} \lambda_i c_i(\mathbf{x})
\label{eq:lagrangian}
\end{equation}
where $\boldsymbol{\lambda} = (\lambda_1, \lambda_2, \ldots, \lambda_m)$ are the **Lagrange multipliers** associated with the constraints.
{{</definition>}}

The KKT conditions emerge as necessary conditions that any optimal solution must satisfy, provided certain regularity assumptions hold. These conditions capture the essential balance that must exist at an optimal point between the desire to improve the objective function and the requirement to respect the constraints.

{{<theorem "Karush-Kuhn-Tucker necessary conditions" kkt_necessary>}}
If $\mathbf{x}^\star$ is a local solution to \eqref{eq:constrained_problem} and the Linear Independence Constraint Qualification (LICQ) holds at $\mathbf{x}^\star$, then there exists a vector $\boldsymbol{\lambda}^\star$ such that:
\begin{equation}
\begin{aligned}
\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star) &= \mathbf{0} && \text{(Stationarity)} \\\\
c_i(\mathbf{x}^\star) &= 0, \quad i \in \mathcal{E} && \text{(Equality feasibility)} \\\\
c_i(\mathbf{x}^\star) &\geq 0, \quad i \in \mathcal{I} && \text{(Inequality feasibility)} \\\\
\lambda_i^\star &\geq 0, \quad i \in \mathcal{I} && \text{(Dual feasibility)} \\\\
\lambda_i^\star c_i(\mathbf{x}^\star) &= 0, \quad i \in \mathcal{E} \cup \mathcal{I} && \text{(Complementarity)}
\end{aligned}
\label{eq:kkt_conditions}
\end{equation}
{{</theorem>}}

While these conditions tell us what an optimal solution must look like, they leave a crucial question unanswered: how should we actually optimize the Lagrangian function to find such a solution? This question leads us to discover one of the most elegant structures in mathematical optimization.

## The emergence of saddle point structure

The key insight that transforms our understanding comes from recognizing that the Lagrangian possesses a special geometric property called a saddle point structure. This property emerges naturally from the mathematical conflict between objectives and constraints, and it explains why constrained optimization requires fundamentally different approaches than unconstrained problems.

To understand why this structure arises, let us examine what happens when we attempt different optimization strategies on the Lagrangian. Consider what would occur if we tried to minimize the Lagrangian with respect to both the primal variables $\mathbf{x}$ and the dual variables $\boldsymbol{\lambda}$ simultaneously.

For an inequality constraint $c_i(\mathbf{x}) \geq 0$, suppose we have a point where $c_i(\mathbf{x}) > 0$, meaning the constraint is satisfied with some slack. The Lagrangian contains the term $-\lambda_i c_i(\mathbf{x})$, which becomes increasingly negative as $\lambda_i$ increases. If we were minimizing over $\lambda_i$, this would drive $\lambda_i$ toward positive infinity, creating an unbounded minimization problem. This mathematical behavior makes no economic sense and violates the dual feasibility requirement $\lambda_i \geq 0$.

The resolution to this apparent contradiction reveals the fundamental insight: we must maximize over the dual variables rather than minimize. When $c_i(\mathbf{x}) > 0$ and we maximize over $\lambda_i \geq 0$, the maximization process naturally drives $\lambda_i$ toward zero, which aligns perfectly with the complementarity condition $\lambda_i c_i(\mathbf{x}) = 0$.

This mathematical necessity gives rise to the saddle point property, which provides both the theoretical foundation and the algorithmic guidance for solving constrained optimization problems.

{{<theorem "Saddle point characterization" saddle_point>}}
A point $(\mathbf{x}^\star, \boldsymbol{\lambda}^\star)$ solves the constrained optimization problem \eqref{eq:constrained_problem} if and only if it constitutes a saddle point of the Lagrangian function:
\begin{equation}
\mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}) \leq \mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda}^\star) \leq \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star)
\label{eq:saddle_point}
\end{equation}
for all feasible $\mathbf{x}$ and all $\boldsymbol{\lambda} \geq 0$.
{{</theorem>}}

The saddle point inequality \eqref{eq:saddle_point} encodes a beautiful mathematical principle. The left inequality tells us that $\mathcal{L}(\mathbf{x}^\star, \boldsymbol{\lambda})$ achieves its maximum over $\boldsymbol{\lambda}$ at $\boldsymbol{\lambda}^\star$, while the right inequality indicates that $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^\star)$ achieves its minimum over $\mathbf{x}$ at $\mathbf{x}^\star$. This creates the characteristic saddle shape: the surface curves downward (like a valley) in the primal direction and upward (like a ridge) in the dual direction.

## Illustrating the saddle point through a concrete example

To make these abstract concepts concrete, let us examine a specific problem that clearly demonstrates how constraint conflicts create saddle point structures. Consider the problem of minimizing $f(x) = -(x-3)^2$ subject to the constraint $x \geq 1$.

This example creates a compelling mathematical conflict. The objective function $f(x) = -(x-3)^2$ wants to make the expression $-(x-3)^2$ as small as possible. Since $(x-3)^2$ is always non-negative, the term $-(x-3)^2$ is always non-positive, reaching its maximum value of zero when $x = 3$. To minimize $-(x-3)^2$, we need $(x-3)^2$ to be as large as possible, which drives $x$ away from 3 toward negative infinity.

However, the constraint $x \geq 1$ acts as a mathematical barrier that prevents this natural tendency. The objective desperately wants to push $x$ toward $-\infty$ where $f(x) \to -\infty$, but the constraint forces the solution to occur at the boundary $x^\star = 1$.

The Lagrangian for this problem becomes:
\begin{equation}
\mathcal{L}(x,\lambda) = -(x-3)^2 - \lambda(x-1)
\label{eq:example_lagrangian}
\end{equation}

To find the optimal point, we apply the stationarity condition:
\begin{equation}
\frac{\partial \mathcal{L}}{\partial x} = -2(x-3) - \lambda = 0
\label{eq:stationarity_condition}
\end{equation}

At the constrained optimum $x^\star = 1$, this gives us:
\begin{equation}
-2(1-3) - \lambda = 0 \Rightarrow 4 - \lambda = 0 \Rightarrow \lambda^\star = 4
\label{eq:optimal_multiplier}
\end{equation}

We can verify the saddle point property by examining cross-sections of the Lagrangian. When we fix $\lambda = 4$ and vary $x$, we obtain:
\begin{equation}
\mathcal{L}(x,4) = -(x-3)^2 - 4(x-1) = -(x-3)^2 - 4x + 4
\label{eq:primal_cross_section}
\end{equation}

This function has a unique minimum at $x = 1$, confirming that we should minimize over the primal variable. When we fix $x = 1$ and vary $\lambda$, we get:
\begin{equation}
\mathcal{L}(1,\lambda) = -(1-3)^2 - \lambda(1-1) = -4
\label{eq:dual_cross_section}
\end{equation}

The Lagrangian becomes constant with respect to $\lambda$ when the constraint is exactly satisfied. This apparent insensitivity to $\lambda$ actually illustrates a profound principle: the dual variable value is uniquely determined by the stationarity requirement, and it encodes the economic value of constraint relaxation.

The Lagrange multiplier $\lambda^\star = 4$ represents the shadow price of the constraint. If we could relax the constraint from $x \geq 1$ to $x \geq 1 - \epsilon$ for some small $\epsilon > 0$, the optimal objective value would improve by approximately $4\epsilon$. We can verify this directly: with the relaxed constraint, the new optimum would be $x^\star = 1 - \epsilon$, giving $f(1-\epsilon) = -((1-\epsilon)-3)^2 = -(2+\epsilon)^2 = -4 - 4\epsilon - \epsilon^2 \approx -4 - 4\epsilon$ for small $\epsilon$. The improvement of $4\epsilon$ confirms that $\lambda^\star = 4$ correctly captures the economic value of constraint relaxation.

## From theory to computation: the projected gradient method

The saddle point structure provides both theoretical insight and algorithmic guidance. Since we need to minimize over primal variables and maximize over dual variables, the natural computational approach involves alternating between these two types of updates. This leads to the projected gradient method, which implements the saddle point structure through iterative optimization.

{{<theorem "Projected gradient algorithm" projected_gradient>}}
Given the Lagrangian $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})$ from \eqref{eq:lagrangian}, the **projected gradient method** alternates between primal minimization and dual maximization:

**Initialize:** $\mathbf{x}^0$, $\boldsymbol{\lambda}^0 \geq \mathbf{0}$

**For** $k = 0, 1, 2, \ldots$ **until convergence:**
\begin{equation}
\begin{aligned}
\mathbf{x}^{k+1} &= \mathbf{x}^k - \alpha_k \nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}^k, \boldsymbol{\lambda}^k) \\\\
\boldsymbol{\lambda}^{k+1} &= \max(\mathbf{0}, \boldsymbol{\lambda}^k + \beta_k \nabla_{\boldsymbol{\lambda}} \mathcal{L}(\mathbf{x}^{k+1}, \boldsymbol{\lambda}^k))
\end{aligned}
\label{eq:projected_gradient}
\end{equation}
where $\alpha_k > 0$ and $\beta_k > 0$ are step size parameters.
{{</theorem>}}

The algorithm embodies the saddle point structure through its alternating updates. The primal step performs gradient descent on the Lagrangian with respect to $\mathbf{x}$, following the downward-curving direction of the saddle surface. The dual step performs projected gradient ascent on the Lagrangian with respect to $\boldsymbol{\lambda}$, following the upward-curving direction while maintaining the constraint $\boldsymbol{\lambda} \geq \mathbf{0}$ through the projection operation $\max(\mathbf{0}, \cdot)$.

To understand the specific update formulas, we need to compute the gradients of the Lagrangian. For our general formulation:

\begin{equation}
\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = \nabla f(\mathbf{x}) - \sum_{i} \lambda_i \nabla c_i(\mathbf{x})
\label{eq:primal_gradient}
\end{equation}

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \lambda_i} = -c_i(\mathbf{x})
\label{eq:dual_gradient}
\end{equation}

These gradients reveal the intuitive behavior of the algorithm. The primal update balances the objective gradient against the weighted constraint gradients, with the dual variables serving as the weights that encode constraint importance. The dual update increases $\lambda_i$ when the constraint $c_i(\mathbf{x}) < 0$ is violated and decreases $\lambda_i$ when $c_i(\mathbf{x}) > 0$ provides slack, naturally driving the algorithm toward complementarity.

## **Exercise**: constrained optimization with mixed constraints

Let us apply our understanding to a concrete problem that illustrates both the theoretical principles and the algorithmic implementation. This exercise demonstrates how the projected gradient method handles a mixture of equality and inequality constraints.

**Problem setup:**
\begin{equation}
\begin{aligned}
\text{minimize} \quad & f(x,y) = (x-2)^2 + (y-2)^2 \\\\
\text{subject to:} \quad & g(x,y) = x + y - 2 = 0 \\\\
& h_1(x,y) = x \geq 0 \\\\
& h_2(x,y) = y \geq 0
\end{aligned}
\label{eq:exercise_problem}
\end{equation}

This problem seeks the point closest to $(2,2)$ that lies on the line $x + y = 2$ while remaining in the first quadrant. The geometric intuition suggests that since the unconstrained minimizer $(2,2)$ lies on the line $x + y = 4$, and our constraint line is $x + y = 2$, the optimal point should be the point on $x + y = 2$ that is closest to $(2,2)$.

**Lagrangian construction:**

We reformulate the inequality constraints in the standard form $c_i(\mathbf{x}) \geq 0$, giving us $h_1(x,y) = x \geq 0$ and $h_2(x,y) = y \geq 0$. The Lagrangian becomes:
\begin{equation}
\mathcal{L}(x,y,\lambda,\mu_1,\mu_2) = (x-2)^2 + (y-2)^2 - \lambda(x + y - 2) - \mu_1(-x) - \mu_2(-y)
\label{eq:exercise_lagrangian}
\end{equation}

Simplifying the inequality constraint terms:
\begin{equation}
\mathcal{L}(x,y,\lambda,\mu_1,\mu_2) = (x-2)^2 + (y-2)^2 - \lambda(x + y - 2) + \mu_1 x + \mu_2 y
\label{eq:simplified_lagrangian}
\end{equation}

**Gradient computations:**

The gradients required for the projected gradient algorithm are:
\begin{equation}
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial x} &= 2(x-2) - \lambda + \mu_1 \\\\
\frac{\partial \mathcal{L}}{\partial y} &= 2(y-2) - \lambda + \mu_2 \\\\
\frac{\partial \mathcal{L}}{\partial \lambda} &= -(x + y - 2) \\\\
\frac{\partial \mathcal{L}}{\partial \mu_1} &= x \\\\
\frac{\partial \mathcal{L}}{\partial \mu_2} &= y
\end{aligned}
\label{eq:exercise_gradients}
\end{equation}

**Projected gradient updates:**

The algorithm updates become:
\begin{equation}
\begin{aligned}
x^{k+1} &= x^k - \alpha(2(x^k-2) - \lambda^k + \mu_1^k) \\\\
y^{k+1} &= y^k - \alpha(2(y^k-2) - \lambda^k + \mu_2^k) \\\\
\lambda^{k+1} &= \lambda^k + \beta(x^{k+1} + y^{k+1} - 2) \\\\
\mu_1^{k+1} &= \max(0, \mu_1^k - \beta x^{k+1}) \\\\
\mu_2^{k+1} &= \max(0, \mu_2^k - \beta y^{k+1})
\end{aligned}
\label{eq:exercise_updates}
\end{equation}

Note that for the equality constraint, we do not apply a projection to $\lambda^{k+1}$ since equality constraint multipliers can take any real value.

**Analytical solution:**

To understand what the algorithm should converge to, let us solve the problem analytically using the KKT conditions. At the optimal point, we expect the inequality constraints $x \geq 0$ and $y \geq 0$ to be inactive since the solution likely lies in the interior of the first quadrant.

If both inequality constraints are inactive, then $\mu_1^\star = \mu_2^\star = 0$ by complementarity. The KKT conditions reduce to:
\begin{equation}
\begin{aligned}
2(x^\star-2) - \lambda^\star &= 0 \\\\
2(y^\star-2) - \lambda^\star &= 0 \\\\
x^\star + y^\star - 2 &= 0
\end{aligned}
\label{eq:reduced_kkt}
\end{equation}

From the first two equations, we see that $2(x^\star-2) = 2(y^\star-2)$, which implies $x^\star = y^\star$. Substituting into the equality constraint:
\begin{equation}
x^\star + x^\star = 2 \Rightarrow x^\star = y^\star = 1
\label{eq:optimal_point}
\end{equation}

The Lagrange multiplier for the equality constraint is:
\begin{equation}
\lambda^\star = 2(1-2) = -2
\label{eq:optimal_lambda}
\end{equation}

The negative value indicates that if we could relax the constraint from $x + y = 2$ to $x + y = 2 + \epsilon$, the objective would worsen by approximately $2\epsilon$, which makes intuitive sense since we would be moving away from the unconstrained optimum.

**Verification:**

We can verify that $(x^\star, y^\star) = (1, 1)$ with $\lambda^\star = -2$ and $\mu_1^\star = \mu_2^\star = 0$ satisfies all KKT conditions:
- **Stationarity:** $\nabla_{\mathbf{x}} \mathcal{L} = \mathbf{0}$ ✓
- **Equality feasibility:** $1 + 1 - 2 = 0$ ✓  
- **Inequality feasibility:** $1 \geq 0$ and $1 \geq 0$ ✓
- **Dual feasibility:** $\mu_1^\star = 0 \geq 0$ and $\mu_2^\star = 0 \geq 0$ ✓
- **Complementarity:** $\mu_1^\star \cdot 1 = 0$ and $\mu_2^\star \cdot 1 = 0$ ✓

The projected gradient algorithm will converge to this solution, automatically determining that the inequality constraints are inactive through the projection steps that drive $\mu_1$ and $\mu_2$ toward zero.

