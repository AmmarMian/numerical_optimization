---
title: Backtracking memo
weight: 100
---

# Backtracking procedure for step size selection

## Introduction

The backtracking line search is a fundamental technique in optimization algorithms for determining an appropriate step size that ensures sufficient decrease in the objective function. This procedure is particularly useful in gradient-based methods where choosing an optimal step size analytically is difficult or computationally expensive.

## Mathematical setup

Consider the optimization problem:
$\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})$

where $f: \mathbb{R}^n \to \mathbb{R}$ is a continuously differentiable function. At iteration $k$, we have:
- Current point: $\mathbf{x}_k$
- Search direction: $\mathbf{p}_k$ (typically $\mathbf{p}_k = -\nabla f(\mathbf{x}_k)$ for steepest descent)
- Step size: $\alpha_k > 0$

## The Armijo condition

The backtracking procedure is based on the Armijo condition, which requires:
$f(\mathbf{x}_k + \alpha_k \mathbf{p}_k) \leq f(\mathbf{x}_k) + c_1 \alpha_k \nabla f(\mathbf{x}_k)^{\mathrm{T}} \mathbf{p}_k$

where $c_1 \in (0, 1)$ is a constant, typically $c_1 = 10^{-4}$.

## Backtracking algorithm steps

### Step 1: Initialize parameters
- Choose initial step size $\alpha_0 > 0$ (e.g., $\alpha_0 = 1$)
- Set reduction factor $\rho \in (0, 1)$ (typically $\rho = 0.5$)
- Set Armijo parameter $c_1 \in (0, 1)$ (typically $c_1 = 10^{-4}$)
- Set $\alpha = \alpha_0$

### Step 2: Check Armijo condition
Evaluate the condition:
$f(\mathbf{x}_k + \alpha \mathbf{p}_k) \leq f(\mathbf{x}_k) + c_1 \alpha \nabla f(\mathbf{x}_k)^{\mathrm{T}} \mathbf{p}_k$

### Step 3: Backtrack if necessary
**If** the Armijo condition is satisfied:
- Accept $\alpha_k = \alpha$
- **Go to** Step 4

**Else:**
- Update $\alpha \leftarrow \rho \alpha$
- **Return to** Step 2

### Step 4: Update iteration
Compute the new iterate:
$\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k$

## Algorithmic description

```
Algorithm: Backtracking Line Search
Input: x_k, p_k, α₀, ρ, c₁
Output: α_k

1. Set α = α₀
2. While f(x_k + α·p_k) > f(x_k) + c₁·α·∇f(x_k)ᵀ·p_k do
3.    α ← ρ·α
4. End while  
5. Return α_k = α
```

## Theoretical properties

### Convergence guarantee
Under mild conditions on $f$ and $\mathbf{p}_k$, the backtracking procedure terminates in finite steps. Specifically, if:
- $f$ is continuously differentiable
- $\mathbf{p}_k$ is a descent direction: $\nabla f(\mathbf{x}_k)^{\mathrm{T}} \mathbf{p}_k < 0$

Then there exists a step size $\alpha > 0$ satisfying the Armijo condition.

### Sufficient decrease property
The accepted step size $\alpha_k$ ensures:
$f(\mathbf{x}_{k+1}) - f(\mathbf{x}_k) \leq c_1 \alpha_k \nabla f(\mathbf{x}_k)^{\mathrm{T}} \mathbf{p}_k < 0$

This guarantees that each iteration decreases the objective function value.

## Implementation considerations

### Choice of parameters
- **Initial step size** $\alpha_0$: Common choices are $\alpha_0 = 1$ for Newton-type methods, or $\alpha_0 = 1/\|\nabla f(\mathbf{x}_k)\|$ for gradient methods
- **Reduction factor** $\rho$: Typically $\rho = 0.5$ or $\rho = 0.8$
- **Armijo parameter** $c_1$: Usually $c_1 = 10^{-4}$ or $c_1 = 10^{-3}$

### Computational complexity
Each backtracking iteration requires:
- One function evaluation: $f(\mathbf{x}_k + \alpha \mathbf{p}_k)$  
- One gradient evaluation: $\nabla f(\mathbf{x}_k)$ (if not already computed)
- One vector operation: $\mathbf{x}_k + \alpha \mathbf{p}_k$

### Practical modifications

**Maximum iterations:** Limit the number of backtracking steps to prevent infinite loops:
```
max_backtracks = 50
iter = 0
while (Armijo condition not satisfied) and (iter < max_backtracks):
    α ← ρ·α
    iter ← iter + 1
```

**Minimum step size:** Set a lower bound $\alpha_{min}$ to avoid numerical issues:
```
if α < α_min:
    α = α_min
    break
```

## Applications

The backtracking procedure is widely used in:
- **Gradient descent:** $\mathbf{p}_k = -\nabla f(\mathbf{x}_k)$
- **Newton's method:** $\mathbf{p}_k = -(\mathbf{H}_k)^{-1} \nabla f(\mathbf{x}_k)$ where $\mathbf{H}_k$ is the Hessian
- **Quasi-Newton methods:** $\mathbf{p}_k = -\mathbf{B}_k^{-1} \nabla f(\mathbf{x}_k)$ where $\mathbf{B}_k$ approximates the Hessian
- **Conjugate gradient methods**

## Example implementation

```python
def backtracking_line_search(f, grad_f, x_k, p_k, alpha_0=1.0, rho=0.5, c1=1e-4):
    """
    Backtracking line search for step size selection
    
    Parameters:
    - f: objective function
    - grad_f: gradient function  
    - x_k: current point
    - p_k: search direction
    - alpha_0: initial step size
    - rho: reduction factor
    - c1: Armijo parameter
    
    Returns:
    - alpha_k: accepted step size
    """
    alpha = alpha_0
    f_k = f(x_k)
    grad_k = grad_f(x_k)
    
    # Armijo condition right-hand side
    armijo_rhs = f_k + c1 * alpha * np.dot(grad_k, p_k)
    
    while f(x_k + alpha * p_k) > armijo_rhs:
        alpha *= rho
        armijo_rhs = f_k + c1 * alpha * np.dot(grad_k, p_k)
    
    return alpha
```

**Exercises**

1. Implement the backtracking line search for the quadratic function $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^{\mathrm{T}} \mathbf{Q} \mathbf{x} - \mathbf{b}^{\mathrm{T}} \mathbf{x}$, where $\mathbf{Q}$ is positive definite.

2. Compare the performance of different values of $\rho$ and $c_1$ on a test optimization problem.

3. Analyze the number of backtracking steps required as a function of the condition number of the Hessian matrix.
