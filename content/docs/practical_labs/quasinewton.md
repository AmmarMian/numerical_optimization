---
title: Quasi-Newton methods memo
weight: 100
---

# BFGS and SR1 Quasi-Newton Methods

## Introduction

Quasi-Newton methods are a class of optimization algorithms that approximate the Newton direction without requiring explicit computation of the Hessian matrix. These methods achieve superlinear convergence while avoiding the computational expense and potential numerical difficulties associated with second derivatives. The two most prominent quasi-Newton methods are the BFGS (Broyden-Fletcher-Goldfarb-Shanno) and SR1 (Symmetric Rank-One) methods.

## Mathematical setup

Consider the optimization problem:
$\min\_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})$

where $f: \mathbb{R}^n \to \mathbb{R}$ is twice continuously differentiable. At iteration $k$, we have:
- Current point: $\mathbf{x}_k$
- Hessian approximation: $\mathbf{B}_k$ (or inverse approximation $\mathbf{H}_k = \mathbf{B}_k^{-1}$)
- Search direction: $\mathbf{p}_k = -\mathbf{B}_k^{-1} \nabla f_k = -\mathbf{H}_k \nabla f_k$

## The secant equation

Quasi-Newton methods are based on the secant equation, which approximates the relationship between gradient changes and the Hessian:

$\mathbf{B}\_{k+1} \mathbf{s}_k = \mathbf{y}_k$

where:
- $\mathbf{s}_k = \mathbf{x}\_{k+1} - \mathbf{x}_k$ (step vector)
- $\mathbf{y}_k = \nabla f\_{k+1} - \nabla f_k$ (gradient difference)

This equation ensures that the Hessian approximation captures the local curvature information from the most recent step.

## BFGS method

### BFGS update formula
The BFGS method uses a rank-two update to maintain positive definiteness:

$\mathbf{B}\_{k+1} = \mathbf{B}_k - \frac{\mathbf{B}_k \mathbf{s}_k \mathbf{s}_k^{\mathrm{T}} \mathbf{B}_k}{\mathbf{s}_k^{\mathrm{T}} \mathbf{B}_k \mathbf{s}_k} + \frac{\mathbf{y}_k \mathbf{y}_k^{\mathrm{T}}}{\mathbf{y}_k^{\mathrm{T}} \mathbf{s}_k}$

### Inverse BFGS formula
For computational efficiency, the inverse approximation is often updated directly:

$\mathbf{H}\_{k+1} = \left(\mathbf{I} - \rho_k \mathbf{s}_k \mathbf{y}_k^{\mathrm{T}}\right) \mathbf{H}_k \left(\mathbf{I} - \rho_k \mathbf{y}_k \mathbf{s}_k^{\mathrm{T}}\right) + \rho_k \mathbf{s}_k \mathbf{s}_k^{\mathrm{T}}$

where $\rho_k = \frac{1}{\mathbf{y}_k^{\mathrm{T}} \mathbf{s}_k}$.

### Properties of BFGS
- Maintains positive definiteness when $\mathbf{s}_k^{\mathrm{T}} \mathbf{y}_k > 0$
- Guarantees descent direction when positive definite
- Superlinear convergence rate
- Most robust and widely used quasi-Newton method

## SR1 method

### SR1 update formula
The Symmetric Rank-One method uses a simpler rank-one update:

$\mathbf{B}\_{k+1} = \mathbf{B}_k + \frac{(\mathbf{y}_k - \mathbf{B}_k \mathbf{s}_k)(\mathbf{y}_k - \mathbf{B}_k \mathbf{s}_k)^{\mathrm{T}}}{(\mathbf{y}_k - \mathbf{B}_k \mathbf{s}_k)^{\mathrm{T}} \mathbf{s}_k}$

### Properties of SR1
- Simpler structure (rank-one vs rank-two)
- Does not guarantee positive definiteness
- Can approximate indefinite Hessians better than BFGS
- May produce non-descent directions
- Often used in trust region methods

## Algorithm steps

### Step 1: Initialize parameters
- Choose initial point $\mathbf{x}_0$
- Set initial Hessian approximation $\mathbf{H}_0 = \mathbf{I}$ (or scaled identity)
- Set convergence tolerance $\epsilon > 0$
- Set $k = 0$

### Step 2: Check convergence
If $\|\nabla f_k\| \leq \epsilon$, **stop** and return $\mathbf{x}_k$.

### Step 3: Compute search direction
$\mathbf{p}_k = -\mathbf{H}_k \nabla f_k$

### Step 4: Line search
Find step size $\alpha_k$ using backtracking line search or other methods.

### Step 5: Update iterate
$\mathbf{x}\_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k$

### Step 6: Update Hessian approximation
Compute $\mathbf{s}_k = \mathbf{x}\_{k+1} - \mathbf{x}_k$ and $\mathbf{y}_k = \nabla f\_{k+1} - \nabla f_k$.
Update $\mathbf{H}\_{k+1}$ using BFGS or SR1 formula.

### Step 7: Increment and repeat
Set $k \leftarrow k + 1$ and **go to** Step 2.

## Algorithmic description

```
Algorithm: BFGS Quasi-Newton Method
Input: x₀, ε, max_iter
Output: x*

1. Set H₀ = I, k = 0
2. While ‖∇f_k‖ > ε and k < max_iter do
3.    p_k = -H_k ∇f_k
4.    α_k = backtracking_line_search(x_k, p_k)
5.    x\_{k+1} = x_k + α_k p_k
6.    s_k = x\_{k+1} - x_k
7.    y_k = ∇f\_{k+1} - ∇f_k
8.    if s_k^T y_k > 0 then
9.       ρ_k = 1/(y_k^T s_k)
10.      H\_{k+1} = (I - ρ_k s_k y_k^T) H_k (I - ρ_k y_k s_k^T) + ρ_k s_k s_k^T
11.   else
12.      H\_{k+1} = H_k  // Skip update
13.   k ← k + 1
14. End while
15. Return x_k
```

## Theoretical properties

### Convergence rate
Under suitable conditions:
- **BFGS:** Superlinear convergence when started sufficiently close to the solution
- **SR1:** Superlinear convergence in trust region frameworks
- Both methods reduce to Newton's method when the Hessian approximation becomes exact

### Curvature condition
For BFGS, the condition $\mathbf{s}_k^{\mathrm{T}} \mathbf{y}_k > 0$ is crucial for maintaining positive definiteness. This is automatically satisfied for strongly convex functions with exact line search.

## Implementation considerations

### Initial Hessian approximation
Common choices for $\mathbf{H}_0$:
- Identity matrix: $\mathbf{H}_0 = \mathbf{I}$
- Scaled identity: $\mathbf{H}_0 = \gamma \mathbf{I}$ where $\gamma > 0$
- Diagonal scaling based on the gradient

### Skipping updates
When $|\mathbf{s}_k^{\mathrm{T}} \mathbf{y}_k|$ is too small, skip the update to avoid numerical instability:
```
if |s_k^T y_k| < epsilon_skip * ‖s_k‖ * ‖y_k‖:
    H\_{k+1} = H_k  // Skip update
```

### Memory considerations
For large-scale problems, consider:
- **L-BFGS:** Limited memory version storing only recent vector pairs
- **Matrix-free implementations:** Avoid storing full matrices

## Applications

Quasi-Newton methods are widely used in:
- **Machine learning:** Training neural networks, logistic regression
- **Engineering optimization:** Design optimization, parameter estimation
- **Economics:** Portfolio optimization, economic modeling
- **Scientific computing:** Parameter fitting, inverse problems

## Example implementation

```python
import numpy as np
from scipy.optimize import line_search

def bfgs_optimizer(f, grad_f, x0, tol=1e-6, max_iter=100):
    """
    BFGS quasi-Newton optimization algorithm
    
    Parameters:
    - f: objective function
    - grad_f: gradient function
    - x0: initial point
    - tol: convergence tolerance
    - max_iter: maximum iterations
    
    Returns:
    - x: optimal point
    - history: optimization history
    """
    n = len(x0)
    x = x0.copy()
    H = np.eye(n)  # Initial inverse Hessian approximation
    
    history = {'x': [x.copy()], 'f': [f(x)], 'grad_norm': [np.linalg.norm(grad_f(x))]}
    
    for k in range(max_iter):
        grad = grad_f(x)
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
            
        # Compute search direction
        p = -H @ grad
        
        # Line search (simplified backtracking)
        alpha = backtracking_line_search(f, grad_f, x, p)
        
        # Update iterate
        x_new = x + alpha * p
        grad_new = grad_f(x_new)
        
        # Compute vectors for BFGS update
        s = x_new - x
        y = grad_new - grad
        
        # BFGS update (if curvature condition satisfied)
        if s @ y > 1e-10:
            rho = 1.0 / (y @ s)
            I = np.eye(n)
            
            # BFGS inverse Hessian update
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        
        # Update for next iteration
        x = x_new
        
        # Store history
        history['x'].append(x.copy())
        history['f'].append(f(x))
        history['grad_norm'].append(np.linalg.norm(grad_new))
    
    return x, history

def sr1_optimizer(f, grad_f, x0, tol=1e-6, max_iter=100):
    """
    SR1 quasi-Newton optimization algorithm
    
    Parameters similar to BFGS
    """
    n = len(x0)
    x = x0.copy()
    H = np.eye(n)
    
    history = {'x': [x.copy()], 'f': [f(x)], 'grad_norm': [np.linalg.norm(grad_f(x))]}
    
    for k in range(max_iter):
        grad = grad_f(x)
        
        if np.linalg.norm(grad) < tol:
            break
            
        p = -H @ grad
        alpha = backtracking_line_search(f, grad_f, x, p)
        
        x_new = x + alpha * p
        grad_new = grad_f(x_new)
        
        s = x_new - x
        y = grad_new - grad
        
        # SR1 update
        y_minus_Hs = y - H @ s
        denominator = y_minus_Hs @ s
        
        # Skip update if denominator too small
        if abs(denominator) > 1e-10:
            H = H + np.outer(y_minus_Hs, y_minus_Hs) / denominator
        
        x = x_new
        history['x'].append(x.copy())
        history['f'].append(f(x))
        history['grad_norm'].append(np.linalg.norm(grad_new))
    
    return x, history

def backtracking_line_search(f, grad_f, x, p, alpha0=1.0, rho=0.5, c1=1e-4):
    """Simple backtracking line search"""
    alpha = alpha0
    f_x = f(x)
    grad_x = grad_f(x)
    
    while f(x + alpha * p) > f_x + c1 * alpha * (grad_x @ p):
        alpha *= rho
        if alpha < 1e-10:  # Prevent infinite loop
            break
    
    return alpha
```

## Exercises

### Exercise 1: Himmelblau's Function
Implement both BFGS and SR1 methods to minimize Himmelblau's function:
$$f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2$$

Starting points to try:
- $(0, 0)$
- $(1, 1)$  
- $(-1, 1)$

Compare convergence rates and final solutions. Himmelblau's function has four global minima.

### Exercise 2: Mixed Function
Minimize the function:
$$f(x_1, x_2) = \frac{1}{2}x_1^2 + x_1 \cos(x_2)$$

Starting points:
- $(2, 0)$
- $(0, \pi)$
- $(1, \pi/2)$

Analyze how the cosine term affects convergence behavior and compare BFGS vs SR1 performance.

### Exercise 3: Comparative Analysis
For both test functions:
1. Plot the convergence history (function values and gradient norms)
2. Count the number of function and gradient evaluations
3. Analyze the condition number of the final Hessian approximations
4. Compare with gradient descent and Newton's method (when applicable)

### Exercise 4: Parameter Sensitivity
Study the effect of:
- Different initial Hessian approximations ($\mathbf{H}_0 = \gamma \mathbf{I}$ for various $\gamma$)
- Line search parameters in the backtracking procedure
- Skipping criteria for Hessian updates
