---
title: "1. Unconstrained optimization : Second-order "
weight: 1
chapter: 1
---

# Unconstrained optimization -  Second-order methods

We have seen in the previous chapter that first-order methods, such as steepest descent, are often used to find a local minimum of a function $f(\mathbf{x})$. However, these methods can be slow to converge, especially when the function has ill-conditioned Hessian or when the initial guess is far from the solution. Second-order methods, which use information about the curvature of the function, can provide faster convergence rates.

## Search directions

Another important search direction-perhaps the most important one of all-is the Newton direction. This direction is derived from the second-order Taylor series approximation to $f\left(\mathbf{x}\_k+\mathbf{p}\right)$, which is
$$
f\left(\mathbf{x}\_k+\mathbf{p}\right) \approx f\_k+\mathbf{p}^{\mathrm{T}} \nabla f\_k+\frac{1}{2} \mathbf{p}^{\mathrm{T}} \nabla^2 f\_k \mathbf{p} \stackrel{\text { def }}{=} m\_k(\mathbf{p})
$$

Assuming for the moment that $\nabla^2 f\_k$ is positive definite, we obtain the Newton direction by finding the vector $\mathbf{p}$ that minimizes $m\_k(\mathbf{p})$. By simply setting the derivative of $m\_k(\mathbf{p})$ to zero, we obtain the following explicit formula:

\begin{equation}
\mathbf{p}\_k^{\mathrm{N}}=-\nabla^2 f\_k^{-1} \nabla f\_k
\label{eq:newton_direction}
\end{equation}

The Newton direction is reliable when the difference between the true function $f\left(\mathbf{x}\_k+ \mathbf{p}\right)$ and its quadratic model $m\_k(\mathbf{p})$ is not too large. By comparing \eqref{eq:newton_direction} with traditional Taylor expansion, we see that the only difference between these functions is that the matrix $\nabla^2 f\left(\mathbf{x}\_k+t \mathbf{p}\right)$ in the third term of the expansion has been replaced by $\nabla^2 f\_k=\nabla^2 f\left(\mathbf{x}\_k\right)$. If $\nabla^2 f(\cdot)$ is sufficiently smooth, this difference introduces a perturbation of only $O\left(\lVert\mathbf{p}\rVert^3\right)$ into the expansion, so that when $\lVert\mathbf{p}\rVert$ is small, the approximation $f\left(\mathbf{x}\_k+\mathbf{p}\right) \approx m\_k(\mathbf{p})$ is very accurate indeed.

The Newton direction can be used in a line search method when $\nabla^2 f\_k$ is positive definite, for in this case we have

$$
\nabla f\_k^{\mathrm{T}} \mathbf{p}\_k^{\mathrm{N}}=-\mathbf{p}\_k^{\mathrm{N} \mathrm{T}} \nabla^2 f\_k \mathbf{p}\_k^{\mathrm{N}} \leq-\sigma\_k\lVert\mathbf{p}\_k^{\mathrm{N}}\rVert^2
$$

for some $\sigma\_k>0$. Unless the gradient $\nabla f\_k$ (and therefore the step $\mathbf{p}\_k^N$) is zero, we have that $\nabla f\_k^{\mathrm{T}} \mathbf{p}\_k^{\mathrm{N}}<0$, so the Newton direction is a descent direction. Unlike the steepest descent direction, there is a "natural" step length of 1 associated with the Newton direction. Most
line search implementations of Newton's method use the unit step $\alpha=1$ where possible and adjust this step length only when it does not produce a satisfactory reduction in the value of $f$.

When $\nabla^2 f\_k$ is not positive definite, the Newton direction may not even be defined, since $\nabla^2 f\_k^{-1}$ may not exist. Even when it is defined, it may not satisfy the descent property $\nabla f\_k^{\mathrm{T}} \mathbf{p}\_k^{\mathrm{N}}<0$, in which case it is unsuitable as a search direction. In these situations, line search methods modify the definition of $\mathbf{p}\_k$ to make it satisfy the downhill condition while retaining the benefit of the second-order information contained in $\nabla^2 f\_k$. 

Methods that use the Newton direction have a fast rate of local convergence, typically quadratic. When a neighborhood of the solution is reached, convergence to high accuracy often occurs in just a few iterations. The main drawback of the Newton direction is the need for the Hessian $\nabla^2 f(\mathbf{x})$. Explicit computation of this matrix of second derivatives is sometimes, though not always, a cumbersome, error-prone, and expensive process.

Quasi-Newton search directions provide an attractive alternative in that they do not require computation of the Hessian and yet still attain a superlinear rate of convergence. In place of the true Hessian $\nabla^2 f\_k$, they use an approximation $\mathbf{B}\_k$, which is updated after each step to take account of the additional knowledge gained during the step. The updates make use of the fact that changes in the gradient $\mathbf{g}$ provide information about the second derivative of $f$ along the search direction. By using the expression from our statement of Taylor's theorem, we have by adding and subtracting the term $\nabla^2 f(\mathbf{x}) \mathbf{p}$ that

$$
\nabla f(\mathbf{x}+\mathbf{p})=\nabla f(\mathbf{x})+\nabla^2 f(\mathbf{x}) \mathbf{p}+\int\_0^1\left[\nabla^2 f(\mathbf{x}+t \mathbf{p})-\nabla^2 f(\mathbf{x})\right] \mathbf{p} d t
$$

Because $\nabla f(\cdot)$ is continuous, the size of the final integral term is $o(\lVert\mathbf{p}\rVert)$. By setting $\mathbf{x}=\mathbf{x}\_k$ and $\mathbf{p}=\mathbf{x}\_{k+1}-\mathbf{x}\_k$, we obtain

$$
\nabla f\_{k+1}=\nabla f\_k+\nabla^2 f\_{k+1}\left(\mathbf{x}\_{k+1}-\mathbf{x}\_k\right)+o\left(\lVert\mathbf{x}\_{k+1}-\mathbf{x}\_k\rVert\right)
$$

When $\mathbf{x}\_k$ and $\mathbf{x}\_{k+1}$ lie in a region near the solution $\mathbf{x}^*$, within which $\nabla f$ is positive definite, the final term in this expansion is eventually dominated by the $\nabla^2 f\_k\left(\mathbf{x}\_{k+1}-\mathbf{x}\_k\right)$ term, and we can write

$$
\nabla^2 f\_{k+1}\left(\mathbf{x}\_{k+1}-\mathbf{x}\_k\right) \approx \nabla f\_{k+1}-\nabla f\_k
$$

We choose the new Hessian approximation $\mathbf{B}\_{k+1}$ so that it mimics this property of the true Hessian, that is, we require it to satisfy the following condition, known as the secant equation:

\begin{equation}
\mathbf{B}\_{k+1} \mathbf{s}\_k=\mathbf{y}\_k
\label{eq:secant_equation}
\end{equation}

where

$$
\mathbf{s}\_k=\mathbf{x}\_{k+1}-\mathbf{x}\_k, \quad \mathbf{y}\_k=\nabla f\_{k+1}-\nabla f\_k
$$

Typically, we impose additional requirements on $\mathbf{B}\_{k+1}$, such as symmetry (motivated by symmetry of the exact Hessian), and a restriction that the difference between successive approximation $\mathbf{B}\_k$ to $\mathbf{B}\_{k+1}$ have low rank. The initial approximation $\mathbf{B}\_0$ must be chosen by the user.

Two of the most popular formulae for updating the Hessian approximation $\mathbf{B}\_k$ are the symmetric-rank-one (SR1) formula, defined by

\begin{equation}
\mathbf{B}\_{k+1}=\mathbf{B}\_k+\frac{\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right)\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right)^{\mathrm{T}}}{\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right)^{\mathrm{T}} \mathbf{s}\_k}
\label{eq:sr1_formula}
\end{equation}

and the BFGS formula, named after its inventors, Broyden, Fletcher, Goldfarb, and Shanno, which is defined by

\begin{equation}
\mathbf{B}\_{k+1}=\mathbf{B}\_k-\frac{\mathbf{B}\_k \mathbf{s}\_k \mathbf{s}\_k^{\mathrm{T}} \mathbf{B}\_k}{\mathbf{s}\_k^{\mathrm{T}} \mathbf{B}\_k \mathbf{s}\_k}+\frac{\mathbf{y}\_k \mathbf{y}\_k^{\mathrm{T}}}{\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k}
\label{eq:bfgs_formula}
\end{equation}

Note that the difference between the matrices $\mathbf{B}\_k$ and $\mathbf{B}\_{k+1}$ is a rank-one matrix in the case of \eqref{eq:sr1_formula}, and a rank-two matrix in the case of \eqref{eq:bfgs_formula}. Both updates satisfy the secant equation and both maintain symmetry. One can show that BFGS update \eqref{eq:bfgs_formula} generates positive definite approximations whenever the initial approximation $\mathbf{B}\_0$ is positive definite and $\mathbf{s}\_k^{\mathrm{T}} \mathbf{y}\_k>0$. 

The quasi-Newton search direction is given by using $\mathbf{B}\_k$ in place of the exact Hessian in the formula \eqref{eq:newton_direction}, that is,

\begin{equation}
\mathbf{p}\_k=-\mathbf{B}\_k^{-1} \nabla f\_k
\label{eq:quasi_newton_direction}
\end{equation}

Some practical implementations of quasi-Newton methods avoid the need to factorize $\mathbf{B}\_k$ at each iteration by updating the inverse of $\mathbf{B}\_k$, instead of $\mathbf{B}\_k$ itself. In fact, the equivalent formula for \eqref{eq:sr1_formula} and \eqref{eq:bfgs_formula}, applied to the inverse approximation $\mathbf{H}\_k \stackrel{\text { def }}{=} \mathbf{B}\_k^{-1}$, is

\begin{equation}
\mathbf{H}\_{k+1}=\left(\mathbf{I}-\rho\_k \mathbf{s}\_k \mathbf{y}\_k^{\mathrm{T}}\right) \mathbf{H}\_k\left(\mathbf{I}-\rho\_k \mathbf{y}\_k \mathbf{s}\_k^{\mathrm{T}}\right)+\rho\_k \mathbf{s}\_k \mathbf{s}\_k^{\mathrm{T}}, \quad \rho\_k=\frac{1}{\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k}
\label{eq:inverse_bfgs}
\end{equation}

Calculation of $\mathbf{p}\_k$ can then be performed by using the formula $\mathbf{p}\_k=-\mathbf{H}\_k \nabla f\_k$. This can be implemented as a matrix-vector multiplication, which is typically simpler than the factorization/back-substitution procedure that is needed to implement the formula \eqref{eq:quasi_newton_direction}.


## Step-size selection

Contrarily to the steepest descent, Newton methods have a "natural" step size of 1 associated with the Newton direction. This is because the Newton direction is derived from the second-order Taylor series approximation, which is designed to minimize the quadratic model of the function. However, in practice, it is often necessary to adjust this step size to ensure sufficient decrease in the function value.

When using a line search method, we can set $\alpha_k=1$ and check if this step size leads to a sufficient decrease in the function value. If it does not, we can use a backtracking line search to find a suitable step size that satisfies the Armijo condition. The Armijo condition ensures that the step size leads to a sufficient decrease in the function value, which is crucial for convergence of the method.


## Convergence of Newton methods

As in first-order methods, we make use of Zoutendijk's condition, that still apllies.

Consider now the Newton-like method with $\mathbf{p}\_k = -\mathbf{B}\_k^{-1} \nabla f\_k$ and assume that the matrices $\mathbf{B}\_k$ are positive definite with a uniformly bounded condition number. That is, there is a constant $M$ such that

\begin{equation}
\\|\mathbf{B}\_k\\|\\|\mathbf{B}\_k^{-1}\\| \leq M, \quad \text { for all } k .
\label{eq:condition_bound}
\end{equation}

It is easy to show from the definition that

$$
\cos \theta\_k \geq 1 / M
$$

By combining this bound with [(4.16)]({{% relref unconstrained_linesearch.md %}}/#eqn-id%3Aeq%3Azoutendijk_condition) we find that

$$
\lim \_{k \rightarrow \infty}\\|\nabla f\_k\\|=0
$$

Therefore, we have shown that Newton and quasi-Newton methods are globally convergent if the matrices $\mathbf{B}\_k$ have a bounded condition number and are positive definite (which is needed to ensure that $\mathbf{p}\_k$ is a descent direction), and if the step lengths satisfy the Wolfe conditions.

For some algorithms, such as conjugate gradient methods, we will not be able to prove the limit [(4.18)]({{% relref unconstrained_linesearch.md %}}), but only the weaker result

\begin{equation}
\liminf \_{k \rightarrow \infty}\\|\nabla f\_k\\|=0
\label{eq:weak_convergence}
\end{equation}

In other words, just a subsequence of the gradient norms $\\|\nabla f\_{k\_j}\\|$ converges to zero, rather than the whole sequence. This result, too, can be proved by using Zoutendijk's condition [(4.16)]({{% relref unconstrained_linesearch.md %}}/#eqn-id%3Aeq%3Azoutendijk_condition), but instead of a constructive proof, we outline a proof by contradiction. Suppose that \eqref{eq:weak_convergence} does not hold, so that the gradients remain bounded away from zero, that is, there exists $\gamma>0$ such that

$$
\\|\nabla f\_k\\| \geq \gamma, \quad \text { for all } k \text { sufficiently large. }
$$

Then from [(4.16)]({{% relref unconstrained_linesearch.md %}}/#eqn-id%3Aeq%3Azoutendijk_condition) we conclude that

$$
\cos \theta\_k \rightarrow 0
$$

that is, the entire sequence $\\{\cos \theta\_k\\}$ converges to 0. To establish \eqref{eq:weak_convergence}, therefore, it is enough to show that a subsequence $\\{\cos \theta\_{k\_j}\\}$ is bounded away from zero. 

<!-- By applying this proof technique, we can prove global convergence in the sense of [(4.18)]({{% relref unconstrained_linesearch.md %}})  or  \eqref{eq:weak_convergence} for a general class of algorithms. Consider any algorithm for which (i) every iteration produces a decrease in the objective function, and (ii) every $m$ th iteration is a steepest descent step, with step length chosen to satisfy the Wolfe or Goldstein conditions. Then since $\cos \theta\_k=1$ for the steepest descent steps, the result [(4.18)]({{% relref unconstrained_linesearch.md %}}) holds. Of course, we would design the algorithm so that it does something "better" than steepest descent at the other $m-1$ iterates; the occasional steepest descent steps may not make much progress, but they at least guarantee overall global convergence. -->


## Rate of convergence

We refer the reader to the textbook:
> "Numerical Optimization" by Nocedal and Wright, 2nd edition, Springer, 2006, pages 47-51,
Peculiarly, see pages 51-53.

