---
title: "1. Unconstrained optimization : Second-order "
weight: 1
chapter: 1
---

# Unconstrained optimization -  Second-order methods




> **Note** : This is in part the content of the book "Numerical Optimization" by Nocedal and Wright, with some modifications to the notations used in this lecture.

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

## Quasi-newton methods

Quasi-Newton methods, like steepest descent, require only the gradient of the objective function to be supplied at each iterate. By measuring the changes in gradients, they construct a model of the objective function that is good enough to produce superlinear convergence. The improvement over steepest descent is dramatic, especially on difficult problems. Moreover, since second derivatives are not required, quasi-Newton methods are sometimes more efficient than Newton's method. Today, optimization software libraries contain a variety of quasi-Newton algorithms for solving unconstrained, constrained, and large-scale optimization problems. In this chapter we discuss quasi-Newton methods for small and medium-sized problems.

The development of automatic differentiation techniques has diminished the appeal of quasi-Newton methods, but only to a limited extent. Automatic differentiation eliminates the tedium of computing second derivatives by hand, as well as the risk of introducing errors in the calculation. Nevertheless, quasi-Newton methods remain competitive on many types of problems.

### The BFGS method

The most popular quasi-Newton algorithm is the BFGS method, named for its discoverers Broyden, Fletcher, Goldfarb, and Shanno. In this section we derive this algorithm (and its close relative, the DFP algorithm) and describe its theoretical properties and practical implementation.

We begin the derivation by forming the following quadratic model of the objective function at the current iterate $\mathbf{x}\_k$:

$$
m\_k(\mathbf{p})=f\_k+\nabla f\_k^{\mathrm{T}} \mathbf{p}+\frac{1}{2} \mathbf{p}^{\mathrm{T}} \mathbf{B}\_k \mathbf{p}
$$

Here $\mathbf{B}\_k$ is an $n \times n$ symmetric positive definite matrix that will be revised or updated at every iteration. Note that the value and gradient of this model at $\mathbf{p}=\mathbf{0}$ match $f\_k$ and $\nabla f\_k$, respectively. The minimizer $\mathbf{p}\_k$ of this convex quadratic model, which we can write explicitly as

$$
\mathbf{p}\_k=-\mathbf{B}\_k^{-1} \nabla f\_k,
$$

is used as the search direction, and the new iterate is

$$
\mathbf{x}\_{k+1}=\mathbf{x}\_k+\alpha\_k \mathbf{p}\_k
$$

where the step length $\alpha\_k$ is chosen to satisfy the Wolfe conditions. This iteration is quite similar to the line search Newton method; the key difference is that the approximate Hessian $\mathbf{B}\_k$ is used in place of the true Hessian.

Instead of computing $\mathbf{B}\_k$ afresh at every iteration, Davidon proposed to update it in a simple manner to account for the curvature measured during the most recent step. Suppose that we have generated a new iterate $\mathbf{x}\_{k+1}$ and wish to construct a new quadratic model, of the form

$$
m\_{k+1}(\mathbf{p})=f\_{k+1}+\nabla f\_{k+1}^{\mathrm{T}} \mathbf{p}+\frac{1}{2} \mathbf{p}^{\mathrm{T}} \mathbf{B}\_{k+1} \mathbf{p} .
$$

What requirements should we impose on $\mathbf{B}\_{k+1}$, based on the knowledge we have gained during the latest step? One reasonable requirement is that the gradient of $m\_{k+1}$ should match the gradient of the objective function $f$ at the latest two iterates $\mathbf{x}\_k$ and $\mathbf{x}\_{k+1}$. Since $\nabla m\_{k+1}(\mathbf{0})$ is precisely $\nabla f\_{k+1}$, the second of these conditions is satisfied automatically. The first condition can be written mathematically as

$$
\nabla m\_{k+1}\left(-\alpha\_k \mathbf{p}\_k\right)=\nabla f\_{k+1}-\alpha\_k \mathbf{B}\_{k+1} \mathbf{p}\_k=\nabla f\_k .
$$

By rearranging, we obtain

$$
\mathbf{B}\_{k+1} \alpha\_k \mathbf{p}\_k=\nabla f\_{k+1}-\nabla f\_k .
$$

To simplify the notation it is useful to define the vectors

$$
\mathbf{s}\_k=\mathbf{x}\_{k+1}-\mathbf{x}\_k, \quad \mathbf{y}\_k=\nabla f\_{k+1}-\nabla f\_k,
$$

so that the equation becomes

\begin{equation}
\mathbf{B}\_{k+1} \mathbf{s}\_k=\mathbf{y}\_k .
\label{eq:secant\_equation\_bfgs}
\end{equation}

{{<definition "Secant equation" secant_equation>}}
We refer to \eqref{eq:secant\_equation\_bfgs} as the **secant equation**.
{{</definition>}}

Given the displacement $\mathbf{s}\_k$ and the change of gradients $\mathbf{y}\_k$, the secant equation requires that the symmetric positive definite matrix $\mathbf{B}\_{k+1}$ map $\mathbf{s}\_k$ into $\mathbf{y}\_k$. This will be possible only if $\mathbf{s}\_k$ and $\mathbf{y}\_k$ satisfy the curvature condition

\begin{equation}
\mathbf{s}\_k^{\mathrm{T}} \mathbf{y}\_k>0,
\label{eq:curvature\_condition}
\end{equation}

as is easily seen by premultiplying \eqref{eq:secant\_equation\_bfgs} by $\mathbf{s}\_k^{\mathrm{T}}$. When $f$ is strongly convex, the inequality \eqref{eq:curvature\_condition} will be satisfied for any two points $\mathbf{x}\_k$ and $\mathbf{x}\_{k+1}$. However, this condition will not always hold for nonconvex functions, and in this case we need to enforce \eqref{eq:curvature\_condition} explicitly, by imposing restrictions on the line search procedure that chooses $\alpha$. In fact, the condition \eqref{eq:curvature\_condition} is guaranteed to hold if we impose the Wolfe or strong Wolfe conditions on the line search. To verify this claim, we note from the definition and the Wolfe condition that $\nabla f\_{k+1}^{\mathrm{T}} \mathbf{s}\_k \geq c\_2 \nabla f\_k^{\mathrm{T}} \mathbf{s}\_k$, and therefore

$$
\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k \geq\left(c\_2-1\right) \alpha\_k \nabla f\_k^{\mathrm{T}} \mathbf{p}\_k .
$$

Since $c\_2<1$ and since $\mathbf{p}\_k$ is a descent direction, the term on the right will be positive, and the curvature condition \eqref{eq:curvature\_condition} holds.

When the curvature condition is satisfied, the secant equation \eqref{eq:secant\_equation\_bfgs} always has a solution $\mathbf{B}\_{k+1}$. In fact, it admits an infinite number of solutions, since there are $n(n+1) / 2$ degrees of freedom in a symmetric matrix, and the secant equation represents only $n$ conditions. The requirement of positive definiteness imposes $n$ additional inequalities—all principal minors must be positive—but these conditions do not absorb the remaining degrees of freedom.

To determine $\mathbf{B}\_{k+1}$ uniquely, then, we impose the additional condition that among all symmetric matrices satisfying the secant equation, $\mathbf{B}\_{k+1}$ is, in some sense, closest to the current matrix $\mathbf{B}\_k$. In other words, we solve the problem

$$
\begin{gathered}
\min \_{\mathbf{B}}\\|\mathbf{B}-\mathbf{B}\_k\\| \\\\
\text { subject to } \quad \mathbf{B}=\mathbf{B}^{\mathrm{T}}, \quad \mathbf{B} \mathbf{s}\_k=\mathbf{y}\_k
\end{gathered}
$$

where $\mathbf{s}\_k$ and $\mathbf{y}\_k$ satisfy \eqref{eq:curvature\_condition} and $\mathbf{B}\_k$ is symmetric and positive definite. Many matrix norms can be used in the objective, and each norm gives rise to a different quasi-Newton method. A norm that allows easy solution of the minimization problem, and that gives rise to a scale-invariant optimization method, is the weighted Frobenius norm

$$
\\|\mathbf{A}\\|\_{\mathbf{W}} \equiv\\|\mathbf{W}^{1 / 2} \mathbf{A} \mathbf{W}^{1 / 2}\\|\_F
$$

where $\\|\cdot\\|\_F$ is defined by $\\|\mathbf{C}\\|\_F^2=\sum\_{i=1}^{n} \sum\_{j=1}^{n} c\_{ij}^2$. The weight $\mathbf{W}$ can be chosen as any matrix satisfying the relation $\mathbf{W} \mathbf{y}\_k=\mathbf{s}\_k$. For concreteness, the reader can assume that $\mathbf{W}=\overline{\mathbf{G}}\_k^{-1}$ where $\overline{\mathbf{G}}\_k$ is the average Hessian defined by

$$
\overline{\mathbf{G}}\_k=\left[\int\_{0}^{1} \nabla^2 f\left(\mathbf{x}\_k+\tau \alpha\_k \mathbf{p}\_k\right) d \tau\right]
$$

The property

$$
\mathbf{y}\_k=\overline{\mathbf{G}}\_k \alpha\_k \mathbf{p}\_k=\overline{\mathbf{G}}\_k \mathbf{s}\_k
$$

follows from Taylor's theorem. With this choice of weighting matrix $\mathbf{W}$, the norm is adimensional, which is a desirable property, since we do not wish the solution to depend on the units of the problem.

With this weighting matrix and this norm, the unique solution is

$$
\text { (DFP) } \quad \mathbf{B}\_{k+1}=\left(\mathbf{I}-\gamma\_k \mathbf{y}\_k \mathbf{s}\_k^{\mathrm{T}}\right) \mathbf{B}\_k\left(\mathbf{I}-\gamma\_k \mathbf{s}\_k \mathbf{y}\_k^{\mathrm{T}}\right)+\gamma\_k \mathbf{y}\_k \mathbf{y}\_k^{\mathrm{T}},
$$

with

$$
\gamma\_k=\frac{1}{\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k}
$$

This formula is called the DFP updating formula, since it is the one originally proposed by Davidon in 1959, and subsequently studied, implemented, and popularized by Fletcher and Powell.

The inverse of $\mathbf{B}\_k$, which we denote by

$$
\mathbf{H}\_k=\mathbf{B}\_k^{-1},
$$

is useful in the implementation of the method, since it allows the search direction to be calculated by means of a simple matrix-vector multiplication. Using the Sherman-Morrison-Woodbury formula, we can derive the following expression for the update of the inverse Hessian approximation $\mathbf{H}\_k$ that corresponds to the DFP update of $\mathbf{B}\_k$:

$$
\text { (DFP) } \quad \mathbf{H}\_{k+1}=\mathbf{H}\_k-\frac{\mathbf{H}\_k \mathbf{y}\_k \mathbf{y}\_k^{\mathrm{T}} \mathbf{H}\_k}{\mathbf{y}\_k^{\mathrm{T}} \mathbf{H}\_k \mathbf{y}\_k}+\frac{\mathbf{s}\_k \mathbf{s}\_k^{\mathrm{T}}}{\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k} \text {. }
$$

Note that the last two terms in the right-hand-side are rank-one matrices, so that $\mathbf{H}\_k$ undergoes a rank-two modification. It is easy to see that the DFP formula is also a rank-two modification of $\mathbf{B}\_k$. This is the fundamental idea of quasi-Newton updating: Instead of recomputing the iteration matrices from scratch at every iteration, we apply a simple modification that combines the most recently observed information about the objective function with the existing knowledge embedded in our current Hessian approximation.

The DFP updating formula is quite effective, but it was soon superseded by the BFGS formula, which is presently considered to be the most effective of all quasi-Newton updating formulae. BFGS updating can be derived by making a simple change in the argument that led to DFP. Instead of imposing conditions on the Hessian approximations $\mathbf{B}\_k$, we impose similar conditions on their inverses $\mathbf{H}\_k$. The updated approximation $\mathbf{H}\_{k+1}$ must be symmetric and positive definite, and must satisfy the secant equation, now written as

$$
\mathbf{H}\_{k+1} \mathbf{y}\_k=\mathbf{s}\_k .
$$

The condition of closeness to $\mathbf{H}\_k$ is now specified by the following analogue:

$$
\begin{gathered}
\min \_{\mathbf{H}}\\|\mathbf{H}-\mathbf{H}\_k\\| \\\\
\text { subject to } \quad \mathbf{H}=\mathbf{H}^{\mathrm{T}}, \quad \mathbf{H} \mathbf{y}\_k=\mathbf{s}\_k .
\end{gathered}
$$

The norm is again the weighted Frobenius norm described above, where the weight matrix $\mathbf{W}$ is now any matrix satisfying $\mathbf{W} \mathbf{s}\_k=\mathbf{y}\_k$. (For concreteness, we assume again that $\mathbf{W}$ is given by the average Hessian $\overline{\mathbf{G}}\_k$.) The unique solution $\mathbf{H}\_{k+1}$ is given by

\begin{equation}
\mathbf{H}\_{k+1}=\left(\mathbf{I}-\rho\_k \mathbf{s}\_k \mathbf{y}\_k^{\mathrm{T}}\right) \mathbf{H}\_k\left(\mathbf{I}-\rho\_k \mathbf{y}\_k \mathbf{s}\_k^{\mathrm{T}}\right)+\rho\_k \mathbf{s}\_k \mathbf{s}\_k^{\mathrm{T}},
\label{eq:bfgs\_update}
\end{equation}

where

$$
\rho\_k=\frac{1}{\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k}
$$

Just one issue has to be resolved before we can define a complete BFGS algorithm: How should we choose the initial approximation $\mathbf{H}\_0$ ? Unfortunately, there is no magic formula that works well in all cases. We can use specific information about the problem, for instance by setting it to the inverse of an approximate Hessian calculated by finite differences at $\mathbf{x}\_0$. Otherwise, we can simply set it to be the identity matrix, or a multiple of the identity matrix, where the multiple is chosen to reflect the scaling of the variables.

```
Algorithm 8.1 (BFGS Method).
    Given starting point $\mathbf{x}_0$, convergence tolerance $\epsilon>0$,
        inverse Hessian approximation $\mathbf{H}_0$;
    $k \leftarrow 0$;
    while $\|\nabla f_k\|>\epsilon$;
        Compute search direction
                $\mathbf{p}_k=-\mathbf{H}_k \nabla f_k ;$
        Set $\mathbf{x}_{k+1}=\mathbf{x}_k+\alpha_k \mathbf{p}_k$ where $\alpha_k$ is computed from a line search
            procedure to satisfy the Wolfe conditions;
        Define $\mathbf{s}_k=\mathbf{x}_{k+1}-\mathbf{x}_k$ and $\mathbf{y}_k=\nabla f_{k+1}-\nabla f_k$;
        Compute $\mathbf{H}_{k+1}$ by means of \eqref{eq:bfgs\_update};
        $k \leftarrow k+1$;
    end (while)
```

Each iteration can be performed at a cost of $O\left(n^{2}\right)$ arithmetic operations (plus the cost of function and gradient evaluations); there are no $O\left(n^{3}\right)$ operations such as linear system solves or matrix-matrix operations. The algorithm is robust, and its rate of convergence is superlinear, which is fast enough for most practical purposes. Even though Newton's method converges more rapidly (that is, quadratically), its cost per iteration is higher because it requires the solution of a linear system. A more important advantage for BFGS is, of course, that it does not require calculation of second derivatives.

We can derive a version of the BFGS algorithm that works with the Hessian approximation $\mathbf{B}\_k$ rather than $\mathbf{H}\_k$. The update formula for $\mathbf{B}\_k$ is obtained by simply applying the Sherman-Morrison-Woodbury formula to \eqref{eq:bfgs\_update} to obtain

\begin{equation}
\text { (BFGS) } \quad \mathbf{B}\_{k+1}=\mathbf{B}\_k-\frac{\mathbf{B}\_k \mathbf{s}\_k \mathbf{s}\_k^{\mathrm{T}} \mathbf{B}\_k}{\mathbf{s}\_k^{\mathrm{T}} \mathbf{B}\_k \mathbf{s}\_k}+\frac{\mathbf{y}\_k \mathbf{y}\_k^{\mathrm{T}}}{\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k} \text {. }
\label{eq:bfgs\_b\_update}
\end{equation}

A naive implementation of this variant is not efficient for unconstrained minimization, because it requires the system $\mathbf{B}\_k \mathbf{p}\_k=-\nabla f\_k$ to be solved for the step $\mathbf{p}\_k$, thereby increasing the cost of the step computation to $O\left(n^{3}\right)$. We discuss later, however, that less expensive implementations of this variant are possible by updating Cholesky factors of $\mathbf{B}\_k$.

### Properties of the BFGS method

It is usually easy to observe the superlinear rate of convergence of the BFGS method on practical problems. Below, we report the last few iterations of the steepest descent, BFGS, and an inexact Newton method on Rosenbrock's function. The table gives the value of $\\|\mathbf{x}\_k-\mathbf{x}^\star\\|$. The Wolfe conditions were imposed on the step length in all three methods. From the starting point $(-1.2,1)$, the steepest descent method required 5264 iterations, whereas BFGS and Newton took only 34 and 21 iterations, respectively to reduce the gradient norm to $10^{-5}$.

| steep. desc. | BFGS | Newton |
| ---: | ---: | ---: |
| $1.827 \mathrm{e}-04$ | $1.70 \mathrm{e}-03$ | $3.48 \mathrm{e}-02$ |
| $1.826 \mathrm{e}-04$ | $1.17 \mathrm{e}-03$ | $1.44 \mathrm{e}-02$ |
| $1.824 \mathrm{e}-04$ | $1.34 \mathrm{e}-04$ | $1.82 \mathrm{e}-04$ |
| $1.823 \mathrm{e}-04$ | $1.01 \mathrm{e}-06$ | $1.17 \mathrm{e}-08$ |

A few points in the derivation of the BFGS and DFP methods merit further discussion. Note that the minimization problem that gives rise to the BFGS update formula does not explicitly require the updated Hessian approximation to be positive definite. It is easy to show, however, that $\mathbf{H}\_{k+1}$ will be positive definite whenever $\mathbf{H}\_k$ is positive definite, by using the following argument. First, note that $\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k$ is positive, so that the updating formula \eqref{eq:bfgs\_update} is well-defined. For any nonzero vector $\mathbf{z}$, we have

$$
\mathbf{z}^{\mathrm{T}} \mathbf{H}\_{k+1} \mathbf{z}=\mathbf{w}^{\mathrm{T}} \mathbf{H}\_k \mathbf{w}+\rho\_k\left(\mathbf{z}^{\mathrm{T}} \mathbf{s}\_k\right)^{2} \geq 0
$$

where we have defined $\mathbf{w}=\mathbf{z}-\rho\_k \mathbf{y}\_k\left(\mathbf{s}\_k^{\mathrm{T}} \mathbf{z}\right)$. The right hand side can be zero only if $\mathbf{s}\_k^{\mathrm{T}} \mathbf{z}=0$, but in this case $\mathbf{w}=\mathbf{z} \neq \mathbf{0}$, which implies that the first term is greater than zero. Therefore, $\mathbf{H}\_{k+1}$ is positive definite.

In order to obtain quasi-Newton updating formulae that are invariant to changes in the variables, it is necessary that the objectives be also invariant. The choice of the weighting matrices $\mathbf{W}$ used to define the norms ensures that this condition holds. Many other choices of the weighting matrix $\mathbf{W}$ are possible, each one of them giving a different update formula. However, despite intensive searches, no formula has been found that is significantly more effective than BFGS.

The BFGS method has many interesting properties when applied to quadratic functions. We will discuss these properties later on, in the more general context of the Broyden family of updating formulae, of which BFGS is a special case.

It is reasonable to ask whether there are situations in which the updating formula such as \eqref{eq:bfgs\_update} can produce bad results. If at some iteration the matrix $\mathbf{H}\_k$ becomes a very poor approximation, is there any hope of correcting it? If, for example, the inner product $\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k$ is tiny (but positive), then it follows from \eqref{eq:bfgs\_update} that $\mathbf{H}\_{k+1}$ becomes huge. Is this behavior reasonable? A related question concerns the rounding errors that occur in finite-precision implementation of these methods. Can these errors grow to the point of erasing all useful information in the quasi-Newton approximate matrix?

These questions have been studied analytically and experimentally, and it is now known that the BFGS formula has very effective self-correcting properties. If the matrix $\mathbf{H}\_k$ incorrectly estimates the curvature in the objective function, and if this bad estimate slows down the iteration, then the Hessian approximation will tend to correct itself within a few steps. It is also known that the DFP method is less effective in correcting bad Hessian approximations; this property is believed to be the reason for its poorer practical performance. The self-correcting properties of BFGS hold only when an adequate line search is performed. In particular, the Wolfe line search conditions ensure that the gradients are sampled at points that allow the model to capture appropriate curvature information.

It is interesting to note that the DFP and BFGS updating formulae are duals of each other, in the sense that one can be obtained from the other by the interchanges $\mathbf{s} \leftrightarrow \mathbf{y}$, $\mathbf{B} \leftrightarrow \mathbf{H}$. This symmetry is not surprising, given the manner in which we derived these methods above.

### Implementation

A few details and enhancements need to be added to Algorithm 8.1 to produce an efficient implementation. The line search, which should satisfy either the Wolfe conditions or the strong Wolfe conditions, should always try the step length $\alpha\_k=1$ first, because this step length will eventually always be accepted (under certain conditions), thereby producing superlinear convergence of the overall algorithm. Computational observations strongly suggest that it is more economical, in terms of function evaluations, to perform a fairly inaccurate line search. The values $c\_1=10^{-4}$ and $c\_2=0.9$ are commonly used.

As mentioned earlier, the initial matrix $\mathbf{H}\_0$ often is set to some multiple $\beta \mathbf{I}$ of the identity, but there is no good general strategy for choosing $\beta$. If $\beta$ is "too large," so that the first step $\mathbf{p}\_0=-\beta \mathbf{g}\_0$ is too long, many function evaluations may be required to find a suitable value for the step length $\alpha\_0$. Some software asks the user to prescribe a value $\delta$ for the norm of the first step, and then set $\mathbf{H}\_0=\delta\\|\mathbf{g}\_0\\|^{-1} \mathbf{I}$ to achieve this norm.

A heuristic that is often quite effective is to scale the starting matrix after the first step has been computed but before the first BFGS update is performed. We change the provisional value $\mathbf{H}\_0=\mathbf{I}$ by setting

$$
\mathbf{H}\_0 \leftarrow \frac{\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k}{\mathbf{y}\_k^{\mathrm{T}} \mathbf{y}\_k} \mathbf{I},
$$

before applying the update \eqref{eq:bfgs\_update} to obtain $\mathbf{H}\_1$. This formula attempts to make the size of $\mathbf{H}\_0$ similar to that of $\left[\nabla^{2} f\left(\mathbf{x}\_0\right)\right]^{-1}$, in the following sense. Assuming that the average Hessian is positive definite, there exists a square root $\overline{\mathbf{G}}\_k^{1 / 2}$ satisfying $\overline{\mathbf{G}}\_k=\overline{\mathbf{G}}\_k^{1 / 2} \overline{\mathbf{G}}\_k^{1 / 2}$. Therefore, by defining $\mathbf{z}\_k=\overline{\mathbf{G}}\_k^{1 / 2} \mathbf{s}\_k$ and using the relation $\mathbf{y}\_k=\overline{\mathbf{G}}\_k \mathbf{s}\_k$, we have

$$
\frac{\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k}{\mathbf{y}\_k^{\mathrm{T}} \mathbf{y}\_k}=\frac{\left(\overline{\mathbf{G}}\_k^{1 / 2} \mathbf{s}\_k\right)^{\mathrm{T}} \overline{\mathbf{G}}\_k^{1 / 2} \mathbf{s}\_k}{\left(\overline{\mathbf{G}}\_k^{1 / 2} \mathbf{s}\_k\right)^{\mathrm{T}} \overline{\mathbf{G}}\_k \overline{\mathbf{G}}\_k^{1 / 2} \mathbf{s}\_k}=\frac{\mathbf{z}\_k^{\mathrm{T}} \mathbf{z}\_k}{\mathbf{z}\_k^{\mathrm{T}} \overline{\mathbf{G}}\_k \mathbf{z}\_k}
$$

The reciprocal of this expression is an approximation to one of the eigenvalues of $\overline{\mathbf{G}}\_k$, which in turn is close to an eigenvalue of $\nabla^{2} f\left(\mathbf{x}\_k\right)$. Hence, the quotient itself approximates an eigenvalue of $\left[\nabla^{2} f\left(\mathbf{x}\_k\right)\right]^{-1}$. Other scaling factors can be used, but the one presented here appears to be the most successful in practice.

We gave an update formula for a BFGS method that works with the Hessian approximation $\mathbf{B}\_k$ instead of the the inverse Hessian approximation $\mathbf{H}\_k$. An efficient implementation of this approach does not store $\mathbf{B}\_k$ explicitly, but rather the Cholesky factorization $\mathbf{L}\_k \mathbf{D}\_k \mathbf{L}\_k^{\mathrm{T}}$ of this matrix. A formula that updates the factors $\mathbf{L}\_k$ and $\mathbf{D}\_k$ directly in $O\left(n^{2}\right)$ operations can be derived from \eqref{eq:bfgs\_b\_update}. Since the linear system $\mathbf{B}\_k \mathbf{p}\_k=-\nabla f\_k$ also can be solved in $O\left(n^{2}\right)$ operations (by performing triangular substitutions with $\mathbf{L}\_k$ and $\mathbf{L}\_k^{\mathrm{T}}$ and a diagonal substitution with $\mathbf{D}\_k$ ), the total cost is quite similar to the variant described in Algorithm 8.1. A potential advantage of this alternative strategy is that it gives us the option of modifying diagonal elements in the $\mathbf{D}\_k$ factor if they are not sufficiently large, to prevent instability when we divide by these elements during the calculation of $\mathbf{p}\_k$. However, computational experience suggests no real advantages for this variant, and we prefer the simpler strategy of Algorithm 8.1.

The performance of the BFGS method can degrade if the line search is not based on the Wolfe conditions. For example, some software implements an Armijo backtracking line search: The unit step length $\alpha\_k=1$ is tried first and is successively decreased until the sufficient decrease condition is satisfied. For this strategy, there is no guarantee that the curvature condition $\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k>0$ will be satisfied by the chosen step, since a step length greater than 1 may be required to satisfy this condition. To cope with this shortcoming, some implementations simply skip the BFGS update by setting $\mathbf{H}\_{k+1}=\mathbf{H}\_k$ when $\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k$ is negative or too close to zero. This approach is not recommended, because the updates may be skipped much too often to allow $\mathbf{H}\_k$ to capture important curvature information for the objective function $f$.

### The SR1 method

In the BFGS and DFP updating formulae, the updated matrix $\mathbf{B}\_{k+1}$ (or $\mathbf{H}\_{k+1}$ ) differs from its predecessor $\mathbf{B}\_k$ (or $\mathbf{H}\_k$ ) by a rank-2 matrix. In fact, as we now show, there is a simpler rank-1 update that maintains symmetry of the matrix and allows it to satisfy the secant equation. Unlike the rank-two update formulae, this symmetric-rank-1, or SR1, update does not guarantee that the updated matrix maintains positive definiteness. Good numerical results have been obtained with algorithms based on SR1, so we derive it here and investigate its properties.

The symmetric rank-1 update has the general form

$$
\mathbf{B}\_{k+1}=\mathbf{B}\_k+\sigma \mathbf{v} \mathbf{v}^{\mathrm{T}}
$$

where $\sigma$ is either +1 or -1 , and $\sigma$ and $\mathbf{v}$ are chosen so that $\mathbf{B}\_{k+1}$ satisfies the secant equation, that is, $\mathbf{y}\_k=\mathbf{B}\_{k+1} \mathbf{s}\_k$. By substituting into this equation, we obtain

$$
\mathbf{y}\_k=\mathbf{B}\_k \mathbf{s}\_k+\left[\sigma \mathbf{v}^{\mathrm{T}} \mathbf{s}\_k\right] \mathbf{v}
$$

Since the term in brackets is a scalar, we deduce that $\mathbf{v}$ must be a multiple of $\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k$, that is, $\mathbf{v}=\delta\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right)$ for some scalar $\delta$. By substituting this form of $\mathbf{v}$ into the equation, we obtain

$$
\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right)=\sigma \delta^{2}\left[\mathbf{s}\_k^{\mathrm{T}}\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right)\right]\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right)
$$

and it is clear that this equation is satisfied if (and only if) we choose the parameters $\delta$ and $\sigma$ to be

$$
\sigma=\operatorname{sign}\left[\mathbf{s}\_k^{\mathrm{T}}\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right)\right], \quad \delta= \pm\left|\mathbf{s}\_k^{\mathrm{T}}\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right)\right|^{-1 / 2} .
$$

Hence, we have shown that the only symmetric rank-1 updating formula that satisfies the secant equation is given by

\begin{equation}
\text { (SR1) } \quad \mathbf{B}\_{k+1}=\mathbf{B}\_k+\frac{\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right)\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right)^{\mathrm{T}}}{\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right)^{\mathrm{T}} \mathbf{s}\_k}
\label{eq:sr1\_update}
\end{equation}

By applying the Sherman-Morrison formula, we obtain the corresponding update formula for the inverse Hessian approximation $\mathbf{H}\_k$ :

\begin{equation}
\mathbf{H}\_{k+1}=\mathbf{H}\_k+\frac{\left(\mathbf{s}\_k-\mathbf{H}\_k \mathbf{y}\_k\right)\left(\mathbf{s}\_k-\mathbf{H}\_k \mathbf{y}\_k\right)^{\mathrm{T}}}{\left(\mathbf{s}\_k-\mathbf{H}\_k \mathbf{y}\_k\right)^{\mathrm{T}} \mathbf{y}\_k}
\label{eq:sr1\_h\_update}
\end{equation}

This derivation is so simple that the SR1 formula has been rediscovered a number of times. It is easy to see that even if $\mathbf{B}\_k$ is positive definite, $\mathbf{B}\_{k+1}$ may not have this property; the same is, of course, true of $\mathbf{H}\_k$. This observation was considered a major drawback in the early days of nonlinear optimization when only line search iterations were used. However, with the advent of trust-region methods, the SR1 updating formula has proved to be quite useful, and its ability to generate indefinite Hessian approximations can actually be regarded as one of its chief advantages.

The main drawback of SR1 updating is that the denominator in \eqref{eq:sr1\_update} or \eqref{eq:sr1\_h\_update} can vanish. In fact, even when the objective function is a convex quadratic, there may be steps on which there is no symmetric rank-1 update that satisfies the secant equation. It pays to reexamine the derivation above in the light of this observation.

By reasoning in terms of $\mathbf{B}\_k$ (similar arguments can be applied to $\mathbf{H}\_k$ ), we see that there are three cases:

1. If $\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right)^{\mathrm{T}} \mathbf{s}\_k \neq 0$, then the arguments above show that there is a unique rank-one updating formula satisfying the secant equation, and that it is given by \eqref{eq:sr1\_update}.
2. If $\mathbf{y}\_k=\mathbf{B}\_k \mathbf{s}\_k$, then the only updating formula satisfying the secant equation is simply $\mathbf{B}\_{k+1}=\mathbf{B}\_k$.
3. If $\mathbf{y}\_k \neq \mathbf{B}\_k \mathbf{s}\_k$ and $\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right)^{\mathrm{T}} \mathbf{s}\_k=0$, then the derivation shows that there is no symmetric rank-one updating formula satisfying the secant equation.

The last case clouds an otherwise simple and elegant derivation, and suggests that numerical instabilities and even breakdown of the method can occur. It suggests that rank-one updating does not provide enough freedom to develop a matrix with all the desired characteristics, and that a rank-two correction is required. This reasoning leads us back to the BFGS method, in which positive definiteness (and thus nonsingularity) of all Hessian approximations is guaranteed.

Nevertheless, we are interested in the SR1 formula for the following reasons.
(i) A simple safeguard seems to adequately prevent the breakdown of the method and the occurrence of numerical instabilities.
(ii) The matrices generated by the SR1 formula tend to be very good approximations of the Hessian matrix—often better than the BFGS approximations.
(iii) In quasi-Newton methods for constrained problems, or in methods for partially separable functions, it may not be possible to impose the curvature condition $\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k>0$, and thus BFGS updating is not recommended. Indeed, in these two settings, indefinite Hessian approximations are desirable insofar as they reflect indefiniteness in the true Hessian.

We now introduce a strategy to prevent the SR1 method from breaking down. It has been observed in practice that SR1 performs well simply by skipping the update if the denominator is small. More specifically, the update \eqref{eq:sr1\_update} is applied only if

\begin{equation}
\left|\mathbf{s}\_k^{\mathrm{T}}\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right)\right| \geq r\\|\mathbf{s}\_k\\|\\|\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\\|,
\label{eq:sr1\_safeguard}
\end{equation}

where $r \in(0,1)$ is a small number, say $r=10^{-8}$. If \eqref{eq:sr1\_safeguard} does not hold, we set $\mathbf{B}\_{k+1}=\mathbf{B}\_k$. Most implementations of the SR1 method use a skipping rule of this kind.

Why do we advocate skipping of updates for the SR1 method, when in the previous section we discouraged this strategy in the case of BFGS? The two cases are quite different. The condition $\mathbf{s}\_k^{\mathrm{T}}\left(\mathbf{y}\_k-\mathbf{B}\_k \mathbf{s}\_k\right) \approx 0$ occurs infrequently, since it requires certain vectors to be aligned in a specific way. When it does occur, skipping the update appears to have no negative effects on the iteration. This is not surprising, since the skipping condition implies that $\mathbf{s}\_k^{\mathrm{T}} \overline{\mathbf{G}} \mathbf{s}\_k \approx \mathbf{s}\_k^{\mathrm{T}} \mathbf{B}\_k \mathbf{s}\_k$, where $\overline{\mathbf{G}}$ is the average Hessian over the last step—meaning that the curvature of $\mathbf{B}\_k$ along $\mathbf{s}\_k$ is already correct. In contrast, the curvature condition $\mathbf{s}\_k^{\mathrm{T}} \mathbf{y}\_k \geq 0$ required for BFGS updating may easily fail if the line search does not impose the Wolfe conditions (e.g., if the step is not long enough), and therefore skipping the BFGS update can occur often and can degrade the quality of the Hessian approximation.

We now give a formal description of an SR1 method using a trust-region framework. We prefer it over a line search framework because it does not require us to modify the Hessian approximations to make them sufficiently positive definite.

Algorithm 8.2 (SR1 Trust-Region Method).

Given starting point $\mathbf{x}\_0$, initial Hessian approximation $\mathbf{B}\_0$,
trust-region radius $\Delta\_0$, convergence tolerance $\epsilon>0$,
parameters $\eta \in\left(0,10^{-3}\right)$ and $r \in(0,1)$;
$k \leftarrow 0$;
while $\\|\nabla f\_k\\|>\epsilon$;
Compute $\mathbf{s}\_k$ by solving the subproblem

$$
\min \_{\mathbf{s}} \nabla f\_k^{\mathrm{T}} \mathbf{s}+\frac{1}{2} \mathbf{s}^{\mathrm{T}} \mathbf{B}\_k \mathbf{s} \quad \text { subject to }\\|\mathbf{s}\\| \leq \Delta\_k .
$$

Compute

$$
\begin{aligned}
\mathbf{y}\_k & =\nabla f\left(\mathbf{x}\_k+\mathbf{s}\_k\right)-\nabla f\_k \\\\
\text { ared } & =f\_k-f\left(\mathbf{x}\_k+\mathbf{s}\_k\right) \quad \text { (actual reduction) } \\\\
\text { pred } & =-\left(\nabla f\_k^{\mathrm{T}} \mathbf{s}\_k+\frac{1}{2} \mathbf{s}\_k^{\mathrm{T}} \mathbf{B}\_k \mathbf{s}\_k\right) \quad \text { (predicted reduction) }
\end{aligned}
$$

if ared/pred > $\eta$
$\mathbf{x}\_{k+1}=\mathbf{x}\_k+\mathbf{s}\_k$
else
$\mathbf{x}\_{k+1}=\mathbf{x}\_k ;$
end (if)
if ared/pred > 0.75

        if $\|\mathbf{s}_k\| \leq 0.8 \Delta_k$
            $\Delta_{k+1}=\Delta_k$
        else
            $\Delta_{k+1}=2 \Delta_k ;$
        end (if)
    elseif $0.1 \leq$ ared/pred $\leq 0.75$
        $\Delta_{k+1}=\Delta_k$
    else
        $\Delta_{k+1}=0.5 \Delta_k ;$
    end (if)
    if \eqref{eq:sr1\_safeguard} holds
        Use \eqref{eq:sr1\_update} to compute $\mathbf{B}_{k+1}$ (even if $\mathbf{x}_{k+1}=\mathbf{x}_k$ )
    else
        $\mathbf{B}_{k+1} \leftarrow \mathbf{B}_k ;$
    end (if)
    $k \leftarrow k+1$;
end (while)

This algorithm has the typical form of a trust region method. For concreteness we have specified a particular strategy for updating the trust region radius, but other heuristics can be used instead.

To obtain a fast rate of convergence, it is important for the matrix $\mathbf{B}\_k$ to be updated even along a failed direction $\mathbf{s}\_k$. The fact that the step was poor indicates that $\mathbf{B}\_k$ is a poor approximation of the true Hessian in this direction. Unless the quality of the approximation is improved, steps along similar directions could be generated on later iterations, and repeated rejection of such steps could prevent superlinear convergence.

### Properties of SR1 updating

One of the main advantages of SR1 updating is its ability to generate very good Hessian approximations. We demonstrate this property by first examining a quadratic function. For functions of this type, the choice of step length does not affect the update, so to examine the effect of the updates, we can assume for simplicity a uniform step length of 1 , that is,

$$
\mathbf{p}\_k=-\mathbf{H}\_k \nabla f\_k, \quad \mathbf{x}\_{k+1}=\mathbf{x}\_k+\mathbf{p}\_k
$$

It follows that $\mathbf{p}\_k=\mathbf{s}\_k$.

{{<theorem "SR1 finite termination" sr1_finite>}}
Suppose that $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ is the strongly convex quadratic function $f(\mathbf{x})=\mathbf{b}^{\mathrm{T}} \mathbf{x}+\frac{1}{2} \mathbf{x}^{\mathrm{T}} \mathbf{A} \mathbf{x}$, where $\mathbf{A}$ is symmetric positive definite. Then for any starting point $\mathbf{x}\_0$ and any symmetric starting matrix $\mathbf{H}\_0$, the iterates $\\{\mathbf{x}\_k\\}$ generated by the SR1 method converge to the minimizer in at most $n$ steps, provided that $\left(\mathbf{s}\_k-\mathbf{H}\_k \mathbf{y}\_k\right)^{\mathrm{T}} \mathbf{y}\_k \neq 0$ for all $k$. Moreover, if $n$ steps are performed, and if the search directions $\mathbf{p}\_i$ are linearly independent, then $\mathbf{H}\_n=\mathbf{A}^{-1}$.
{{</theorem>}}

{{<proof>}}
Because of our assumption $\left(\mathbf{s}\_k-\mathbf{H}\_k \mathbf{y}\_k\right)^{\mathrm{T}} \mathbf{y}\_k \neq 0$, the SR1 update is always well-defined. We start by showing inductively that

$$
\mathbf{H}\_k \mathbf{y}\_j=\mathbf{s}\_j \quad \text { for } \quad j=0, \ldots, k-1
$$

In other words, we claim that the secant equation is satisfied not only along the most recent search direction, but along all previous directions.

By definition, the SR1 update satisfies the secant equation, so we have $\mathbf{H}\_1 \mathbf{y}\_0=\mathbf{s}\_0$. Let us now assume that this relation holds for some value $k>1$ and show that it holds also for $k+1$. From this assumption, we have

$$
\left(\mathbf{s}\_k-\mathbf{H}\_k \mathbf{y}\_k\right)^{\mathrm{T}} \mathbf{y}\_j=\mathbf{s}\_k^{\mathrm{T}} \mathbf{y}\_j-\mathbf{y}\_k^{\mathrm{T}}\left(\mathbf{H}\_k \mathbf{y}\_j\right)=\mathbf{s}\_k^{\mathrm{T}} \mathbf{y}\_j-\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_j=0, \quad \text { for all } j<k,
$$

where the last equality follows because $\mathbf{y}\_i=\mathbf{A} \mathbf{s}\_i$ for the quadratic function we are considering here. By using this result and the induction hypothesis in \eqref{eq:sr1\_h\_update}, we have

$$
\mathbf{H}\_{k+1} \mathbf{y}\_j=\mathbf{H}\_k \mathbf{y}\_j=\mathbf{s}\_j, \quad \text { for all } j<k
$$

Since $\mathbf{H}\_{k+1} \mathbf{y}\_k=\mathbf{s}\_k$ by the secant equation, we have shown that the relation holds when $k$ is replaced by $k+1$. By induction, then, this relation holds for all $k$.

If the algorithm performs $n$ steps and if these steps $\\{\mathbf{s}\_j\\}$ are linearly independent, we have

$$
\mathbf{s}\_j=\mathbf{H}\_n \mathbf{y}\_j=\mathbf{H}\_n \mathbf{A} \mathbf{s}\_j, \quad \text { for } j=0, \ldots, n-1
$$

It follows that $\mathbf{H}\_n \mathbf{A}=\mathbf{I}$, that is, $\mathbf{H}\_n=\mathbf{A}^{-1}$. Therefore, the step taken at $\mathbf{x}\_n$ is the Newton step, and so the next iterate $\mathbf{x}\_{n+1}$ will be the solution, and the algorithm terminates.

Consider now the case in which the steps become linearly dependent. Suppose that $\mathbf{s}\_k$ is a linear combination of the previous steps, that is,

$$
\mathbf{s}\_k=\xi\_0 \mathbf{s}\_0+\cdots+\xi\_{k-1} \mathbf{s}\_{k-1}
$$

for some scalars $\xi\_i$. From the relations above we have that

$$
\begin{aligned}
\mathbf{H}\_k \mathbf{y}\_k & =\mathbf{H}\_k \mathbf{A} \mathbf{s}\_k \\\\
& =\xi\_0 \mathbf{H}\_k \mathbf{A} \mathbf{s}\_0+\cdots+\xi\_{k-1} \mathbf{H}\_k \mathbf{A} \mathbf{s}\_{k-1} \\\\
& =\xi\_0 \mathbf{H}\_k \mathbf{y}\_0+\cdots+\xi\_{k-1} \mathbf{H}\_k \mathbf{y}\_{k-1} \\\\
& =\xi\_0 \mathbf{s}\_0+\cdots+\xi\_{k-1} \mathbf{s}\_{k-1} \\\\
& =\mathbf{s}\_k
\end{aligned}
$$

Since $\mathbf{y}\_k=\nabla f\_{k+1}-\nabla f\_k$ and since $\mathbf{s}\_k=\mathbf{p}\_k=-\mathbf{H}\_k \nabla f\_k$, we have that

$$
\mathbf{H}\_k\left(\nabla f\_{k+1}-\nabla f\_k\right)=-\mathbf{H}\_k \nabla f\_k
$$

which, by the nonsingularity of $\mathbf{H}\_k$, implies that $\nabla f\_{k+1}=\mathbf{0}$. Therefore, $\mathbf{x}\_{k+1}$ is the solution point.
{{</proof>}}

The relation from the theorem shows that when $f$ is quadratic, the secant equation is satisfied along all previous search directions, regardless of how the line search is performed. A result like this can be established for BFGS updating only under the restrictive assumption that the line search is exact.

For general nonlinear functions, the SR1 update continues to generate good Hessian approximations under certain conditions.

{{<theorem "SR1 superlinear convergence" sr1_superlinear>}}
Suppose that $f$ is twice continuously differentiable, and that its Hessian is bounded and Lipschitz continuous in a neighborhood of a point $\mathbf{x}^\star$. Let $\\{\mathbf{x}\_k\\}$ be any sequence of iterates such that $\mathbf{x}\_k \rightarrow \mathbf{x}^\star$ for some $\mathbf{x}^\star \in \mathbb{R}^{n}$. Suppose in addition that the inequality \eqref{eq:sr1\_safeguard} holds for all $k$, for some $r \in(0,1)$, and that the steps $\mathbf{s}\_k$ are uniformly linearly independent. Then the matrices $\mathbf{B}\_k$ generated by the SR1 updating formula satisfy

$$
\lim \_{k \rightarrow \infty}\\|\mathbf{B}\_k-\nabla^{2} f\left(\mathbf{x}^\star\right)\\|=0
$$
{{</theorem>}}

The term "uniformly linearly independent steps" means, roughly speaking, that the steps do not tend to fall in a subspace of dimension less than $n$. This assumption is usually, but not always, satisfied in practice.

### The Broyden class

So far, we have described the BFGS, DFP, and SR1 quasi-Newton updating formulae, but there are many others. Of particular interest is the Broyden class, a family of updates specified by the following general formula:

\begin{equation}
\mathbf{B}\_{k+1}=\mathbf{B}\_k-\frac{\mathbf{B}\_k \mathbf{s}\_k \mathbf{s}\_k^{\mathrm{T}} \mathbf{B}\_k}{\mathbf{s}\_k^{\mathrm{T}} \mathbf{B}\_k \mathbf{s}\_k}+\frac{\mathbf{y}\_k \mathbf{y}\_k^{\mathrm{T}}}{\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k}+\phi\_k\left(\mathbf{s}\_k^{\mathrm{T}} \mathbf{B}\_k \mathbf{s}\_k\right) \mathbf{v}\_k \mathbf{v}\_k^{\mathrm{T}},
\label{eq:broyden\_class}
\end{equation}

where $\phi\_k$ is a scalar parameter and

$$
\mathbf{v}\_k=\left[\frac{\mathbf{y}\_k}{\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k}-\frac{\mathbf{B}\_k \mathbf{s}\_k}{\mathbf{s}\_k^{\mathrm{T}} \mathbf{B}\_k \mathbf{s}\_k}\right] .
$$

The BFGS and DFP methods are members of the Broyden class—we recover BFGS by setting $\phi\_k=0$ and DFP by setting $\phi\_k=1$ in \eqref{eq:broyden\_class}. We can therefore rewrite \eqref{eq:broyden\_class} as a "linear combination" of these two methods, that is,

$$
\mathbf{B}\_{k+1}=\left(1-\phi\_k\right) \mathbf{B}\_{k+1}^{\mathrm{BFGS}}+\phi\_k \mathbf{B}\_{k+1}^{\mathrm{DFP}} .
$$

This relationship indicates that all members of the Broyden class satisfy the secant equation, since the BFGS and DFP matrices themselves satisfy this equation. Also, since BFGS and DFP updating preserve positive definiteness of the Hessian approximations when $\mathbf{s}\_k^{\mathrm{T}} \mathbf{y}\_k>0$, this relation implies that the same property will hold for the Broyden family if $0 \leq \phi\_k \leq 1$.

Much attention has been given to the so-called restricted Broyden class, which is obtained by restricting $\phi\_k$ to the interval $[0,1]$. It enjoys the following property when applied to quadratic functions. Since the analysis is independent of the step length, we assume for simplicity that each iteration has the form

$$
\mathbf{p}\_k=-\mathbf{B}\_k^{-1} \nabla f\_k, \quad \mathbf{x}\_{k+1}=\mathbf{x}\_k+\mathbf{p}\_k
$$

{{<theorem "Broyden class monotonicity" broyden_monotonic>}}
Suppose that $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ is the strongly convex quadratic function $f(\mathbf{x})=\mathbf{b}^{\mathrm{T}} \mathbf{x}+\frac{1}{2} \mathbf{x}^{\mathrm{T}} \mathbf{A} \mathbf{x}$, where $\mathbf{A}$ is symmetric and positive definite. Let $\mathbf{x}\_0$ be any starting point for the iteration above and $\mathbf{B}\_0$ be any symmetric positive definite starting matrix, and suppose that the matrices $\mathbf{B}\_k$ are updated by the Broyden formula \eqref{eq:broyden\_class} with $\phi\_k \in[0,1]$. Define $\lambda\_1^k \leq \lambda\_2^k \leq \cdots \leq \lambda\_n^k$ to be the eigenvalues of the matrix

$$
\mathbf{A}^{\frac{1}{2}} \mathbf{B}\_k^{-1} \mathbf{A}^{\frac{1}{2}}
$$

Then for all $k$, we have

$$
\min \left\\{\lambda\_i^k, 1\right\\} \leq \lambda\_i^{k+1} \leq \max \left\\{\lambda\_i^k, 1\right\\}, \quad i=1, \ldots, n .
$$

Moreover, the property above does not hold if the Broyden parameter $\phi\_k$ is chosen outside the interval $[0,1]$.
{{</theorem>}}

Let us discuss the significance of this result. If the eigenvalues $\lambda\_i^k$ of the matrix are all 1, then the quasi-Newton approximation $\mathbf{B}\_k$ is identical to the Hessian $\mathbf{A}$ of the quadratic objective function. This situation is the ideal one, so we should be hoping for these eigenvalues to be as close to 1 as possible. In fact, the relation tells us that the eigenvalues $\\{\lambda\_i^k\\}$ converge monotonically (but not strictly monotonically) to 1. Suppose, for example, that at iteration $k$ the smallest eigenvalue is $\lambda\_1^k=0.7$. Then the theorem tells us that at the next iteration $\lambda\_1^{k+1} \in[0.7,1]$. We cannot be sure that this eigenvalue has actually gotten closer to 1 , but it is reasonable to expect that it has. In contrast, the first eigenvalue can become smaller than 0.7 if we allow $\phi\_k$ to be outside $[0,1]$. Significantly, the result holds even if the linear searches are not exact.

Although {{<theoremref broyden_monotonic>}} seems to suggest that the best update formulas belong to the restricted Broyden class, the situation is not at all clear. Some analysis and computational testing suggest that algorithms that allow $\phi\_k$ to be negative (in a strictly controlled manner) may in fact be superior to the BFGS method. The SR1 formula is a case in point: It is a member of the Broyden class, obtained by setting

$$
\phi\_k=\frac{\mathbf{s}\_k^{\mathrm{T}} \mathbf{y}\_k}{\mathbf{s}\_k^{\mathrm{T}} \mathbf{y}\_k-\mathbf{s}\_k^{\mathrm{T}} \mathbf{B}\_k \mathbf{s}\_k},
$$

but it does not belong to the restricted Broyden class, because this value of $\phi\_k$ may fall outside the interval $[0,1]$.

We complete our discussion of the Broyden class by informally stating some of its main properties.

### Properties of the Broyden class

We have noted already that if $\mathbf{B}\_k$ is positive definite, $\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k>0$, and $\phi\_k \geq 0$, then $\mathbf{B}\_{k+1}$ is also positive definite if a restricted Broyden class update, with $\phi\_k \in[0,1]$, is used. We would like to determine more precisely the range of values of $\phi\_k$ that preserve positive definiteness.

The last term in \eqref{eq:broyden\_class} is a rank-one correction, which by the interlacing eigenvalue theorem decreases the eigenvalues of the matrix when $\phi\_k$ is negative. As we decrease $\phi\_k$, this matrix eventually becomes singular and then indefinite. A little computation shows that $\mathbf{B}\_{k+1}$ is singular when $\phi\_k$ has the value

$$
\phi\_k^c=\frac{1}{1-\mu\_k}
$$

where

$$
\mu\_k=\frac{\left(\mathbf{y}\_k^{\mathrm{T}} \mathbf{B}\_k^{-1} \mathbf{y}\_k\right)\left(\mathbf{s}\_k^{\mathrm{T}} \mathbf{B}\_k \mathbf{s}\_k\right)}{\left(\mathbf{y}\_k^{\mathrm{T}} \mathbf{s}\_k\right)^{2}}
$$

By applying the Cauchy-Schwarz inequality we see that $\mu\_k \geq 1$ and therefore $\phi\_k^c \leq 0$. Hence, if the initial Hessian approximation $\mathbf{B}\_0$ is symmetric and positive definite, and if $\mathbf{s}\_k^{\mathrm{T}} \mathbf{y}\_k>0$ and $\phi\_k>\phi\_k^c$ for each $k$, then all the matrices $\mathbf{B}\_k$ generated by Broyden's formula \eqref{eq:broyden\_class} remain symmetric and positive definite.

When the line search is exact, all methods in the Broyden class with $\phi\_k \geq \phi\_k^c$ generate the same sequence of iterates. This result applies to general nonlinear functions and is based on the observation that when all the line searches are exact, the directions generated by Broyden-class methods differ only in their lengths. The line searches identify the same minima along the chosen search direction, though the values of the line search parameter may differ because of the different scaling.

The Broyden class has several remarkable properties when applied with exact line searches to quadratic functions. We state some of these properties in the next theorem, whose proof is omitted.

{{<theorem "Broyden class quadratic properties" broyden_quadratic>}}
Suppose that a method in the Broyden class is applied to a strongly convex quadratic function $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$, where $\mathbf{x}\_0$ is the starting point and $\mathbf{B}\_0$ is any symmetric and positive definite matrix. Assume that $\alpha\_k$ is the exact step length and that $\phi\_k \geq \phi\_k^c$ for all $k$. Then the following statements are true.
(i) The iterates converge to the solution in at most $n$ iterations.
(ii) The secant equation is satisfied for all previous search directions, that is,

$$
\mathbf{B}\_k \mathbf{s}\_j=\mathbf{y}\_j, \quad j=k-1, \ldots, 1
$$

(iii) If the starting matrix is $\mathbf{B}\_0=\mathbf{I}$, then the iterates are identical to those generated by the conjugate gradient method. In particular, the search directions are conjugate, that is,

$$
\mathbf{s}\_i^{\mathrm{T}} \mathbf{A} \mathbf{s}\_j=0 \quad \text { for } i \neq j
$$

where $\mathbf{A}$ is the Hessian of the quadratic function.
(iv) If $n$ iterations are performed, we have $\mathbf{B}\_{n+1}=\mathbf{A}$.
{{</theorem>}}

Note that parts (i), (ii), and (iv) of this result echo the statement and proof of {{<theoremref sr1_finite>}}, where similar results were derived for the SR1 update formula.

In fact, we can generalize {{<theoremref broyden_quadratic>}} slightly: It continues to hold if the Hessian approximations remain nonsingular but not necessarily positive definite. (Hence, we could allow $\phi\_k$ to be smaller than $\phi\_k^c$, provided that the chosen value did not produce a singular updated matrix.) We also can generalize point (iii) as follows: If the starting matrix $\mathbf{B}\_0$ is not the identity matrix, then the Broyden-class method is identical to the preconditioned conjugate gradient method that uses $\mathbf{B}\_0$ as preconditioner.

We conclude by commenting that results like {{<theoremref broyden_quadratic>}} would appear to be mainly of theoretical interest, since the inexact line searches used in practical implementations of Broyden-class methods (and all other quasi-Newton methods) cause their performance to differ markedly. Nevertheless, this type of analysis guided most of the development of quasi-Newton methods.
