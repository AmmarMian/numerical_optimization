---
title: "4. Unconstrained optimization : linesearch"
weight: 4
chapter: 4
---

# Unconstrained optimization - Linesearch methods

All algorithms for unconstrained minimization require the user to supply a starting point, which we usually denote by $\mathbf{x}\_0$. The user with knowledge about the application and the data set may be in a good position to choose $\mathbf{x}\_0$ to be a reasonable estimate of the solution. Otherwise, the starting point must be chosen in some arbitrary manner.

Beginning at $\mathbf{x}\_0$, optimization algorithms generate a sequence of iterates $\left\\{\mathbf{x}\_k\right\\}\_{k=0}^{\infty}$ that terminate when either no more progress can be made or when it seems that a solution point has been approximated with sufficient accuracy. In deciding how to move from one iterate $\mathbf{x}\_k$ to the next, the algorithms use information about the function $f$ at $\mathbf{x}\_k$, and possibly also information from earlier iterates $\mathbf{x}\_0, \mathbf{x}\_1, \ldots, \mathbf{x}\_{k-1}$. They use this information to find a new iterate $\mathbf{x}\_{k+1}$ with a lower function value than $\mathbf{x}\_k$. (There exist nonmonotone algorithms that do not insist on a decrease in $f$ at every step, but even these algorithms require $f$ to be decreased after some prescribed number $m$ of iterations. That is, they enforce $f\left(\mathbf{x}\_k\right)<f\left(\mathbf{x}\_{k-m}\right)$.)

There are two fundamental strategies for moving from the current point $\mathbf{x}\_k$ to a new iterate $\mathbf{x}\_{k+1}$. Most of the algorithms described in this book follow one of these approaches.

## Two strategies: line search and trust region

In the line search strategy, the algorithm chooses a direction $\mathbf{p}\_k$ and searches along this direction from the current iterate $\mathbf{x}\_k$ for a new iterate with a lower function value. The distance to move along $\mathbf{p}\_k$ can be found by approximately solving the following one-dimensional minimization problem to find a step length $\alpha$ :

\begin{equation}
\min \_{\alpha>0} f\left(\mathbf{x}\_k+\alpha \mathbf{p}\_k\right)
\label{eq:line_search_min}
\end{equation}

By solving \eqref{eq:line_search_min} exactly, we would derive the maximum benefit from the direction $\mathbf{p}\_k$, but an exact minimization is expensive and unnecessary. Instead, the line search algorithm generates a limited number of trial step lengths until it finds one that loosely approximates the minimum of \eqref{eq:line_search_min}. At the new point a new search direction and step length are computed, and the process is repeated.

In the second algorithmic strategy, known as trust region, the information gathered about $f$ is used to construct a model function $m\_k$ whose behavior near the current point $\mathbf{x}\_k$ is similar to that of the actual objective function $f$. Because the model $m\_k$ may not be a good approximation of $f$ when $\mathbf{x}$ is far from $\mathbf{x}\_k$, we restrict the search for a minimizer of $m\_k$ to some region around $\mathbf{x}\_k$. In other words, we find the candidate step $\mathbf{p}$ by approximately
solving the following subproblem:

\begin{equation}
\min \_{\mathbf{p}} m\_k\left(\mathbf{x}\_k+\mathbf{p}\right), \quad \text { where } \mathbf{x}\_k+\mathbf{p} \text { lies inside the trust region. }
\label{eq:trust_region_subproblem}
\end{equation}

If the candidate solution does not produce a sufficient decrease in $f$, we conclude that the trust region is too large, and we shrink it and re-solve \eqref{eq:trust_region_subproblem}. Usually, the trust region is a ball defined by $\lVert\mathbf{p}\rVert\_2 \leq \Delta$, where the scalar $\Delta>0$ is called the trust-region radius. Elliptical and box-shaped trust regions may also be used.

The model $m\_k$ in \eqref{eq:trust_region_subproblem} is usually defined to be a quadratic function of the form

\begin{equation}
m\_k\left(\mathbf{x}\_k+\mathbf{p}\right)=f\_k+\mathbf{p}^{\mathrm{T}} \nabla f\_k+\frac{1}{2} \mathbf{p}^{\mathrm{T}} \mathbf{B}\_k \mathbf{p}
\label{eq:quadratic_model}
\end{equation}

where $f\_k$, $\nabla f\_k$, and $\mathbf{B}\_k$ are a scalar, vector, and matrix, respectively. As the notation indicates, $f\_k$ and $\nabla f\_k$ are chosen to be the function and gradient values at the point $\mathbf{x}\_k$, so that $m\_k$ and $f$ are in agreement to first order at the current iterate $\mathbf{x}\_k$. The matrix $\mathbf{B}\_k$ is either the Hessian $\nabla^2 f\_k$ or some approximation to it.

Suppose that the objective function is given by $f(\mathbf{x})=10\left(x\_2-x\_1^2\right)^2+\left(1-x\_1\right)^2$. At the point $\mathbf{x}\_k=(0,1)$ its gradient and Hessian are

$$
\nabla f\_k=\left[\begin{array}{c}
-2 \\\\
20
\end{array}\right], \quad \nabla^2 f\_k=\left[\begin{array}{cc}
-38 & 0 \\\\
0 & 20
\end{array}\right]
$$

Note that each time we decrease the size of the trust region after failure of a candidate iterate, the step from $\mathbf{x}\_k$ to the new candidate will be shorter, and it usually points in a different direction from the previous candidate. The trust-region strategy differs in this respect from line search, which stays with a single search direction.

In a sense, the line search and trust-region approaches differ in the order in which they choose the direction and distance of the move to the next iterate. Line search starts by fixing the direction $\mathbf{p}\_k$ and then identifying an appropriate distance, namely the step length $\alpha\_k$. In trust region, we first choose a maximum distance-the trust-region radius $\Delta\_k$-and then seek a direction and step that attain the best improvement possible subject to this distance constraint. If this step proves to be unsatisfactory, we reduce the distance measure $\Delta\_k$ and try again.

The line search approach is discussed in more detail in this lecture while the trust-region strategy, is left to the reader to study. 

## Search directions for line search methods

The steepest-descent direction $-\nabla f\_k$ is the most obvious choice for search direction for a line search method. It is intuitive; among all the directions we could move from $\mathbf{x}\_k$, it is the one along which $f$ decreases most rapidly. To verify this claim, we appeal again to Taylor's theorem, which tells us that for any search direction $\mathbf{p}$ and step-length parameter $\alpha$, we have

\begin{equation}
f\left(\mathbf{x}\_k+\alpha \mathbf{p}\right)=f\left(\mathbf{x}\_k\right)+\alpha \mathbf{p}^{\mathrm{T}} \nabla f\_k+\frac{1}{2} \alpha^2 \mathbf{p}^{\mathrm{T}} \nabla^2 f\left(\mathbf{x}\_k+t \mathbf{p}\right) \mathbf{p}, \quad \text { for some } t \in(0, \alpha)
\label{eq:taylor_expansion}
\end{equation}

The rate of change in $f$ along the direction $\mathbf{p}$ at $\mathbf{x}\_k$ is simply the coefficient of $\alpha$, namely, $\mathbf{p}^{\mathrm{T}} \nabla f\_k$. Hence, the unit direction $\mathbf{p}$ of most rapid decrease is the solution to the problem

\begin{equation}
\min \_{\mathbf{p}} \mathbf{p}^{\mathrm{T}} \nabla f\_k, \quad \text { subject to }\lVert\mathbf{p}\rVert=1
\label{eq:steepest_descent_problem}
\end{equation}

Since $\mathbf{p}^{\mathrm{T}} \nabla f\_k=\lVert\mathbf{p}\rVert\lVert\nabla f\_k\rVert \cos \theta$, where $\theta$ is the angle between $\mathbf{p}$ and $\nabla f\_k$, we have from $\lVert\mathbf{p}\rVert=1$ that $\mathbf{p}^{\mathrm{T}} \nabla f\_k=\lVert\nabla f\_k\rVert \cos \theta$, so the objective in \eqref{eq:steepest_descent_problem} is minimized when $\cos \theta$ takes on its minimum value of -1 at $\theta=\pi$ radians. In other words, the solution to \eqref{eq:steepest_descent_problem} is

$$
\mathbf{p}=-\nabla f\_k /\lVert\nabla f\_k\rVert
$$

as claimed. This direction is orthogonal to the contours of the function.

The steepest descent method is a line search method that moves along $\mathbf{p}\_k=-\nabla f\_k$ at every step. It can choose the step length $\alpha\_k$ in a variety of ways, as we will see in next chapter. One advantage of the steepest descent direction is that it requires calculation of the gradient $\nabla f\_k$ but not of second derivatives. However, it can be excruciatingly slow on difficult problems.

Line search methods may use search directions other than the steepest descent direction. In general, any descent direction-one that makes an angle of strictly less than $\pi / 2$ radians with $-\nabla f\_k$-is guaranteed to produce a decrease in $f$, provided that the step length is sufficiently small. We can verify this claim by using Taylor's theorem. From \eqref{eq:taylor_expansion}, we have that

$$
f\left(\mathbf{x}\_k+\epsilon \mathbf{p}\_k\right)=f\left(\mathbf{x}\_k\right)+\epsilon \mathbf{p}\_k^{\mathrm{T}} \nabla f\_k+O\left(\epsilon^2\right)
$$

When $\mathbf{p}\_k$ is a downhill direction, the angle $\theta\_k$ between $\mathbf{p}\_k$ and $\nabla f\_k$ has $\cos \theta\_k<0$, so that

$$
\mathbf{p}\_k^{\mathrm{T}} \nabla f\_k=\lVert\mathbf{p}\_k\rVert\lVert\nabla f\_k\rVert \cos \theta\_k<0
$$

It follows that $f\left(\mathbf{x}\_k+\epsilon \mathbf{p}\_k\right)<f\left(\mathbf{x}\_k\right)$ for all positive but sufficiently small values of $\epsilon$.

Another important search direction-perhaps the most important one of all-is the Newton direction. This direction is derived from the second-order Taylor series approximation to $f\left(\mathbf{x}\_k+\mathbf{p}\right)$, which is
$$
f\left(\mathbf{x}\_k+\mathbf{p}\right) \approx f\_k+\mathbf{p}^{\mathrm{T}} \nabla f\_k+\frac{1}{2} \mathbf{p}^{\mathrm{T}} \nabla^2 f\_k \mathbf{p} \stackrel{\text { def }}{=} m\_k(\mathbf{p})
$$

Assuming for the moment that $\nabla^2 f\_k$ is positive definite, we obtain the Newton direction by finding the vector $\mathbf{p}$ that minimizes $m\_k(\mathbf{p})$. By simply setting the derivative of $m\_k(\mathbf{p})$ to zero, we obtain the following explicit formula:

\begin{equation}
\mathbf{p}\_k^{\mathrm{N}}=-\nabla^2 f\_k^{-1} \nabla f\_k
\label{eq:newton_direction}
\end{equation}

The Newton direction is reliable when the difference between the true function $f\left(\mathbf{x}\_k+ \mathbf{p}\right)$ and its quadratic model $m\_k(\mathbf{p})$ is not too large. By comparing \eqref{eq:quadratic_model} with \eqref{eq:taylor_expansion}, we see that the only difference between these functions is that the matrix $\nabla^2 f\left(\mathbf{x}\_k+t \mathbf{p}\right)$ in the third term of the expansion has been replaced by $\nabla^2 f\_k=\nabla^2 f\left(\mathbf{x}\_k\right)$. If $\nabla^2 f(\cdot)$ is sufficiently smooth, this difference introduces a perturbation of only $O\left(\lVert\mathbf{p}\rVert^3\right)$ into the expansion, so that when $\lVert\mathbf{p}\rVert$ is small, the approximation $f\left(\mathbf{x}\_k+\mathbf{p}\right) \approx m\_k(\mathbf{p})$ is very accurate indeed.

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

All of the search directions discussed so far can be used directly in a line search framework. They give rise to the steepest descent, Newton, quasi-Newton, and conjugate gradient line search methods. All except conjugate gradients have an analogue in the trustregion framework, as we now discuss.


## Step-length conditions
### Wolfe conditions

<iframe style="border:none;" scrolling="no" src="../../../../interactive/line-search-conditions.html" width="700px" height="500px" title="Wolfe conditions visualisation"></iframe>


### Goldenstein conditions

<iframe style="border:none;" scrolling="no" src="../../../../interactive/goldstein-conditions-visualization.html" width="700px" height="700px" title="Wolfe conditions visualisation"></iframe>

## Search directions


## Rate of convergence


