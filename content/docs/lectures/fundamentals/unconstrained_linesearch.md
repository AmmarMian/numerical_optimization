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


All of the search directions discussed so far can be used directly in a line search framework. They give rise to the steepest descent, Newton, quasi-Newton, and conjugate gradient line search methods. For Newton and quasi-Newton methods, see the next chapter.

## Step-length conditions

In computing the step length $\alpha\_{k}$, we face a tradeoff. We would like to choose $\alpha\_{k}$ to give a substantial reduction of $f$, but at the same time, we do not want to spend too much time making the choice. The ideal choice would be the global minimizer of the univariate function $\phi(\cdot)$ defined by

\begin{equation}
\phi(\alpha)=f\left(\mathbf{x}\_{k}+\alpha \mathbf{p}\_{k}\right), \quad \alpha>0
\label{eq:phi\_def}
\end{equation}

but in general, it is too expensive to identify this value. To find even a local minimizer of $\phi$ to moderate precision generally requires too many evaluations of the objective function $f$ and possibly the gradient $\nabla f$. More practical strategies perform an inexact line search to identify a step length that achieves adequate reductions in $f$ at minimal cost.

Typical line search algorithms try out a sequence of candidate values for $\alpha$, stopping to accept one of these values when certain conditions are satisfied. The line search is done in two stages: A bracketing phase finds an interval containing desirable step lengths, and a bisection or interpolation phase computes a good step length within this interval. Sophisticated line search algorithms can be quite complicated, so we defer a full description until the end of this chapter. We now discuss various termination conditions for the line search algorithm and show that effective step lengths need not lie near minimizers of the univariate function $\phi(\alpha)$ defined in \eqref{eq:phi\_def}.

A simple condition we could impose on $\alpha\_{k}$ is that it provide a reduction in $f$, i.e., $f\left(\mathbf{x}\_{k}+\alpha\_{k} \mathbf{p}\_{k}\right)<f\left(\mathbf{x}\_{k}\right)$. The difficulty is that we do not have sufficient reduction in $f$, a concept we discuss next.

## The Wolfe conditions

A popular inexact line search condition stipulates that $\alpha\_{k}$ should first of all give sufficient decrease in the objective function $f$, as measured by the following inequality:

\begin{equation}
f\left(\mathbf{x}\_{k}+\alpha \mathbf{p}\_{k}\right) \leq f\left(\mathbf{x}\_{k}\right)+c\_{1} \alpha \nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k},
\label{eq:armijo}
\end{equation}

for some constant $c\_{1} \in(0,1)$. In other words, the reduction in $f$ should be proportional to both the step length $\alpha\_{k}$ and the directional derivative $\nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k}$. Inequality \eqref{eq:armijo} is sometimes called the Armijo condition.


The right-hand-side of \eqref{eq:armijo}, which is a linear function, can be denoted by $l(\alpha)$. The function $l(\cdot)$ has negative slope $c\_{1} \nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k}$, but because $c\_{1} \in(0,1)$, it lies above the graph of $\phi$ for small positive values of $\alpha$. The sufficient decrease condition states that $\alpha$ is acceptable only if $\phi(\alpha) \leq l(\alpha)$. In practice, $c\_{1}$ is chosen to be quite small, say $c\_{1}=10^{-4}$.

The sufficient decrease condition is not enough by itself to ensure that the algorithm makes reasonable progress, because it is satisfied for all sufficiently small values of $\alpha$. To rule out unacceptably short steps we introduce a second requirement, called the curvature condition, which requires $\alpha\_{k}$ to satisfy

\begin{equation}
\nabla f\left(\mathbf{x}\_{k}+\alpha\_{k} \mathbf{p}\_{k}\right)^{\mathrm{T}} \mathbf{p}\_{k} \geq c\_{2} \nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k}
\label{eq:curvature}
\end{equation}

for some constant $c\_{2} \in\left(c\_{1}, 1\right)$, where $c\_{1}$ is the constant from \eqref{eq:armijo}. Note that the left-handside is simply the derivative $\phi^{\prime}\left(\alpha\_{k}\right)$, so the curvature condition ensures that the slope of $\phi\left(\alpha\_{k}\right)$ is greater than $c\_{2}$ times the gradient $\phi^{\prime}(0)$. This makes sense because if the slope $\phi^{\prime}(\alpha)$ is strongly negative, we have an indication that we can reduce $f$ significantly by moving further along the chosen direction. On the other hand, if the slope is only slightly negative or even positive, it is a sign that we cannot expect much more decrease in $f$ in this direction, so it might make sense to terminate the line search. Typical values of $c\_{2}$ are 0.9 when the search direction $\mathbf{p}\_{k}$ is chosen by a Newton or quasi-Newton method, and 0.1 when $\mathbf{p}\_{k}$ is obtained from a nonlinear conjugate gradient method.

The sufficient decrease and curvature conditions are known collectively as the Wolfe conditions. We restate them here for future reference:

\begin{equation}
\begin{aligned}
f\left(\mathbf{x}\_{k}+\alpha\_{k} \mathbf{p}\_{k}\right) & \leq f\left(\mathbf{x}\_{k}\right)+c\_{1} \alpha\_{k} \nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k}, \\\\
\nabla f\left(\mathbf{x}\_{k}+\alpha\_{k} \mathbf{p}\_{k}\right)^{\mathrm{T}} \mathbf{p}\_{k} & \geq c\_{2} \nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k},
\end{aligned}
\label{eq:wolfe}
\end{equation}

with $0<c\_{1}<c\_{2}<1$.

A step length may satisfy the Wolfe conditions without being particularly close to a minimizer of $\phi$. We can, however, modify the curvature condition to force $\alpha\_{k}$ to lie in at least a broad neighborhood of a local minimizer or stationary point of $\phi$. The strong Wolfe conditions require $\alpha\_{k}$ to satisfy

\begin{equation}
\begin{aligned}
f\left(\mathbf{x}\_{k}+\alpha\_{k} \mathbf{p}\_{k}\right) & \leq f\left(\mathbf{x}\_{k}\right)+c\_{1} \alpha\_{k} \nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k}, \\\\
\left|\nabla f\left(\mathbf{x}\_{k}+\alpha\_{k} \mathbf{p}\_{k}\right)^{\mathrm{T}} \mathbf{p}\_{k}\right| & \leq c\_{2}\left|\nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k}\right|,
\end{aligned}
\label{eq:strong\_wolfe}
\end{equation}

with $0<c\_{1}<c\_{2}<1$. The only difference with the Wolfe conditions is that we no longer allow the derivative $\phi^{\prime}\left(\alpha\_{k}\right)$ to be too positive. Hence, we exclude points that are far from stationary points of $\phi$.

It is not difficult to prove that there exist step lengths that satisfy the Wolfe conditions for every function $f$ that is smooth and bounded below.

{{<lemma "Existence of step lengths" wolfe\_existence>}}
Suppose that $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ is continuously differentiable. Let $\mathbf{p}\_{k}$ be a descent direction at $\mathbf{x}\_{k}$, and assume that $f$ is bounded below along the ray $\left\\{\mathbf{x}\_{k}+\alpha \mathbf{p}\_{k} \mid \alpha>0\right\\}$. Then if $0<c\_{1}<c\_{2}<1$, there exist intervals of step lengths satisfying the Wolfe conditions \eqref{eq:wolfe} and the strong Wolfe conditions \eqref{eq:strong\_wolfe}.
{{</lemma>}}

{{<proof>}}
Since $\phi(\alpha)=f\left(\mathbf{x}\_{k}+\alpha \mathbf{p}\_{k}\right)$ is bounded below for all $\alpha>0$ and since $0<c\_{1}<1$, the line $l(\alpha)=f\left(\mathbf{x}\_{k}\right)+\alpha c\_{1} \nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k}$ must intersect the graph of $\phi$ at least once. Let $\alpha^{\prime}>0$ be the smallest intersecting value of $\alpha$, that is,

\begin{equation}
f\left(\mathbf{x}\_{k}+\alpha^{\prime} \mathbf{p}\_{k}\right)=f\left(\mathbf{x}\_{k}\right)+\alpha^{\prime} c\_{1} \nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k}
\label{eq:intersection}
\end{equation}

The sufficient decrease condition \eqref{eq:armijo} clearly holds for all step lengths less than $\alpha^{\prime}$.

By the mean value theorem, there exists $\alpha^{\prime \prime} \in\left(0, \alpha^{\prime}\right)$ such that

\begin{equation}
f\left(\mathbf{x}\_{k}+\alpha^{\prime} \mathbf{p}\_{k}\right)-f\left(\mathbf{x}\_{k}\right)=\alpha^{\prime} \nabla f\left(\mathbf{x}\_{k}+\alpha^{\prime \prime} \mathbf{p}\_{k}\right)^{\mathrm{T}} \mathbf{p}\_{k}
\label{eq:mean\_value}
\end{equation}

By combining \eqref{eq:intersection} and \eqref{eq:mean\_value}, we obtain

\begin{equation}
\nabla f\left(\mathbf{x}\_{k}+\alpha^{\prime \prime} \mathbf{p}\_{k}\right)^{\mathrm{T}} \mathbf{p}\_{k}=c\_{1} \nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k}>c\_{2} \nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k},
\label{eq:inequality\_proof}
\end{equation}

since $c\_{1}<c\_{2}$ and $\nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k}<0$. Therefore, $\alpha^{\prime \prime}$ satisfies the Wolfe conditions \eqref{eq:wolfe}, and the inequalities hold strictly in both conditions. Hence, by our smoothness assumption on $f$, there is an interval around $\alpha^{\prime \prime}$ for which the Wolfe conditions hold. Moreover, since the term in the left-hand side of \eqref{eq:inequality\_proof} is negative, the strong Wolfe conditions \eqref{eq:strong\_wolfe} hold in the same interval.
{{</proof>}}

The Wolfe conditions are scale-invariant in a broad sense: Multiplying the objective function by a constant or making an affine change of variables does not alter them. They can be used in most line search methods, and are particularly important in the implementation of quasi-Newton methods.

To summarize, see the following interactive visualisation of the Wolfe conditions, which illustrates the sufficient decrease and curvature conditions in action:

<iframe style="border:none;" scrolling="no" src="../../../../interactive/line-search-conditions.html" width="700px" height="500px" title="Wolfe conditions visualisation"></iframe>


## The Goldstein conditions

Like the Wolfe conditions, the Goldstein conditions also ensure that the step length $\alpha$ achieves sufficient decrease while preventing $\alpha$ from being too small. The Goldstein conditions can also be stated as a pair of inequalities, in the following way:

\begin{equation}
f\left(\mathbf{x}\_{k}\right)+(1-c) \alpha\_{k} \nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k} \leq f\left(\mathbf{x}\_{k}+\alpha\_{k} \mathbf{p}\_{k}\right) \leq f\left(\mathbf{x}\_{k}\right)+c \alpha\_{k} \nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k},
\label{eq:goldstein}
\end{equation}

with $0<c<\frac{1}{2}$. The second inequality is the sufficient decrease condition \eqref{eq:armijo}, whereas the first inequality is introduced to control the step length from below.

A disadvantage of the Goldstein conditions vis-à-vis the Wolfe conditions is that the first inequality in \eqref{eq:goldstein} may exclude all minimizers of $\phi$. However, the Goldstein and Wolfe conditions have much in common, and their convergence theories are quite similar. The Goldstein conditions are often used in Newton-type methods but are not well suited for quasi-Newton methods that maintain a positive definite Hessian approximation.

An illustration of the Goldstein conditions is shown in the following interactive visualisation:

<iframe style="border:none;" scrolling="no" src="../../../../interactive/goldstein-conditions-visualization.html" width="700px" height="700px" title="Wolfe conditions visualisation"></iframe>


## Sufficient decrease and backtracking

We have mentioned that the sufficient decrease condition \eqref{eq:armijo} alone is not sufficient to ensure that the algorithm makes reasonable progress along the given search direction. However, if the line search algorithm chooses its candidate step lengths appropriately, by using a so-called backtracking approach, we can dispense with the extra condition \eqref{eq:curvature} and use just the sufficient decrease condition to terminate the line search procedure. In its most basic form, backtracking proceeds as follows.

**Procedure (Backtracking Line Search).**

Choose $\bar{\alpha}>0, \rho, c \in(0,1)$;

set $\alpha \leftarrow \bar{\alpha}$;

repeat until $f\left(\mathbf{x}\_{k}+\alpha \mathbf{p}\_{k}\right) \leq f\left(\mathbf{x}\_{k}\right)+c \alpha \nabla f\_{k}^{\mathrm{T}} \mathbf{p}\_{k}$

$\alpha \leftarrow\rho\alpha$;

end (repeat)

terminate with $\alpha\_{k}=\alpha$.

In this procedure, the initial step length $\bar{\alpha}$ is chosen to be 1 in Newton and quasi-Newton methods, but can have different values in other algorithms such as steepest descent or conjugate gradient. An acceptable step length will be found after a finite number of trials because $\alpha\_{k}$ will eventually become small enough that the sufficient decrease condition holds. In practice, the contraction factor $\rho$ is often allowed to vary at each iteration of the line search. For example, it can be chosen by safeguarded interpolation, as we describe later. We need ensure only that at each iteration we have $\rho \in\left[\rho\_{\mathrm{lo}}, \rho\_{\mathrm{hi}}\right]$, for some fixed constants $0<\rho\_{\text {lo }}<\rho\_{\text {hi }}<1$.

The backtracking approach ensures either that the selected step length $\alpha\_{k}$ is some fixed value (the initial choice $\bar{\alpha}$ ), or else that it is short enough to satisfy the sufficient decrease condition but not too short. The latter claim holds because the accepted value $\alpha\_{k}$ is within striking distance of the previous trial value, $\alpha\_{k} / \rho$, which was rejected for violating the sufficient decrease condition, that is, for being too long.


## Convergence of line search methods

To obtain global convergence, we must not only have well-chosen step lengths but also well-chosen search directions $\mathbf{p}\_k$. We discuss requirements on the search direction in this section, focusing on one key property: the angle $\theta\_k$ between $\mathbf{p}\_k$ and the steepest descent direction $-\nabla f\_k$, defined by

$$
\cos \theta\_k=\frac{-\nabla f\_k^{\mathrm{T}} \mathbf{p}\_k}{\\|\nabla f\_k\\|\\|\mathbf{p}\_k\\|}
$$

The following theorem, due to Zoutendijk, has far-reaching consequences. It shows, for example, that the steepest descent method is globally convergent. For other algorithms it describes how far $\mathbf{p}\_k$ can deviate from the steepest descent direction and still give rise to a globally convergent iteration. Various line search termination conditions can be used to establish this result, but for concreteness we will consider only the Wolfe conditions. Though Zoutendijk's result appears, at first, to be technical and obscure, its power will soon become evident.

{{<theorem "Zoutendijk's theorem" zoutendijk_theorem>}}
Consider any iteration of the form $\mathbf{x}\_{k+1} = \mathbf{x}\_k + \alpha\_k \mathbf{p}\_k$, where $\mathbf{p}\_k$ is a descent direction and $\alpha\_k$ satisfies the Wolfe conditions. Suppose that $f$ is bounded below in $\mathbb{R}^n$ and that $f$ is continuously differentiable in an open set $\mathcal{N}$ containing the level set $\mathcal{L} = \\{\mathbf{x}: f(\mathbf{x}) \leq f(\mathbf{x}\_0)\\}$, where $\mathbf{x}\_0$ is the starting point of the iteration. Assume also that the gradient $\nabla f$ is Lipschitz continuous on $\mathcal{N}$, that is, there exists a constant $L>0$ such that

\begin{equation}
\\|\nabla f(\mathbf{x})-\nabla f(\tilde{\mathbf{x}})\\| \leq L\\|\mathbf{x}-\tilde{\mathbf{x}}\\|, \quad \text { for all } \mathbf{x}, \tilde{\mathbf{x}} \in \mathcal{N} .
\label{eq:lipschitz}
\end{equation}

Then

\begin{equation}
\sum\_{k \geq 0} \cos ^{2} \theta\_k\\|\nabla f\_k\\|^{2}<\infty
\label{eq:zoutendijk_condition}
\end{equation}
{{</theorem>}}

{{<proof>}}
From the second Wolfe condition and the iteration formula we have that

$$
(\nabla f\_{k+1}-\nabla f\_k)^{\mathrm{T}} \mathbf{p}\_k \geq(c\_2-1) \nabla f\_k^{\mathrm{T}} \mathbf{p}\_k
$$

while the Lipschitz condition \eqref{eq:lipschitz} implies that

$$
(\nabla f\_{k+1}-\nabla f\_k)^{\mathrm{T}} \mathbf{p}\_k \leq \alpha\_k L\\|\mathbf{p}\_k\\|^{2}
$$

By combining these two relations, we obtain

$$
\alpha\_k \geq \frac{c\_2-1}{L} \frac{\nabla f\_k^{\mathrm{T}} \mathbf{p}\_k}{\\|\mathbf{p}\_k\\|^{2}}
$$

By substituting this inequality into the first Wolfe condition, we obtain

$$
f\_{k+1} \leq f\_k-c\_1 \frac{1-c\_2}{L} \frac{(\nabla f\_k^{\mathrm{T}} \mathbf{p}\_k)^{2}}{\\|\mathbf{p}\_k\\|^{2}}
$$

From the definition of $\cos \theta\_k$, we can write this relation as

$$
f\_{k+1} \leq f\_k-c \cos ^{2} \theta\_k\\|\nabla f\_k\\|^{2}
$$

where $c=c\_1(1-c\_2) / L$. By summing this expression over all indices less than or equal to $k$, we obtain

$$
f\_{k+1} \leq f\_0-c \sum\_{j=0}^{k} \cos ^{2} \theta\_j\\|\nabla f\_j\\|^{2}
$$

Since $f$ is bounded below, we have that $f\_0-f\_{k+1}$ is less than some positive constant, for all $k$. Hence by taking limits, we obtain

$$
\sum\_{k=0}^{\infty} \cos ^{2} \theta\_k\\|\nabla f\_k\\|^{2}<\infty
$$

which concludes the proof.
{{</proof>}}

Similar results to this theorem hold when the Goldstein conditions or strong Wolfe conditions are used in place of the Wolfe conditions.

Note that the assumptions of {{<theoremref zoutendijk_theorem>}} are not too restrictive. If the function $f$ were not bounded below, the optimization problem would not be well-defined. The smoothness assumption—Lipschitz continuity of the gradient—is implied by many of the smoothness conditions that are used in local convergence theorems and are often satisfied in practice.

Inequality \eqref{eq:zoutendijk_condition}, which we call the Zoutendijk condition, implies that

$$
\cos ^{2} \theta\_k\\|\nabla f\_k\\|^{2} \rightarrow 0
$$

This limit can be used in turn to derive global convergence results for line search algorithms.
If our method for choosing the search direction $\mathbf{p}\_k$ in the iteration ensures that the angle $\theta\_k$ is bounded away from $90^{\circ}$, there is a positive constant $\delta$ such that

\begin{equation}
\cos \theta\_k \geq \delta>0, \quad \text { for all } k
\label{eq:angle_bound}
\end{equation}

It follows immediately from \eqref{eq:zoutendijk_condition} that

\begin{equation}
\lim \_{k \rightarrow \infty}\\|\nabla f\_k\\|=0
\label{eq:global_convergence}
\end{equation}

In other words, we can be sure that the gradient norms $\\|\nabla f\_k\\|$ converge to zero, provided that the search directions are never too close to orthogonality with the gradient. In particular, the method of steepest descent (for which the search direction $\mathbf{p}\_k$ makes an angle of zero degrees with the negative gradient) produces a gradient sequence that converges to zero, provided that it uses a line search satisfying the Wolfe or Goldstein conditions.

We use the term globally convergent to refer to algorithms for which the property \eqref{eq:global_convergence} is satisfied, but note that this term is sometimes used in other contexts to mean different things. For line search methods of the general form $\mathbf{x}\_{k+1} = \mathbf{x}\_k + \alpha\_k \mathbf{p}\_k$, the limit \eqref{eq:global_convergence} is the strongest global convergence result that can be obtained: We cannot guarantee that the method converges to a minimizer, but only that it is attracted by stationary points. Only by making additional requirements on the search direction $\mathbf{p}\_k$—by introducing negative curvature information from the Hessian $\nabla^{2} f(\mathbf{x}\_k)$, for example—can we strengthen these results to include convergence to a local minimum.

Note that throughout this section we have used only the fact that Zoutendijk's condition implies the limit \eqref{eq:zoutendijk_condition}. 

## Rate of convergence

We refer the reader to the textbook
> "Numerical Optimization" by Nocedal and Wright, 2nd edition, Springer, 2006, pages 47-51,

for a detailed discussion of the rate of convergence of line search methods.

Peculiarly, see pages 47-51. In general, the rate of convergence depends on the choice of search direction and the step length conditions used.
