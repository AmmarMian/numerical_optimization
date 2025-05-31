---
title: 5. Constrained optimization - Introduction
weight: 5
math: true
chapter: 5
---

# Constrained optimization methods

> **Note** : This is in part the content of the book "Numerical Optimization" by Nocedal and Wright, with some modifications to the notations used in this lecture.


The second part of this lecture is about minimizing functions subject to constraints on the variables. A general formulation for these problems is

$$
\min\_{\mathbf{x} \in \mathrm{R}^{n}} f(\mathbf{x}) \quad \text { subject to } \quad \begin{cases}c\_{i}(\mathbf{x})=0, & i \in \mathcal{E}, \\\\ c\_{i}(\mathbf{x}) \geq 0, & i \in \mathcal{I},\end{cases}
$$

where $f$ and the functions $c\_{i}$ are all smooth, real-valued functions on a subset of $\mathbb{R}^{n}$, and $\mathcal{I}$ and $\mathcal{E}$ are two finite sets of indices. As before, we call $f$ the objective function, while $c\_{i}$, $i \in \mathcal{E}$ are the equality constraints and $c\_{i}, i \in \mathcal{I}$ are the inequality constraints. We define the feasible set $\Omega$ to be the set of points $\mathbf{x}$ that satisfy the constraints; that is,

$$
\Omega=\left\\{\mathbf{x} \mid c\_{i}(\mathbf{x})=0, \quad i \in \mathcal{E} ; \quad c\_{i}(\mathbf{x}) \geq 0, \quad i \in \mathcal{I}\right\\}
$$

so that we can rewrite the problem more compactly as

\begin{equation}
\min\_{\mathbf{x} \in \Omega} f(\mathbf{x}).
\label{eq:constrained_problem}
\end{equation}

In this chapter we derive mathematical characterizations of the solutions of \eqref{eq:constrained_problem}. Recall that for the unconstrained optimization problem, we characterized solution points $\mathbf{x}^{\star}$ in the following way:

Necessary conditions: Local minima of unconstrained problems have $\nabla f\left(\mathbf{x}^{\star}\right)=0$ and $\nabla^{2} f\left(\mathbf{x}^{\star}\right)$ positive semidefinite.

Sufficient conditions: Any point $\mathbf{x}^{\star}$ at which $\nabla f\left(\mathbf{x}^{\star}\right)=0$ and $\nabla^{2} f\left(\mathbf{x}^{\star}\right)$ is positive definite is a strong local minimizer of $f$.

Our aim in this chapter is to derive similar conditions to characterize the solutions of constrained optimization problems.

## Local and global solutions

We have seen already that global solutions are difficult to find even when there are no constraints. The situation may be improved when we add constraints, since the feasible set might exclude many of the local minima and it may be comparatively easy to pick the global minimum from those that remain. However, constraints can also make things much more difficult. As an example, consider the problem

$$
\min\_{\mathbf{x} \in \mathrm{R}^{n}}\\|\mathbf{x}\\|\_{2}^{2}, \quad \text { subject to }\\|\mathbf{x}\\|\_{2}^{2} \geq 1
$$

Without the constraint, this is a convex quadratic problem with unique minimizer $\mathbf{x}=\mathbf{0}$. When the constraint is added, any vector $\mathbf{x}$ with $\\|\mathbf{x}\\|\_{2}=1$ solves the problem. There are infinitely many such vectors (hence, infinitely many local minima) whenever $n \geq 2$.

A second example shows how addition of a constraint produces a large number of local solutions that do not form a connected set. Consider

$$
\min \left(x\_{2}+100\right)^{2}+0.01 x\_{1}^{2}, \quad \text { subject to } x\_{2}-\cos x\_{1} \geq 0
$$

Without the constraint, the problem has the unique solution $(-100,0)$. With the constraint there are local solutions near the points

$$
\left(x\_{1}, x\_{2}\right)=(k \pi,-1), \quad \text { for } \quad k= \pm 1, \pm 3, \pm 5, \ldots
$$

{{<definition "Local solution" local_solution>}}
A vector $\mathbf{x}^{\star}$ is a **local solution** of the problem \eqref{eq:constrained_problem} if $\mathbf{x}^{\star} \in \Omega$ and there is a neighborhood $\mathcal{N}$ of $\mathbf{x}^{\star}$ such that $f(\mathbf{x}) \geq f\left(\mathbf{x}^{\star}\right)$ for $\mathbf{x} \in \mathcal{N} \cap \Omega$.
{{</definition>}}

Similarly, we can make the following definitions:

{{<definition "Strict local solution" strict_local_solution>}}
A vector $\mathbf{x}^{\star}$ is a **strict local solution** (also called a strong local solution) if $\mathbf{x}^{\star} \in \Omega$ and there is a neighborhood $\mathcal{N}$ of $\mathbf{x}^{\star}$ such that $f(\mathbf{x})>f\left(\mathbf{x}^{\star}\right)$ for all $\mathbf{x} \in \mathcal{N} \cap \Omega$ with $\mathbf{x} \neq \mathbf{x}^{\star}$.
{{</definition>}}

{{<definition "Isolated local solution" isolated_local_solution>}}
A point $\mathbf{x}^{\star}$ is an **isolated local solution** if $\mathbf{x}^{\star} \in \Omega$ and there is a neighborhood $\mathcal{N}$ of $\mathbf{x}^{\star}$ such that $\mathbf{x}^{\star}$ is the only local minimizer in $\mathcal{N} \cap \Omega$.
{{</definition>}}

At times, we replace the word "solution" by "minimizer" in our discussion. This alternative is frequently used in the literature, but it is slightly less satisfying because it does not account for the role of the constraints in defining the point in question.

## Smoothness

Smoothness of objective functions and constraints is an important issue in characterizing solutions, just as in the unconstrained case. It ensures that the objective function and the constraints all behave in a reasonably predictable way and therefore allows algorithms to make good choices for search directions.

We saw earlier that graphs of nonsmooth functions contain "kinks" or "jumps" where the smoothness breaks down. If we plot the feasible region for any given constrained optimization problem, we usually observe many kinks and sharp edges. Does this mean that the constraint functions that describe these regions are nonsmooth? The answer is often no, because the nonsmooth boundaries can often be described by a collection of smooth constraint functions. A diamond-shaped feasible region in $\mathbb{R}^{2}$ could be described by the single nonsmooth constraint

$$
\\|\mathbf{x}\\|\_{1}=\left|x\_{1}\right|+\left|x\_{2}\right| \leq 1 .
$$

{{<center>}}
{{<NumberedFigure
  src=" ../../../../../../tikZ/nonsmooth_tosmooth_constraints/main.svg"
  alt="nonsmooth constraints can be described by smooth constraints"
  width="300px"
  caption="Nonsmooth constraints can be described by smooth constraints"
>}}
{{</center>}}

It can also be described by the following set of smooth (in fact, linear) constraints:

$$
x\_{1}+x\_{2} \leq 1, \quad x\_{1}-x\_{2} \leq 1, \quad -x\_{1}+x\_{2} \leq 1, \quad -x\_{1}-x\_{2} \leq 1
$$

Each of the four constraints represents one edge of the feasible polytope. In general, the constraint functions are chosen so that each one represents a smooth piece of the boundary of $\Omega$.

Nonsmooth, unconstrained optimization problems can sometimes be reformulated as smooth constrained problems. An example is given by the unconstrained scalar problem of minimizing a nonsmooth function $f(x)$ defined by

$$
f(x)=\max \left(x^{2}, x\right),
$$

which has kinks at $x=0$ and $x=1$, and the solution at $x^{\star}=0$. We obtain a smooth, constrained formulation of this problem by adding an artificial variable $t$ and writing

$$
\min t \quad \text { s.t. } \quad t \geq x, \quad t \geq x^{2} .
$$

Reformulation techniques such as these are used often in cases where $f$ is a maximum of a collection of functions or when $f$ is a 1 -norm or $\infty$-norm of a vector function.

In the examples above we expressed inequality constraints in a slightly different way from the form $c\_{i}(\mathbf{x}) \geq 0$ that appears in the definition. However, any collection of inequality constraints with $\geq$ and $\leq$ and nonzero right-hand-sides can be expressed in the form $c\_{i}(\mathbf{x}) \geq 0$ by simple rearrangement of the inequality. In general, it is good practice to state the constraint in a way that is intuitive and easy to understand.

## Examples

To introduce the basic principles behind the characterization of solutions of constrained optimization problems, we work through three simple examples. The ideas discussed here will be made rigorous in the sections that follow.

We start by noting one item of terminology that recurs throughout the rest of the lecture: At a feasible point $\mathbf{x}$, the inequality constraint $i \in \mathcal{I}$ is said to be active if $c\_{i}(\mathbf{x})=0$ and inactive if the strict inequality $c\_{i}(\mathbf{x})>0$ is satisfied.

### A single equality constraint

**Example 1**

Our first example is a two-variable problem with a single equality constraint:

\begin{equation}
\min x\_{1}+x\_{2} \quad \text { s.t. } \quad x\_{1}^{2}+x\_{2}^{2}-2=0.
\label{eq:equality_example}
\end{equation}

{{<center>}}
{{<NumberedFigure
  src=" ../../../../../../tikZ/single_equality_cosntraint/main.svg"
  alt="constraints and gradient of function"
  width="300px"
  caption="Constraints and gradient of function"
>}}
{{</center>}}


In the general form, we have $f(\mathbf{x})=x\_{1}+x\_{2}, \mathcal{I}=\emptyset, \mathcal{E}=\\{1\\}$, and $c\_{1}(\mathbf{x})=x\_{1}^{2}+x\_{2}^{2}-2$. We can see by inspection that the feasible set for this problem is the circle of radius $\sqrt{2}$ centered at the origin-just the boundary of this circle, not its interior. The solution $\mathbf{x}^{\star}$ is obviously $(-1,-1)^{\mathrm{T}}$. From any other point on the circle, it is easy to find a way to move that stays feasible (that is, remains on the circle) while decreasing $f$. For instance, from the point $\mathbf{x}=(\sqrt{2}, 0)^{\mathrm{T}}$ any move in the clockwise direction around the circle has the desired effect.

We also see that at the solution $\mathbf{x}^{\star}$, the constraint normal $\nabla c\_{1}\left(\mathbf{x}^{\star}\right)$ is parallel to $\nabla f\left(\mathbf{x}^{\star}\right)$. That is, there is a scalar $\lambda\_{1}^{\star}$ such that

\begin{equation}
\nabla f\left(\mathbf{x}^{\star}\right)=\lambda\_{1}^{\star} \nabla c\_{1}\left(\mathbf{x}^{\star}\right).
\label{eq:parallel_gradients}
\end{equation}

(In this particular case, we have $\lambda\_{1}^{\star}=-\frac{1}{2}$.)

We can derive \eqref{eq:parallel_gradients} by examining first-order Taylor series approximations to the objective and constraint functions. To retain feasibility with respect to the function $c\_{1}(\mathbf{x})=0$, we require that $c\_{1}(\mathbf{x}+\mathbf{d})=0$; that is,

$$
0=c\_{1}(\mathbf{x}+\mathbf{d}) \approx c\_{1}(\mathbf{x})+\nabla c\_{1}(\mathbf{x})^{\mathrm{T}} \mathbf{d}=\nabla c\_{1}(\mathbf{x})^{\mathrm{T}} \mathbf{d}
$$

Hence, the direction $\mathbf{d}$ retains feasibility with respect to $c\_{1}$, to first order, when it satisfies

$$
\nabla c\_{1}(\mathbf{x})^{\mathrm{T}} \mathbf{d}=0
$$

Similarly, a direction of improvement must produce a decrease in $f$, so that

$$
0>f(\mathbf{x}+\mathbf{d})-f(\mathbf{x}) \approx \nabla f(\mathbf{x})^{\mathrm{T}} \mathbf{d}
$$

or, to first order,

$$
\nabla f(\mathbf{x})^{\mathrm{T}} \mathbf{d}<0
$$

If there exists a direction $\mathbf{d}$ that satisfies both conditions, we conclude that improvement on our current point $\mathbf{x}$ is possible. It follows that a necessary condition for optimality for the problem \eqref{eq:equality_example} is that there exist no direction $\mathbf{d}$ satisfying both conditions.

By drawing a picture (see visualization below), the reader can check that the only way that such a direction cannot exist is if $\nabla f(\mathbf{x})$ and $\nabla c\_{1}(\mathbf{x})$ are parallel, that is, if the condition $\nabla f(\mathbf{x})=\lambda\_{1} \nabla c\_{1}(\mathbf{x})$ holds at $\mathbf{x}$, for some scalar $\lambda\_{1}$. If this condition is not satisfied, the direction defined by

$$
\mathbf{d}=-\left(\mathbf{I}-\frac{\nabla c\_{1}(\mathbf{x}) \nabla c\_{1}(\mathbf{x})^{\mathrm{T}}}{\\|\nabla c\_{1}(\mathbf{x})\\|^{2}}\right) \nabla f(\mathbf{x})
$$

satisfies both conditions.

By introducing the Lagrangian function

$$
\mathcal{L}\left(\mathbf{x}, \lambda\_{1}\right)=f(\mathbf{x})-\lambda\_{1} c\_{1}(\mathbf{x}),
$$

and noting that $\nabla\_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}, \lambda\_{1}\right)=\nabla f(\mathbf{x})-\lambda\_{1} \nabla c\_{1}(\mathbf{x})$, we can state the condition \eqref{eq:parallel_gradients} equivalently as follows: At the solution $\mathbf{x}^{\star}$, there is a scalar $\lambda\_{1}^{\star}$ such that

\begin{equation}
\nabla\_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \lambda\_{1}^{\star}\right)=0.
\label{eq:lagrangian_gradient_zero}
\end{equation}

This observation suggests that we can search for solutions of the equality-constrained problem \eqref{eq:equality_example} by searching for stationary points of the Lagrangian function. The scalar quantity $\lambda\_{1}$ is called a Lagrange multiplier for the constraint $c\_{1}(\mathbf{x})=0$.

Though the condition \eqref{eq:parallel_gradients} (equivalently, \eqref{eq:lagrangian_gradient_zero}) appears to be necessary for an optimal solution of the problem \eqref{eq:equality_example}, it is clearly not sufficient. For instance, in this example, \eqref{eq:parallel_gradients} is satisfied at the point $\mathbf{x}=(1,1)$ (with $\lambda\_{1}=\frac{1}{2}$ ), but this point is obviously not a solution-in fact, it maximizes the function $f$ on the circle. Moreover, in the case of equality-constrained problems, we cannot turn the condition \eqref{eq:parallel_gradients} into a sufficient condition simply by placing some restriction on the sign of $\lambda\_{1}$. To see this, consider replacing the constraint $x\_{1}^{2}+x\_{2}^{2}-2=0$ by its negative $2-x\_{1}^{2}-x\_{2}^{2}=0$. The solution of the problem is not affected, but the value of $\lambda\_{1}^{\star}$ that satisfies the condition \eqref{eq:parallel_gradients} changes from $\lambda\_{1}^{\star}=-\frac{1}{2}$ to $\lambda\_{1}^{\star}=\frac{1}{2}$.

This situation is illustrated in following visualization:

<iframe style="border:none;" scrolling="no" src="../../../../interactive/onesingle_constraint.html" width="700px" height="500px" title="One single constraint"></iframe>

### A single inequality constraint

**Example 2**

This is a slight modification of Example 1, in which the equality constraint is replaced by an inequality. Consider

\begin{equation}
\min x\_{1}+x\_{2} \quad \text { s.t. } \quad 2-x\_{1}^{2}-x\_{2}^{2} \geq 0,
\label{eq:inequality_example}
\end{equation}

for which the feasible region consists of the circle of problem \eqref{eq:equality_example} and its interior. Note that the constraint normal $\nabla c\_{1}$ points toward the interior of the feasible region at each point on the boundary of the circle. By inspection, we see that the solution is still $(-1,-1)$ and that the condition \eqref{eq:parallel_gradients} holds for the value $\lambda\_{1}^{\star}=\frac{1}{2}$. However, this inequality-constrained problem differs from the equality-constrained problem \eqref{eq:equality_example} in that the sign of the Lagrange multiplier plays a significant role, as we now argue.

As before, we conjecture that a given feasible point $\mathbf{x}$ is not optimal if we can find a step $\mathbf{d}$ that both retains feasibility and decreases the objective function $f$ to first order. The main difference between problems \eqref{eq:equality_example} and \eqref{eq:inequality_example} comes in the handling of the feasibility condition. The direction $\mathbf{d}$ improves the objective function, to first order, if $\nabla f(\mathbf{x})^{\mathrm{T}} \mathbf{d}<0$. Meanwhile, the direction $\mathbf{d}$ retains feasibility if

$$
0 \leq c\_{1}(\mathbf{x}+\mathbf{d}) \approx c\_{1}(\mathbf{x})+\nabla c\_{1}(\mathbf{x})^{\mathrm{T}} \mathbf{d}
$$

so, to first order, feasibility is retained if

$$
c\_{1}(\mathbf{x})+\nabla c\_{1}(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0
$$

In determining whether a direction $\mathbf{d}$ exists that satisfies both conditions, we consider the following two cases:

**Case I:** Consider first the case in which $\mathbf{x}$ lies strictly inside the circle, so that the strict inequality $c\_{1}(\mathbf{x})>0$ holds. In this case, any vector $\mathbf{d}$ satisfies the feasibility condition, provided only that its length is sufficiently small. In particular, whenever $\nabla f\left(\mathbf{x}^{\star}\right) \neq \mathbf{0}$, we can obtain a direction $\mathbf{d}$ that satisfies both conditions by setting

$$
\mathbf{d}=-c\_{1}(\mathbf{x}) \frac{\nabla f(\mathbf{x})}{\\|\nabla f(\mathbf{x})\\|}
$$

The only situation in which such a direction fails to exist is when

$$
\nabla f(\mathbf{x})=\mathbf{0} .
$$

This situation is summarized through the following interactive visualization:
<iframe style="border:none;" scrolling="no" src="../../../../interactive/one_inequality_constraint_case1.html" width="700px" height="500px" title="One single constraint"></iframe>



**Case II:** Consider now the case in which $\mathbf{x}$ lies on the boundary of the circle, so that $c\_{1}(\mathbf{x})=0$. The conditions therefore become

$$
\nabla f(\mathbf{x})^{\mathrm{T}} \mathbf{d}<0, \quad \nabla c\_{1}(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0
$$

The first of these conditions defines an open half-space, while the second defines a closed half-space. It is clear that the two regions fail to intersect only when $\nabla f(\mathbf{x})$ and $\nabla c\_{1}(\mathbf{x})$ point in the same direction, that is, when

\begin{equation}
\nabla f(\mathbf{x})=\lambda\_{1} \nabla c\_{1}(\mathbf{x}), \quad \text { for some } \lambda\_{1} \geq 0.
\label{eq:inequality_optimality}
\end{equation}

Note that the sign of the multiplier is significant here. If \eqref{eq:parallel_gradients} were satisfied with a negative value of $\lambda\_{1}$, then $\nabla f(\mathbf{x})$ and $\nabla c\_{1}(\mathbf{x})$ would point in opposite directions, and we see that the set of directions that satisfy both conditions would make up an entire open half-plane.

The optimality conditions for both cases I and II can again be summarized neatly with reference to the Lagrangian function. When no first-order feasible descent direction exists at some point $\mathbf{x}^{\star}$, we have that

\begin{equation}
\nabla\_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \lambda\_{1}^{\star}\right)=\mathbf{0}, \quad \text { for some } \lambda\_{1}^{\star} \geq 0,
\label{eq:kkt_gradient}
\end{equation}

where we also require that

\begin{equation}
\lambda\_{1}^{\star} c\_{1}\left(\mathbf{x}^{\star}\right)=0.
\label{eq:complementarity}
\end{equation}

This condition is known as a complementarity condition; it implies that the Lagrange multiplier $\lambda\_{1}$ can be strictly positive only when the corresponding constraint $c\_{1}$ is active. Conditions of this type play a central role in constrained optimization, as we see in the sections that follow. In case I, we have that $c\_{1}\left(\mathbf{x}^{\star}\right)>0$, so \eqref{eq:complementarity} requires that $\lambda\_{1}^{\star}=0$. Hence, \eqref{eq:kkt_gradient} reduces to $\nabla f\left(\mathbf{x}^{\star}\right)=\mathbf{0}$, as required. In case II, \eqref{eq:complementarity} allows $\lambda\_{1}^{\star}$ to take on a nonnegative value, so \eqref{eq:kkt_gradient} becomes equivalent to \eqref{eq:inequality_optimality}.

This situation is summarized through the following interactive visualization:
<iframe style="border:none;" scrolling="no" src="../../../../interactive/one_inequality_constraint_case2.html" width="700px" height="500px" title="One single constraint"></iframe>

This situation is also well visualized for quadratic functions:
<iframe style="border:none;" scrolling="no" src="one_inequality_quadratic" width="700px" height="500px" title="One single constraint quadratic"></iframe>



### Two inequality constraints

**Example 3**

Suppose we add an extra constraint to the problem \eqref{eq:inequality_example} to obtain

\begin{equation}
\min x\_{1}+x\_{2} \quad \text { s.t. } \quad 2-x\_{1}^{2}-x\_{2}^{2} \geq 0, \quad x\_{2} \geq 0,
\label{eq:two_inequality_example}
\end{equation}

for which the feasible region is the half-disk. It is easy to see that the solution lies at $(-\sqrt{2}, 0)^{\mathrm{T}}$, a point at which both constraints are active. By repeating the arguments for the previous examples, we conclude that a direction $\mathbf{d}$ is a feasible descent direction, to first order, if it satisfies the following conditions:

$$
\nabla c\_{i}(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0, \quad i \in \mathcal{I}=\\{1,2\\}, \quad \nabla f(\mathbf{x})^{\mathrm{T}} \mathbf{d}<0
$$

However, it is clear that no such direction can exist when $\mathbf{x}=(-\sqrt{2}, 0)^{\mathrm{T}}$. The conditions $\nabla c\_{i}(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0, i=1,2$, are both satisfied only if $\mathbf{d}$ lies in the quadrant defined by $\nabla c\_{1}(\mathbf{x})$ and $\nabla c\_{2}(\mathbf{x})$, but it is clear by inspection that all vectors $\mathbf{d}$ in this quadrant satisfy $\nabla f(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0$.

Let us see how the Lagrangian and its derivatives behave for the problem \eqref{eq:two_inequality_example} and the solution point $(-\sqrt{2}, 0)^{\mathrm{T}}$. First, we include an additional term $\lambda\_{i} c\_{i}(\mathbf{x})$ in the Lagrangian for each additional constraint, so we have

\begin{equation}
\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})=f(\mathbf{x})-\lambda\_{1} c\_{1}(\mathbf{x})-\lambda\_{2} c\_{2}(\mathbf{x}),
\label{eq:two_constraint_lagrangian}
\end{equation}

where $\boldsymbol{\lambda}=\left(\lambda\_{1}, \lambda\_{2}\right)^{\mathrm{T}}$ is the vector of Lagrange multipliers. The extension of condition \eqref{eq:kkt_gradient} to this case is

\begin{equation}
\nabla\_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)=\mathbf{0}, \quad \text { for some } \boldsymbol{\lambda}^{\star} \geq \mathbf{0},
\label{eq:two_constraint_kkt}
\end{equation}

where the inequality $\boldsymbol{\lambda}^{\star} \geq \mathbf{0}$ means that all components of $\boldsymbol{\lambda}^{\star}$ are required to be nonnegative. By applying the complementarity condition \eqref{eq:complementarity} to both inequality constraints, we obtain

\begin{equation}
\lambda\_{1}^{\star} c\_{1}\left(\mathbf{x}^{\star}\right)=0, \quad \lambda\_{2}^{\star} c\_{2}\left(\mathbf{x}^{\star}\right)=0.
\label{eq:two_constraint_complementarity}
\end{equation}

When $\mathbf{x}^{\star}=(-\sqrt{2}, 0)^{\mathrm{T}}$, we have

$$
\nabla f\left(\mathbf{x}^{\star}\right)=\begin{bmatrix} 1 \\\\ 1 \end{bmatrix}, \quad \nabla c\_{1}\left(\mathbf{x}^{\star}\right)=\begin{bmatrix} 2 \sqrt{2} \\\\ 0 \end{bmatrix}, \quad \nabla c\_{2}\left(\mathbf{x}^{\star}\right)=\begin{bmatrix} 0 \\\\ 1 \end{bmatrix},
$$

so that it is easy to verify that $\nabla\_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)=\mathbf{0}$ when we select $\boldsymbol{\lambda}^{\star}$ as follows:

$$
\boldsymbol{\lambda}^{\star}=\begin{bmatrix} 1 /(2 \sqrt{2}) \\\\ 1 \end{bmatrix}
$$

Note that both components of $\boldsymbol{\lambda}^{\star}$ are positive.

We consider now some other feasible points that are not solutions of \eqref{eq:two_inequality_example}, and examine the properties of the Lagrangian and its gradient at these points.

For the point $\mathbf{x}=(\sqrt{2}, 0)^{\mathrm{T}}$, we again have that both constraints are active. However, the objective gradient $\nabla f(\mathbf{x})$ no longer lies in the quadrant defined by the conditions $\nabla c\_{i}(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0, i=1,2$. One first-order feasible descent direction from this point-a vector $\mathbf{d}$ that satisfies the required conditions-is simply $\mathbf{d}=(-1,0)^{\mathrm{T}}$; there are many others. For this value of $\mathbf{x}$ it is easy to verify that the condition $\nabla\_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})=\mathbf{0}$ is satisfied only when $\boldsymbol{\lambda}=(-1 /(2 \sqrt{2}), 1)$. Note that the first component $\lambda\_{1}$ is negative, so that the conditions \eqref{eq:two_constraint_kkt} are not satisfied at this point.

Finally, let us consider the point $\mathbf{x}=(1,0)^{\mathrm{T}}$, at which only the second constraint $c\_{2}$ is active. At this point, linearization of $f$ and $c$ gives the following conditions, which must be satisfied for $\mathbf{d}$ to be a feasible descent direction, to first order:

$$
1+\nabla c\_{1}(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0, \quad \nabla c\_{2}(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0, \quad \nabla f(\mathbf{x})^{\mathrm{T}} \mathbf{d}<0 .
$$

In fact, we need worry only about satisfying the second and third conditions, since we can always satisfy the first condition by multiplying $\mathbf{d}$ by a sufficiently small positive quantity. By noting that

$
\nabla f(\mathbf{x})=\begin{bmatrix} 1 \\\\ 1 \end{bmatrix}, \quad \nabla c\_{2}(\mathbf{x})=\begin{bmatrix} 0 \\\\ 1 \end{bmatrix}
$

it is easy to verify that the vector $\mathbf{d}=\left(-\frac{1}{2}, \frac{1}{4}\right)$ satisfies the required conditions and is therefore a descent direction.

To show that optimality conditions \eqref{eq:two_constraint_kkt} and \eqref{eq:two_constraint_complementarity} fail, we note first from \eqref{eq:two_constraint_complementarity} that since $c\_{1}(\mathbf{x})>0$, we must have $\lambda\_{1}=0$. Therefore, in trying to satisfy $\nabla\_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})=\mathbf{0}$, we are left to search for a value $\lambda\_{2}$ such that $\nabla f(\mathbf{x})-\lambda\_{2} \nabla c\_{2}(\mathbf{x})=\mathbf{0}$. No such $\lambda\_{2}$ exists, and thus this point fails to satisfy the optimality conditions.

The following visualization summarizes the discussion of this example on the active constraints:
<iframe style="border:none;" scrolling="no" src="../../../../interactive/twoinequality_cosntraints_active.html" width="700px" height="500px" title="One single constraint"></iframe>


## First-order optimality conditions

### Statement of first-order necessary conditions

The three examples above suggest that a number of conditions are important in the characterization of solutions for the general problem. These include the relation $\nabla\_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})=\mathbf{0}$, the nonnegativity of $\lambda\_{i}$ for all inequality constraints $c\_{i}(\mathbf{x})$, and the complementarity condition $\lambda\_{i} c\_{i}(\mathbf{x})=0$ that is required for all the inequality constraints. We now generalize the observations made in these examples and state the first-order optimality conditions in a rigorous fashion.

In general, the Lagrangian for the constrained optimization problem is defined as

\begin{equation}
\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})=f(\mathbf{x})-\sum\_{i \in \mathcal{E} \cup \mathcal{I}} \lambda\_{i} c\_{i}(\mathbf{x}).
\label{eq:general_lagrangian}
\end{equation}

The active set $\mathcal{A}(\mathbf{x})$ at any feasible $\mathbf{x}$ is the union of the set $\mathcal{E}$ with the indices of the active inequality constraints; that is,

\begin{equation}
\mathcal{A}(\mathbf{x})=\mathcal{E} \cup\left\\{i \in \mathcal{I} \mid c\_{i}(\mathbf{x})=0\right\\}.
\label{eq:active_set}
\end{equation}

Next, we need to give more attention to the properties of the constraint gradients. The vector $\nabla c\_{i}(\mathbf{x})$ is often called the normal to the constraint $c\_{i}$ at the point $\mathbf{x}$, because it is usually a vector that is perpendicular to the contours of the constraint $c\_{i}$ at $\mathbf{x}$, and in the case of an inequality constraint, it points toward the feasible side of this constraint. It is possible, however, that $\nabla c\_{i}(\mathbf{x})$ vanishes due to the algebraic representation of $c\_{i}$, so that the term $\lambda\_{i} \nabla c\_{i}(\mathbf{x})$ vanishes for all values of $\lambda\_{i}$ and does not play a role in the Lagrangian gradient $\nabla\_{\mathbf{x}} \mathcal{L}$. For instance, if we replaced the constraint in \eqref{eq:equality_example} by the equivalent condition

$
c\_{1}(\mathbf{x})=\left(x\_{1}^{2}+x\_{2}^{2}-2\right)^{2}=0
$

we would have that $\nabla c\_{1}(\mathbf{x})=\mathbf{0}$ for all feasible points $\mathbf{x}$, and in particular that the condition $\nabla f(\mathbf{x})=\lambda\_{1} \nabla c\_{1}(\mathbf{x})$ no longer holds at the optimal point $(-1,-1)^{\mathrm{T}}$. We usually make an assumption called a constraint qualification to ensure that such degenerate behavior does not occur at the value of $\mathbf{x}$ in question. One such constraint qualification-probably the one most often used in the design of algorithms-is the one defined as follows:

{{<definition "Linear independence constraint qualification (LICQ)" licq>}}
Given the point $\mathbf{x}^{\star}$ and the active set $\mathcal{A}\left(\mathbf{x}^{\star}\right)$ defined by \eqref{eq:active_set}, we say that the **linear independence constraint qualification (LICQ)** holds if the set of active constraint gradients $\left\\{\nabla c\_{i}\left(\mathbf{x}^{\star}\right), i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)\right\\}$ is linearly independent.
{{</definition>}}

Note that if this condition holds, none of the active constraint gradients can be zero.

This condition allows us to state the following optimality conditions for a general nonlinear programming problem. These conditions provide the foundation for many of the algorithms described in the remaining chapters of the lecture. They are called first-order conditions because they concern themselves with properties of the gradients (first-derivative vectors) of the objective and constraint functions.

{{<theorem "First-order necessary conditions" first_order_necessary>}}
Suppose that $\mathbf{x}^{\star}$ is a local solution and that the LICQ holds at $\mathbf{x}^{\star}$. Then there is a Lagrange multiplier vector $\boldsymbol{\lambda}^{\star}$, with components $\lambda\_{i}^{\star}, i \in \mathcal{E} \cup \mathcal{I}$, such that the following conditions are satisfied at $\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)$:

\begin{align}
\nabla\_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)&=\mathbf{0}, \label{eq:kkt_gradient_zero} \\\\
c\_{i}\left(\mathbf{x}^{\star}\right)&=0, \quad && \text { for all } i \in \mathcal{E}, \label{eq:kkt_equality} \\\\
c\_{i}\left(\mathbf{x}^{\star}\right)&\geq 0, \quad && \text { for all } i \in \mathcal{I}, \label{eq:kkt_inequality} \\\\
\lambda\_{i}^{\star} &\geq 0, \quad && \text { for all } i \in \mathcal{I}, \label{eq:kkt_multiplier_sign} \\\\
\lambda\_{i}^{\star} c\_{i}\left(\mathbf{x}^{\star}\right)&=0, \quad && \text { for all } i \in \mathcal{E} \cup \mathcal{I}. \label{eq:kkt_complementarity}
\end{align}
{{</theorem>}}

The conditions \eqref{eq:kkt_gradient_zero}-\eqref{eq:kkt_complementarity} are often known as the Karush-Kuhn-Tucker conditions, or KKT conditions for short. Because the complementarity condition implies that the Lagrange multipliers corresponding to inactive inequality constraints are zero, we can omit the terms for indices $i \notin \mathcal{A}\left(\mathbf{x}^{\star}\right)$ from \eqref{eq:kkt_gradient_zero} and rewrite this condition as

\begin{equation}
\mathbf{0}=\nabla\_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)=\nabla f\left(\mathbf{x}^{\star}\right)-\sum\_{i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)} \lambda\_{i}^{\star} \nabla c\_{i}\left(\mathbf{x}^{\star}\right).
\label{eq:kkt_active_gradients}
\end{equation}

A special case of complementarity is important and deserves its own definition:

{{<definition "Strict complementarity" strict_complementarity>}}
Given a local solution $\mathbf{x}^{\star}$ and a vector $\boldsymbol{\lambda}^{\star}$ satisfying \eqref{eq:kkt_gradient_zero}-\eqref{eq:kkt_complementarity}, we say that the **strict complementarity condition** holds if exactly one of $\lambda\_{i}^{\star}$ and $c\_{i}\left(\mathbf{x}^{\star}\right)$ is zero for each index $i \in \mathcal{I}$. In other words, we have that $\lambda\_{i}^{\star}>0$ for each $i \in \mathcal{I} \cap \mathcal{A}\left(\mathbf{x}^{\star}\right)$.
{{</definition>}}

For a given problem and solution point $\mathbf{x}^{\star}$, there may be many vectors $\boldsymbol{\lambda}^{\star}$ for which the conditions \eqref{eq:kkt_gradient_zero}-\eqref{eq:kkt_complementarity} are satisfied. When the LICQ holds, however, the optimal $\boldsymbol{\lambda}^{\star}$ is unique.

**Example 4**

Consider the feasible region described by the four constraints:

\begin{equation}
\min\_{\mathbf{x}}\left(x\_{1}-\frac{3}{2}\right)^{2}+\left(x\_{2}-\frac{1}{8}\right)^{4} \quad \text { s.t. } \quad\begin{bmatrix} 1-x\_{1}-x\_{2} \\\\ 1-x\_{1}+x\_{2} \\\\ 1+x\_{1}-x\_{2} \\\\ 1+x\_{1}+x\_{2} \end{bmatrix} \geq \mathbf{0}.
\label{eq:diamond_example}
\end{equation}

It is fairly clear that the solution is $\mathbf{x}^{\star}=(1,0)$. The first and second constraints are active at this point. Denoting them by $c\_{1}$ and $c\_{2}$ (and the inactive constraints by $c\_{3}$ and $c\_{4}$ ), we have

$
\nabla f\left(\mathbf{x}^{\star}\right)=\begin{bmatrix} -1 \\\\ -\frac{1}{2} \end{bmatrix}, \quad \nabla c\_{1}\left(\mathbf{x}^{\star}\right)=\begin{bmatrix} -1 \\\\ -1 \end{bmatrix}, \quad \nabla c\_{2}\left(\mathbf{x}^{\star}\right)=\begin{bmatrix} -1 \\\\ 1 \end{bmatrix} .
$

Therefore, the KKT conditions \eqref{eq:kkt_gradient_zero}-\eqref{eq:kkt_complementarity} are satisfied when we set

$
\boldsymbol{\lambda}^{\star}=\left(\frac{3}{4}, \frac{1}{4}, 0,0\right)^{\mathrm{T}}.
$

### Sensitivity

The convenience of using Lagrange multipliers should now be clear, but what of their intuitive significance? The value of each Lagrange multiplier $\lambda\_{i}^{\star}$ tells us something about the sensitivity of the optimal objective value $f\left(\mathbf{x}^{\star}\right)$ to the presence of constraint $c\_{i}$. To put it another way, $\lambda\_{i}^{\star}$ indicates how hard $f$ is "pushing" or "pulling" against the particular constraint $c\_{i}$. We illustrate this point with a little analysis. When we choose an inactive constraint $i \notin \mathcal{A}\left(\mathbf{x}^{\star}\right)$ such that $c\_{i}\left(\mathbf{x}^{\star}\right)>0$, the solution $\mathbf{x}^{\star}$ and function value $f\left(\mathbf{x}^{\star}\right)$ are quite indifferent to whether this constraint is present or not. If we perturb $c\_{i}$ by a tiny amount, it will still be inactive and $\mathbf{x}^{\star}$ will still be a local solution of the optimization problem. Since $\lambda\_{i}^{\star}=0$ from \eqref{eq:kkt_complementarity}, the Lagrange multiplier indicates accurately that constraint $i$ is not significant.

Suppose instead that constraint $i$ is active, and let us perturb the right-hand-side of this constraint a little, requiring, say, that $c\_{i}(\mathbf{x}) \geq-\epsilon\left\\|\nabla c\_{i}\left(\mathbf{x}^{\star}\right)\right\\|$ instead of $c\_{i}(\mathbf{x}) \geq 0$. Suppose that $\epsilon$ is sufficiently small that the perturbed solution $\mathbf{x}^{\star}(\epsilon)$ still has the same set of active constraints, and that the Lagrange multipliers are not much affected by the perturbation. (These conditions can be made more rigorous with the help of strict complementarity and second-order conditions, as discussed later in the lecture.) We then find that

\begin{equation}
\begin{aligned}
-\epsilon\left\\|\nabla c\_{i}\left(\mathbf{x}^{\star}\right)\right\\| & =c\_{i}\left(\mathbf{x}^{\star}(\epsilon)\right)-c\_{i}\left(\mathbf{x}^{\star}\right) \approx\left(\mathbf{x}^{\star}(\epsilon)-\mathbf{x}^{\star}\right)^{\mathrm{T}} \nabla c\_{i}\left(\mathbf{x}^{\star}\right), \\\\
0 & =c\_{j}\left(\mathbf{x}^{\star}(\epsilon)\right)-c\_{j}\left(\mathbf{x}^{\star}\right) \approx\left(\mathbf{x}^{\star}(\epsilon)-\mathbf{x}^{\star}\right)^{\mathrm{T}} \nabla c\_{j}\left(\mathbf{x}^{\star}\right),
\end{aligned}
\label{eq:sensitivity_perturbation}
\end{equation}

for all $j \in \mathcal{A}\left(\mathbf{x}^{\star}\right)$ with $j \neq i$.

The value of $f\left(\mathbf{x}^{\star}(\epsilon)\right)$, meanwhile, can be estimated with the help of \eqref{eq:kkt_gradient_zero}. We have

\begin{equation}
\begin{aligned}
f\left(\mathbf{x}^{\star}(\epsilon)\right)-f\left(\mathbf{x}^{\star}\right) & \approx\left(\mathbf{x}^{\star}(\epsilon)-\mathbf{x}^{\star}\right)^{\mathrm{T}} \nabla f\left(\mathbf{x}^{\star}\right) \\\\
& =\sum\_{j \in \mathcal{A}\left(\mathbf{x}^{\star}\right)} \lambda\_{j}^{\star}\left(\mathbf{x}^{\star}(\epsilon)-\mathbf{x}^{\star}\right)^{\mathrm{T}} \nabla c\_{j}\left(\mathbf{x}^{\star}\right) \\\\
& \approx-\epsilon\left\\|\nabla c\_{i}\left(\mathbf{x}^{\star}\right)\right\\| \lambda\_{i}^{\star}
\end{aligned}
\label{eq:sensitivity_objective}
\end{equation}

By taking limits, we see that the family of solutions $\mathbf{x}^{\star}(\epsilon)$ satisfies

\begin{equation}
\frac{d f\left(\mathbf{x}^{\star}(\epsilon)\right)}{d \epsilon}=-\lambda\_{i}^{\star}\left\\|\nabla c\_{i}\left(\mathbf{x}^{\star}\right)\right\\|
\label{eq:sensitivity_derivative}
\end{equation}

A sensitivity analysis of this problem would conclude that if $\lambda\_{i}^{\star}\left\\|\nabla c\_{i}\left(\mathbf{x}^{\star}\right)\right\\|$ is large, then the optimal value is sensitive to the placement of the $i$th constraint, while if this quantity is small, the dependence is not too strong. If $\lambda\_{i}^{\star}$ is exactly zero for some active constraint, small perturbations to $c\_{i}$ in some directions will hardly affect the optimal objective value at all; the change is zero, to first order.

This discussion motivates the definition below, which classifies constraints according to whether or not their corresponding Lagrange multiplier is zero.

{{<definition "Strongly active and weakly active constraints" active_constraints>}}
Let $\mathbf{x}^{\star}$ be a solution of the optimization problem, and suppose that the KKT conditions are satisfied. We say that an inequality constraint $c\_{i}$ is **strongly active** or **binding** if $i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)$ and $\lambda\_{i}^{\star}>0$ for some Lagrange multiplier $\boldsymbol{\lambda}^{\star}$ satisfying the KKT conditions. We say that $c\_{i}$ is **weakly active** if $i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)$ and $\lambda\_{i}^{\star}=0$ for all $\boldsymbol{\lambda}^{\star}$ satisfying the KKT conditions.
{{</definition>}}

Note that the analysis above is independent of scaling of the individual constraints. For instance, we might change the formulation of the problem by replacing some active constraint $c\_{i}$ by $10 c\_{i}$. The new problem will actually be equivalent (that is, it has the same feasible set and same solution), but the optimal multiplier $\lambda\_{i}^{\star}$ corresponding to $c\_{i}$ will be replaced by $\lambda\_{i}^{\star} / 10$. However, since $\left\\|\nabla c\_{i}\left(\mathbf{x}^{\star}\right)\right\\|$ is replaced by $10\left\\|\nabla c\_{i}\left(\mathbf{x}^{\star}\right)\right\\|$, the product $\lambda\_{i}^{\star}\left\\|\nabla c\_{i}\left(\mathbf{x}^{\star}\right)\right\\|$ does not change. If, on the other hand, we replace the objective function $f$ by $10 f$, the multipliers $\lambda\_{i}^{\star}$ in the KKT conditions all will need to be replaced by $10 \lambda\_{i}^{\star}$. Hence in \eqref{eq:sensitivity_derivative} we see that the sensitivity of $f$ to perturbations has increased by a factor of 10, which is exactly what we would expect.

## Derivation of the first-order conditions

Having studied some motivating examples, observed the characteristics of optimal and nonoptimal points, and stated the KKT conditions, we now describe a complete proof of {{<theoremref first_order_necessary>}}. This analysis is not just of esoteric interest, but is rather the key to understanding all constrained optimization algorithms.

### Feasible sequences

The first concept we introduce is that of a feasible sequence. Given a feasible point $\mathbf{x}^{\star}$, a sequence $\left\\{\mathbf{z}\_{k}\right\\}\_{k=0}^{\infty}$ with $\mathbf{z}\_{k} \in \mathbb{R}^{n}$ is a feasible sequence if the following properties hold:
(i) $\mathbf{z}\_{k} \neq \mathbf{x}^{\star}$ for all $k$;
(ii) $\lim\_{k \rightarrow \infty} \mathbf{z}\_{k}=\mathbf{x}^{\star}$;
(iii) $\mathbf{z}\_{k}$ is feasible for all sufficiently large values of $k$.

For later reference, we denote the set of all possible feasible sequences approaching $\mathbf{x}$ by $\mathcal{T}(\mathbf{x})$.

We characterize a local solution as a point $\mathbf{x}$ at which all feasible sequences have the property that $f\left(\mathbf{z}\_{k}\right) \geq f(\mathbf{x})$ for all $k$ sufficiently large. We derive practical, verifiable conditions under which this property holds. To do so we will make use of the concept of a limiting direction of a feasible sequence.

Limiting directions of a feasible sequence are vectors $\mathbf{d}$ such that we have

\begin{equation}
\lim\_{\mathbf{z}\_{k} \in \mathcal{S}\_{\mathbf{d}}} \frac{\mathbf{z}\_{k}-\mathbf{x}}{\left\\|\mathbf{z}\_{k}-\mathbf{x}\right\\|} \rightarrow \mathbf{d}
\label{eq:limiting_direction}
\end{equation}

where $\mathcal{S}\_{\mathbf{d}}$ is some subsequence of $\left\\{\mathbf{z}\_{k}\right\\}\_{k=0}^{\infty}$. In general, a feasible sequence has at least one limiting direction and may have more than one. To see this, note that the sequence of vectors defined by

\begin{equation}
\mathbf{d}\_{k}=\frac{\mathbf{z}\_{k}-\mathbf{x}}{\left\\|\mathbf{z}\_{k}-\mathbf{x}\right\\|}
\label{eq:normalized_direction}
\end{equation}

lies on the surface of the unit sphere, which is a compact set, and thus there is at least one limit point $\mathbf{d}$. Moreover, all such points are limiting directions by the definition \eqref{eq:limiting_direction}. If we have some sequence $\left\\{\mathbf{z}\_{k}\right\\}$ with limiting direction $\mathbf{d}$ and corresponding subsequence $\mathcal{S}\_{\mathbf{d}}$, we can construct another feasible sequence $\left\\{\overline{\mathbf{z}}\_{k}\right\\}$ such that

\begin{equation}
\lim\_{k \rightarrow \infty} \frac{\overline{\mathbf{z}}\_{k}-\mathbf{x}}{\left\\|\overline{\mathbf{z}}\_{k}-\mathbf{x}\right\\|}=\mathbf{d}
\label{eq:unique_limiting_direction}
\end{equation}

(that is, with a unique limit point) by simply defining each $\overline{\mathbf{z}}\_{k}$ to be an element from the subsequence $\mathcal{S}\_{\mathbf{d}}$.

We illustrate these concepts by revisiting the equality-constrained example.

**Example 1** (Equality-constrained example, revisited)

The figure shows a closeup of the equality-constrained problem in which the feasible set is a circle of radius $\sqrt{2}$, near the nonoptimal point $\mathbf{x}=(-\sqrt{2}, 0)^{\mathrm{T}}$. The figure also shows a feasible sequence approaching $\mathbf{x}$. This sequence could be defined analytically by the formula

\begin{equation}
\mathbf{z}\_{k}=\begin{bmatrix} -\sqrt{2-1 / k^{2}} \\\\ -1 / k \end{bmatrix}.
\label{eq:example_sequence_1}
\end{equation}

The vector $\mathbf{d}=(0,-1)^{\mathrm{T}}$ is a limiting direction of this feasible sequence. Note that $\mathbf{d}$ is tangent to the feasible sequence at $\mathbf{x}$ but points in the opposite direction. The objective function $f(\mathbf{x})=x\_{1}+x\_{2}$ increases as we move along the sequence \eqref{eq:example_sequence_1}; in fact, we have $f\left(\mathbf{z}\_{k+1}\right)>f\left(\mathbf{z}\_{k}\right)$ for all $k=2,3, \ldots$. It follows that $f\left(\mathbf{z}\_{k}\right)<f(\mathbf{x})$ for $k=2,3, \ldots$. Hence, $\mathbf{x}$ cannot be a solution.

Another feasible sequence is one that approaches $\mathbf{x}^{\star}=(-\sqrt{2}, 0)^{\mathrm{T}}$ from the opposite direction. Its elements are defined by

\begin{equation}
\mathbf{z}\_{k}=\begin{bmatrix} -\sqrt{2-1 / k^{2}} \\\\ 1 / k \end{bmatrix}.
\label{eq:example_sequence_2}
\end{equation}

It is easy to show that $f$ decreases along this sequence and that its limiting direction is $\mathbf{d}=(0,1)^{\mathrm{T}}$. Other feasible sequences are obtained by combining elements from the two sequences already discussed, for instance

\begin{equation}
\mathbf{z}\_{k}= \begin{cases}\left(-\sqrt{2-1 / k^{2}}, 1 / k\right)^{\mathrm{T}}, & \text { when } k \text { is a multiple of } 3 \\\\ \left(-\sqrt{2-1 / k^{2}},-1 / k\right)^{\mathrm{T}}, & \text { otherwise. }\end{cases}
\label{eq:example_sequence_3}
\end{equation}

In general, feasible sequences of points approaching $(-\sqrt{2}, 0)^{\mathrm{T}}$ will have two limiting directions, $(0,1)^{\mathrm{T}}$ and $(0,-1)^{\mathrm{T}}$.

We now consider feasible sequences and limiting directions for an example that involves inequality constraints.

**Example 2** (Inequality-constrained example, revisited)

We now reconsider the inequality-constrained problem. The solution $\mathbf{x}^{\star}=(-1,-1)^{\mathrm{T}}$ is the same as in the equality-constrained case, but there is a much more extensive collection of feasible sequences that converge to any given feasible point. From the point $\mathbf{x}=(-\sqrt{2}, 0)^{\mathrm{T}}$, the various feasible sequences defined above for the equality-constrained problem are still feasible for the inequality-constrained problem. There are also infinitely many feasible sequences that converge to $\mathbf{x}=(-\sqrt{2}, 0)^{\mathrm{T}}$ along a straight line from the interior of the circle. These are defined by

\begin{equation}
\mathbf{z}\_{k}=(-1,0)^{\mathrm{T}}+(1 / k) \mathbf{w},
\label{eq:straight_line_sequence}
\end{equation}

where $\mathbf{w}$ is any vector whose first component is positive ($w\_{1}>0$). Now, $\mathbf{z}\_{k}$ is feasible, provided that $\left\\|\mathbf{z}\_{k}\right\\| \leq 1$, that is,

\begin{equation}
\left(-1+w\_{1} / k\right)^{2}+\left(w\_{2} / k\right)^{2} \leq 1,
\label{eq:feasibility_condition}
\end{equation}

a condition that is satisfied, provided that $k>\left(2 w\_{1}\right) /\left(w\_{1}^{2}+w\_{2}^{2}\right)$. In addition to these straight-line feasible sequences, we can also define an infinite variety of sequences that approach $(-\sqrt{2}, 0)^{\mathrm{T}}$ along a curve from the interior of the circle or that make the approach in a seemingly random fashion.

Given a point $\mathbf{x}$, if it is possible to choose a feasible sequence from $\mathcal{T}(\mathbf{x})$ such that the first-order approximation to the objective function actually increases monotonically along the sequence, then $\mathbf{x}$ must not be optimal. This condition is the fundamental first-order necessary condition, and we state it formally in the following theorem.

{{<theorem "First-order necessary condition for feasible sequences" feasible_sequence_necessary>}}
If $\mathbf{x}^{\star}$ is a local solution, then all feasible sequences $\left\\{\mathbf{z}\_{k}\right\\}$ in $\mathcal{T}\left(\mathbf{x}^{\star}\right)$ must satisfy

\begin{equation}
\nabla f\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{d} \geq 0
\label{eq:first_order_feasible_condition}
\end{equation}

where $\mathbf{d}$ is any limiting direction of the feasible sequence.
{{</theorem>}}

{{<proof>}}
Suppose that there is a feasible sequence $\left\\{\mathbf{z}\_{k}\right\\}$ with the property $\nabla f\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{d}<0$, for some limiting direction $\mathbf{d}$, and let $\mathcal{S}\_{\mathbf{d}}$ be the subsequence of $\left\\{\mathbf{z}\_{k}\right\\}$ that approaches $\mathbf{x}^{\star}$. By Taylor's theorem, we have for any $\mathbf{z}\_{k} \in \mathcal{S}\_{\mathbf{d}}$ that

\begin{equation}
\begin{aligned}
f\left(\mathbf{z}\_{k}\right) & =f\left(\mathbf{x}^{\star}\right)+\left(\mathbf{z}\_{k}-\mathbf{x}^{\star}\right)^{\mathrm{T}} \nabla f\left(\mathbf{x}^{\star}\right)+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|\right) \\\\
& =f\left(\mathbf{x}^{\star}\right)+\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\| \mathbf{d}^{\mathrm{T}} \nabla f\left(\mathbf{x}^{\star}\right)+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|\right)
\end{aligned}
\label{eq:taylor_expansion_proof}
\end{equation}

Since $\mathbf{d}^{\mathrm{T}} \nabla f\left(\mathbf{x}^{\star}\right)<0$, we have that the remainder term is eventually dominated by the first-order term, that is,

\begin{equation}
f\left(\mathbf{z}\_{k}\right)<f\left(\mathbf{x}^{\star}\right)+\frac{1}{2}\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\| \mathbf{d}^{\mathrm{T}} \nabla f\left(\mathbf{x}^{\star}\right), \quad \text { for all } k \text { sufficiently large. }
\label{eq:contradiction_inequality}
\end{equation}

Hence, given any open neighborhood of $\mathbf{x}^{\star}$, we can choose $k$ sufficiently large that $\mathbf{z}\_{k}$ lies within this neighborhood and has a lower value of the objective $f$. Therefore, $\mathbf{x}^{\star}$ is not a local solution.
{{</proof>}}

This theorem tells us why we can ignore constraints that are strictly inactive (that is, constraints for which $c\_{i}(\mathbf{x})>0$) in formulating optimality conditions. The theorem does not use the whole range of properties of the feasible sequence, but rather one specific property: the limiting directions of $\left\\{\mathbf{z}\_{k}\right\\}$. Because of the way in which the limiting directions are defined, it is clear that only the asymptotic behavior of the sequence is relevant, that is, its behavior for large values of the index $k$. If some constraint $i \in \mathcal{I}$ is inactive at $\mathbf{x}$, then we have $c\_{i}\left(\mathbf{z}\_{k}\right)>0$ for all $k$ sufficiently large, so that a constraint that is inactive at $\mathbf{x}$ is also inactive at all sufficiently advanced elements of the feasible sequence $\left\\{\mathbf{z}\_{k}\right\\}$.

### Characterizing limiting directions: constraint qualifications

{{<theoremref feasible_sequence_necessary>}} is quite general, but it is not very useful as stated, because it seems to require knowledge of all possible limiting directions for all feasible sequences $\mathcal{T}\left(\mathbf{x}^{\star}\right)$. In this section we show that constraint qualifications allow us to characterize the salient properties of $\mathcal{T}\left(\mathbf{x}^{\star}\right)$, and therefore make the condition \eqref{eq:first_order_feasible_condition} easier to verify.

One frequently used constraint qualification is the linear independence constrained qualification (LICQ) given in {{<definitionref "licq">}}. The following lemma shows that when LICQ holds, there is a neat way to characterize the set of all possible limiting directions $\mathbf{d}$ in terms of the gradients $\nabla c\_{i}\left(\mathbf{x}^{\star}\right)$ of the active constraints at $\mathbf{x}^{\star}$.

In subsequent results we introduce the notation $\mathbf{A}$ to represent the matrix whose rows are the active constraint gradients at the optimal point, that is,

\begin{equation}
\nabla c\_{i}^{\star}=\nabla c\_{i}\left(\mathbf{x}^{\star}\right), \quad \mathbf{A}^{\mathrm{T}}=\left[\nabla c\_{i}^{\star}\right]\_{i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)}, \quad \nabla f^{\star}=\nabla f\left(\mathbf{x}^{\star}\right),
\label{eq:matrix_notation}
\end{equation}

where the active set $\mathcal{A}\left(\mathbf{x}^{\star}\right)$ is defined as in \eqref{eq:active_set}.

{{<lemma "Characterization of limiting directions" limiting_directions>}}
The following two statements are true.
(i) If $\mathbf{d} \in \mathbb{R}^{n}$ is a limiting direction of a feasible sequence, then

\begin{equation}
\mathbf{d}^{\mathrm{T}} \nabla c\_{i}^{\star}=0, \quad \text { for all } i \in \mathcal{E}, \quad \mathbf{d}^{\mathrm{T}} \nabla c\_{i}^{\star} \geq 0, \quad \text { for all } i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I}.
\label{eq:limiting_direction_conditions}
\end{equation}

(ii) If \eqref{eq:limiting_direction_conditions} holds with $\left\\|\mathbf{d}\right\\|=1$ and the LICQ condition is satisfied, then $\mathbf{d} \in \mathbb{R}^{n}$ is a limiting direction of some feasible sequence.
{{</lemma>}}

{{<proof>}}
Without loss of generality, let us assume that all the constraints $c\_{i}(\cdot), i=1,2, \ldots, m$, are active. (We can arrive at this convenient ordering by simply dropping all inactive constraints—which are irrelevant in some neighborhood of $\mathbf{x}^{\star}$—and renumbering the active constraints that remain.)

To prove (i), let $\left\\{\mathbf{z}\_{k}\right\\} \in \mathcal{T}\left(\mathbf{x}^{\star}\right)$ be some feasible sequence for which $\mathbf{d}$ is a limiting direction, and assume (by taking a subsequence if necessary) that

\begin{equation}
\lim\_{k \rightarrow \infty} \frac{\mathbf{z}\_{k}-\mathbf{x}^{\star}}{\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|}=\mathbf{d}
\label{eq:limiting_definition}
\end{equation}

From this definition, we have that

\begin{equation}
\mathbf{z}\_{k}=\mathbf{x}^{\star}+\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\| \mathbf{d}+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|\right).
\label{eq:sequence_expansion}
\end{equation}

By taking $i \in \mathcal{E}$ and using Taylor's theorem, we have that

\begin{equation}
\begin{aligned}
0 & =\frac{1}{\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|} c\_{i}\left(\mathbf{z}\_{k}\right) \\\\
& =\frac{1}{\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|}\left[c\_{i}\left(\mathbf{x}^{\star}\right)+\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\| \nabla c\_{i}^{\star \mathrm{T}} \mathbf{d}+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|\right)\right] \\\\
& =\nabla c\_{i}^{\star \mathrm{T}} \mathbf{d}+\frac{o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|\right)}{\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|}
\end{aligned}
\label{eq:equality_constraint_proof}
\end{equation}

By taking the limit as $k \rightarrow \infty$, the last term in this expression vanishes, and we have $\nabla c\_{i}^{\star \mathrm{T}} \mathbf{d}=0$, as required. For the active inequality constraints $i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I}$, we have similarly that

\begin{equation}
\begin{aligned}
0 & \leq \frac{1}{\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|} c\_{i}\left(\mathbf{z}\_{k}\right) \\\\
& =\frac{1}{\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|}\left[c\_{i}\left(\mathbf{x}^{\star}\right)+\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\| \nabla c\_{i}^{\star \mathrm{T}} \mathbf{d}+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|\right)\right] \\\\
& =\nabla c\_{i}^{\star \mathrm{T}} \mathbf{d}+\frac{o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|\right)}{\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|}
\end{aligned}
\label{eq:inequality_constraint_proof}
\end{equation}

Hence, by a similar limiting argument, we have that $\nabla c\_{i}^{\star \mathrm{T}} \mathbf{d} \geq 0$, as required.

For (ii), we use the implicit function theorem. First, since the LICQ holds, we have from {{<definitionref licq>}} that the $m \times n$ matrix $\mathbf{A}$ of active constraint gradients has full row rank $m$. Let $\mathbf{Z}$ be a matrix whose columns are a basis for the null space of $\mathbf{A}$; that is,

\begin{equation}
\mathbf{Z} \in \mathbb{R}^{n \times(n-m)}, \quad \mathbf{Z} \text{ has full column rank }, \quad \mathbf{A} \mathbf{Z}=\mathbf{0}.
\label{eq:null_space_basis}
\end{equation}

Let $\mathbf{d}$ have the properties \eqref{eq:limiting_direction_conditions}, and suppose that $\left\\{t\_{k}\right\\}\_{k=0}^{\infty}$ is any sequence of positive scalars such $\lim\_{k \rightarrow \infty} t\_{k}=0$. Define the parametrized system of equations $\mathbf{R}: \mathbb{R}^{n} \times \mathbb{R} \rightarrow \mathbb{R}^{n}$ by

\begin{equation}
\mathbf{R}(\mathbf{z}, t)=\begin{bmatrix} \mathbf{c}(\mathbf{z})-t \mathbf{A} \mathbf{d} \\\\ \mathbf{Z}^{\mathrm{T}}\left(\mathbf{z}-\mathbf{x}^{\star}-t \mathbf{d}\right) \end{bmatrix}=\begin{bmatrix} \mathbf{0} \\\\ \mathbf{0} \end{bmatrix}
\label{eq:parametrized_system}
\end{equation}

We claim that for each $t=t\_{k}$, the solutions $\mathbf{z}=\mathbf{z}\_{k}$ of this system for small $t>0$ give a feasible sequence that approaches $\mathbf{x}^{\star}$.

Clearly, for $t=0$, the solution of \eqref{eq:parametrized_system} is $\mathbf{z}=\mathbf{x}^{\star}$, and the Jacobian of $\mathbf{R}$ at this point is

\begin{equation}
\nabla\_{\mathbf{z}} \mathbf{R}\left(\mathbf{x}^{\star}, 0\right)=\begin{bmatrix} \mathbf{A} \\\\ \mathbf{Z}^{\mathrm{T}} \end{bmatrix},
\label{eq:jacobian_matrix}
\end{equation}

which is nonsingular by construction of $\mathbf{Z}$. Hence, according to the implicit function theorem, the system \eqref{eq:parametrized_system} has a unique solution $\mathbf{z}\_{k}$ for all values of $t\_{k}$ sufficiently small. Moreover, we have from \eqref{eq:parametrized_system} and \eqref{eq:limiting_direction_conditions} that

\begin{equation}
\begin{aligned}
i \in \mathcal{E} & \Rightarrow c\_{i}\left(\mathbf{z}\_{k}\right)=t\_{k} \nabla c\_{i}^{\star \mathrm{T}} \mathbf{d}=0, \\\\
i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I} & \Rightarrow c\_{i}\left(\mathbf{z}\_{k}\right)=t\_{k} \nabla c\_{i}^{\star \mathrm{T}} \mathbf{d} \geq 0,
\end{aligned}
\label{eq:feasibility_proof}
\end{equation}

so that $\mathbf{z}\_{k}$ is indeed feasible. Also, for any positive value $t=\bar{t}>0$, we cannot have $\mathbf{z}(t)=\mathbf{x}^{\star}$, since otherwise by substituting $(\mathbf{z}, t)=\left(\mathbf{x}^{\star}, \bar{t}\right)$ into \eqref{eq:parametrized_system}, we obtain

\begin{equation}
\begin{bmatrix} \mathbf{c}\left(\mathbf{x}^{\star}\right)-\bar{t} \mathbf{A} \mathbf{d} \\\\ -\mathbf{Z}^{\mathrm{T}}(\bar{t} \mathbf{d}) \end{bmatrix}=\begin{bmatrix} \mathbf{0} \\\\ \mathbf{0} \end{bmatrix}.
\label{eq:contradiction_system}
\end{equation}

Since $\mathbf{c}\left(\mathbf{x}^{\star}\right)=\mathbf{0}$ (we have assumed that all constraints are active) and $\bar{t}>0$, we have from the full rank of the matrix in \eqref{eq:jacobian_matrix} that $\mathbf{d}=\mathbf{0}$, which contradicts $\left\\|\mathbf{d}\right\\|=1$. It follows that $\mathbf{z}\_{k}=\mathbf{z}\left(t\_{k}\right) \neq \mathbf{x}^{\star}$ for all $k$.

It remains to show that $\mathbf{d}$ is a limiting direction of $\left\\{\mathbf{z}\_{k}\right\\}$. Using the fact that $\mathbf{R}\left(\mathbf{z}\_{k}, t\_{k}\right)=\mathbf{0}$ for all $k$ together with Taylor's theorem, we find that

\begin{equation}
\begin{aligned}
\mathbf{0}=\mathbf{R}\left(\mathbf{z}\_{k}, t\_{k}\right) & =\begin{bmatrix} \mathbf{c}\left(\mathbf{z}\_{k}\right)-t\_{k} \mathbf{A} \mathbf{d} \\\\ \mathbf{Z}^{\mathrm{T}}\left(\mathbf{z}\_{k}-\mathbf{x}^{\star}-t\_{k} \mathbf{d}\right) \end{bmatrix} \\\\
& =\begin{bmatrix} \mathbf{A}\left(\mathbf{z}\_{k}-\mathbf{x}^{\star}\right)+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|\right)-t\_{k} \mathbf{A} \mathbf{d} \\\\ \mathbf{Z}^{\mathrm{T}}\left(\mathbf{z}\_{k}-\mathbf{x}^{\star}-t\_{k} \mathbf{d}\right) \end{bmatrix} \\\\
& =\begin{bmatrix} \mathbf{A} \\\\ \mathbf{Z}^{\mathrm{T}} \end{bmatrix}\left(\mathbf{z}\_{k}-\mathbf{x}^{\star}-t\_{k} \mathbf{d}\right)+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|\right)
\end{aligned}
\label{eq:taylor_system}
\end{equation}

By dividing this expression by $\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|$ and using nonsingularity of the coefficient matrix in the first term, we obtain

\begin{equation}
\lim\_{k \rightarrow \infty} \mathbf{d}\_{k}-\frac{t\_{k}}{\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|} \mathbf{d}=\mathbf{0}, \quad \text { where } \mathbf{d}\_{k}=\frac{\mathbf{z}\_{k}-\mathbf{x}^{\star}}{\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|}
\label{eq:direction_limit}
\end{equation}

Since $\left\\|\mathbf{d}\_{k}\right\\|=1$ for all $k$ and since $\left\\|\mathbf{d}\right\\|=1$, we must have

\begin{equation}
\lim\_{k \rightarrow \infty} \frac{t\_{k}}{\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|}=1
\label{eq:scaling_limit}
\end{equation}

(We leave the simple proof by contradiction of this statement as an exercise.) Hence, from \eqref{eq:direction_limit}, we have $\lim\_{k \rightarrow \infty} \mathbf{d}\_{k}=\mathbf{d}$, as required.
{{</proof>}}

The set of directions defined by \eqref{eq:limiting_direction_conditions} plays a central role in the optimality conditions, so for future reference we give this set a name and define it formally.

{{<definition "Linearized feasible directions" linearized_feasible>}}
Given a point $\mathbf{x}^{\star}$ and the active constraint set $\mathcal{A}\left(\mathbf{x}^{\star}\right)$ defined by \eqref{eq:active_set}, the set $F\_{1}$ is defined by

\begin{equation}
F\_{1}=\left\\{\alpha \mathbf{d} \mid \alpha>0, \quad \begin{array}{ll}
\mathbf{d}^{\mathrm{T}} \nabla c\_{i}^{\star}=0, & \text { for all } i \in \mathcal{E} \\\\
\mathbf{d}^{\mathrm{T}} \nabla c\_{i}^{\star} \geq 0, & \text { for all } i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I}
\end{array}\right\\}
\label{eq:linearized_feasible_set}
\end{equation}
{{</definition>}}

Note that $F\_{1}$ is a cone. In fact, when a constraint qualification is satisfied, $F\_{1}$ is the tangent cone to the feasible set at $\mathbf{x}^{\star}$.

### Introducing Lagrange multipliers
{{<lemmaref limiting_directions>}} tells us that when the LICQ holds, the cone $F\_{1}$ is simply the set of all positive multiples of all limiting directions of all possible feasible sequences. Therefore, the condition \eqref{eq:first_order_feasible_condition} of {{<theoremref feasible_sequence_necessary>}} holds if $\nabla f\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{d}<0$ for all $\mathbf{d} \in F\_{1}$. This condition, too, would appear to be impossible to check, since the set $F\_{1}$ contains infinitely many vectors in general. The next lemma gives an alternative, practical way to check this condition that makes use of the Lagrange multipliers, the variables $\lambda\_{i}$ that were introduced in the definition of the Lagrangian $\mathcal{L}$.
{{<lemma "Characterization using Lagrange multipliers" lagrange_characterization>}}
There is no direction $\mathbf{d} \in F\_{1}$ for which $\mathbf{d}^{\mathrm{T}} \nabla f^{\star}<0$ if and only if there exists a vector $\boldsymbol{\lambda} \in \mathbb{R}^{m}$ with
\begin{equation}
\nabla f^{\star}=\sum\_{i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)} \lambda\_{i} \nabla c\_{i}^{\star}=\mathbf{A}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \boldsymbol{\lambda}, \quad \lambda\_{i} \geq 0 \text { for } i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I}
\label{eq:lagrange_condition}
\end{equation}
{{</lemma>}}
{{<proof>}}
If we define the cone $N$ by
\begin{equation}
N=\left\\{\mathbf{s} \mid \mathbf{s}=\sum\_{i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)} \lambda\_{i} \nabla c\_{i}^{\star}, \quad \lambda\_{i} \geq 0 \text { for } i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I}\right\\}
\label{eq:cone_n_definition}
\end{equation}
then the condition \eqref{eq:lagrange_condition} is equivalent to $\nabla f^{\star} \in N$. We note first that the set $N$ is closed—a fact that is intuitively clear but nontrivial to prove rigorously.
We prove the forward implication by supposing that \eqref{eq:lagrange_condition} holds and choosing $\mathbf{d}$ to be any vector satisfying \eqref{eq:limiting_direction_conditions}. We then have that
\begin{equation}
\mathbf{d}^{\mathrm{T}} \nabla f^{\star}=\sum\_{i \in \mathcal{E}} \lambda\_{i}\left(\mathbf{d}^{\mathrm{T}} \nabla c\_{i}^{\star}\right)+\sum\_{i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I}} \lambda\_{i}\left(\mathbf{d}^{\mathrm{T}} \nabla c\_{i}^{\star}\right)
\label{eq:forward_implication}
\end{equation}
The first summation is zero because $\mathbf{d}^{\mathrm{T}} \nabla c\_{i}^{\star}=0$ for $i \in \mathcal{E}$, while the second term is nonnegative because $\lambda\_{i} \geq 0$ and $\mathbf{d}^{\mathrm{T}} \nabla c\_{i}^{\star} \geq 0$ for $i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I}$. Hence $\mathbf{d}^{\mathrm{T}} \nabla f^{\star} \geq 0$.
For the reverse implication, we show that if $\nabla f^{\star}$ does not satisfy \eqref{eq:lagrange_condition} (that is, $\nabla f^{\star} \notin N$), then we can find a vector $\mathbf{d}$ for which $\mathbf{d}^{\mathrm{T}} \nabla f^{\star}<0$ and \eqref{eq:limiting_direction_conditions} holds.
Let $\hat{\mathbf{s}}$ be the vector in $N$ that is closest to $\nabla f^{\star}$. Because $N$ is closed, $\hat{\mathbf{s}}$ is well-defined. In fact, $\hat{\mathbf{s}}$ solves the constrained optimization problem
\begin{equation}
\min \left\|\mathbf{s}-\nabla f^{\star}\right\|\_{2}^{2} \quad \text { subject to } \mathbf{s} \in N
\label{eq:projection_problem}
\end{equation}
Since $\hat{\mathbf{s}} \in N$, we also have $t \hat{\mathbf{s}} \in N$ for all scalars $t \geq 0$. Since $\left\|t \hat{\mathbf{s}}-\nabla f^{\star}\right\|\_{2}^{2}$ is minimized at $t=1$, we have
\begin{equation}
\begin{aligned}
\left.\frac{d}{d t}\left\|t \hat{\mathbf{s}}-\nabla f^{\star}\right\|{2}^{2}\right|{t=1}=0 & \Rightarrow\left.\left(-2 \hat{\mathbf{s}}^{\mathrm{T}} \nabla f^{\star}+2 t \hat{\mathbf{s}}^{\mathrm{T}} \hat{\mathbf{s}}\right)\right|\_{t=1}=0 \\
& \Rightarrow \hat{\mathbf{s}}^{\mathrm{T}}\left(\hat{\mathbf{s}}-\nabla f^{\star}\right)=0
\end{aligned}
\label{eq:orthogonality_condition}
\end{equation}
Now, let $\mathbf{s}$ be any other vector in $N$. Since $N$ is convex, we have by the minimizing property of $\hat{\mathbf{s}}$ that
\begin{equation}
\left\|\hat{\mathbf{s}}+\theta(\mathbf{s}-\hat{\mathbf{s}})-\nabla f^{\star}\right\|{2}^{2} \geq\left\|\hat{\mathbf{s}}-\nabla f^{\star}\right\|{2}^{2} \quad \text { for all } \theta \in[0,1]
\label{eq:convexity_property}
\end{equation}
and hence
\begin{equation}
2 \theta(\mathbf{s}-\hat{\mathbf{s}})^{\mathrm{T}}\left(\hat{\mathbf{s}}-\nabla f^{\star}\right)+\theta^{2}\left\|\mathbf{s}-\hat{\mathbf{s}}\right\|\_{2}^{2} \geq 0
\label{eq:quadratic_expansion}
\end{equation}
By dividing this expression by $\theta$ and taking the limit as $\theta \downarrow 0$, we have $(\mathbf{s}-\hat{\mathbf{s}})^{\mathrm{T}}\left(\hat{\mathbf{s}}-\nabla f^{\star}\right) \geq 0$. Therefore, because of \eqref{eq:orthogonality_condition},
\begin{equation}
\mathbf{s}^{\mathrm{T}}\left(\hat{\mathbf{s}}-\nabla f^{\star}\right) \geq 0, \quad \text { for all } \mathbf{s} \in N
\label{eq:separation_property}
\end{equation}
We claim now that the vector
\begin{equation}
\mathbf{d}=\hat{\mathbf{s}}-\nabla f^{\star}
\label{eq:descent_direction}
\end{equation}
satisfies both \eqref{eq:limiting_direction_conditions} and $\mathbf{d}^{\mathrm{T}} \nabla f^{\star}<0$. Note that $\mathbf{d} \neq \mathbf{0}$ because $\nabla f^{\star}$ does not belong to the cone $N$. We have from \eqref{eq:orthogonality_condition} that
\begin{equation}
\mathbf{d}^{\mathrm{T}} \nabla f^{\star}=\mathbf{d}^{\mathrm{T}}(\hat{\mathbf{s}}-\mathbf{d})=\left(\hat{\mathbf{s}}-\nabla f^{\star}\right)^{\mathrm{T}} \hat{\mathbf{s}}-\mathbf{d}^{\mathrm{T}} \mathbf{d}=-\left\|\mathbf{d}\right\|\_{2}^{2}<0
\label{eq:descent_property}
\end{equation}
so that $\mathbf{d}$ satisfies the descent property.
By making appropriate choices of coefficients $\lambda\_{i}, i=1,2, \ldots, m$, it is easy to see that
\begin{equation}
\begin{aligned}
i \in \mathcal{E} & \Rightarrow \nabla c\_{i}^{\star} \in N \quad \text { and } \quad -\nabla c\_{i}^{\star} \in N \\
i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I} & \Rightarrow \nabla c\_{i}^{\star} \in N
\end{aligned}
\label{eq:gradient_membership}
\end{equation}
Hence, from \eqref{eq:separation_property}, we have by substituting $\mathbf{d}=\hat{\mathbf{s}}-\nabla f^{\star}$ and the particular choices $\mathbf{s}=\nabla c\_{i}^{\star}$ and $\mathbf{s}=-\nabla c\_{i}^{\star}$ that
\begin{equation}
\begin{aligned}
i \in \mathcal{E} & \Rightarrow \mathbf{d}^{\mathrm{T}} \nabla c\_{i}^{\star} \geq 0 \quad \text { and } \quad -\mathbf{d}^{\mathrm{T}} \nabla c\_{i}^{\star} \geq 0 \Rightarrow \mathbf{d}^{\mathrm{T}} \nabla c\_{i}^{\star}=0 \\\\
i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I} & \Rightarrow \mathbf{d}^{\mathrm{T}} \nabla c\_{i}^{\star} \geq 0
\end{aligned}
\label{eq:direction_conditions_verified}
\end{equation}
Therefore, $\mathbf{d}$ also satisfies \eqref{eq:limiting_direction_conditions}, so the reverse implication is proved.
{{</proof>}}
Proof of the first-order necessary conditions
{{<lemmaref limiting_directions>}} and {{<lemmaref lagrange_characterization>}} can be combined to give the KKT conditions described in {{<theoremref first_order_necessary>}}. Suppose that $\mathbf{x}^{\star} \in \mathbb{R}^{n}$ is a feasible point at which the LICQ holds. The theorem claims that if $\mathbf{x}^{\star}$ is a local solution, then there is a vector $\boldsymbol{\lambda}^{\star} \in \mathbb{R}^{m}$ that satisfies the KKT conditions.
We show first that there are multipliers $\lambda\_{i}, i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)$, such that \eqref{eq:lagrange_condition} is satisfied. {{<theoremref feasible_sequence_necessary>}} tells us that $\mathbf{d}^{\mathrm{T}} \nabla f^{\star} \geq 0$ for all vectors $\mathbf{d}$ that are limiting directions of feasible sequences. From {{<lemmaref limiting_directions>}}, we know that when LICQ holds, the set of all possible limiting directions is exactly the set of vectors that satisfy the conditions \eqref{eq:limiting_direction_conditions}. By putting these two statements together, we find that all directions $\mathbf{d}$ that satisfy \eqref{eq:limiting_direction_conditions} must also have $\mathbf{d}^{\mathrm{T}} \nabla f^{\star} \geq 0$. Hence, from {{<lemmaref lagrange_characterization>}}, we have that there is a vector $\boldsymbol{\lambda}$ for which \eqref{eq:lagrange_condition} holds, as claimed.
We now define the vector $\boldsymbol{\lambda}^{\star}$ by
\begin{equation}
\lambda\_{i}^{\star}= \begin{cases}\lambda\_{i}, & i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \\ 0, & \text { otherwise }\end{cases}
\label{eq:multiplier_definition}
\end{equation}
and show that this choice of $\boldsymbol{\lambda}^{\star}$, together with our local solution $\mathbf{x}^{\star}$, satisfies the KKT conditions. We check these conditions in turn.

The condition \eqref{eq:kkt_gradient_zero} follows immediately from \eqref{eq:lagrange_condition} and the definitions of the Lagrangian function and \eqref{eq:multiplier_definition} of $\boldsymbol{\lambda}^{\star}$.
Since $\mathbf{x}^{\star}$ is feasible, the conditions \eqref{eq:kkt_equality} and \eqref{eq:kkt_inequality} are satisfied.
We have from \eqref{eq:lagrange_condition} that $\lambda\_{i}^{\star} \geq 0$ for $i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I}$, while from \eqref{eq:multiplier_definition}, $\lambda\_{i}^{\star}=0$ for $i \in \mathcal{I} \backslash \mathcal{A}\left(\mathbf{x}^{\star}\right)$. Hence, $\lambda\_{i}^{\star} \geq 0$ for $i \in \mathcal{I}$, so that \eqref{eq:kkt_multiplier_sign} holds.
We have for $i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I}$ that $c\_{i}\left(\mathbf{x}^{\star}\right)=0$, while for $i \in \mathcal{I} \backslash \mathcal{A}\left(\mathbf{x}^{\star}\right)$, we have $\lambda\_{i}^{\star}=0$. Hence $\lambda\_{i}^{\star} c\_{i}\left(\mathbf{x}^{\star}\right)=0$ for $i \in \mathcal{I}$, so that \eqref{eq:kkt_complementarity} is satisfied as well.

This completes the proof.

## Second-order conditions

So far, we have described the first-order conditions—the KKT conditions—which tell us how the first derivatives of $f$ and the active constraints $c\_{i}$ are related at $\mathbf{x}^{\star}$. When these conditions are satisfied, a move along any vector $\mathbf{w}$ from $F\_{1}$ either increases the first-order approximation to the objective function (that is, $\mathbf{w}^{\mathrm{T}} \nabla f\left(\mathbf{x}^{\star}\right)>0$), or else keeps this value the same (that is, $\mathbf{w}^{\mathrm{T}} \nabla f\left(\mathbf{x}^{\star}\right)=0$).

What implications does optimality have for the second derivatives of $f$ and the constraints $c\_{i}$? We see in this section that these derivatives play a "tiebreaking" role. For the directions $\mathbf{w} \in F\_{1}$ for which $\mathbf{w}^{\mathrm{T}} \nabla f\left(\mathbf{x}^{\star}\right)=0$, we cannot determine from first derivative information alone whether a move along this direction will increase or decrease the objective function $f$. Second-order conditions examine the second derivative terms in the Taylor series expansions of $f$ and $c\_{i}$, to see whether this extra information resolves the issue of increase or decrease in $f$. Essentially, the second-order conditions concern the curvature of the Lagrangian function in the "undecided" directions—the directions $\mathbf{w} \in F\_{1}$ for which $\mathbf{w}^{\mathrm{T}} \nabla f\left(\mathbf{x}^{\star}\right)=0$.

Since we are discussing second derivatives, stronger smoothness assumptions are needed here than in the previous sections. For the purpose of this section, $f$ and $c\_{i}, i \in \mathcal{E} \cup \mathcal{I}$, are all assumed to be twice continuously differentiable.

Given $F\_{1}$ from {{<definitionref linearized_feasible>}} and some Lagrange multiplier vector $\boldsymbol{\lambda}^{\star}$ satisfying the KKT conditions, we define a subset $F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$ of $F\_{1}$ by

\begin{equation}
F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)=\left\\{\mathbf{w} \in F\_{1} \mid \nabla c\_{i}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{w}=0, \text { all } i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I} \text { with } \lambda\_{i}^{\star}>0\right\\}
\label{eq:f2_definition}
\end{equation}

Equivalently,

\begin{equation}
\mathbf{w} \in F\_{2}\left(\boldsymbol{\lambda}^{\star}\right) \Leftrightarrow \begin{cases}\nabla c\_{i}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{w}=0, & \text { for all } i \in \mathcal{E}, \\\\ \nabla c\_{i}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{w}=0, & \text { for all } i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I} \text { with } \lambda\_{i}^{\star}>0, \\\\ \nabla c\_{i}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{w} \geq 0, & \text { for all } i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I} \text { with } \lambda\_{i}^{\star}=0 .\end{cases}
\label{eq:f2_conditions}
\end{equation}

The subset $F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$ contains the directions $\mathbf{w}$ that tend to "adhere" to the active inequality constraints for which the Lagrange multiplier component $\lambda\_{i}^{\star}$ is positive, as well as to the equality constraints. From the definition \eqref{eq:f2_definition} and the fact that $\lambda\_{i}^{\star}=0$ for all inactive components $i \in \mathcal{I} \backslash \mathcal{A}\left(\mathbf{x}^{\star}\right)$, it follows immediately that

\begin{equation}
\mathbf{w} \in F\_{2}\left(\boldsymbol{\lambda}^{\star}\right) \Rightarrow \lambda\_{i}^{\star} \nabla c\_{i}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{w}=0 \text { for all } i \in \mathcal{E} \cup \mathcal{I}.
\label{eq:f2_property}
\end{equation}

Hence, from the first KKT condition \eqref{eq:kkt_gradient_zero} and the definition of the Lagrangian function, we have that

\begin{equation}
\mathbf{w} \in F\_{2}\left(\boldsymbol{\lambda}^{\star}\right) \Rightarrow \mathbf{w}^{\mathrm{T}} \nabla f\left(\mathbf{x}^{\star}\right)=\sum\_{i \in \mathcal{E} \cup \mathcal{I}} \lambda\_{i}^{\star} \mathbf{w}^{\mathrm{T}} \nabla c\_{i}\left(\mathbf{x}^{\star}\right)=0
\label{eq:f2_gradient_zero}
\end{equation}

Hence the set $F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$ contains directions from $F\_{1}$ for which it is not clear from first derivative information alone whether $f$ will increase or decrease.

The first theorem defines a necessary condition involving the second derivatives: If $\mathbf{x}^{\star}$ is a local solution, then the curvature of the Lagrangian along directions in $F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$ must be nonnegative.

{{<theorem "Second-order necessary conditions" second_order_necessary>}}
Suppose that $\mathbf{x}^{\star}$ is a local solution and that the LICQ condition is satisfied. Let $\boldsymbol{\lambda}^{\star}$ be a Lagrange multiplier vector such that the KKT conditions are satisfied, and let $F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$ be defined as above. Then

\begin{equation}
\mathbf{w}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \mathbf{w} \geq 0, \quad \text { for all } \mathbf{w} \in F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)
\label{eq:second_order_necessary}
\end{equation}
{{</theorem>}}

{{<proof>}}
Since $\mathbf{x}^{\star}$ is a local solution, all feasible sequences $\left\\{\mathbf{z}\_{k}\right\\}$ approaching $\mathbf{x}^{\star}$ must have $f\left(\mathbf{z}\_{k}\right) \geq f\left(\mathbf{x}^{\star}\right)$ for all $k$ sufficiently large. Our approach in this proof is to construct a feasible sequence whose limiting direction is $\mathbf{w} /\left\\|\mathbf{w}\right\\|$ and show that the property $f\left(\mathbf{z}\_{k}\right) \geq f\left(\mathbf{x}^{\star}\right)$ implies that \eqref{eq:second_order_necessary} holds.

Since $\mathbf{w} \in F\_{2}\left(\boldsymbol{\lambda}^{\star}\right) \subset F\_{1}$, we can use the technique in the proof of {{<lemmaref limiting_directions>}} to construct a feasible sequence $\left\\{\mathbf{z}\_{k}\right\\}$ such that

\begin{equation}
\lim\_{k \rightarrow \infty} \frac{\mathbf{z}\_{k}-\mathbf{x}^{\star}}{\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|}=\frac{\mathbf{w}}{\left\\|\mathbf{w}\right\\|}
\label{eq:sequence_limit}
\end{equation}

In particular, we have from the construction that

\begin{equation}
c\_{i}\left(\mathbf{z}\_{k}\right)=\frac{t\_{k}}{\left\\|\mathbf{w}\right\\|} \nabla c\_{i}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{w}, \quad \text { for all } i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)
\label{eq:constraint_values}
\end{equation}

where $\left\\{t\_{k}\right\\}$ is some sequence of positive scalars decreasing to zero. Moreover, we have that

\begin{equation}
\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|=t\_{k}+o\left(t\_{k}\right)
\label{eq:distance_relation}
\end{equation}

and so by substitution, we obtain

\begin{equation}
\mathbf{z}\_{k}-\mathbf{x}^{\star}=\frac{t\_{k}}{\left\\|\mathbf{w}\right\\|} \mathbf{w}+o\left(t\_{k}\right)
\label{eq:difference_expansion}
\end{equation}

From the definition of the Lagrangian and \eqref{eq:constraint_values}, we have that

\begin{equation}
\begin{aligned}
\mathcal{L}\left(\mathbf{z}\_{k}, \boldsymbol{\lambda}^{\star}\right) &=f\left(\mathbf{z}\_{k}\right)-\sum\_{i \in \mathcal{E} \cup \mathcal{I}} \lambda\_{i}^{\star} c\_{i}\left(\mathbf{z}\_{k}\right) \\\\
&=f\left(\mathbf{z}\_{k}\right)-\frac{t\_{k}}{\left\\|\mathbf{w}\right\\|} \sum\_{i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)} \lambda\_{i}^{\star} \nabla c\_{i}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{w} \\\\
&=f\left(\mathbf{z}\_{k}\right)
\end{aligned}
\label{eq:lagrangian_simplification}
\end{equation}

where the last equality follows from the critical property \eqref{eq:f2_property}. On the other hand, we can perform a Taylor series expansion to obtain an estimate of $\mathcal{L}\left(\mathbf{z}\_{k}, \boldsymbol{\lambda}^{\star}\right)$ near $\mathbf{x}^{\star}$. By using Taylor's theorem and continuity of the Hessians $\nabla^{2} f$ and $\nabla^{2} c\_{i}, i \in \mathcal{E} \cup \mathcal{I}$, we obtain

\begin{equation}
\begin{aligned}
\mathcal{L}\left(\mathbf{z}\_{k}, \boldsymbol{\lambda}^{\star}\right)= & \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)+\left(\mathbf{z}\_{k}-\mathbf{x}^{\star}\right)^{\mathrm{T}} \nabla\_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \\\\
& +\frac{1}{2}\left(\mathbf{z}\_{k}-\mathbf{x}^{\star}\right)^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)\left(\mathbf{z}\_{k}-\mathbf{x}^{\star}\right)+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|^{2}\right)
\end{aligned}
\label{eq:taylor_expansion}
\end{equation}

By the complementarity conditions \eqref{eq:kkt_complementarity} the first term on the right-hand-side of this expression is equal to $f\left(\mathbf{x}^{\star}\right)$. From \eqref{eq:kkt_gradient_zero}, the second term is zero. Hence we can rewrite \eqref{eq:taylor_expansion} as

\begin{equation}
\mathcal{L}\left(\mathbf{z}\_{k}, \boldsymbol{\lambda}^{\star}\right)=f\left(\mathbf{x}^{\star}\right)+\frac{1}{2}\left(\mathbf{z}\_{k}-\mathbf{x}^{\star}\right)^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)\left(\mathbf{z}\_{k}-\mathbf{x}^{\star}\right)+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|^{2}\right)
\label{eq:simplified_taylor}
\end{equation}

By using \eqref{eq:difference_expansion} and \eqref{eq:distance_relation}, we have for the second-order term and the remainder term that

\begin{equation}
\begin{aligned}
& \frac{1}{2}\left(\mathbf{z}\_{k}-\mathbf{x}^{\star}\right)^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)\left(\mathbf{z}\_{k}-\mathbf{x}^{\star}\right)+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|^{2}\right) \\\\
& =\frac{1}{2}\left(t\_{k} /\left\\|\mathbf{w}\right\\|\right)^{2} \mathbf{w}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \mathbf{w}+o\left(t\_{k}^{2}\right)
\end{aligned}
\label{eq:second_order_term}
\end{equation}

Hence, by substituting this expression together with \eqref{eq:lagrangian_simplification} into \eqref{eq:simplified_taylor}, we obtain

\begin{equation}
f\left(\mathbf{z}\_{k}\right)=f\left(\mathbf{x}^{\star}\right)+\frac{1}{2}\left(t\_{k} /\left\\|\mathbf{w}\right\\|\right)^{2} \mathbf{w}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \mathbf{w}+o\left(t\_{k}^{2}\right)
\label{eq:objective_expansion}
\end{equation}

If $\mathbf{w}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \mathbf{w}<0$, then \eqref{eq:objective_expansion} would imply that $f\left(\mathbf{z}\_{k}\right)<f\left(\mathbf{x}^{\star}\right)$ for all $k$ sufficiently large, contradicting the fact that $\mathbf{x}^{\star}$ is a local solution. Hence, the condition \eqref{eq:second_order_necessary} must hold, as claimed.
{{</proof>}}

Sufficient conditions are conditions on $f$ and $c\_{i}, i \in \mathcal{E} \cup \mathcal{I}$, that ensure that $\mathbf{x}^{\star}$ is a local solution. (They take the opposite tack to necessary conditions, which assume that $\mathbf{x}^{\star}$ is a local solution and deduce properties of $f$ and $c\_{i}$.) The second-order sufficient condition stated in the next theorem looks very much like the necessary condition just discussed, but it differs in that the constraint qualification is not required, and the inequality in \eqref{eq:second_order_necessary} is replaced by a strict inequality.

{{<theorem "Second-order sufficient conditions" second_order_sufficient>}}
Suppose that for some feasible point $\mathbf{x}^{\star} \in \mathbb{R}^{n}$ there is a Lagrange multiplier vector $\boldsymbol{\lambda}^{\star}$ such that the KKT conditions are satisfied. Suppose also that

\begin{equation}
\mathbf{w}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \mathbf{w}>0, \quad \text { for all } \mathbf{w} \in F\_{2}\left(\boldsymbol{\lambda}^{\star}\right), \mathbf{w} \neq \mathbf{0}
\label{eq:second_order_sufficient}
\end{equation}

Then $\mathbf{x}^{\star}$ is a strict local solution.
{{</theorem>}}

{{<proof>}}
The result is proved if we can show that for any feasible sequence $\left\\{\mathbf{z}\_{k}\right\\}$ approaching $\mathbf{x}^{\star}$, we have that $f\left(\mathbf{z}\_{k}\right)>f\left(\mathbf{x}^{\star}\right)$ for all $k$ sufficiently large.

Given any feasible sequence, we have from {{<lemmaref limiting_directions>}}(i) and {{<definitionref linearized_feasible>}} that all its limiting directions $\mathbf{d}$ satisfy $\mathbf{d} \in F\_{1}$. Choose a particular limiting direction $\mathbf{d}$ whose associated subsequence $\mathcal{S}\_{\mathbf{d}}$ satisfies \eqref{eq:limiting_direction}. In other words, we have for all $k \in \mathcal{S}\_{\mathbf{d}}$ that

\begin{equation}
\mathbf{z}\_{k}-\mathbf{x}^{\star}=\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\| \mathbf{d}+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|\right)
\label{eq:limiting_direction_expansion}
\end{equation}

From the definition of the Lagrangian, we have that

\begin{equation}
\mathcal{L}\left(\mathbf{z}\_{k}, \boldsymbol{\lambda}^{\star}\right)=f\left(\mathbf{z}\_{k}\right)-\sum\_{i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)} \lambda\_{i}^{\star} c\_{i}\left(\mathbf{z}\_{k}\right) \leq f\left(\mathbf{z}\_{k}\right)
\label{eq:lagrangian_bound}
\end{equation}

while the Taylor series approximation \eqref{eq:simplified_taylor} from the proof of {{<theoremref second_order_necessary>}} continues to hold.

We know that $\mathbf{d} \in F\_{1}$, but suppose first that it is not in $F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$. We can then identify some index $j \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \cap \mathcal{I}$ such that the strict positivity condition

\begin{equation}
\lambda\_{j}^{\star} \nabla c\_{j}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{d}>0
\label{eq:strict_positivity}
\end{equation}

is satisfied, while for the remaining indices $i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)$, we have

\begin{equation}
\lambda\_{i}^{\star} \nabla c\_{i}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{d} \geq 0
\label{eq:nonnegativity}
\end{equation}

From Taylor's theorem and \eqref{eq:limiting_direction_expansion}, we have for all $k \in \mathcal{S}\_{\mathbf{d}}$ and for this particular value of $j$ that

\begin{equation}
\begin{aligned}
\lambda\_{j}^{\star} c\_{j}\left(\mathbf{z}\_{k}\right) & =\lambda\_{j}^{\star} c\_{j}\left(\mathbf{x}^{\star}\right)+\lambda\_{j}^{\star} \nabla c\_{j}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}}\left(\mathbf{z}\_{k}-\mathbf{x}^{\star}\right)+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|\right) \\\\
& =\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\| \lambda\_{j}^{\star} \nabla c\_{j}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{d}+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|\right)
\end{aligned}
\label{eq:constraint_taylor}
\end{equation}

Hence, from \eqref{eq:lagrangian_bound}, we have for $k \in \mathcal{S}\_{\mathbf{d}}$ that

\begin{equation}
\begin{aligned}
\mathcal{L}\left(\mathbf{z}\_{k}, \boldsymbol{\lambda}^{\star}\right) & =f\left(\mathbf{z}\_{k}\right)-\sum\_{i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)} \lambda\_{i}^{\star} c\_{i}\left(\mathbf{z}\_{k}\right) \\\\
& \leq f\left(\mathbf{z}\_{k}\right)-\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\| \lambda\_{j}^{\star} \nabla c\_{j}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{d}+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|\right)
\end{aligned}
\label{eq:lagrangian_inequality}
\end{equation}

From the Taylor series estimate \eqref{eq:simplified_taylor}, we have meanwhile that

\begin{equation}
\mathcal{L}\left(\mathbf{z}\_{k}, \boldsymbol{\lambda}^{\star}\right)=f\left(\mathbf{x}^{\star}\right)+O\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|^{2}\right)
\label{eq:lagrangian_taylor}
\end{equation}

and by combining with \eqref{eq:lagrangian_inequality}, we obtain

\begin{equation}
f\left(\mathbf{z}\_{k}\right) \geq f\left(\mathbf{x}^{\star}\right)+\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\| \lambda\_{j}^{\star} \nabla c\_{j}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{d}+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|\right)
\label{eq:objective_lower_bound}
\end{equation}

Therefore, because of \eqref{eq:strict_positivity}, we have $f\left(\mathbf{z}\_{k}\right)>f\left(\mathbf{x}^{\star}\right)$ for all $k \in \mathcal{S}\_{\mathbf{d}}$ sufficiently large.

For the other case of $\mathbf{d} \in F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$, we use \eqref{eq:limiting_direction_expansion}, \eqref{eq:lagrangian_bound}, and \eqref{eq:simplified_taylor} to write

\begin{equation}
\begin{aligned}
f\left(\mathbf{z}\_{k}\right) & \geq f\left(\mathbf{x}^{\star}\right)+\frac{1}{2}\left(\mathbf{z}\_{k}-\mathbf{x}^{\star}\right)^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)\left(\mathbf{z}\_{k}-\mathbf{x}^{\star}\right)+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|^{2}\right) \\\\
& =f\left(\mathbf{x}^{\star}\right)+\frac{1}{2}\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|^{2} \mathbf{d}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \mathbf{d}+o\left(\left\\|\mathbf{z}\_{k}-\mathbf{x}^{\star}\right\\|^{2}\right)
\end{aligned}
\label{eq:second_order_case}
\end{equation}

Because of \eqref{eq:second_order_sufficient}, we again have $f\left(\mathbf{z}\_{k}\right)>f\left(\mathbf{x}^{\star}\right)$ for all $k \in \mathcal{S}\_{\mathbf{d}}$ sufficiently large.

Since this reasoning applies to all limiting directions of $\left\\{\mathbf{z}\_{k}\right\\}$, and since each element $\mathbf{z}\_{k}$ of the sequence can be assigned to one of the subsequences $\mathcal{S}\_{\mathbf{d}}$ that converge to one of these limiting directions, we conclude that $f\left(\mathbf{z}\_{k}\right)>f\left(\mathbf{x}^{\star}\right)$ for all $k$ sufficiently large.
{{</proof>}}

**Example 3** (Inequality-constrained example, one more time)

We now return to the inequality-constrained example to check the second-order conditions. In this problem we have $f(\mathbf{x})=x\_{1}+x\_{2}$, $c\_{1}(\mathbf{x})=2-x\_{1}^{2}-x\_{2}^{2}$, $\mathcal{E}=\emptyset$, and $\mathcal{I}=\\{1\\}$. The Lagrangian is

\begin{equation}
\mathcal{L}(\mathbf{x}, \lambda)=\left(x\_{1}+x\_{2}\right)-\lambda\_{1}\left(2-x\_{1}^{2}-x\_{2}^{2}\right)
\label{eq:example_lagrangian}
\end{equation}

and it is easy to show that the KKT conditions are satisfied by $\mathbf{x}^{\star}=(-1,-1)^{\mathrm{T}}$, with $\lambda\_{1}^{\star}=\frac{1}{2}$. The Lagrangian Hessian at this point is

\begin{equation}
\nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)=\begin{bmatrix} 2 \lambda\_{1}^{\star} & 0 \\\\ 0 & 2 \lambda\_{1}^{\star} \end{bmatrix}=\begin{bmatrix} 1 & 0 \\\\ 0 & 1 \end{bmatrix}
\label{eq:example_hessian}
\end{equation}

This matrix is positive definite, that is, it satisfies $\mathbf{w}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \mathbf{w}>0$ for all $\mathbf{w} \neq \mathbf{0}$, so it certainly satisfies the conditions of {{<theoremref second_order_sufficient>}}. We conclude that $\mathbf{x}^{\star}=(-1,-1)^{\mathrm{T}}$ is a strict local solution. (In fact, it is the global solution of this problem, since this problem is a convex programming problem.)

**Example 4**

For an example in which the issues are more complex, consider the problem

\begin{equation}
\min -0.1\left(x\_{1}-4\right)^{2}+x\_{2}^{2} \quad \text { s.t. } \quad x\_{1}^{2}+x\_{2}^{2}-1 \geq 0,
\label{eq:nonconvex_example}
\end{equation}

in which we seek to minimize a nonconvex function over the exterior of the unit circle. Obviously, the objective function is not bounded below on the feasible region, since we can take the feasible sequence

\begin{equation}
\begin{bmatrix} 10 \\\\ 0 \end{bmatrix}, \quad \begin{bmatrix} 20 \\\\ 0 \end{bmatrix}, \quad \begin{bmatrix} 30 \\\\ 0 \end{bmatrix}, \quad \begin{bmatrix} 40 \\\\ 0 \end{bmatrix},
\label{eq:unbounded_sequence}
\end{equation}

and note that $f(\mathbf{x})$ approaches $-\infty$ along this sequence. Therefore, no global solution exists, but it may still be possible to identify a strict local solution on the boundary of the constraint. We search for such a solution by using the KKT conditions and the second-order conditions of {{<theoremref second_order_sufficient>}}.

By defining the Lagrangian for \eqref{eq:nonconvex_example} in the usual way, it is easy to verify that

\begin{equation}
\begin{aligned}
\nabla\_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \lambda) & =\begin{bmatrix} -0.2\left(x\_{1}-4\right)-2 \lambda x\_{1} \\\\ 2 x\_{2}-2 \lambda x\_{2} \end{bmatrix}, \\\\
\nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}(\mathbf{x}, \lambda) & =\begin{bmatrix} -0.2-2 \lambda & 0 \\\\ 0 & 2-2 \lambda \end{bmatrix}.
\end{aligned}
\label{eq:example_derivatives}
\end{equation}

The point $\mathbf{x}^{\star}=(1,0)^{\mathrm{T}}$ satisfies the KKT conditions with $\lambda\_{1}^{\star}=0.3$ and the active set $\mathcal{A}\left(\mathbf{x}^{\star}\right)=\\{1\\}$. To check that the second-order sufficient conditions are satisfied at this point, we note that

\begin{equation}
\nabla c\_{1}\left(\mathbf{x}^{\star}\right)=\begin{bmatrix} 2 \\\\ 0 \end{bmatrix},
\label{eq:constraint_gradient}
\end{equation}

so that the space $F\_{2}$ defined in \eqref{eq:f2_definition} is simply

\begin{equation}
F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)=\left\\{\mathbf{w} \mid w\_{1}=0\right\\}=\left\\{\left(0, w\_{2}\right)^{\mathrm{T}} \mid w\_{2} \in \mathbb{R}\right\\}
\label{eq:f2_example}
\end{equation}

Now, by substituting $\mathbf{x}^{\star}$ and $\boldsymbol{\lambda}^{\star}$ into \eqref{eq:example_derivatives}, we have for any $\mathbf{w} \in F\_{2}$ with $\mathbf{w} \neq \mathbf{0}$ that

\begin{equation}
\mathbf{w}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \mathbf{w}=\begin{bmatrix} 0 \\\\ w\_{2} \end{bmatrix}^{\mathrm{T}}\begin{bmatrix} -0.4 & 0 \\\\ 0 & 1.4 \end{bmatrix}\begin{bmatrix} 0 \\\\ w\_{2} \end{bmatrix}=1.4 w\_{2}^{2}>0
\label{eq:positive_definiteness}
\end{equation}

Hence, the second-order sufficient conditions are satisfied, and we conclude from {{<theoremref second_order_sufficient>}} that $(1,0)^{\mathrm{T}}$ is a strict local solution for \eqref{eq:nonconvex_example}.

## Second-order conditions and projected Hessians

The second-order conditions are sometimes stated in a form that is weaker but easier to verify than \eqref{eq:second_order_necessary} and \eqref{eq:second_order_sufficient}. This form uses a two-sided projection of the Lagrangian Hessian $\nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)$ onto subspaces that are related to $F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$.

The simplest case is obtained when the multiplier $\boldsymbol{\lambda}^{\star}$ that satisfies the KKT conditions is unique (as happens, for example, when the LICQ condition holds) and strict complementarity holds. In this case, the definition \eqref{eq:f2_definition} of $F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$ reduces to

\begin{equation}
F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)=\operatorname{Null}\left[\nabla c\_{i}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}}\right]\_{i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)}=\operatorname{Null} \mathbf{A},
\label{eq:f2_null_space}
\end{equation}

where $\mathbf{A}$ is defined as in \eqref{eq:matrix_notation}. In other words, $F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$ is the null space of the matrix whose rows are the active constraint gradients at $\mathbf{x}^{\star}$. As in \eqref{eq:null_space_basis}, we can define the matrix $\mathbf{Z}$ with full column rank whose columns span the space $F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$. Any vector $\mathbf{w} \in F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$ can be written as $\mathbf{w}=\mathbf{Z} \mathbf{u}$ for some vector $\mathbf{u}$, and conversely, we have that $\mathbf{Z} \mathbf{u} \in F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$ for all $\mathbf{u}$. Hence, the condition \eqref{eq:second_order_necessary} in {{<theoremref second_order_necessary>}} can be restated as

\begin{equation}
\mathbf{u}^{\mathrm{T}} \mathbf{Z}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \mathbf{Z} \mathbf{u} \geq 0 \text { for all } \mathbf{u}
\label{eq:projected_necessary}
\end{equation}

or, more succinctly,

\begin{equation}
\mathbf{Z}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \mathbf{Z} \text { is positive semidefinite.}
\label{eq:projected_psd}
\end{equation}

Similarly, the condition \eqref{eq:second_order_sufficient} in {{<theoremref second_order_sufficient>}} can be restated as

\begin{equation}
\mathbf{Z}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \mathbf{Z} \text { is positive definite.}
\label{eq:projected_pd}
\end{equation}

We see at the end of this section that $\mathbf{Z}$ can be computed numerically, so that the positive (semi)definiteness conditions can actually be checked by forming these matrices and finding their eigenvalues.

When the optimal multiplier $\boldsymbol{\lambda}^{\star}$ is unique but the strict complementarity condition is not satisfied, $F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$ is no longer a subspace. Instead, it is an intersection of planes (defined by the first two conditions in \eqref{eq:f2_conditions}) and half-spaces (defined by the third condition in \eqref{eq:f2_conditions}). We can still, however, define two subspaces $\overline{F}\_{2}$ and $\underline{F}\_{2}$ that "bound" $F\_{2}$ above and below, in the sense that $\overline{F}\_{2}$ is the smallest-dimensional subspace that contains $F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$, while $\underline{F}\_{2}$ is the largest-dimensional subspace contained in $F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$. To be precise, we have

\begin{equation}
\underline{F}\_{2}=\left\\{\mathbf{d} \in F\_{1} \mid \nabla c\_{i}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{d}=0, \text { all } i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)\right\\}
\label{eq:f2_lower}
\end{equation}

\begin{equation}
\overline{F}\_{2}=\left\\{\mathbf{d} \in F\_{1} \mid \nabla c\_{i}\left(\mathbf{x}^{\star}\right)^{\mathrm{T}} \mathbf{d}=0, \text { all } i \in \mathcal{A}\left(\mathbf{x}^{\star}\right) \text { with } i \in \mathcal{E} \text { or } \lambda\_{i}^{\star}>0\right\\}
\label{eq:f2_upper}
\end{equation}

so that

\begin{equation}
\underline{F}\_{2} \subset F\_{2}\left(\boldsymbol{\lambda}^{\star}\right) \subset \overline{F}\_{2}.
\label{eq:f2_containment}
\end{equation}

As in the previous case, we can construct matrices $\underline{\mathbf{Z}}$ and $\overline{\mathbf{Z}}$ whose columns span the subspaces $\underline{F}\_{2}$ and $\overline{F}\_{2}$, respectively. If the condition \eqref{eq:second_order_necessary} of {{<theoremref second_order_necessary>}} holds, we can be sure that

\begin{equation}
\mathbf{w}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \mathbf{w} \geq 0, \quad \text { for all } \mathbf{w} \in \underline{F}\_{2},
\label{eq:lower_bound_condition}
\end{equation}

because $\underline{F}\_{2} \subset F\_{2}\left(\boldsymbol{\lambda}^{\star}\right)$. Therefore, an immediate consequence of \eqref{eq:second_order_necessary} is that the matrix $\underline{\mathbf{Z}}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \underline{\mathbf{Z}}$ is positive semidefinite.

Analogously, we have from $F\_{2}\left(\boldsymbol{\lambda}^{\star}\right) \subset \overline{F}\_{2}$ that condition \eqref{eq:second_order_sufficient} is implied by the condition

\begin{equation}
\mathbf{w}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \mathbf{w}>0, \quad \text { for all } \mathbf{w} \in \overline{F}\_{2}
\label{eq:upper_bound_condition}
\end{equation}

Hence, given that the $\boldsymbol{\lambda}^{\star}$ satisfying the KKT conditions is unique, a sufficient condition for \eqref{eq:second_order_sufficient} is that the matrix $\overline{\mathbf{Z}}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \overline{\mathbf{Z}}$ be positive definite. Again, this condition provides a practical way to check the second-order sufficient condition.

The matrices $\underline{\mathbf{Z}}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \underline{\mathbf{Z}}$ and $\overline{\mathbf{Z}}^{\mathrm{T}} \nabla\_{\mathbf{x}\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right) \overline{\mathbf{Z}}$ are sometimes called two-sided projected Hessian matrices, or simply projected Hessians for short.

One way to compute the matrix $\mathbf{Z}$ (and its counterparts $\underline{\mathbf{Z}}$ and $\overline{\mathbf{Z}}$) is to apply a QR factorization to the matrix of active constraint gradients whose null space we seek. In the simplest case above (in which the multiplier $\boldsymbol{\lambda}^{\star}$ is unique and strictly complementary), we define $\mathbf{A}$ as in \eqref{eq:matrix_notation} and write the QR factorization of $\mathbf{A}^{\mathrm{T}}$ as

\begin{equation}
\mathbf{A}^{\mathrm{T}}=\mathbf{Q}\begin{bmatrix} \mathbf{R} \\\\ \mathbf{0} \end{bmatrix}=\begin{bmatrix} \mathbf{Q}\_{1} & \mathbf{Q}\_{2} \end{bmatrix}\begin{bmatrix} \mathbf{R} \\\\ \mathbf{0} \end{bmatrix}=\mathbf{Q}\_{1} \mathbf{R}
\label{eq:qr_factorization}
\end{equation}

where $\mathbf{R}$ is a square upper triangular matrix, and $\mathbf{Q}$ is $n \times n$ orthogonal. If $\mathbf{R}$ is nonsingular, we can set $\mathbf{Z}=\mathbf{Q}\_{2}$. If $\mathbf{R}$ is singular (indicating that the active constraint gradients are linearly dependent), a slight enhancement of this procedure that makes use of column pivoting during the QR procedure can be used to identify $\mathbf{Z}$.
