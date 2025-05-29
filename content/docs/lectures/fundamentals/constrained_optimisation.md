---
title: 6. Constrained optimization - introduction
weight: 6
math: true
chapter: 6
---

# Constrained optimization methods

The second part of this lecture is about minimizing functions subject to constraints on the variables. A general formulation for these problems is

$$
\min_{\mathbf{x} \in \mathrm{R}^{n}} f(\mathbf{x}) \quad \text { subject to } \quad \begin{cases}c_{i}(\mathbf{x})=0, & i \in \mathcal{E}, \\\\ c_{i}(\mathbf{x}) \geq 0, & i \in \mathcal{I},\end{cases}
$$

where $f$ and the functions $c_{i}$ are all smooth, real-valued functions on a subset of $\mathbb{R}^{n}$, and $\mathcal{I}$ and $\mathcal{E}$ are two finite sets of indices. As before, we call $f$ the objective function, while $c_{i}$, $i \in \mathcal{E}$ are the equality constraints and $c_{i}, i \in \mathcal{I}$ are the inequality constraints. We define the feasible set $\Omega$ to be the set of points $\mathbf{x}$ that satisfy the constraints; that is,

$$
\Omega=\left\\{\mathbf{x} \mid c_{i}(\mathbf{x})=0, \quad i \in \mathcal{E} ; \quad c_{i}(\mathbf{x}) \geq 0, \quad i \in \mathcal{I}\right\\}
$$

so that we can rewrite the problem more compactly as

\begin{equation}
\min_{\mathbf{x} \in \Omega} f(\mathbf{x}).
\label{eq:constrained_problem}
\end{equation}

In this chapter we derive mathematical characterizations of the solutions of \eqref{eq:constrained_problem}. Recall that for the unconstrained optimization problem, we characterized solution points $\mathbf{x}^{\star}$ in the following way:

Necessary conditions: Local minima of unconstrained problems have $\nabla f\left(\mathbf{x}^{\star}\right)=0$ and $\nabla^{2} f\left(\mathbf{x}^{\star}\right)$ positive semidefinite.

Sufficient conditions: Any point $\mathbf{x}^{\star}$ at which $\nabla f\left(\mathbf{x}^{\star}\right)=0$ and $\nabla^{2} f\left(\mathbf{x}^{\star}\right)$ is positive definite is a strong local minimizer of $f$.

Our aim in this chapter is to derive similar conditions to characterize the solutions of constrained optimization problems.

## Local and global solutions

We have seen already that global solutions are difficult to find even when there are no constraints. The situation may be improved when we add constraints, since the feasible set might exclude many of the local minima and it may be comparatively easy to pick the global minimum from those that remain. However, constraints can also make things much more difficult. As an example, consider the problem

$$
\min_{\mathbf{x} \in \mathrm{R}^{n}}\\|\mathbf{x}\\|_{2}^{2}, \quad \text { subject to }\\|\mathbf{x}\\|\_{2}^{2} \geq 1
$$

Without the constraint, this is a convex quadratic problem with unique minimizer $\mathbf{x}=\mathbf{0}$. When the constraint is added, any vector $\mathbf{x}$ with $\\|\mathbf{x}\\|_{2}=1$ solves the problem. There are infinitely many such vectors (hence, infinitely many local minima) whenever $n \geq 2$.

A second example shows how addition of a constraint produces a large number of local solutions that do not form a connected set. Consider

$$
\min \left(x_{2}+100\right)^{2}+0.01 x_{1}^{2}, \quad \text { subject to } x_{2}-\cos x_{1} \geq 0
$$

Without the constraint, the problem has the unique solution $(-100,0)$. With the constraint there are local solutions near the points

$$
\left(x_{1}, x_{2}\right)=(k \pi,-1), \quad \text { for } \quad k= \pm 1, \pm 3, \pm 5, \ldots
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
\\|\mathbf{x}\\|_{1}=\left|x\_{1}\right|+\left|x\_{2}\right| \leq 1 .
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
x_{1}+x_{2} \leq 1, \quad x_{1}-x_{2} \leq 1, \quad -x_{1}+x_{2} \leq 1, \quad -x_{1}-x_{2} \leq 1
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

In the examples above we expressed inequality constraints in a slightly different way from the form $c_{i}(\mathbf{x}) \geq 0$ that appears in the definition. However, any collection of inequality constraints with $\geq$ and $\leq$ and nonzero right-hand-sides can be expressed in the form $c_{i}(\mathbf{x}) \geq 0$ by simple rearrangement of the inequality. In general, it is good practice to state the constraint in a way that is intuitive and easy to understand.

### Examples

To introduce the basic principles behind the characterization of solutions of constrained optimization problems, we work through three simple examples. The ideas discussed here will be made rigorous in the sections that follow.

We start by noting one item of terminology that recurs throughout the rest of the lecture: At a feasible point $\mathbf{x}$, the inequality constraint $i \in \mathcal{I}$ is said to be active if $c_{i}(\mathbf{x})=0$ and inactive if the strict inequality $c_{i}(\mathbf{x})>0$ is satisfied.

#### A single equality constraint

**Example 1**

Our first example is a two-variable problem with a single equality constraint:

\begin{equation}
\min x_{1}+x_{2} \quad \text { s.t. } \quad x_{1}^{2}+x_{2}^{2}-2=0.
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


In the general form, we have $f(\mathbf{x})=x_{1}+x_{2}, \mathcal{I}=\emptyset, \mathcal{E}=\\{1\\}$, and $c_{1}(\mathbf{x})=x_{1}^{2}+x_{2}^{2}-2$. We can see by inspection that the feasible set for this problem is the circle of radius $\sqrt{2}$ centered at the origin-just the boundary of this circle, not its interior. The solution $\mathbf{x}^{\star}$ is obviously $(-1,-1)^{\mathrm{T}}$. From any other point on the circle, it is easy to find a way to move that stays feasible (that is, remains on the circle) while decreasing $f$. For instance, from the point $\mathbf{x}=(\sqrt{2}, 0)^{\mathrm{T}}$ any move in the clockwise direction around the circle has the desired effect.

We also see that at the solution $\mathbf{x}^{\star}$, the constraint normal $\nabla c_{1}\left(\mathbf{x}^{\star}\right)$ is parallel to $\nabla f\left(\mathbf{x}^{\star}\right)$. That is, there is a scalar $\lambda_{1}^{\star}$ such that

\begin{equation}
\nabla f\left(\mathbf{x}^{\star}\right)=\lambda_{1}^{\star} \nabla c_{1}\left(\mathbf{x}^{\star}\right).
\label{eq:parallel_gradients}
\end{equation}

(In this particular case, we have $\lambda_{1}^{\star}=-\frac{1}{2}$.)

We can derive \eqref{eq:parallel_gradients} by examining first-order Taylor series approximations to the objective and constraint functions. To retain feasibility with respect to the function $c_{1}(\mathbf{x})=0$, we require that $c_{1}(\mathbf{x}+\mathbf{d})=0$; that is,

$$
0=c_{1}(\mathbf{x}+\mathbf{d}) \approx c_{1}(\mathbf{x})+\nabla c_{1}(\mathbf{x})^{\mathrm{T}} \mathbf{d}=\nabla c_{1}(\mathbf{x})^{\mathrm{T}} \mathbf{d}
$$

Hence, the direction $\mathbf{d}$ retains feasibility with respect to $c_{1}$, to first order, when it satisfies

$$
\nabla c_{1}(\mathbf{x})^{\mathrm{T}} \mathbf{d}=0
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

By drawing a picture (see visualization below), the reader can check that the only way that such a direction cannot exist is if $\nabla f(\mathbf{x})$ and $\nabla c_{1}(\mathbf{x})$ are parallel, that is, if the condition $\nabla f(\mathbf{x})=\lambda_{1} \nabla c_{1}(\mathbf{x})$ holds at $\mathbf{x}$, for some scalar $\lambda_{1}$. If this condition is not satisfied, the direction defined by

$$
\mathbf{d}=-\left(\mathbf{I}-\frac{\nabla c_{1}(\mathbf{x}) \nabla c_{1}(\mathbf{x})^{\mathrm{T}}}{\\|\nabla c_{1}(\mathbf{x})\\|^{2}}\right) \nabla f(\mathbf{x})
$$

satisfies both conditions.

By introducing the Lagrangian function

$$
\mathcal{L}\left(\mathbf{x}, \lambda_{1}\right)=f(\mathbf{x})-\lambda_{1} c_{1}(\mathbf{x}),
$$

and noting that $\nabla_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}, \lambda_{1}\right)=\nabla f(\mathbf{x})-\lambda_{1} \nabla c_{1}(\mathbf{x})$, we can state the condition \eqref{eq:parallel_gradients} equivalently as follows: At the solution $\mathbf{x}^{\star}$, there is a scalar $\lambda_{1}^{\star}$ such that

\begin{equation}
\nabla_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \lambda_{1}^{\star}\right)=0.
\label{eq:lagrangian_gradient_zero}
\end{equation}

This observation suggests that we can search for solutions of the equality-constrained problem \eqref{eq:equality_example} by searching for stationary points of the Lagrangian function. The scalar quantity $\lambda_{1}$ is called a Lagrange multiplier for the constraint $c_{1}(\mathbf{x})=0$.

Though the condition \eqref{eq:parallel_gradients} (equivalently, \eqref{eq:lagrangian_gradient_zero}) appears to be necessary for an optimal solution of the problem \eqref{eq:equality_example}, it is clearly not sufficient. For instance, in this example, \eqref{eq:parallel_gradients} is satisfied at the point $\mathbf{x}=(1,1)$ (with $\lambda_{1}=\frac{1}{2}$ ), but this point is obviously not a solution-in fact, it maximizes the function $f$ on the circle. Moreover, in the case of equality-constrained problems, we cannot turn the condition \eqref{eq:parallel_gradients} into a sufficient condition simply by placing some restriction on the sign of $\lambda_{1}$. To see this, consider replacing the constraint $x_{1}^{2}+x_{2}^{2}-2=0$ by its negative $2-x_{1}^{2}-x_{2}^{2}=0$. The solution of the problem is not affected, but the value of $\lambda_{1}^{\star}$ that satisfies the condition \eqref{eq:parallel_gradients} changes from $\lambda_{1}^{\star}=-\frac{1}{2}$ to $\lambda_{1}^{\star}=\frac{1}{2}$.

This situation is illustrated in following visualization:

<iframe style="border:none;" scrolling="no" src="../../../../interactive/onesingle_constraint.html" width="700px" height="500px" title="One single constraint"></iframe>

#### A single inequality constraint

**Example 2**

This is a slight modification of Example 1, in which the equality constraint is replaced by an inequality. Consider

\begin{equation}
\min x_{1}+x_{2} \quad \text { s.t. } \quad 2-x_{1}^{2}-x_{2}^{2} \geq 0,
\label{eq:inequality_example}
\end{equation}

for which the feasible region consists of the circle of problem \eqref{eq:equality_example} and its interior. Note that the constraint normal $\nabla c_{1}$ points toward the interior of the feasible region at each point on the boundary of the circle. By inspection, we see that the solution is still $(-1,-1)$ and that the condition \eqref{eq:parallel_gradients} holds for the value $\lambda_{1}^{\star}=\frac{1}{2}$. However, this inequality-constrained problem differs from the equality-constrained problem \eqref{eq:equality_example} in that the sign of the Lagrange multiplier plays a significant role, as we now argue.

As before, we conjecture that a given feasible point $\mathbf{x}$ is not optimal if we can find a step $\mathbf{d}$ that both retains feasibility and decreases the objective function $f$ to first order. The main difference between problems \eqref{eq:equality_example} and \eqref{eq:inequality_example} comes in the handling of the feasibility condition. The direction $\mathbf{d}$ improves the objective function, to first order, if $\nabla f(\mathbf{x})^{\mathrm{T}} \mathbf{d}<0$. Meanwhile, the direction $\mathbf{d}$ retains feasibility if

$$
0 \leq c_{1}(\mathbf{x}+\mathbf{d}) \approx c_{1}(\mathbf{x})+\nabla c_{1}(\mathbf{x})^{\mathrm{T}} \mathbf{d}
$$

so, to first order, feasibility is retained if

$$
c_{1}(\mathbf{x})+\nabla c_{1}(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0
$$

In determining whether a direction $\mathbf{d}$ exists that satisfies both conditions, we consider the following two cases:

**Case I:** Consider first the case in which $\mathbf{x}$ lies strictly inside the circle, so that the strict inequality $c_{1}(\mathbf{x})>0$ holds. In this case, any vector $\mathbf{d}$ satisfies the feasibility condition, provided only that its length is sufficiently small. In particular, whenever $\nabla f\left(\mathbf{x}^{\star}\right) \neq \mathbf{0}$, we can obtain a direction $\mathbf{d}$ that satisfies both conditions by setting

$$
\mathbf{d}=-c_{1}(\mathbf{x}) \frac{\nabla f(\mathbf{x})}{\\|\nabla f(\mathbf{x})\\|}
$$

The only situation in which such a direction fails to exist is when

$$
\nabla f(\mathbf{x})=\mathbf{0} .
$$

**Case II:** Consider now the case in which $\mathbf{x}$ lies on the boundary of the circle, so that $c_{1}(\mathbf{x})=0$. The conditions therefore become

$$
\nabla f(\mathbf{x})^{\mathrm{T}} \mathbf{d}<0, \quad \nabla c_{1}(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0
$$

The first of these conditions defines an open half-space, while the second defines a closed half-space. It is clear that the two regions fail to intersect only when $\nabla f(\mathbf{x})$ and $\nabla c_{1}(\mathbf{x})$ point in the same direction, that is, when

\begin{equation}
\nabla f(\mathbf{x})=\lambda_{1} \nabla c_{1}(\mathbf{x}), \quad \text { for some } \lambda_{1} \geq 0.
\label{eq:inequality_optimality}
\end{equation}

Note that the sign of the multiplier is significant here. If \eqref{eq:parallel_gradients} were satisfied with a negative value of $\lambda_{1}$, then $\nabla f(\mathbf{x})$ and $\nabla c_{1}(\mathbf{x})$ would point in opposite directions, and we see that the set of directions that satisfy both conditions would make up an entire open half-plane.

The optimality conditions for both cases I and II can again be summarized neatly with reference to the Lagrangian function. When no first-order feasible descent direction exists at some point $\mathbf{x}^{\star}$, we have that

\begin{equation}
\nabla_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \lambda_{1}^{\star}\right)=\mathbf{0}, \quad \text { for some } \lambda_{1}^{\star} \geq 0,
\label{eq:kkt_gradient}
\end{equation}

where we also require that

\begin{equation}
\lambda_{1}^{\star} c_{1}\left(\mathbf{x}^{\star}\right)=0.
\label{eq:complementarity}
\end{equation}

This condition is known as a complementarity condition; it implies that the Lagrange multiplier $\lambda_{1}$ can be strictly positive only when the corresponding constraint $c_{1}$ is active. Conditions of this type play a central role in constrained optimization, as we see in the sections that follow. In case I, we have that $c_{1}\left(\mathbf{x}^{\star}\right)>0$, so \eqref{eq:complementarity} requires that $\lambda_{1}^{\star}=0$. Hence, \eqref{eq:kkt_gradient} reduces to $\nabla f\left(\mathbf{x}^{\star}\right)=\mathbf{0}$, as required. In case II, \eqref{eq:complementarity} allows $\lambda_{1}^{\star}$ to take on a nonnegative value, so \eqref{eq:kkt_gradient} becomes equivalent to \eqref{eq:inequality_optimality}.

## Two inequality constraints

**Example 3**

Suppose we add an extra constraint to the problem \eqref{eq:inequality_example} to obtain

\begin{equation}
\min x_{1}+x_{2} \quad \text { s.t. } \quad 2-x_{1}^{2}-x_{2}^{2} \geq 0, \quad x_{2} \geq 0,
\label{eq:two_inequality_example}
\end{equation}

for which the feasible region is the half-disk. It is easy to see that the solution lies at $(-\sqrt{2}, 0)^{\mathrm{T}}$, a point at which both constraints are active. By repeating the arguments for the previous examples, we conclude that a direction $\mathbf{d}$ is a feasible descent direction, to first order, if it satisfies the following conditions:

$$
\nabla c_{i}(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0, \quad i \in \mathcal{I}=\\{1,2\\}, \quad \nabla f(\mathbf{x})^{\mathrm{T}} \mathbf{d}<0
$$

However, it is clear that no such direction can exist when $\mathbf{x}=(-\sqrt{2}, 0)^{\mathrm{T}}$. The conditions $\nabla c_{i}(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0, i=1,2$, are both satisfied only if $\mathbf{d}$ lies in the quadrant defined by $\nabla c_{1}(\mathbf{x})$ and $\nabla c_{2}(\mathbf{x})$, but it is clear by inspection that all vectors $\mathbf{d}$ in this quadrant satisfy $\nabla f(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0$.

Let us see how the Lagrangian and its derivatives behave for the problem \eqref{eq:two_inequality_example} and the solution point $(-\sqrt{2}, 0)^{\mathrm{T}}$. First, we include an additional term $\lambda_{i} c_{i}(\mathbf{x})$ in the Lagrangian for each additional constraint, so we have

\begin{equation}
\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})=f(\mathbf{x})-\lambda_{1} c_{1}(\mathbf{x})-\lambda_{2} c_{2}(\mathbf{x}),
\label{eq:two_constraint_lagrangian}
\end{equation}

where $\boldsymbol{\lambda}=\left(\lambda_{1}, \lambda_{2}\right)^{\mathrm{T}}$ is the vector of Lagrange multipliers. The extension of condition \eqref{eq:kkt_gradient} to this case is

\begin{equation}
\nabla_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)=\mathbf{0}, \quad \text { for some } \boldsymbol{\lambda}^{\star} \geq \mathbf{0},
\label{eq:two_constraint_kkt}
\end{equation}

where the inequality $\boldsymbol{\lambda}^{\star} \geq \mathbf{0}$ means that all components of $\boldsymbol{\lambda}^{\star}$ are required to be nonnegative. By applying the complementarity condition \eqref{eq:complementarity} to both inequality constraints, we obtain

\begin{equation}
\lambda_{1}^{\star} c_{1}\left(\mathbf{x}^{\star}\right)=0, \quad \lambda_{2}^{\star} c_{2}\left(\mathbf{x}^{\star}\right)=0.
\label{eq:two_constraint_complementarity}
\end{equation}

When $\mathbf{x}^{\star}=(-\sqrt{2}, 0)^{\mathrm{T}}$, we have

$$
\nabla f\left(\mathbf{x}^{\star}\right)=\begin{bmatrix} 1 \\\\ 1 \end{bmatrix}, \quad \nabla c_{1}\left(\mathbf{x}^{\star}\right)=\begin{bmatrix} 2 \sqrt{2} \\\\ 0 \end{bmatrix}, \quad \nabla c_{2}\left(\mathbf{x}^{\star}\right)=\begin{bmatrix} 0 \\\\ 1 \end{bmatrix},
$$

so that it is easy to verify that $\nabla_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)=\mathbf{0}$ when we select $\boldsymbol{\lambda}^{\star}$ as follows:

$$
\boldsymbol{\lambda}^{\star}=\begin{bmatrix} 1 /(2 \sqrt{2}) \\\\ 1 \end{bmatrix}
$$

Note that both components of $\boldsymbol{\lambda}^{\star}$ are positive.

We consider now some other feasible points that are not solutions of \eqref{eq:two_inequality_example}, and examine the properties of the Lagrangian and its gradient at these points.

For the point $\mathbf{x}=(\sqrt{2}, 0)^{\mathrm{T}}$, we again have that both constraints are active. However, the objective gradient $\nabla f(\mathbf{x})$ no longer lies in the quadrant defined by the conditions $\nabla c_{i}(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0, i=1,2$. One first-order feasible descent direction from this point-a vector $\mathbf{d}$ that satisfies the required conditions-is simply $\mathbf{d}=(-1,0)^{\mathrm{T}}$; there are many others. For this value of $\mathbf{x}$ it is easy to verify that the condition $\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})=\mathbf{0}$ is satisfied only when $\boldsymbol{\lambda}=(-1 /(2 \sqrt{2}), 1)$. Note that the first component $\lambda_{1}$ is negative, so that the conditions \eqref{eq:two_constraint_kkt} are not satisfied at this point.

Finally, let us consider the point $\mathbf{x}=(1,0)^{\mathrm{T}}$, at which only the second constraint $c_{2}$ is active. At this point, linearization of $f$ and $c$ gives the following conditions, which must be satisfied for $\mathbf{d}$ to be a feasible descent direction, to first order:

$$
1+\nabla c_{1}(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0, \quad \nabla c_{2}(\mathbf{x})^{\mathrm{T}} \mathbf{d} \geq 0, \quad \nabla f(\mathbf{x})^{\mathrm{T}} \mathbf{d}<0 .
$$

In fact, we need worry only about satisfying the second and third conditions, since we can always satisfy the first condition by multiplying $\mathbf{d}$ by a sufficiently small positive quantity. By noting that

$
\nabla f(\mathbf{x})=\begin{bmatrix} 1 \\\\ 1 \end{bmatrix}, \quad \nabla c_{2}(\mathbf{x})=\begin{bmatrix} 0 \\\\ 1 \end{bmatrix}
$

it is easy to verify that the vector $\mathbf{d}=\left(-\frac{1}{2}, \frac{1}{4}\right)$ satisfies the required conditions and is therefore a descent direction.

To show that optimality conditions \eqref{eq:two_constraint_kkt} and \eqref{eq:two_constraint_complementarity} fail, we note first from \eqref{eq:two_constraint_complementarity} that since $c_{1}(\mathbf{x})>0$, we must have $\lambda_{1}=0$. Therefore, in trying to satisfy $\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})=\mathbf{0}$, we are left to search for a value $\lambda_{2}$ such that $\nabla f(\mathbf{x})-\lambda_{2} \nabla c_{2}(\mathbf{x})=\mathbf{0}$. No such $\lambda_{2}$ exists, and thus this point fails to satisfy the optimality conditions.

### First-order optimality conditions

## Statement of first-order necessary conditions

The three examples above suggest that a number of conditions are important in the characterization of solutions for the general problem. These include the relation $\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})=\mathbf{0}$, the nonnegativity of $\lambda_{i}$ for all inequality constraints $c_{i}(\mathbf{x})$, and the complementarity condition $\lambda_{i} c_{i}(\mathbf{x})=0$ that is required for all the inequality constraints. We now generalize the observations made in these examples and state the first-order optimality conditions in a rigorous fashion.

In general, the Lagrangian for the constrained optimization problem is defined as

\begin{equation}
\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})=f(\mathbf{x})-\sum_{i \in \mathcal{E} \cup \mathcal{I}} \lambda_{i} c_{i}(\mathbf{x}).
\label{eq:general_lagrangian}
\end{equation}

The active set $\mathcal{A}(\mathbf{x})$ at any feasible $\mathbf{x}$ is the union of the set $\mathcal{E}$ with the indices of the active inequality constraints; that is,

\begin{equation}
\mathcal{A}(\mathbf{x})=\mathcal{E} \cup\left\\{i \in \mathcal{I} \mid c_{i}(\mathbf{x})=0\right\\}.
\label{eq:active_set}
\end{equation}

Next, we need to give more attention to the properties of the constraint gradients. The vector $\nabla c_{i}(\mathbf{x})$ is often called the normal to the constraint $c_{i}$ at the point $\mathbf{x}$, because it is usually a vector that is perpendicular to the contours of the constraint $c_{i}$ at $\mathbf{x}$, and in the case of an inequality constraint, it points toward the feasible side of this constraint. It is possible, however, that $\nabla c_{i}(\mathbf{x})$ vanishes due to the algebraic representation of $c_{i}$, so that the term $\lambda_{i} \nabla c_{i}(\mathbf{x})$ vanishes for all values of $\lambda_{i}$ and does not play a role in the Lagrangian gradient $\nabla_{\mathbf{x}} \mathcal{L}$. For instance, if we replaced the constraint in \eqref{eq:equality_example} by the equivalent condition

$
c_{1}(\mathbf{x})=\left(x_{1}^{2}+x_{2}^{2}-2\right)^{2}=0
$

we would have that $\nabla c_{1}(\mathbf{x})=\mathbf{0}$ for all feasible points $\mathbf{x}$, and in particular that the condition $\nabla f(\mathbf{x})=\lambda_{1} \nabla c_{1}(\mathbf{x})$ no longer holds at the optimal point $(-1,-1)^{\mathrm{T}}$. We usually make an assumption called a constraint qualification to ensure that such degenerate behavior does not occur at the value of $\mathbf{x}$ in question. One such constraint qualification-probably the one most often used in the design of algorithms-is the one defined as follows:

{{<definition "Linear independence constraint qualification (LICQ)" licq>}}
Given the point $\mathbf{x}^{\star}$ and the active set $\mathcal{A}\left(\mathbf{x}^{\star}\right)$ defined by \eqref{eq:active_set}, we say that the **linear independence constraint qualification (LICQ)** holds if the set of active constraint gradients $\left\\{\nabla c_{i}\left(\mathbf{x}^{\star}\right), i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)\right\\}$ is linearly independent.
{{</definition>}}

Note that if this condition holds, none of the active constraint gradients can be zero.

This condition allows us to state the following optimality conditions for a general nonlinear programming problem. These conditions provide the foundation for many of the algorithms described in the remaining chapters of the lecture. They are called first-order conditions because they concern themselves with properties of the gradients (first-derivative vectors) of the objective and constraint functions.

{{<theorem "First-order necessary conditions" first_order_necessary>}}
Suppose that $\mathbf{x}^{\star}$ is a local solution and that the LICQ holds at $\mathbf{x}^{\star}$. Then there is a Lagrange multiplier vector $\boldsymbol{\lambda}^{\star}$, with components $\lambda_{i}^{\star}, i \in \mathcal{E} \cup \mathcal{I}$, such that the following conditions are satisfied at $\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)$:

\begin{align}
\nabla_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)&=\mathbf{0}, \label{eq:kkt_gradient_zero} \\\\
c_{i}\left(\mathbf{x}^{\star}\right)&=0, \quad && \text { for all } i \in \mathcal{E}, \label{eq:kkt_equality} \\\\
c_{i}\left(\mathbf{x}^{\star}\right)&\geq 0, \quad && \text { for all } i \in \mathcal{I}, \label{eq:kkt_inequality} \\\\
\lambda_{i}^{\star} &\geq 0, \quad && \text { for all } i \in \mathcal{I}, \label{eq:kkt_multiplier_sign} \\\\
\lambda_{i}^{\star} c_{i}\left(\mathbf{x}^{\star}\right)&=0, \quad && \text { for all } i \in \mathcal{E} \cup \mathcal{I}. \label{eq:kkt_complementarity}
\end{align}
{{</theorem>}}

The conditions \eqref{eq:kkt_gradient_zero}-\eqref{eq:kkt_complementarity} are often known as the Karush-Kuhn-Tucker conditions, or KKT conditions for short. Because the complementarity condition implies that the Lagrange multipliers corresponding to inactive inequality constraints are zero, we can omit the terms for indices $i \notin \mathcal{A}\left(\mathbf{x}^{\star}\right)$ from \eqref{eq:kkt_gradient_zero} and rewrite this condition as

\begin{equation}
\mathbf{0}=\nabla_{\mathbf{x}} \mathcal{L}\left(\mathbf{x}^{\star}, \boldsymbol{\lambda}^{\star}\right)=\nabla f\left(\mathbf{x}^{\star}\right)-\sum_{i \in \mathcal{A}\left(\mathbf{x}^{\star}\right)} \lambda_{i}^{\star} \nabla c_{i}\left(\mathbf{x}^{\star}\right).
\label{eq:kkt_active_gradients}
\end{equation}

A special case of complementarity is important and deserves its own definition:

{{<definition "Strict complementarity" strict_complementarity>}}
Given a local solution $\mathbf{x}^{\star}$ and a vector $\boldsymbol{\lambda}^{\star}$ satisfying \eqref{eq:kkt_gradient_zero}-\eqref{eq:kkt_complementarity}, we say that the **strict complementarity condition** holds if exactly one of $\lambda_{i}^{\star}$ and $c_{i}\left(\mathbf{x}^{\star}\right)$ is zero for each index $i \in \mathcal{I}$. In other words, we have that $\lambda_{i}^{\star}>0$ for each $i \in \mathcal{I} \cap \mathcal{A}\left(\mathbf{x}^{\star}\right)$.
{{</definition>}}

For a given problem and solution point $\mathbf{x}^{\star}$, there may be many vectors $\boldsymbol{\lambda}^{\star}$ for which the conditions \eqref{eq:kkt_gradient_zero}-\eqref{eq:kkt_complementarity} are satisfied. When the LICQ holds, however, the optimal $\boldsymbol{\lambda}^{\star}$ is unique.

**Example 4**

Consider the feasible region described by the four constraints:

\begin{equation}
\min_{\mathbf{x}}\left(x_{1}-\frac{3}{2}\right)^{2}+\left(x_{2}-\frac{1}{8}\right)^{4} \quad \text { s.t. } \quad\begin{bmatrix} 1-x_{1}-x_{2} \\\\ 1-x_{1}+x_{2} \\\\ 1+x_{1}-x_{2} \\\\ 1+x_{1}+x_{2} \end{bmatrix} \geq \mathbf{0}.
\label{eq:diamond_example}
\end{equation}

It is fairly clear that the solution is $\mathbf{x}^{\star}=(1,0)$. The first and second constraints are active at this point. Denoting them by $c_{1}$ and $c_{2}$ (and the inactive constraints by $c_{3}$ and $c_{4}$ ), we have

$
\nabla f\left(\mathbf{x}^{\star}\right)=\begin{bmatrix} -1 \\\\ -\frac{1}{2} \end{bmatrix}, \quad \nabla c_{1}\left(\mathbf{x}^{\star}\right)=\begin{bmatrix} -1 \\\\ -1 \end{bmatrix}, \quad \nabla c_{2}\left(\mathbf{x}^{\star}\right)=\begin{bmatrix} -1 \\\\ 1 \end{bmatrix} .
$

Therefore, the KKT conditions \eqref{eq:kkt_gradient_zero}-\eqref{eq:kkt_complementarity} are satisfied when we set

$
\boldsymbol{\lambda}^{\star}=\left(\frac{3}{4}, \frac{1}{4}, 0,0\right)^{\mathrm{T}}.
$

**Exercises**

1. For Example 1, verify that the direction $\mathbf{d}$ defined by 
   $\mathbf{d}=-\left(\mathbf{I}-\frac{\nabla c_{1}(\mathbf{x}) \nabla c_{1}(\mathbf{x})^{\mathrm{T}}}{\\|\nabla c_{1}(\mathbf{x})\\|^{2}}\right) \nabla f(\mathbf{x})$
   satisfies both $\nabla c_{1}(\mathbf{x})^{\mathrm{T}} \mathbf{d}=0$ and $\nabla f(\mathbf{x})^{\mathrm{T}} \mathbf{d}<0$, provided that $\nabla f(\mathbf{x})$ and $\nabla c_{1}(\mathbf{x})$ are not parallel.

2. For Example 3, find another first-order feasible descent direction from the point $\mathbf{x}=(\sqrt{2}, 0)^{\mathrm{T}}$, in addition to $\mathbf{d}=(-1,0)^{\mathrm{T}}$.

3. Show that when the LICQ holds at a KKT point $\mathbf{x}^{\star}$ for problem, the Lagrange multiplier vector $\boldsymbol{\lambda}^{\star}$ is uniquely defined.

4. Consider the problem
   $\min x_{1}^{2}+x_{2}^{2} \quad \text{s.t.} \quad (x_{1}-1)^{3}-x_{2}^{2}=0.$
   
   a) Sketch the feasible region and solve the problem geometrically.
   
   b) Write down the KKT conditions and verify that they are satisfied at the solution you found in part (a).
   
   c) Does the LICQ hold at the solution?

5. Consider the problem
   $\min (x_{1}-2)^{2}+(x_{2}-1)^{2} \quad \text{s.t.} \quad x_{1}^{2}+x_{2}^{2} \leq 1, \quad x_{1}+x_{2} \geq 1.$
   
   Find all points that satisfy the KKT conditions and identify which ones are local solutions.
