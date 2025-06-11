---
title: 6. Constrained optimization - Linear programming
weight: 7
math: true
chapter: 6
---

## Linear programming

Linear programs have a linear objective function and linear constraints, which may include both equalities and inequalities. The feasible set is a polytope, that is, a convex, connected set with flat, polygonal faces. The contours of the objective function are planar. The solution in this case is unique-a single vertex. A simple reorientation of the polytope or the objective gradient $\mathbf{c}$ could, however, make the solution nonunique; the optimal value $\mathbf{c}^{\mathrm{T}} \mathbf{x}$ could be the same on an entire edge. In higher dimensions, the set of optimal points can be a single vertex, an edge or face, or even the entire feasible set!

Linear programs are usually stated and analyzed in the following standard form:

\begin{equation}
\min \mathbf{c}^{\mathrm{T}} \mathbf{x}, \text { subject to } \mathbf{A} \mathbf{x}=\mathbf{b}, \mathbf{x} \geq \mathbf{0},
\label{eq:standard_form}
\end{equation}

where $\mathbf{c}$ and $\mathbf{x}$ are vectors in $\mathbb{R}^{n}$, $\mathbf{b}$ is a vector in $\mathbb{R}^{m}$, and $\mathbf{A}$ is an $m \times n$ matrix. Simple devices can be used to transform any linear program to this form. For instance, given the problem

$$
\min \mathbf{c}^{\mathrm{T}} \mathbf{x}, \text { subject to } \mathbf{A} \mathbf{x} \geq \mathbf{b}
$$

(without any bounds on $\mathbf{x}$ ), we can convert the inequality constraints to equalities by introducing a vector of surplus variables $\mathbf{z}$ and writing

$$
\min \mathbf{c}^{\mathrm{T}} \mathbf{x}, \text { subject to } \mathbf{A} \mathbf{x}-\mathbf{z}=\mathbf{b}, \mathbf{z} \geq \mathbf{0}
$$

This form is still not quite standard, since not all the variables are constrained to be nonnegative. We deal with this by splitting $\mathbf{x}$ into its nonnegative and nonpositive parts, $\mathbf{x}=\mathbf{x}^{+}-\mathbf{x}^{-}$, where $\mathbf{x}^{+}=\max (\mathbf{x}, \mathbf{0}) \geq \mathbf{0}$ and $\mathbf{x}^{-}=\max (-\mathbf{x}, \mathbf{0}) \geq \mathbf{0}$. The problem can now be written as

$$
\min \begin{bmatrix}
\mathbf{c} \\\\\\
-\mathbf{c} \\\\\\
\mathbf{0}
\end{bmatrix}^{\mathrm{T}}\begin{bmatrix}
\mathbf{x}^{+} \\\\\\
\mathbf{x}^{-} \\\\\\
\mathbf{z}
\end{bmatrix}, \text { s.t. }\begin{bmatrix}\mathbf{A} & -\mathbf{A} & -\mathbf{I}\end{bmatrix}\begin{bmatrix}
\mathbf{x}^{+} \\\\\\
\mathbf{x}^{-} \\\\\\
\mathbf{z}
\end{bmatrix}=\mathbf{b},\begin{bmatrix}
\mathbf{x}^{+} \\\\\\
\mathbf{x}^{-} \\\\\\
\mathbf{z}
\end{bmatrix} \geq \mathbf{0}
$$

which clearly has the same form as \eqref{eq:standard_form}. Inequality constraints of the form $\mathbf{x} \leq \mathbf{u}$ or $\mathbf{A} \mathbf{x} \leq \mathbf{b}$ can be dealt with by adding slack variables to make up the difference between the left- and right-hand-sides. Hence

\begin{align}
\mathbf{x} & \leq \mathbf{u} \Leftrightarrow \mathbf{x}+\mathbf{w}=\mathbf{u}, \mathbf{w} \geq \mathbf{0} \\\\\\
\mathbf{A} \mathbf{x} & \leq \mathbf{b} \Leftrightarrow \mathbf{A} \mathbf{x}+\mathbf{y}=\mathbf{b}, \mathbf{y} \geq \mathbf{0}
\end{align}

We can also convert a "maximize" objective $\max \mathbf{c}^{\mathrm{T}} \mathbf{x}$ into the "minimize" form of \eqref{eq:standard_form} by simply negating $\mathbf{c}$ to obtain $\min (-\mathbf{c})^{\mathrm{T}} \mathbf{x}$.

Many linear programs arise from models of transshipment and distribution networks. These problems have much additional structure in their constraints; special-purpose simplex algorithms that exploit this structure are highly efficient. We do not discuss these network-flow problems further in this lecture, except to note that the subject is important and complex, and that a number of excellent texts are available (see, for example, Ahuja, Magnanti, and Orlin [1]).

For the standard formulation \eqref{eq:standard_form}, we will assume throughout that $m<n$. Otherwise, the system $\mathbf{A} \mathbf{x}=\mathbf{b}$ contains redundant rows, is infeasible, or defines a unique point. When $m \geq n$, factorizations such as the QR or LU factorization can be used to transform the system $\mathbf{A} \mathbf{x}=\mathbf{b}$ to one with a coefficient matrix of full row rank and, in some cases, decide that the feasible region is empty or consists of a single point.

## Optimality and duality

### Optimality conditions

Optimality conditions for the problem \eqref{eq:standard_form} can be derived from the theory of the previous chapter. Only the first-order conditions-the Karush-Kuhn-Tucker (KKT) conditions-are needed. Convexity of the problem ensures that these conditions are sufficient for a global minimum, as we show below by a simple argument. (We do not need to refer to the second-order conditions from the previous chapter, which are not informative in any case because the Hessian of the Lagrangian for \eqref{eq:standard_form} is zero.)

The tools we developed in the previous chapter make derivation of optimality and duality theory for linear programming much easier than in other treatments of the subject, where this theory has to be developed more or less from scratch.

The KKT conditions follow from {{< theoremref first_order_necessary >}}. As stated in the previous chapter, this theorem requires linear independence of the active constraint gradients (LICQ). However, as we showed in the section on constraint qualifications, the result continues to hold for dependent constraints, provided that they are linear, as is the case here.

We partition the Lagrange multipliers for the problem \eqref{eq:standard_form} into two vectors $\boldsymbol{\pi}$ and $\mathbf{s}$, where $\boldsymbol{\pi} \in \mathbb{R}^{m}$ is the multiplier vector for the equality constraints $\mathbf{A} \mathbf{x}=\mathbf{b}$, while $\mathbf{s} \in \mathbb{R}^{n}$ is the multiplier vector for the bound constraints $\mathbf{x} \geq \mathbf{0}$. Using the definition from the previous chapter, we can write the Lagrangian function for \eqref{eq:standard_form} as

\begin{equation}
\mathcal{L}(\mathbf{x}, \boldsymbol{\pi}, \mathbf{s})=\mathbf{c}^{\mathrm{T}} \mathbf{x}-\boldsymbol{\pi}^{\mathrm{T}}(\mathbf{A} \mathbf{x}-\mathbf{b})-\mathbf{s}^{\mathrm{T}} \mathbf{x} .
\label{eq:lp_lagrangian}
\end{equation}

Applying {{< theoremref first_order_necessary >}}, we find that the first-order necessary conditions for $\mathbf{x}^{\star}$ to be a solution of \eqref{eq:standard_form} are that there exist vectors $\boldsymbol{\pi}$ and $\mathbf{s}$ such that

\begin{align}
\mathbf{A}^{\mathrm{T}} \boldsymbol{\pi}+\mathbf{s} & =\mathbf{c}, \label{eq:lp_kkt_gradient} \\\\\\
\mathbf{A} \mathbf{x} & =\mathbf{b}, \label{eq:lp_kkt_equality} \\\\\\
\mathbf{x} & \geq \mathbf{0}, \label{eq:lp_kkt_primal_feasible} \\\\\\
\mathbf{s} & \geq \mathbf{0}, \label{eq:lp_kkt_dual_feasible} \\\\\\
x\_{i} s\_{i} & =0, \quad i=1,2, \ldots, n . \label{eq:lp_kkt_complementarity}
\end{align}

The complementarity condition \eqref{eq:lp_kkt_complementarity}, which essentially says that at least one of the components $x\_{i}$ and $s\_{i}$ must be zero for each $i=1,2, \ldots, n$, is often written in the alternative form $\mathbf{x}^{\mathrm{T}} \mathbf{s}=0$. Because of the nonnegativity conditions \eqref{eq:lp_kkt_primal_feasible}, \eqref{eq:lp_kkt_dual_feasible}, the two forms are identical.

Let $\left(\mathbf{x}^{\star}, \boldsymbol{\pi}^{\star}, \mathbf{s}^{\star}\right)$ denote a vector triple that satisfies \eqref{eq:lp_kkt_gradient}-\eqref{eq:lp_kkt_complementarity}. By combining the three equalities \eqref{eq:lp_kkt_gradient}, \eqref{eq:lp_kkt_dual_feasible}, and \eqref{eq:lp_kkt_complementarity}, we find that

\begin{equation}
\mathbf{c}^{\mathrm{T}} \mathbf{x}^{\star}=\left(\mathbf{A}^{\mathrm{T}} \boldsymbol{\pi}^{\star}+\mathbf{s}^{\star}\right)^{\mathrm{T}} \mathbf{x}^{\star}=\left(\mathbf{A} \mathbf{x}^{\star}\right)^{\mathrm{T}} \boldsymbol{\pi}^{\star}=\mathbf{b}^{\mathrm{T}} \boldsymbol{\pi}^{\star} .
\label{eq:primal_dual_equality}
\end{equation}

As we shall see in a moment, $\mathbf{b}^{\mathrm{T}} \boldsymbol{\pi}$ is the objective function for the dual problem to \eqref{eq:standard_form}, so \eqref{eq:primal_dual_equality} indicates that the primal and dual objectives are equal for vector triples $(\mathbf{x}, \boldsymbol{\pi}, \mathbf{s})$ that satisfy \eqref{eq:lp_kkt_gradient}-\eqref{eq:lp_kkt_complementarity}.

It is easy to show directly that the conditions \eqref{eq:lp_kkt_gradient}-\eqref{eq:lp_kkt_complementarity} are sufficient for $\mathbf{x}^{\star}$ to be a global solution of \eqref{eq:standard_form}. Let $\overline{\mathbf{x}}$ be any other feasible point, so that $\mathbf{A} \overline{\mathbf{x}}=\mathbf{b}$ and $\overline{\mathbf{x}} \geq \mathbf{0}$. Then

\begin{equation}
\mathbf{c}^{\mathrm{T}} \overline{\mathbf{x}}=\left(\mathbf{A} \boldsymbol{\pi}^{\star}+\mathbf{s}^{\star}\right)^{\mathrm{T}} \overline{\mathbf{x}}=\mathbf{b}^{\mathrm{T}} \boldsymbol{\pi}^{\star}+\overline{\mathbf{x}}^{\mathrm{T}} \mathbf{s}^{\star} \geq \mathbf{b}^{\mathrm{T}} \boldsymbol{\pi}^{\star}=\mathbf{c}^{\mathrm{T}} \mathbf{x}^{\star}
\label{eq:sufficiency_proof}
\end{equation}

We have used \eqref{eq:lp_kkt_gradient}-\eqref{eq:lp_kkt_complementarity} and \eqref{eq:primal_dual_equality} here; the inequality relation follows trivially from $\overline{\mathbf{x}} \geq \mathbf{0}$ and $\mathbf{s}^{\star} \geq \mathbf{0}$. The inequality \eqref{eq:sufficiency_proof} tells us that no other feasible point can have a lower objective value than $\mathbf{c}^{\mathrm{T}} \mathbf{x}^{\star}$. We can say more: The feasible point $\overline{\mathbf{x}}$ is optimal if and only if

$$
\overline{\mathbf{x}}^{\mathrm{T}} \mathbf{s}^{\star}=0
$$

since otherwise the inequality in \eqref{eq:sufficiency_proof} is strict. In other words, when $s\_{i}^{\star}>0$, then we must have $\overline{x}\_{i}=0$ for all solutions $\overline{\mathbf{x}}$ of \eqref{eq:standard_form}.

### The dual problem

Given the data $\mathbf{c}$, $\mathbf{b}$, and $\mathbf{A}$, which define the problem \eqref{eq:standard_form}, we can define another, closely related, problem as follows:

\begin{equation}
\max \mathbf{b}^{\mathrm{T}} \boldsymbol{\pi}, \quad \text { subject to } \mathbf{A}^{\mathrm{T}} \boldsymbol{\pi} \leq \mathbf{c} .
\label{eq:dual_problem}
\end{equation}

This problem is called the dual problem for \eqref{eq:standard_form}. In contrast, \eqref{eq:standard_form} is often referred to as the primal.

The primal and dual problems are two sides of the same coin, as we see when we write down the KKT conditions for \eqref{eq:dual_problem}. Let us first rewrite \eqref{eq:dual_problem} in the form

$$
\min -\mathbf{b}^{\mathrm{T}} \boldsymbol{\pi} \quad \text { subject to } \mathbf{c}-\mathbf{A}^{\mathrm{T}} \boldsymbol{\pi} \geq \mathbf{0}
$$

to fit the formulation from the previous chapter. By using $\mathbf{x} \in \mathbb{R}^{n}$ to denote the Lagrange multipliers for the constraints $\mathbf{A}^{\mathrm{T}} \boldsymbol{\pi} \leq \mathbf{c}$, we write the Lagrangian function as

$$
\overline{\mathcal{L}}(\boldsymbol{\pi}, \mathbf{x})=-\mathbf{b}^{\mathrm{T}} \boldsymbol{\pi}-\mathbf{x}^{\mathrm{T}}\left(\mathbf{c}-\mathbf{A}^{\mathrm{T}} \boldsymbol{\pi}\right) .
$$

Noting again that the conclusions of {{< theoremref first_order_necessary >}} continue to hold if the linear independence assumption is replaced by linearity of all constraints, we find the first-order necessary condition for $\boldsymbol{\pi}$ to be optimal for \eqref{eq:dual_problem} to be that there exist a vector $\mathbf{x}$ such that

\begin{align}
\mathbf{A} \mathbf{x} & =\mathbf{b}, \label{eq:dual_kkt_a} \\\\\\
\mathbf{A}^{\mathrm{T}} \boldsymbol{\pi} & \leq \mathbf{c}, \label{eq:dual_kkt_b} \\\\\\
\mathbf{x} & \geq \mathbf{0}, \label{eq:dual_kkt_c} \\\\\\
x\_{i}\left(\mathbf{c}-\mathbf{A}^{\mathrm{T}} \boldsymbol{\pi}\right)\_{i} & =0, \quad i=1,2, \ldots, n . \label{eq:dual_kkt_d}
\end{align}

If we define $\mathbf{s}=\mathbf{c}-\mathbf{A}^{\mathrm{T}} \boldsymbol{\pi}$ and substitute in \eqref{eq:dual_kkt_a}-\eqref{eq:dual_kkt_d}, we find that the conditions \eqref{eq:dual_kkt_a}-\eqref{eq:dual_kkt_d} and \eqref{eq:lp_kkt_gradient}-\eqref{eq:lp_kkt_complementarity} are identical! The optimal Lagrange multipliers $\boldsymbol{\pi}$ in the primal problem are the optimal variables in the dual problem, while the optimal Lagrange multipliers $\mathbf{x}$ in the dual problem are the optimal variables in the primal problem.

The primal-dual relationship is symmetric; by taking the dual of the dual, we recover the primal. To see this, we restate \eqref{eq:dual_problem} in standard form by introducing the slack vector $\mathbf{s}$ (so that $\mathbf{A}^{\mathrm{T}} \boldsymbol{\pi}+\mathbf{s}=\mathbf{c}$ ) and splitting the unbounded variables $\boldsymbol{\pi}$ as $\boldsymbol{\pi}=\boldsymbol{\pi}^{+}-\boldsymbol{\pi}^{-}$, where $\boldsymbol{\pi}^{+} \geq \mathbf{0}$, and $\boldsymbol{\pi}^{-} \geq \mathbf{0}$. We can now write the dual as

$$
\min \begin{bmatrix}
-\mathbf{b} \\\\\\
\mathbf{b} \\\\\\
\mathbf{0}
\end{bmatrix}^{\mathrm{T}}\begin{bmatrix}
\boldsymbol{\pi}^{+} \\\\\\
\boldsymbol{\pi}^{-} \\\\\\
\mathbf{s}
\end{bmatrix} \text { s.t. }\begin{bmatrix}\mathbf{A}^{\mathrm{T}} & -\mathbf{A}^{\mathrm{T}} & \mathbf{I}\end{bmatrix}\begin{bmatrix}
\boldsymbol{\pi}^{+} \\\\\\
\boldsymbol{\pi}^{-} \\\\\\
\mathbf{s}
\end{bmatrix}=\mathbf{c},\begin{bmatrix}
\boldsymbol{\pi}^{+} \\\\\\
\boldsymbol{\pi}^{-} \\\\\\
\mathbf{s}
\end{bmatrix} \geq \mathbf{0}
$$

which clearly has the standard form \eqref{eq:standard_form}. The dual of this problem is now

$$
\max \mathbf{c}^{\mathrm{T}} \mathbf{z} \text { subject to }\begin{bmatrix}
\mathbf{A} \\\\\\
-\mathbf{A} \\\\\\
\mathbf{I}
\end{bmatrix} \mathbf{z} \leq\begin{bmatrix}
-\mathbf{b} \\\\\\
\mathbf{b} \\\\\\
\mathbf{0}
\end{bmatrix}
$$

Now $\mathbf{A} \mathbf{z} \leq-\mathbf{b}$ and $-\mathbf{A} \mathbf{z} \leq \mathbf{b}$ together imply that $\mathbf{A} \mathbf{z}=-\mathbf{b}$, so we obtain the equivalent problem

$$
\min -\mathbf{c}^{\mathrm{T}} \mathbf{z} \text { subject to } \mathbf{A} \mathbf{z}=-\mathbf{b}, \mathbf{z} \leq \mathbf{0}
$$

By making the identification $\mathbf{z}=-\mathbf{x}$, we recover \eqref{eq:standard_form}, as claimed.

Given a feasible vector $\mathbf{x}$ for the primal-that is, $\mathbf{A} \mathbf{x}=\mathbf{b}$ and $\mathbf{x} \geq \mathbf{0}$-and a feasible point $(\boldsymbol{\pi}, \mathbf{s})$ for the dual-that is, $\mathbf{A}^{\mathrm{T}} \boldsymbol{\pi}+\mathbf{s}=\mathbf{c}, \mathbf{s} \geq \mathbf{0}$-we have as in \eqref{eq:sufficiency_proof} that

\begin{equation}
0 \leq \mathbf{x}^{\mathrm{T}} \mathbf{s}=\mathbf{x}^{\mathrm{T}}\left(\mathbf{c}-\mathbf{A}^{\mathrm{T}} \boldsymbol{\pi}\right)=\mathbf{c}^{\mathrm{T}} \mathbf{x}-\mathbf{b}^{\mathrm{T}} \boldsymbol{\pi}
\label{eq:weak_duality}
\end{equation}

Therefore, we have $\mathbf{c}^{\mathrm{T}} \mathbf{x} \geq \mathbf{b}^{\mathrm{T}} \boldsymbol{\pi}$ when both the primal and dual variables are feasible-the dual objective function is a lower bound on the primal objective function. At a solution, the gap between primal and dual shrinks to zero, as we show in the following theorem.

{{<theorem "Duality theorem of linear programming" duality_theorem>}}
(i) If either problem \eqref{eq:standard_form} or \eqref{eq:dual_problem} has a solution with finite optimal objective value, then so does the other, and the objective values are equal.
(ii) If either problem \eqref{eq:standard_form} or \eqref{eq:dual_problem} has an unbounded objective, then the other problem has no feasible points.
{{</theorem>}}

{{<proof>}}
For (i), suppose that \eqref{eq:standard_form} has a finite optimal solution. Then because of {{< theoremref first_order_necessary >}}, there are vectors $\boldsymbol{\pi}$ and $\mathbf{s}$ such that $(\mathbf{x}, \boldsymbol{\pi}, \mathbf{s})$ satisfies \eqref{eq:lp_kkt_gradient}-\eqref{eq:lp_kkt_complementarity}. Since \eqref{eq:lp_kkt_gradient}-\eqref{eq:lp_kkt_complementarity} and \eqref{eq:dual_kkt_a}-\eqref{eq:dual_kkt_d} are equivalent, it follows that $\boldsymbol{\pi}$ is a solution of the dual problem \eqref{eq:dual_problem}, since there exists a vector $\mathbf{x}$ that satisfies \eqref{eq:dual_kkt_a}-\eqref{eq:dual_kkt_d}. Because $\mathbf{x}^{\mathrm{T}} \mathbf{s}=0$, it follows from \eqref{eq:weak_duality} that $\mathbf{c}^{\mathrm{T}} \mathbf{x}=\mathbf{b}^{\mathrm{T}} \boldsymbol{\pi}$, so the optimal objective values are equal.

We can make a symmetric argument if we start by assuming that the dual problem \eqref{eq:dual_problem} has a solution.

For (ii), suppose that the primal objective value is unbounded below. Then there must exist a direction $\mathbf{d} \in \mathbb{R}^{n}$ along which $\mathbf{c}^{\mathrm{T}} \mathbf{x}$ decreases without violating feasibility. That is,

$$
\mathbf{c}^{\mathrm{T}} \mathbf{d}<0, \quad \mathbf{A} \mathbf{d}=\mathbf{0}, \quad \mathbf{d} \geq \mathbf{0}
$$

Suppose now that there does exist a feasible point $\boldsymbol{\pi}$ for the dual problem \eqref{eq:dual_problem}, that is $\mathbf{A}^{\mathrm{T}} \boldsymbol{\pi} \leq \mathbf{c}$. Multiplying from the left by $\mathbf{d}^{\mathrm{T}}$, using the nonnegativity of $\mathbf{d}$, we obtain

$$
0=\mathbf{d}^{\mathrm{T}} \mathbf{A}^{\mathrm{T}} \boldsymbol{\pi} \leq \mathbf{d}^{\mathrm{T}} \mathbf{c}<0
$$

giving a contradiction.

Again, we can make a symmetric argument to prove (ii) if we start by assuming that the dual objective is unbounded below.
{{</proof>}}

As we showed in the discussion following {{< theoremref first_order_necessary >}}, the multiplier values $\boldsymbol{\pi}$ and $\mathbf{s}$ for \eqref{eq:standard_form} tell us how sensitive the optimal objective value is to perturbations in the constraints. In fact, the process of finding $(\boldsymbol{\pi}, \mathbf{s})$ for a given optimal $\mathbf{x}$ is often called sensitivity analysis. We can make a simple direct argument to illustrate this dependence. If a small change $\Delta \mathbf{b}$ is made to the vector $\mathbf{b}$ (the right-hand-side in \eqref{eq:standard_form} and objective gradient in \eqref{eq:dual_problem}), then we would usually expect small perturbations in the primal and dual solutions. If these perturbations $(\Delta \mathbf{x}, \Delta \boldsymbol{\pi}, \Delta \mathbf{s})$ are small enough, we know that provided the problem is not degenerate (defined below), the vectors $\Delta \mathbf{s}$ and $\Delta \mathbf{x}$ have zeros in the same locations as $\mathbf{s}$ and $\mathbf{x}$, respectively. Since $\mathbf{x}$ and $\mathbf{s}$ are complementary (see \eqref{eq:lp_kkt_complementarity}), it follows that

$$
\mathbf{x}^{\mathrm{T}} \mathbf{s}=\mathbf{x}^{\mathrm{T}} \Delta \mathbf{s}=(\Delta \mathbf{x})^{\mathrm{T}} \mathbf{s}=(\Delta \mathbf{x})^{\mathrm{T}} \Delta \mathbf{s}=0
$$

Now we have from the duality theorem that

$$
\mathbf{c}^{\mathrm{T}}(\mathbf{x}+\Delta \mathbf{x})=(\mathbf{b}+\Delta \mathbf{b})^{\mathrm{T}}(\boldsymbol{\pi}+\Delta \boldsymbol{\pi})
$$

Since

$$
\mathbf{c}^{\mathrm{T}} \mathbf{x}=\mathbf{b}^{\mathrm{T}} \boldsymbol{\pi}, \quad \mathbf{A}(\mathbf{x}+\Delta \mathbf{x})=\mathbf{b}+\Delta \mathbf{b}, \quad \mathbf{A}^{\mathrm{T}} \Delta \boldsymbol{\pi}=-\Delta \mathbf{s}
$$

we have

\begin{align}
\mathbf{c}^{\mathrm{T}} \Delta \mathbf{x} & =(\mathbf{b}+\Delta \mathbf{b})^{\mathrm{T}} \Delta \boldsymbol{\pi}+\Delta \mathbf{b}^{\mathrm{T}} \boldsymbol{\pi} \\\\\\
& =(\mathbf{x}+\Delta \mathbf{x})^{\mathrm{T}} \mathbf{A}^{\mathrm{T}} \Delta \boldsymbol{\pi}+\Delta \mathbf{b}^{\mathrm{T}} \boldsymbol{\pi} \\\\\\
& =-(\mathbf{x}+\Delta \mathbf{x})^{\mathrm{T}} \Delta \mathbf{s}+\Delta \mathbf{b}^{\mathrm{T}} \boldsymbol{\pi}=\Delta \mathbf{b}^{\mathrm{T}} \boldsymbol{\pi} .
\end{align}

In particular, if $\Delta \mathbf{b}=\epsilon \mathbf{e}\_{j}$, where $\mathbf{e}\_{j}$ is the $j$ th unit vector in $\mathbb{R}^{m}$, we have for all $\epsilon$ sufficiently small that

$$
\mathbf{c}^{\mathrm{T}} \Delta \mathbf{x}=\epsilon \pi\_{j}
$$

That is, the change in primal objective is linear in the value of $\pi\_{j}$ for small perturbations in the components of the right-hand-side $b\_{j}$.

## Geometry of the feasible set

### Basic feasible points

We assume for the remainder of the chapter that

$$
\text { The matrix } \mathbf{A} \text { in } \eqref{eq:standard_form} \text{ has full row rank. }
$$

In practice, a preprocessing phase is applied to the user-supplied data to remove some redundancies from the given constraints and eliminate some of the variables. Reformulation by adding slack, surplus, and artificial variables is also used to force $\mathbf{A}$ to satisfy this property.

Suppose that $\mathbf{x}$ is a feasible point with at most $m$ nonzero components. Suppose, too, that we can identify a subset $\mathcal{B}(\mathbf{x})$ of the index set $\\{1,2, \ldots, n\\}$ such that

- $\mathcal{B}(\mathbf{x})$ contains exactly $m$ indices;
- $i \notin \mathcal{B}(\mathbf{x}) \Rightarrow x\_{i}=0$;
- the $m \times m$ matrix $\mathbf{B}$ defined by

$$
\mathbf{B}=\left[\mathbf{A}\_{i}\right]\_{i \in \mathcal{B}(\mathbf{x})}
$$

is nonsingular, where $\mathbf{A}\_{i}$ is the $i$ th column of $\mathbf{A}$.

{{<definition "Basic feasible point" basic_feasible_point>}}
If all these conditions are true, we call $\mathbf{x}$ a **basic feasible point** for \eqref{eq:standard_form}.
{{</definition>}}

The simplex method generates a sequence of iterates $\mathbf{x}^{k}$ all of which are basic feasible points. Since we want the iterates to converge to a solution of \eqref{eq:standard_form}, the simplex strategy will make sense only if
(a) the problem has basic feasible points; and
(b) at least one such point is a basic optimal point, that is, a solution of \eqref{eq:standard_form} that is also a basic feasible point.

Happily, both (a) and (b) are true under minimal assumptions.

{{<theorem "Fundamental theorem of linear programming" fundamental_lp_theorem>}}
(i) If there is a feasible point for \eqref{eq:standard_form}, then there is a basic feasible point.
(ii) If \eqref{eq:standard_form} has solutions, then at least one such solution is a basic optimal point.
(iii) If \eqref{eq:standard_form} is feasible and bounded, then it has an optimal solution.
{{</theorem>}}

{{<proof>}}
Among all feasible vectors $\mathbf{x}$, choose one with the minimal number of nonzero components, and denote this number by $p$. Without loss of generality, assume that the nonzeros are $x\_{1}, x\_{2}, \ldots, x\_{p}$, so we have

$$
\sum\_{i=1}^{p} \mathbf{A}\_{i} x\_{i}=\mathbf{b}
$$

Suppose first that the columns $\mathbf{A}\_{1}, \mathbf{A}\_{2}, \ldots, \mathbf{A}\_{p}$ are linearly dependent. Then we can express one of them ($\mathbf{A}\_{p}$, say) in terms of the others, and write

$$
\mathbf{A}\_{i}=\sum\_{i=1}^{p-1} \mathbf{A}\_{i} z\_{i},
$$

for some scalars $z\_{1}, z\_{2}, \ldots, z\_{p-1}$. It is easy to check that the vector

$$
\mathbf{x}(\epsilon)=\mathbf{x}+\epsilon\left(z\_{1}, z\_{2}, \ldots, z\_{p-1},-1,0,0, \ldots, 0\right)^{\mathrm{T}}=\mathbf{x}+\epsilon \mathbf{z}
$$

satisfies $\mathbf{A} \mathbf{x}(\epsilon)=\mathbf{b}$ for any scalar $\epsilon$. In addition, since $x\_{i}>0$ for $i=1,2, \ldots, p$, we also have $x\_{i}(\epsilon)>0$ for the same indices $i=1,2, \ldots, p$ and all $\epsilon$ sufficiently small in magnitude. However, there is a value $\bar{\epsilon} \in\left(0, x\_{p}\right]$ such that $x\_{i}(\bar{\epsilon})=0$ for some $i=1,2, \ldots, p$. Hence, $\mathbf{x}(\bar{\epsilon})$ is feasible and has at most $p-1$ nonzero components, contradicting our choice of $p$ as the minimal number of nonzeros.

Therefore, columns $\mathbf{A}\_{1}, \mathbf{A}\_{2}, \ldots, \mathbf{A}\_{p}$ must be linearly independent, and so $p \leq m$. If $p=m$, we are done, since then $\mathbf{x}$ is a basic feasible point and $\mathcal{B}(\mathbf{x})$ is simply $\\{1,2, \ldots, m\\}$. Otherwise, $p<m$, and because $\mathbf{A}$ has full row rank, we can choose $m-p$ columns from among $\mathbf{A}\_{p+1}, \mathbf{A}\_{p+2}, \ldots, \mathbf{A}\_{n}$ to build up a set of $m$ linearly independent vectors. We construct $\mathcal{B}(\mathbf{x})$ by adding the corresponding indices to $\\{1,2, \ldots, p\\}$. The proof of (i) is complete.

The proof of (ii) is quite similar. Let $\mathbf{x}^{\star}$ be a solution with a minimal number of nonzero components $p$, and assume again that $x\_{1}^{\star}, x\_{2}^{\star}, \ldots, x\_{p}^{\star}$ are the nonzeros. If the columns $\mathbf{A}\_{1}, \mathbf{A}\_{2}, \ldots, \mathbf{A}\_{p}$ are linearly dependent, we define

$$
\mathbf{x}^{\star}(\epsilon)=\mathbf{x}^{\star}+\epsilon \mathbf{z},
$$

where $\mathbf{z}$ is chosen exactly as above. It is easy to check that $\mathbf{x}^{\star}(\epsilon)$ will be feasible for all $\epsilon$ sufficiently small, both positive and negative. Hence, since $\mathbf{x}^{\star}$ is optimal, we must have

$$
\mathbf{c}^{\mathrm{T}}\left(\mathbf{x}^{\star}+\epsilon \mathbf{z}\right) \geq \mathbf{c}^{\mathrm{T}} \mathbf{x}^{\star} \Rightarrow \epsilon \mathbf{c}^{\mathrm{T}} \mathbf{z} \geq 0
$$

for all $|\epsilon|$ sufficiently small. Therefore, $\mathbf{c}^{\mathrm{T}} \mathbf{z}=0$, and so $\mathbf{c}^{\mathrm{T}} \mathbf{x}^{\star}(\epsilon)=\mathbf{c}^{\mathrm{T}} \mathbf{x}^{\star}$ for all $\epsilon$. The same logic as in the proof of (i) can be applied to find $\bar{\epsilon}>0$ such that $\mathbf{x}^{\star}(\bar{\epsilon})$ is feasible and optimal, with at most $p-1$ nonzero components. This contradicts our choice of $p$ as the minimal number of nonzeros, so the columns $\mathbf{A}\_{1}, \mathbf{A}\_{2}, \ldots, \mathbf{A}\_{p}$ must be linearly independent. We can now apply the same logic as above to conclude that $\mathbf{x}^{\star}$ is already a basic feasible point and therefore a basic optimal point.

The final statement (iii) is a consequence of finite termination of the simplex method. We comment on the latter property in the next section.
{{</proof>}}

The terminology we use here is not standard, as the following table shows:

| our terminology | standard terminology |
| :--- | :--- |
| basic feasible point | basic feasible solution |
| basic optimal point | optimal basic feasible solution |

The standard terms arose because "solution" and "feasible solution" were originally used as synonyms for "feasible point." However, as the discipline of optimization developed, the word "solution" took on a more specific and intuitive meaning (as in "solution to the problem . . ."). We keep the terminology of this chapter consistent with the rest of the lecture by sticking to this more modern usage.

### Vertices of the feasible polytope

The feasible set defined by the linear constraints is a polytope, and the vertices of this polytope are the points that do not lie on a straight line between two other points in the set. Geometrically, they are easily recognizable.

Algebraically, the vertices are exactly the basic feasible points that we described above. We therefore have an important relationship between the algebraic and geometric viewpoints and a useful aid to understanding how the simplex method works.

{{<theorem "Vertices and basic feasible points" vertices_bfp_theorem>}}
All basic feasible points for \eqref{eq:standard_form} are vertices of the feasible polytope $\\{\mathbf{x} \mid \mathbf{A} \mathbf{x}=\mathbf{b}, \mathbf{x} \geq \mathbf{0}\\}$, and vice versa.
{{</theorem>}}

{{<proof>}}
Let $\mathbf{x}$ be a basic feasible point and assume without loss of generality that $\mathcal{B}(\mathbf{x})=\\{1,2, \ldots, m\\}$. The matrix $\mathbf{B}=\left[\mathbf{A}\_{i}\right]\_{i=1,2, \ldots, m}$ is therefore nonsingular, and

$$
x\_{m+1}=x\_{m+2}=\cdots=x\_{n}=0
$$

Suppose that $\mathbf{x}$ lies on a straight line between two other feasible points $\mathbf{y}$ and $\mathbf{z}$. Then we can find $\alpha \in(0,1)$ such that $\mathbf{x}=\alpha \mathbf{y}+(1-\alpha) \mathbf{z}$. Because of the above condition and the fact that $\alpha$ and $1-\alpha$ are both positive, we must have $y\_{i}=z\_{i}=0$ for $i=m+1, m+2, \ldots, n$. Writing $\mathbf{x}\_{\mathrm{B}}=\left(x\_{1}, x\_{2}, \ldots, x\_{m}\right)^{\mathrm{T}}$ and defining $\mathbf{y}\_{\mathrm{B}}$ and $\mathbf{z}\_{\mathrm{B}}$ likewise, we have from $\mathbf{A} \mathbf{x}=\mathbf{A} \mathbf{y}=\mathbf{A} \mathbf{z}=\mathbf{b}$ that

$$
\mathbf{B} \mathbf{x}\_{\mathrm{B}}=\mathbf{B} \mathbf{y}\_{\mathrm{B}}=\mathbf{B} \mathbf{z}\_{\mathrm{B}}=\mathbf{b},
$$

and so, by nonsingularity of $\mathbf{B}$, we have $\mathbf{x}\_{\mathrm{B}}=\mathbf{y}\_{\mathrm{B}}=\mathbf{z}\_{\mathrm{B}}$. Therefore, $\mathbf{x}=\mathbf{y}=\mathbf{z}$, contradicting our assertion that $\mathbf{y}$ and $\mathbf{z}$ are two feasible points other than $\mathbf{x}$. Therefore, $\mathbf{x}$ is a vertex.

Conversely, let $\mathbf{x}$ be a vertex of the feasible polytope, and suppose that the nonzero components of $\mathbf{x}$ are $x\_{1}, x\_{2}, \ldots, x\_{p}$. If the corresponding columns $\mathbf{A}\_{1}, \mathbf{A}\_{2}, \ldots, \mathbf{A}\_{p}$ are linearly dependent, then we can construct the vector $\mathbf{x}(\epsilon)=\mathbf{x}+\epsilon \mathbf{z}$ as before. Since $\mathbf{x}(\epsilon)$ is feasible for all $\epsilon$ with sufficiently small magnitude, we can define $\hat{\epsilon}>0$ such that $\mathbf{x}(\hat{\epsilon})$ and $\mathbf{x}(-\hat{\epsilon})$ are both feasible. Since $\mathbf{x}=\mathbf{x}(0)$ obviously lies on a straight line between these two points, it cannot be a vertex. Hence our assertion that $\mathbf{A}\_{1}, \mathbf{A}\_{2}, \ldots, \mathbf{A}\_{p}$ are linearly dependent must be incorrect, so these columns must be linearly independent and $p \leq m$. The same arguments as in the proof of {{< theoremref fundamental_lp_theorem >}} can now be used to show that $\mathbf{x}$ is a basic feasible point, completing our proof.
{{</proof>}}

We conclude this discussion of the geometry of the feasible set with a definition of degeneracy. This term has a variety of meanings in optimization, as we discuss later. For the purposes of this chapter, we use the following definition.

{{<definition "Degenerate linear program" degenerate_lp>}}
A linear program \eqref{eq:standard_form} is said to be **degenerate** if there exists at least one basic feasible point that has fewer than $m$ nonzero components.
{{</definition>}}

Naturally, nondegenerate linear programs are those for which this definition is not satisfied.

## The simplex method

### Outline of the method

As we just described, all iterates of the simplex method are basic feasible points for \eqref{eq:standard_form} and therefore vertices of the feasible polytope. Most steps consist of a move from one vertex to an adjacent one for which the set of basic indices $\mathcal{B}(\mathbf{x})$ differs in exactly one component. On most steps (but not all), the value of the primal objective function $\mathbf{c}^{\mathrm{T}} \mathbf{x}$ is decreased. Another type of step occurs when the problem is unbounded: The step is an edge along which the objective function is reduced, and along which we can move infinitely far without ever reaching a vertex.

The major issue at each simplex iteration is to decide which index to change in the basis set $\mathcal{B}$. Unless the step is a direction of unboundedness, one index must be removed from $\mathcal{B}$ and replaced by another from outside $\mathcal{B}$. We can get some insight into how this decision is made by looking again at the KKT conditions \eqref{eq:lp_kkt_gradient}-\eqref{eq:lp_kkt_complementarity} to see how they relate to the algorithm.

From $\mathcal{B}$ and \eqref{eq:lp_kkt_gradient}-\eqref{eq:lp_kkt_complementarity}, we can derive values for not just the primal variable $\mathbf{x}$ but also the dual variables $\boldsymbol{\pi}$ and $\mathbf{s}$, as we now show. We define the index set $\mathcal{N}$ as the complement of $\mathcal{B}$, that is,

$$
\mathcal{N}=\\{1,2, \ldots, n\\} \backslash \mathcal{B} .
$$

Just as $\mathbf{B}$ is the column submatrix of $\mathbf{A}$ that corresponds to the indices $i \in \mathcal{B}$, we use $\mathbf{N}$ to denote the submatrix $\mathbf{N}=\left[\mathbf{A}\_{i}\right]\_{i \in \mathcal{N}}$. We also partition the $n$-element vectors $\mathbf{x}$, $\mathbf{s}$, and $\mathbf{c}$ according to the index sets $\mathcal{B}$ and $\mathcal{N}$, using the notation

$$
\mathbf{x}\_{\mathrm{B}}=\left[x\_{i}\right]\_{i \in \mathcal{B}}, \quad \mathbf{x}\_{\mathrm{N}}=\left[x\_{i}\right]\_{i \in \mathcal{N}}, \quad \mathbf{s}\_{\mathrm{B}}=\left[s\_{i}\right]\_{i \in \mathcal{B}}, \quad \mathbf{s}\_{\mathrm{N}}=\left[s\_{i}\right]\_{i \in \mathcal{N}} .
$$

From the KKT condition \eqref{eq:lp_kkt_equality}, we have that

$$
\mathbf{A} \mathbf{x}=\mathbf{B} \mathbf{x}\_{\mathrm{B}}+\mathbf{N} \mathbf{x}\_{\mathrm{N}}=\mathbf{b} .
$$

The primal variable $\mathbf{x}$ for this simplex iterate is defined as

$$
\mathbf{x}\_{\mathrm{B}}=\mathbf{B}^{-1} \mathbf{b}, \quad \mathbf{x}\_{\mathrm{N}}=\mathbf{0} .
$$

Since we are dealing only with basic feasible points, we know that $\mathbf{B}$ is nonsingular and that $\mathbf{x}\_{\mathrm{B}} \geq \mathbf{0}$, so this choice of $\mathbf{x}$ satisfies two of the KKT conditions: the equality constraints \eqref{eq:lp_kkt_equality} and the nonnegativity condition \eqref{eq:lp_kkt_primal_feasible}.

We choose $\mathbf{s}$ to satisfy the complementarity condition \eqref{eq:lp_kkt_complementarity} by setting $\mathbf{s}\_{\mathrm{B}}=\mathbf{0}$. The remaining components $\boldsymbol{\pi}$ and $\mathbf{s}\_{\mathrm{N}}$ can be found by partitioning this condition into $\mathcal{B}$ and $\mathcal{N}$ components and using $\mathbf{s}\_{\mathrm{B}}=\mathbf{0}$ to obtain

$$
\mathbf{B}^{\mathrm{T}} \boldsymbol{\pi}=\mathbf{c}\_{\mathrm{B}}, \quad \mathbf{N}^{\mathrm{T}} \boldsymbol{\pi}+\mathbf{s}\_{\mathrm{N}}=\mathbf{c}\_{\mathrm{N}} .
$$

Since $\mathbf{B}$ is square and nonsingular, the first equation uniquely defines $\boldsymbol{\pi}$ as

$$
\boldsymbol{\pi}=\mathbf{B}^{-\mathrm{T}} \mathbf{c}\_{\mathrm{B}} .
$$

The second equation implies a value for $\mathbf{s}\_{\mathrm{N}}$:

\begin{equation}
\mathbf{s}\_{\mathrm{N}}=\mathbf{c}\_{\mathrm{N}}-\mathbf{N}^{\mathrm{T}} \boldsymbol{\pi}=\mathbf{c}\_{\mathrm{N}}-\left(\mathbf{B}^{-1} \mathbf{N}\right)^{\mathrm{T}} \mathbf{c}\_{\mathrm{B}} .
\label{eq:reduced_costs}
\end{equation}

Computation of the vector $\mathbf{s}\_{\mathrm{N}}$ is often referred to as pricing. The components of $\mathbf{s}\_{\mathrm{N}}$ are often called the reduced costs of the nonbasic variables $\mathbf{x}\_{\mathrm{N}}$.

The only KKT condition that we have not enforced explicitly is the nonnegativity condition $\mathbf{s} \geq \mathbf{0}$. The basic components $\mathbf{s}\_{\mathrm{B}}$ certainly satisfy this condition, by our choice $\mathbf{s}\_{\mathrm{B}}=\mathbf{0}$. If the vector $\mathbf{s}\_{\mathrm{N}}$ defined by \eqref{eq:reduced_costs} also satisfies $\mathbf{s}\_{\mathrm{N}} \geq \mathbf{0}$, we have found an optimal vector triple $(\mathbf{x}, \boldsymbol{\pi}, \mathbf{s})$, so the algorithm can terminate and declare success. The usual case, however, is that one or more of the components of $\mathbf{s}\_{\mathrm{N}}$ are negative, so the condition $\mathbf{s} \geq \mathbf{0}$ is violated. The new index to enter the basic index set $\mathcal{B}$-the entering index-is now chosen to be one of the indices $q \in \mathcal{N}$ for which $s\_{q}<0$. As we show below, the objective $\mathbf{c}^{\mathrm{T}} \mathbf{x}$ will decrease when we allow $x\_{q}$ to become positive if and only if $q$ has the property that $s\_{q}<0$. Our procedure for altering $\mathcal{B}$ and changing $\mathbf{x}$ and $\mathbf{s}$ accordingly is as follows:

- allow $x\_{q}$ to increase from zero during the next step;
- fix all other components of $\mathbf{x}\_{\mathrm{N}}$ at zero;
- figure out the effect of increasing $x\_{q}$ on the current basic vector $\mathbf{x}\_{\mathrm{B}}$, given that we want to stay feasible with respect to the equality constraints $\mathbf{A} \mathbf{x}=\mathbf{b}$;
- keep increasing $x\_{q}$ until one of the components of $\mathbf{x}\_{\mathrm{B}}$ (corresponding to $x\_{p}$, say) is driven to zero, or determining that no such component exists (the unbounded case);
- remove index $p$ (known as the leaving index) from $\mathcal{B}$ and replace it with the entering index $q$.

It is easy to formalize this procedure in algebraic terms. Since we want both the new iterate $\mathbf{x}^{+}$and the current iterate $\mathbf{x}$ to satisfy $\mathbf{A} \mathbf{x}=\mathbf{b}$, and since $\mathbf{x}\_{\mathrm{N}}=\mathbf{0}$ and $x\_{i}^{+}=0$ for $i \in \mathcal{N} \backslash\\{q\\}$, we have

$$
\mathbf{A} \mathbf{x}^{+}=\mathbf{B} \mathbf{x}\_{\mathrm{B}}^{+}+\mathbf{A}\_{q} x\_{q}^{+}=\mathbf{B} \mathbf{x}\_{\mathrm{B}}=\mathbf{A} \mathbf{x} .
$$

By multiplying this expression by $\mathbf{B}^{-1}$ and rearranging, we obtain

\begin{equation}
\mathbf{x}\_{\mathrm{B}}^{+}=\mathbf{x}\_{\mathrm{B}}-\mathbf{B}^{-1} \mathbf{A}\_{q} x\_{q}^{+} .
\label{eq:simplex_step}
\end{equation}

We show in a moment that the direction $-\mathbf{B}^{-1} \mathbf{A}\_{q}$ is a descent direction for $\mathbf{c}^{\mathrm{T}} \mathbf{x}$. Geometrically speaking, \eqref{eq:simplex_step} is a move along an edge of the feasible polytope that decreases $\mathbf{c}^{\mathrm{T}} \mathbf{x}$. We continue to move along this edge until a new vertex is encountered. We have to stop at this vertex, since by definition we cannot move any further without leaving the feasible region. At the new vertex, a new constraint $x\_{i} \geq 0$ must have become active, that is, one of the components $x\_{i}, i \in \mathcal{B}$, has decreased to zero. This index $i$ is the one that is removed from the basis.

### Finite termination of the simplex method

Let us now verify that the step defined by \eqref{eq:simplex_step} leads to a decrease in $\mathbf{c}^{\mathrm{T}} \mathbf{x}$. By using the definition \eqref{eq:simplex_step} of $\mathbf{x}\_{\mathrm{B}}^{+}$together with

$$
\mathbf{x}\_{\mathrm{N}}^{+}=\left(0, \ldots, 0, x\_{q}^{+}, 0, \ldots, 0\right)^{\mathrm{T}}
$$

we have

\begin{align}
\mathbf{c}^{\mathrm{T}} \mathbf{x}^{+} & =\mathbf{c}\_{\mathrm{B}}^{\mathrm{T}} \mathbf{x}\_{\mathrm{B}}^{+}+\mathbf{c}\_{\mathrm{N}}^{\mathrm{T}} \mathbf{x}\_{\mathrm{N}}^{+} \\\\\\
& =\mathbf{c}\_{\mathrm{B}}^{\mathrm{T}} \mathbf{x}\_{\mathrm{B}}^{+}+c\_{q} x\_{q}^{+} \\\\\\
& =\mathbf{c}\_{\mathrm{B}}^{\mathrm{T}} \mathbf{x}\_{\mathrm{B}}-\mathbf{c}\_{\mathrm{B}}^{\mathrm{T}} \mathbf{B}^{-1} \mathbf{A}\_{q} x\_{q}^{+}+c\_{q} x\_{q}^{+}
\end{align}

Now, from the definition of $\boldsymbol{\pi}$ we have $\mathbf{c}\_{\mathrm{B}}^{\mathrm{T}} \mathbf{B}^{-1}=\boldsymbol{\pi}^{\mathrm{T}}$, while from the second equation above we have $\mathbf{A}\_{q}^{\mathrm{T}} \boldsymbol{\pi}=c\_{q}-s\_{q}$, since $\mathbf{A}\_{q}$ is a column of $\mathbf{N}$. Therefore,

$$
\mathbf{c}\_{\mathrm{B}}^{\mathrm{T}} \mathbf{B}^{-1} \mathbf{A}\_{q} x\_{q}^{+}=\boldsymbol{\pi}^{\mathrm{T}} \mathbf{A}\_{q} x\_{q}^{+}=\left(c\_{q}-s\_{q}\right) x\_{q}^{+}
$$

so by substituting above we obtain

$$
\mathbf{c}^{\mathrm{T}} \mathbf{x}^{+}=\mathbf{c}\_{\mathrm{B}}^{\mathrm{T}} \mathbf{x}\_{\mathrm{B}}-\left(c\_{q}-s\_{q}\right) x\_{q}^{+}+c\_{q} x\_{q}^{+}=\mathbf{c}\_{\mathrm{B}}^{\mathrm{T}} \mathbf{x}\_{\mathrm{B}}-s\_{q} x\_{q}^{+} .
$$

Since $\mathbf{x}\_{\mathrm{N}}=\mathbf{0}$, we have $\mathbf{c}^{\mathrm{T}} \mathbf{x}=\mathbf{c}\_{\mathrm{B}}^{\mathrm{T}} \mathbf{x}\_{\mathrm{B}}$ and therefore

\begin{equation}
\mathbf{c}^{\mathrm{T}} \mathbf{x}^{+}=\mathbf{c}^{\mathrm{T}} \mathbf{x}-s\_{q} x\_{q}^{+}
\label{eq:objective_decrease}
\end{equation}

Since we chose $q$ such that $s\_{q}<0$, and since $x\_{q}^{+}>0$ if we are able to move at all along the edge, it follows from \eqref{eq:objective_decrease} that the step \eqref{eq:simplex_step} produces a decrease in the primal objective function $\mathbf{c}^{\mathrm{T}} \mathbf{x}$.

If the problem is nondegenerate (see {{< definitionref degenerate_lp >}}), then we are guaranteed that $x\_{q}^{+}>0$, so we can be assured of a strict decrease in the objective function $\mathbf{c}^{\mathrm{T}} \mathbf{x}$ at every simplex step. We can therefore prove the following result concerning termination of the simplex method.

{{<theorem "Finite termination of simplex method" simplex_termination>}}
Provided that the linear program \eqref{eq:standard_form} is nondegenerate and bounded, the simplex method terminates at a basic optimal point.
{{</theorem>}}

{{<proof>}}
The simplex method cannot visit the same basic feasible point $\mathbf{x}$ at two different iterations, because it attains a strict decrease at each iteration. Since each subset of $m$ indices drawn from the set $\\{1,2, \ldots, n\\}$ is associated with at most one basic feasible point, it follows that no basis $\mathcal{B}$ can be visited at two different simplex iterations. The number of possible bases is at most $\binom{n}{m}$ (which is the number of ways to choose the $m$ elements of a basis $\mathcal{B}$ from among the $n$ possible indices), so there can be only a finite number of iterations. Since the method is always able to take a step away from a nonoptimal basic feasible point, the point of termination must be a basic optimal point.
{{</proof>}}

Note that this result gives us a proof of {{< theoremref fundamental_lp_theorem >}} (iii) for the nondegenerate case.

### A single step of the method

We have covered most of the mechanics of taking a single step of the simplex method. To make subsequent discussions easier to follow, we summarize our description in a semiformal way.

**Procedure: One step of simplex**

Given $\mathcal{B}, \mathcal{N}, \mathbf{x}\_{\mathrm{B}}=\mathbf{B}^{-1} \mathbf{b} \geq \mathbf{0}, \mathbf{x}\_{\mathrm{N}}=\mathbf{0}$;

Solve $\mathbf{B}^{\mathrm{T}} \boldsymbol{\pi}=\mathbf{c}\_{\mathrm{B}}$ for $\boldsymbol{\pi}$,

Compute $\mathbf{s}\_{\mathrm{N}}=\mathbf{c}\_{\mathrm{N}}-\mathbf{N}^{\mathrm{T}} \boldsymbol{\pi}$;

if $\mathbf{s}\_{\mathrm{N}} \geq \mathbf{0}$

STOP; (*optimal point found*)

Select $q \in \mathcal{N}$ with $s\_{q}<0$ as the entering index;

Solve $\mathbf{B} \mathbf{t}=\mathbf{A}\_{q}$ for $\mathbf{t}$;

if $\mathbf{t} \leq \mathbf{0}$

STOP; (*problem is unbounded*)

Calculate $x\_{q}^{+}=\min \_{i \mid t\_{i}>0}\left(\mathbf{x}\_{\mathrm{B}}\right)\_{i} / t\_{i}$, and use $p$ to denote the index of the basic variable for which this minimum is achieved;

Update $\mathbf{x}\_{\mathrm{B}}^{+}=\mathbf{x}\_{\mathrm{B}}-\mathbf{t} x\_{q}^{+}, \mathbf{x}\_{\mathrm{N}}^{+}=\left(0, \ldots, 0, x\_{q}^{+}, 0, \ldots, 0\right)^{\mathrm{T}}$;

Change $\mathcal{B}$ by adding $q$ and removing $p$.

We need to flesh out this description with specifics of three important points. These are as follows.

- Linear algebra issues-maintaining an LU factorization of $\mathbf{B}$ that can be used to solve for $\boldsymbol{\pi}$ and $\mathbf{t}$.
- Selection of the entering index $q$ from among the negative components of $\mathbf{s}\_{\mathrm{N}}$. (In general, there are many such components.)
- Handling of degenerate steps, in which $x\_{q}^{+}=0$, so that $\mathbf{x}$ is not changed.

Proper handling of these issues is crucial to the efficiency of a simplex implementation. 
