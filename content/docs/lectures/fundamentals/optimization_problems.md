---
title: 1. Optimization problems
weight: 1
chapter: 1
---
# Optimization problems


## unconstrained vs constrained

What we are interested in these lectures is to solve problems of the form :

\begin{equation}
  \underset{\mathbf{x}\in\mathbb{R}^d}{\operatorname{(arg)min}} f(\mathbf{x}),
  \label{eq: optim general unconstrained}
\end{equation}
where $\mathbf{x}\in\mathbb{R}^d$ and $f:\mathcal{D}_f \mapsto \mathbb{R} $ is a scalar-valued function with domain $\mathcal{D}_f$. Under this formulation, the problem is said to be an **unconstrained Optimization**  problem.

If additionally, we add a set of equalities constraints functions:
$$
\\{h\_i : \mathbb{R}^d \mapsto \mathbb{R} \\, /\\,   1 \leq i \leq N    \\}
$$
and inequalities constraints functions:
$$
\\{g\_j : \mathbb{R}^d \mapsto \mathbb{R} \\, /\\,  1 \leq j \leq M  \\}
$$
and define the set $\mathcal{S} = \\{\mathbf{x} \in \mathbb{R}^d \\,/\\, \forall\\,(i, j),\\, h\_i(\mathbf{x})=0,\\, g\_j(\mathbf{x})\leq 0\\}$ and want to solve:
\begin{equation}
  \underset{\mathbf{x}\in\mathcal{S}}{\operatorname{(arg)min}} f(\mathbf{x}),
  \label{eq: optim general constrained}
\end{equation}
then the problem is said to be a **constrained optimization** problem.

> Note that here, the constraints and the function domain are not the same sets. Constraints usually stem from modelling of the problem whilst the function domain only characterizes for which values of $\mathbf{x}$ it is possible to compute a value of the function.

Sunch problems arise when ...
TODO. Lorem 

## Global optimization vs local optimization

{{< center >}}
{{< NumberedFigure
  src="./tikZ/himmelblau/main_2D.svg"
  caption="An example of multiple local minima"
  alt="Himmelblau's Function"
  width="500px"
  label="fig:himmelblau"
>}}
{{< /NumberedFigure >}}


