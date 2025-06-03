---
title: 3. Convexity theory
weight: 3
chapter: 3
---

# Convexity theory

Convexity is a powerful property of functions and sets that simplifies the analysis of optimization problems and the characterization of global minimizers. In this chapter, we will explore the concepts of convex sets, convex functions, and their implications for unconstrained optimization.

## Convex sets

Let us first start by defining the convexity of a given set $\mathcal{S}\subset\mathbb{R}^d$:

{{<definition "Convex set" convex_set>}}
Let $\mathcal{S}\subset\mathbb{R}^d$ be a set. The set $\mathcal{S}$ is convex if, for any two points $\mathbf{x}, \mathbf{y} \in \mathcal{S}$, the line segment that connects them is also contained in $\mathcal{S}$, that is,
\begin{equation}
\mathbf{x}, \mathbf{y} \in \mathcal{S} \implies \lambda \mathbf{x} + (1-\lambda) \mathbf{y} \in \mathcal{S}, \quad \forall \lambda \in [0, 1].
\label{eq:convex_set}
\end{equation}

{{</definition>}}

{{<center>}}
{{<NumberedFigure
  src=" ../../../../../../tikZ/convex_set/main.svg"
  alt="Zig zag"
  width="400px"
  caption="Convex set"
  label="convex_set"
>}}
{{</center>}}


This is illustrated by Figure {{<NumberedFigureRef convex_set>}}, where the set $\mathcal{S}$ is convex, as the line segment between any two points $\mathbf{x}$ and $\mathbf{y}$ lies entirely within $\mathcal{S}$. If this property does not hold, then the set is called non-convex.

While we will not do deeper now, this property is desirable for the constraints of an optimization problem, .because it means that for a given algorithm, any subsequent step is feasible by staying true to the given constraints for the problem.


## Convex functions

A function $f:\mathbb{R}^n\to\mathbb{R}$ is convex if its domain is a convex set and it satisfies the following property:

{{<definition "Convex function" convex_function>}}
A function $f:\mathbb{R}^n\to\mathbb{R}$ is convex if, for any two points $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ and for all $\lambda \in [0, 1]$, the following inequality holds:
\begin{equation}
f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y}).
\label{eq:convex_function}
\end{equation}
{{</definition>}}


{{<center>}}
{{<NumberedFigure
  src=" ../../../../../../tikZ/convex_func/main.svg"
  alt="Zig zag"
  width="600px"
  caption="Convex function"
  label="convex_func"
>}}
{{</center>}}


This means that the line segment connecting the points $(\mathbf{x}, f(\mathbf{x}))$ and $(\mathbf{y}, f(\mathbf{y}))$ lies above the graph of the function $f$. In other words, the function is "bowl-shaped" or "curves upwards". Such an illustration is given for a 1-dimensional function in Figure {{<NumberedFigureRef convex_func>}}, where the function $f$ is convex, as the line segment between any two points $(\mathbf{x}, f(\mathbf{x}))$ and $(\mathbf{y}, f(\mathbf{y}))$ lies above the graph of $f$.

In practice to show that a function is convex, we can make use of the following properties, given convex functions $f$ and $g$:
* let $\alpha, \beta>0$, then $\alpha f + \beta g$ is convex
* $f \circ g$ is convex





## Convexity and unconstrained optimization

When the objective function is convex, local and global minimizers are simple to characterize.


{{<theorem "" local_global_convex>}}
When $f$ is convex, any local minimizer $\mathbf{x}^\star$ is a global minimizer of $f$. If in addition $f$ is differentiable, then any stationary point $\mathbf{x}^\star$ is a global minimizer of $f$.
{{</theorem>}}
{{<proof>}}
Suppose that $\mathbf{x}^\star$ is a local but not a global minimizer. Then we can find a point $\mathbf{z} \in \mathbb{R}^n$ with $f(\mathbf{z})<f\left(\mathbf{x}^\star\right)$. Consider the line segment that joins $\mathbf{x}^\star$ to $\mathbf{z}$, that is,
\begin{equation}
\mathbf{x}=\lambda \mathbf{z}+(1-\lambda) \mathbf{x}^\star, \quad \text { for some } \lambda \in(0,1]
\label{eq:line_segment}
\end{equation}
By the convexity property for $f$, we have
\begin{equation}
f(\mathbf{x}) \leq \lambda f(\mathbf{z})+(1-\lambda) f\left(\mathbf{x}^\star\right)<f\left(\mathbf{x}^\star\right)
\label{eq:convexity}
\end{equation}

Any neighborhood $\mathcal{N}$ of $\mathbf{x}^\star$ contains a piece of the line segment \eqref{eq:line_segment}, so there will always be points $\mathbf{x} \in \mathcal{N}$ at which \eqref{eq:convexity} is satisfied. Hence, $\mathbf{x}^\star$ is not a local minimizer.
For the second part of the theorem, suppose that $\mathbf{x}^\star$ is not a global minimizer and choose $\mathbf{z}$ as above. Then, from convexity, we have

\begin{equation}
\begin{aligned}
\nabla f\left(\mathbf{x}^\star\right)^{\mathrm{T}}\left(\mathbf{z}-\mathbf{x}^\star\right) \& =\left.\frac{d}{d \lambda} f\left(\mathbf{x}^\star+\lambda\left(\mathbf{z}-\mathbf{x}^\star\right)\right)\right|_{\lambda=0}  \\\\
\& =\lim _{\lambda \downarrow 0} \frac{f\left(\mathbf{x}^\star+\lambda\left(\mathbf{z}-\mathbf{x}^\star\right)\right)-f\left(\mathbf{x}^\star\right)}{\lambda} \\\\
& \leq \lim _{\lambda \downarrow 0} \frac{\lambda f(\mathbf{z})+(1-\lambda) f\left(\mathbf{x}^\star\right)-f\left(\mathbf{x}^\star\right)}{\lambda} \\\\
\& =f(\mathbf{z})-f\left(\mathbf{x}^\star\right)<0
\end{aligned}
\end{equation}

Therefore, $\nabla f\left(\mathbf{x}^\star\right) \neq 0$, and so $\mathbf{x}^\star$ is not a stationary point.
{{</proof>}}


This result is fundamental in optimization, as it guarantees that if we find a local minimizer of a convex function, we can be sure that it is also the global minimizer. This property greatly simplifies the search for optimal solutions. As such, finding that the function we minimize is convex often means that the problem is easier to solve, as we can use algorithms that are guaranteed to converge to the global minimum.

Conversely, in the design stage, we might prefer to design a convex function, or try to find a convex approximation of a non-convex function, to ensure that the optimization problem is well-behaved and that we can find the global minimum efficiently.
