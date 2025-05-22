---
title: "3. Unconstrained optimization : basics"
weight: 3
chapter: 3
---

# Unconstrained optimization - basics


We hereby consider problems without any constraints on the set of admissible solutions, i.e we aim to solve:
$$
  \underset{\mathbf{x}\in\mathbb{R}^d}{\operatorname{argmin}} f(\mathbf{x}).
$$

Let us try to characterizes the nature of the solutions under this setup.

## What is a solution ?

{{<NumberedFigure
  src=" ../../../../../../tikZ/local_global_minima/main.svg"
  alt="Local vs global"
  width="600px"
  caption="Local and global minimum can coexist."
>}}

Generally, we would be happiest if we found a global minimizer of $f$ , a point where the
function attains its least value. A formal definition is :

{{<definition "Global minimizer" >}} 
  A point $\mathbf{x}^\star$ is a **global minimizer** if $f(\mathbf{x}^\star)\leq f(\mathbf{x})$,
where $\mathbf{x}$ ranges over all of $\mathbb{R}^d$  (or at least over the domain of interest to the modeler).
{{</definition>}}

The global minimizer can be difficult to find, because our knowledge of $f$ is usually only local.
Since our algorithm does not visit many points (we hope!), we usually do not have a good
picture of the overall shape of $f$ , and we can never be sure that the function does not take a
sharp dip in some region that has not been sampled by the algorithm. Most algorithms are
able to find only a local minimizer, which is a point that achieves the smallest value of f in
its neighborhood. Formally, we say:

{{<definition "Local minimizer" >}} 
  A point $\mathbf{x}^\star$ is a **local minimizer** if $\exists r>0,\\, f(\mathbf{x}^\star)\leq f(\mathbf{x})$, $\forall \mathbf{x}\in\mathcal{B}(\mathbf{x}^\star, r)$.

{{</definition>}}

A point that satisfies
this definition is sometimes called a **weak local minimizer**. Alternatively, when $f(\mathbf{x}^\star)<f(\mathbf{x})$, we say that the minimum is a **strict local minimizer**.

## Taylor's theorem

From the definitions given above, it might seem that the only way to find out whether
a point $\mathbf{x}^\star$ is a local minimum is to examine all the points in its immediate vicinity, to make
sure that none of them has a smaller function value. When the function $f$ is smooth, however,
there are much more efficient and practical ways to identify local minima. In particular, if $f$
is twice continuously differentiable, we may be able to tell that $\mathbf{x}^\star$ is a local minimizer (and
possibly a strict local minimizer) by examining just the gradient $\nabla f (\mathbf{x}^\star)$ and the Hessian
$\nabla^2 f (\mathbf{x}^\star)$.
The mathematical tool used to study minimizers of smooth functions is Taylor’s the-
orem. Because this theorem is central to our analysis  we state it now.

{{<theorem "Taylor's theorem" taylor_theory>}}
Suppose that $f:\mathbb{R}^d\mapsto\mathbb{R}$ is continuously differentiable and that we have $\mathbf{p}\in\mathbb{R}^d $. The we have :
\begin{equation}
  f(\mathbf{x}+\mathbf{p}) = f(\mathbf{x}) + \nabla f(\mathbf{x}+t\mathbf{p})^\mathrm{T}\mathbf{p},
\end{equation}
for some $t\in [0,1]$.

Moreover, if $f$ is twice continuously differentiable, we have :
\begin{equation}
  f(\mathbf{x}+\mathbf{p}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^\mathrm{T}\mathbf{p} + \frac{1}{2}\mathbf{p}^\mathrm{T}\nabla^2 f(\mathbf{x}+t\mathbf{p})\mathbf{p}),
\end{equation}
for some $t\in [0,1]$.
{{</theorem>}}
{{<proof>}}
See any calculus book
{{</proof>}}

Note that in this formulation, the definition is exact and the $t$ scalar is usually unknown. The interest lies in skeching proofs. In practical matters, we rather use the following approximation:
{{<theorem "Taylor's approximation" taylor_approximation>}}
First order approximation:
\begin{equation}
  f(\mathbf{x}+\mathbf{p}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^\mathrm{T}\mathbf{p} + o(\lVert\mathbf{p}\rVert),
\end{equation}

Second-order approximation:

\begin{equation}
  f(\mathbf{x}+\mathbf{p}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^\mathrm{T}\mathbf{p} + \frac{1}{2}\mathbf{p}^\mathrm{T}\nabla^2 f(\mathbf{x})\mathbf{p} + o(\lVert\mathbf{p}\rVert^2),
\end{equation}

where $o(\lVert\mathbf{p}\rVert)$ and $o(\lVert\mathbf{p}\rVert^2)$ represent terms that grow slower than $\lVert\mathbf{p}\rVert$ and $\lVert\mathbf{p}\rVert^2$ respectively as $\lVert\mathbf{p}\rVert \to 0$.
{{</theorem>}}

## Sufficient and necessary conditions for local minima

Let us consider a local minimum and see how they can be characterized to later design appropriate solution finding methods. The first well-known result is as follows:
{{<theorem "First-order necessary conditions" first-order_necessary>}}
if $\mathbf{x}^\star$ is a local minimize, and $f$ is continuously differentiable in a neighborhood of $\mathbf{x}^\star$, then $\nabla f(\mathbf{x}^\star) = \mathbf{0}$.
{{</theorem>}}
{{<proof>}}
  Suppose for contradiction that $\nabla f(\mathbf{x}^\star) \neq 0$, and define vector $\mathbf{p}=-\nabla f(\mathbf{x}^\star)$ such that by construction $\mathbf{p}^\mathrm{T}\nabla f(\mathbf{x}^\star) = - \lVert f(\mathbf{x}^\star) \rVert^2 < 0$.

Since $f$ is a continuous function, we can define a scalar $T>0$ such that $\forall t\in [0,T[$, we still have:
$$
\mathbf{p}^\mathrm{T}f(\mathbf{x}+t\mathbf{p}) < 0.
$$

Using {{< theoremref taylor_theory >}} first-order result, we can write:
$$
f(\mathbf{x}^\star+t\mathbf{p}) = f(\mathbf{x}^\star) + t\mathbf{p}^\mathrm{T}\nabla f(\mathbf{x}^\star+\overline{t}\mathbf{p}),
$$
for some $\overline{t}\in[0,T[$ and any $t\in[0,T[$. Given previous inequality, we obtain:
$$
  f(\mathbf{x}^\star+t\mathbf{p}) < f(\mathbf{x}^\star),
$$
which contradicts the fact that $\mathbf{x}^\star$ is a local minimizer.
{{</proof>}}

Henceforth, we will call **stationary point**, any $\mathbf{x}$ such that $\nabla f(\mathbf{x}) = 0$.

For the next result we recall that a matrix $\mathbf{B}$ is positive definite if $\mathbf{p}^\mathrm{T} \mathbf{B} \mathbf{p}>0$ for all $\mathbf{p} \neq \mathbf{0}$, and positive semidefinite if $\mathbf{p}^\mathrm{T} \mathbf{B} \mathbf{p} \geq 0$ for all $\mathbf{p}$.

{{<theorem "Second-order necessary conditions">}}
If $\mathbf{x}^\star$ is a local minimizer of $f$ and $\nabla^2 f$ is continuous in an open neighborhood of $\mathbf{x}^\star$, then $\nabla f\left(\mathbf{x}^\star\right)=0$ and $\nabla^2 f\left(\mathbf{x}^\star\right)$ is positive semidefinite.
{{</theorem>}}

{{<proof>}}
Proof. We know from {{<theoremref first-order_necessary>}} that $\nabla f\left(\mathbf{x}^\star\right)=0$. For contradiction, assume that $\nabla^2 f\left(\mathbf{x}^\star\right)$ is not positive semidefinite. Then we can choose a vector $\mathbf{p}$ such that $\mathbf{p}^T \nabla^2 f\left(\mathbf{x}^\star\right) \mathbf{p}<0$, and because $\nabla^2 f$ is continuous near $\mathbf{x}^\star$, there is a scalar $T>0$ such that $\mathbf{p}^\mathrm{T} \nabla^2 f\left(\mathbf{x}^*+\overline{t} \mathbf{p}\right) \mathbf{p}<0$ for all $t \in[0, T[$.

By doing a Taylor series expansion around $\mathbf{x}^\star$, we have for all $\bar{t} \in[0, T[$ and some $t \in[0, \bar{t}]$ that

$$
f\left(\mathbf{x}^\star+\bar{t} \mathbf{p}\right) = f\left(\mathbf{x}^\star\right)+\bar{t} \mathbf{p}^\mathrm{T} \nabla f\left(\mathbf{x}^\star\right)+\frac{1}{2} \bar{t}^2 \mathbf{p}^\mathrm{T} \nabla^2 f\left(\mathbf{x}^\star+t \mathbf{p}\right) \mathbf{p}<f\left(\mathbf{x}^\star\right) .
$$


As in {{<theoremref first-order_necessary>}}, we have found a direction from $\mathbf{x}^\star$ along which $f$ is decreasing, and so again, $\mathbf{x}^\star$ is not a local minimizer.
{{</proof>}}

We now describe sufficient conditions, which are conditions on the derivatives of $f$ at the point $\mathbf{z}^\star$ that guarantee that $\mathbf{x}^\star$ is a local minimizer.


{{<theorem "Second-Order Sufficient Conditions" second-order_sufficient>}}
Suppose that $\nabla^2 f$ is continuous in an open neighborhood of $\mathbf{x}^\star$ and that $\nabla f\left(\mathbf{x}^\star\right)=0$ and $\nabla^2 f\left(\mathbf{x}^\star\right)$ is positive definite. Then $\mathbf{x}^\star$ is a strict local minimizer of $f$.
{{</theorem>}}
{{<proof>}}
Because the Hessian is continuous and positive definite at $\mathbf{x}^\star$, we can choose a radius $r>0$ so that $\nabla^2 f(x)$ remains positive definite for all $x$ in the open ball $\mathcal{D}=\left\\{\mathbf{z} \mid\left\lVert\mathbf{z}-\mathbf{x}^\star\right\rVert<\right.$ $r\\}$. Taking any nonzero vector $p$ with $\lVert\mathbf{p}\rVert<r$, we have $\mathbf{x}^\star+\mathbf{p} \in \mathcal{D}$ and so

$$
\begin{aligned}
f\left(\mathbf{x}^\star+p\right) & =f\left(\mathbf{x}^\star\right)+\mathbf{p}^\mathrm{T} \nabla f\left(\mathbf{x}^\star\right)+\frac{1}{2} \mathbf{p}^\mathrm{T} \nabla^2 f(\mathbf{z}) \mathbf{p} \\
& =f\left(\mathbf{x}^\star\right)+\frac{1}{2} \mathbf{p}^\mathrm{T} \nabla^2 f(\mathbf{z}) \mathbf{p}
\end{aligned}
$$

where $\mathbf{z}=\mathbf{x}^\star+t \mathbf{p}$ for some $t \in(0,1)$. Since $\mathbf{z} \in \mathcal{D}$, we have $\mathbf{p}^{\mathrm{T}} \nabla^2 f(\mathbf{z}) \mathbf{p}>0$, and therefore $f\left(\mathbf{x}^\star+\mathbf{p}\right)>f\left(\mathbf{x}^\star\right)$, giving the result.
{{</proof>}}

Note that the second-order sufficient conditions of {{<theoremref second-order_sufficient>}} guarantee something stronger than the necessary conditions discussed earlier; namely, that the minimizer is a strict local minimizer. Note too that the second-order sufficient conditions are not necessary: A point $\mathbf{x}^\star$ may be a strict local minimizer, and yet may fail to satisfy the sufficient conditions. A simple example is given by the function $f(x)=x^4$, for which the point $x^\star=0$ is a strict local minimizer at which the Hessian matrix vanishes (and is therefore not positive definite).
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

These results, which are based on elementary calculus, provide the foundations for unconstrained optimization algorithms. In one way or another, all algorithms seek a point where $\nabla f(\cdot)$ vanishes.



## The need for algorithms


## Steepest-descent approach


## Newton method
