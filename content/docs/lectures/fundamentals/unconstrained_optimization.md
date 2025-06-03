---
title: "2. Unconstrained optimization : basics"
weight: 2
chapter: 2
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
These results, which are based on elementary calculus, provide the foundations for unconstrained optimization algorithms. In one way or another, all algorithms seek a point where $\nabla f(\cdot)$ vanishes.



## The need for algorithms

On question we might ask here is why do we need algorithms to find local minima? After all, we have just shown that if $\nabla f(\mathbf{x}^\star)=0$, then $\mathbf{x}^\star$ is a local minimizer. The answer is that in practice, we do not always have the luxury to know the exact solution to $\nabla f(\mathbf{x})=0$. Moreover, we can't always compute the Hessian matrix to check the second-order conditions. 

Thus, to circumvent the need to solve analytically the equations $\nabla f(\mathbf{x})=0$, we will design algorithms that iteratively update a point $\mathbf{x}$ until it converges to a local minimizer. The algorithms will be based on the properties of the gradient and Hessian, and will use the information they provide to guide the search for a local minimum. When hessian is not computable or too expensive to compute, we will use the gradient only, and the algorithms will be called **gradient-based methods**. When the Hessian is available, we will use it to accelerate convergence, and the algorithms will be called **Newton methods**. As a between between these methods lie **quasi-Newton methods**, which use an approximation of the Hessian to guide the search for a local minimum.

But before we dive in more complicated algorithms, let us consider the most obvious approaches and try to understand their limitations.

## Steepest-descent approach

The first algorithm we consider is the so-called **Steepest-descent** algorithm which involves choosing an initial point $\mathbf{x}_0$ and compute a series of subsequent points with the following formula:

\begin{equation}
\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \nabla f(\mathbf{x}_k),
\label{eq: steepest descent}
\end{equation}

where $\alpha_k$ are a series of scalar values called **step-size** (or learning rate in machine learning context).

The intuition behind this approach is beautifully simple yet profound. Imagine yourself standing on a mountainside in thick fog, trying to find the bottom of the valley. Since you can't see the overall landscape, the most sensible strategy is to feel the slope beneath your feet and take a step in the direction that descends most steeply. This is precisely what the steepest-descent algorithm does mathematically.

The negative gradient $-\nabla f(\mathbf{x}_k)$ points in the direction of steepest decrease of the function at point $\mathbf{x}_k$. This isn't just a convenient mathematical fact—it's the fundamental geometric property that makes gradient-based optimization possible. By moving in this direction, we ensure that we're making the most aggressive local progress toward reducing the function value.

{{<center>}}
{{<NumberedFigure
  src=" ../../../../../../tikZ/succesful_unconstained_optim/main.svg"
  alt="Zig zag"
  width="400px"
  caption=" Optimization with steepest descent"
  label="succesful_unconstained_optim"
>}}
{{</center>}}


### Understanding the algorithm step by step

Let's walk through what happens in each iteration of steepest descent. Starting from point $\mathbf{x}_k$, we compute the gradient $\nabla f(\mathbf{x}_k)$. This vector tells us which direction the function increases most rapidly. Since we want to minimize, we go in the opposite direction: $-\nabla f(\mathbf{x}_k)$.

The step size $\alpha_k$ determines how far we travel in this direction. Think of it as the length of your stride as you walk down the mountain. The choice of step size involves a fundamental trade-off: too small and you make painfully slow progress; too large and you might overshoot the valley bottom or even start climbing uphill again.

### Why steepest descent can struggle

{{<center>}}
{{<NumberedFigure
  src=" ../../../../../../tikZ/zigzag_valley_optimization/main.svg"
  alt="Zig zag"
  width="400px"
  caption=" Problem of stepsize"
  label="zigzag"
>}}
{{</center>}}


Here's where the method reveals its first major limitation. Consider a function that looks like a long, narrow valley—mathematically, this corresponds to a function with a large condition number. Steepest descent exhibits what we call "zigzag behavior" in such cases as illustrated in Figure {{<NumberedFigureRef zigzag>}}.

Picture this scenario: you're in a narrow canyon, and the steepest direction points toward one wall rather than down the canyon. You take steps toward that wall, then the gradient changes direction and points toward the other wall. Instead of walking efficiently down the canyon, you find yourself bouncing back and forth between the walls, making very slow progress toward your destination.

This zigzag pattern occurs because steepest descent is fundamentally myopic. At each step, it only considers the immediate local slope and ignores the broader geometric structure of the function. The algorithm doesn't "remember" where it came from or "anticipate" where the function is heading.



### Convergence properties

Despite these limitations, steepest descent does have reliable convergence properties. Under reasonable mathematical conditions—essentially requiring that the function is well-behaved and doesn't have any pathological features—the algorithm will eventually reach a stationary point where the gradient vanishes.

The convergence is what we call "linear," meaning that the error decreases by a constant factor at each iteration. While this sounds reasonable, it can be frustratingly slow in practice, especially for poorly conditioned problems where that constant factor is very close to one.

## Newton method

If steepest descent is like navigating with only your immediate sense of slope, Newton's method is like having a detailed topographic map of your local surroundings. This method incorporates not just information about which way is downhill, but also how the slope itself is changing—what we call the curvature of the function.

### The mathematical foundation

Newton's method emerges from a clever idea: instead of trying to minimize the original function directly, let's create a simpler approximation and minimize that instead. We use the second-order Taylor approximation around our current point $\mathbf{x}_k$:

$$f(\mathbf{x}_k + \mathbf{p}) \approx f(\mathbf{x}_k) + \nabla f(\mathbf{x}_k)^T\mathbf{p} + \frac{1}{2}\mathbf{p}^T\nabla^2 f(\mathbf{x}_k)\mathbf{p}$$

This quadratic approximation captures both the slope (first-order term) and the curvature (second-order term) at our current location. The brilliant insight is that quadratic functions are easy to minimize—we simply set the gradient of the approximation equal to zero and solve for the optimal step $\mathbf{p}$.

Taking the gradient of our quadratic model and setting it to zero gives us:
$$\nabla f(\mathbf{x}_k) + \nabla^2 f(\mathbf{x}_k)\mathbf{p} = \mathbf{0}$$

Solving for the Newton step:
$$\mathbf{p}_k = -[\nabla^2 f(\mathbf{x}_k)]^{-1}\nabla f(\mathbf{x}_k)$$

The complete Newton iteration becomes:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - [\nabla^2 f(\mathbf{x}_k)]^{-1}\nabla f(\mathbf{x}_k)$$

{{<center>}}
{{<NumberedFigure
  src=" ../../../../../../tikZ/newton_approx/main.svg"
  alt="Zig zag"
  width="400px"
  caption="Newton optimization step"
  label="newton_approx"
>}}
{{</center>}}


### The geometric insight

What makes Newton's method so powerful becomes clear when we think geometrically. The Hessian matrix $\nabla^2 f(\mathbf{x}_k)$ encodes information about how the gradient changes in different directions. If the function curves sharply in one direction and gently in another, the Hessian "knows" this and adjusts the step accordingly.

Consider our narrow valley example again. While steepest descent keeps pointing toward the valley walls, Newton's method recognizes the elongated shape of the valley and naturally takes larger steps along the valley floor and smaller steps perpendicular to it. This geometric awareness eliminates the zigzag behavior that plagues steepest descent.

For quadratic functions, this geometric understanding leads to a remarkable property: Newton's method finds the exact minimum in a single step, regardless of how poorly conditioned the function might be. This happens because our second-order approximation is exact for quadratic functions.

### The power of quadratic convergence

Near a solution that satisfies our second-order sufficient conditions, Newton's method exhibits quadratic convergence. This technical term describes an almost magical property: the number of correct digits in your answer roughly doubles with each iteration.

To appreciate this, consider what linear convergence means: if you have one correct digit, you need about three more iterations to get two correct digits. But with quadratic convergence, if you have one correct digit, the next iteration gives you two, then four, then eight. The improvement accelerates dramatically as you approach the solution.

This rapid convergence makes Newton's method incredibly efficient for high-precision optimization, which is why it forms the backbone of many sophisticated algorithms.

### The computational cost

Newton's method's power comes with a price. At each iteration, we must compute the Hessian matrix, which requires evaluating all second partial derivatives of our function. For a function of $d$ variables, this means computing and storing $d(d+1)/2$ distinct second derivatives.

Even more expensive is solving the linear system $\nabla^2 f(\mathbf{x}_k)\mathbf{p}_k = -\nabla f(\mathbf{x}_k)$ at each iteration. For general matrices, this requires roughly $d^3/3$ arithmetic operations, which becomes prohibitive as the dimension grows.

### When Newton's method can fail

Pure Newton's method isn't foolproof. The Hessian matrix might not be positive definite away from the minimum, which means our quadratic model might not have a minimum—it could have a maximum or a saddle point instead. In such cases, the Newton step might point in completely the wrong direction.

Additionally, if we start too far from a minimum, the quadratic approximation might be a poor representation of the true function, leading to steps that actually increase the function value.

## Looking ahead: the bridge between methods

The contrasting strengths and weaknesses of steepest descent and Newton's method naturally lead to interesting questions. Can we capture some of Newton's geometric insight without the full computational burden? Can we ensure the global reliability of gradient methods while achieving faster local convergence?

These questions motivate more sophisticated approaches. Quasi-Newton methods, which we'll explore in later chapters, build approximations to the Hessian using only gradient information. Methods like BFGS achieve superlinear convergence—faster than linear but not quite quadratic—while requiring much less computation than full Newton steps.

Similarly, trust region methods and linesearch strategies, which we'll study later, provide systematic ways to ensure that our algorithms make reliable progress even when our local approximations aren't perfect.



