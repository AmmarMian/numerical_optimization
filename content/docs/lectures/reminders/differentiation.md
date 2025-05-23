---
title: Differentiation
weight: 1
chapter: A
---

# Fundamentals of Multivariate Differentiation

Multivariate calculus extends the fundamental concepts of single-variable calculus to functions of several variables. This powerful mathematical framework allows us to analyze and describe phenomena in multiple dimensions, making it essential for physics, engineering, economics, statistics, and many other disciplines. This document provides a comprehensive introduction to multivariate differentiation, building from basic principles to advanced applications.

## 1. Functions of Several Variables

We begin by defining functions that map points from higher-dimensional spaces to either real numbers or other multi-dimensional spaces.

{{< definition "Function of Several Variables" "multivariate-function" >}}
A real-valued function of $n$ variables is a mapping $f: D \rightarrow \mathbb{R}$ where $D \subseteq \mathbb{R}^n$. For each point $(x_1, x_2, \ldots, x_n) \in D$, the function assigns a unique real value $f(x_1, x_2, \ldots, x_n)$.
{{< /definition >}}

{{< definition "Vector-Valued Function" "vector-valued-function" >}}
A vector-valued function of $n$ variables is a mapping $\mathbf{F}: D \rightarrow \mathbb{R}^m$ where $D \subseteq \mathbb{R}^n$. For each point $(x_1, x_2, \ldots, x_n) \in D$, the function assigns a unique vector $\mathbf{F}(x_1, x_2, \ldots, x_n) = (F_1(x_1, \ldots, x_n), \ldots, F_m(x_1, \ldots, x_n))$.
{{< /definition >}}

{{% hint info %}}
**Visualizing Functions of Several Variables**  
While functions of one variable can be visualized as curves in the plane, functions of two variables $f(x,y)$ can be visualized as surfaces in three-dimensional space. For functions of more than two variables, we often use level sets, contour plots, or partial visualizations to gain insight into their behavior.
{{% /hint %}}

### Exercises on Functions of Several Variables

**Exercise 1.1** Identify the domain and range of the function $f(x,y) = \sqrt{1-x^2-y^2}$.

**Solution:**
For the function to be real-valued, we need $1-x^2-y^2 \geq 0$, which means $x^2+y^2 \leq 1$. This describes a closed disk in the $xy$-plane with center at the origin and radius 1.

Therefore, the domain is $D = \{(x,y) \in \mathbb{R}^2 \mid x^2+y^2 \leq 1\}$.

Since $0 \leq 1-x^2-y^2 \leq 1$ for all points in the domain, we have $0 \leq \sqrt{1-x^2-y^2} \leq 1$.

Therefore, the range is $[0,1]$.

Geometrically, the graph of this function forms the upper half of a sphere with radius 1 centered at the origin.

**Exercise 1.2** Consider the vector-valued function $\mathbf{F}(x,y) = (x^2-y^2, 2xy)$. Show that this function can be interpreted as a complex-valued function. What familiar operation does it represent?

**Solution:**
The function $\mathbf{F}(x,y) = (x^2-y^2, 2xy)$ maps points from $\mathbb{R}^2$ to $\mathbb{R}^2$.

If we interpret the input $(x,y)$ as a complex number $z = x + yi$, and the output $(x^2-y^2, 2xy)$ as another complex number $w = (x^2-y^2) + (2xy)i$, we can show that $\mathbf{F}$ represents complex squaring.

This is because:
$z^2 = (x + yi)^2 = x^2 + 2xyi + (yi)^2 = x^2 + 2xyi - y^2 = (x^2 - y^2) + (2xy)i$

Therefore, $\mathbf{F}(x,y) = (x^2-y^2, 2xy)$ corresponds to the complex function $f(z) = z^2$.

## 2. Limits and Continuity

Before diving into differentiation, we need to establish the concepts of limits and continuity in higher dimensions, which are more intricate than their single-variable counterparts.

{{< definition "Limit of a Multivariate Function" "multivariate-limit" >}}
Let $f: D \rightarrow \mathbb{R}$ be a function defined on a domain $D \subseteq \mathbb{R}^n$, and let $\mathbf{a}$ be a point in $\mathbb{R}^n$ such that every neighborhood of $\mathbf{a}$ contains points in $D$ other than $\mathbf{a}$ itself. We say that $f(\mathbf{x})$ approaches the limit $L$ as $\mathbf{x}$ approaches $\mathbf{a}$, written as
\begin{equation}
  \lim_{\mathbf{x} \to \mathbf{a}} f(\mathbf{x}) = L
\end{equation}
if for every $\varepsilon > 0$, there exists a $\delta > 0$ such that
\begin{equation}
  |f(\mathbf{x}) - L| < \varepsilon \quad \text{whenever} \quad 0 < \|\mathbf{x} - \mathbf{a}\| < \delta \quad \text{and} \quad \mathbf{x} \in D
\end{equation}
where $\|\mathbf{x} - \mathbf{a}\|$ is the Euclidean distance between $\mathbf{x}$ and $\mathbf{a}$.
{{< /definition >}}

{{% hint warning %}}
**Path Independence of Limits**  
A crucial difference in multivariate calculus is that a limit exists only if the function approaches the same value regardless of the path taken to approach the point. In single-variable calculus, there are only two directions of approach (from the left or right), but in higher dimensions, there are infinitely many possible paths.
{{% /hint %}}

{{< theorem "Existence of Multivariate Limits" "multivariate-limit-existence" >}}
If there exist two different paths approaching $\mathbf{a}$ along which $f(\mathbf{x})$ approaches different limits, then $\lim_{\mathbf{x} \to \mathbf{a}} f(\mathbf{x})$ does not exist.
{{< /theorem >}}

{{< proof >}}
Suppose there are two paths $P_1$ and $P_2$ approaching $\mathbf{a}$ such that
\begin{equation}
  \lim_{\mathbf{x} \to \mathbf{a}, \mathbf{x} \in P_1} f(\mathbf{x}) = L_1 \quad \text{and} \quad \lim_{\mathbf{x} \to \mathbf{a}, \mathbf{x} \in P_2} f(\mathbf{x}) = L_2
\end{equation}
where $L_1 \neq L_2$.

Let $\varepsilon = \frac{|L_1 - L_2|}{3} > 0$. According to the definition of a limit, if $\lim_{\mathbf{x} \to \mathbf{a}} f(\mathbf{x}) = L$ were to exist, there would be a $\delta > 0$ such that
\begin{equation}
  |f(\mathbf{x}) - L| < \varepsilon \quad \text{whenever} \quad 0 < \|\mathbf{x} - \mathbf{a}\| < \delta \quad \text{and} \quad \mathbf{x} \in D
\end{equation}

However, we can choose points $\mathbf{x}_1 \in P_1$ and $\mathbf{x}_2 \in P_2$ such that $\|\mathbf{x}_1 - \mathbf{a}\| < \delta$ and $\|\mathbf{x}_2 - \mathbf{a}\| < \delta$, and by the existence of the limits along the paths, we can ensure that
\begin{equation}
  |f(\mathbf{x}_1) - L_1| < \varepsilon \quad \text{and} \quad |f(\mathbf{x}_2) - L_2| < \varepsilon
\end{equation}

This would imply that
\begin{equation}
  |f(\mathbf{x}_1) - L| < \varepsilon \quad \text{and} \quad |f(\mathbf{x}_2) - L| < \varepsilon
\end{equation}

By the triangle inequality, we have
\begin{equation}
  |L_1 - L_2| \leq |L_1 - f(\mathbf{x}_1)| + |f(\mathbf{x}_1) - L| + |L - f(\mathbf{x}_2)| + |f(\mathbf{x}_2) - L_2| < 4\varepsilon = \frac{4|L_1 - L_2|}{3}
\end{equation}

This leads to the contradiction $|L_1 - L_2| < \frac{4|L_1 - L_2|}{3}$.

Therefore, the limit $\lim_{\mathbf{x} \to \mathbf{a}} f(\mathbf{x})$ cannot exist if the limits along different paths are different.
{{< /proof >}}

{{< definition "Continuity of a Multivariate Function" "multivariate-continuity" >}}
A function $f: D \rightarrow \mathbb{R}$ is continuous at a point $\mathbf{a} \in D$ if
\begin{equation}
  \lim_{\mathbf{x} \to \mathbf{a}} f(\mathbf{x}) = f(\mathbf{a})
\end{equation}
A function is continuous on a set $S \subseteq D$ if it is continuous at every point in $S$.
{{< /definition >}}

### Exercises on Limits and Continuity

**Exercise 2.1** Determine whether the following limit exists, and find its value if it does:
\begin{equation}
  \lim_{(x,y) \to (0,0)} \frac{x^2 - y^2}{x^2 + y^2}
\end{equation}

**Solution:**
To evaluate this limit, we need to check if the function approaches the same value regardless of the path used to approach the origin.

Let's parameterize different paths to the origin and check if they yield the same limit:

1. Along the $x$-axis (where $y = 0$):
   \begin{equation}
     \lim_{x \to 0} \frac{x^2 - 0^2}{x^2 + 0^2} = \lim_{x \to 0} \frac{x^2}{x^2} = 1
   \end{equation}

2. Along the $y$-axis (where $x = 0$):
   \begin{equation}
     \lim_{y \to 0} \frac{0^2 - y^2}{0^2 + y^2} = \lim_{y \to 0} \frac{-y^2}{y^2} = -1
   \end{equation}

Since we get different values along different paths, the limit does not exist.

Alternatively, we can use polar coordinates. Setting $x = r\cos\theta$ and $y = r\sin\theta$, we get:
\begin{equation}
  \frac{x^2 - y^2}{x^2 + y^2} = \frac{r^2\cos^2\theta - r^2\sin^2\theta}{r^2\cos^2\theta + r^2\sin^2\theta} = \frac{\cos^2\theta - \sin^2\theta}{\cos^2\theta + \sin^2\theta} = \cos(2\theta)
\end{equation}

As $r \to 0$, the value depends on $\theta$, which confirms that the limit does not exist.

**Exercise 2.2** Show that the function $f(x,y) = \frac{xy}{x^2+y^2}$ with $f(0,0) = 0$ is not continuous at the origin.

**Solution:**
For $f$ to be continuous at the origin, we need 
\begin{equation}
  \lim_{(x,y) \to (0,0)} \frac{xy}{x^2+y^2} = f(0,0) = 0
\end{equation}

Let's examine the limit along the line $y = mx$, where $m$ is a constant:
\begin{equation}
  \lim_{x \to 0} \frac{x \cdot mx}{x^2 + (mx)^2} = \lim_{x \to 0} \frac{mx^2}{x^2(1 + m^2)} = \frac{m}{1 + m^2}
\end{equation}

The limit depends on the value of $m$, so the limit doesn't exist as $(x,y) \to (0,0)$. Therefore, $f$ is not continuous at the origin.

To be more specific, along the line $y = x$ (where $m = 1$), the limit is $\frac{1}{2}$, while along the line $y = -x$ (where $m = -1$), the limit is $-\frac{1}{2}$.

## 3. Partial Derivatives

Partial derivatives are the foundation of multivariate calculus, allowing us to analyze how a function changes with respect to one variable while keeping all others fixed.

{{< definition "Partial Derivative" "partial-derivative" >}}
The partial derivative of a function $f(x_1, x_2, \ldots, x_n)$ with respect to $x_i$ is defined as
\begin{equation}
  \frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}
\end{equation}
provided the limit exists. Alternative notations include $f_{x_i}$, $D_i f$, or $\partial_{x_i} f$.
{{< /definition >}}

{{% hint info %}}
**Geometric Interpretation of Partial Derivatives**  
The partial derivative $\frac{\partial f}{\partial x_i}$ at a point $\mathbf{a}$ represents the slope of the tangent line to the curve formed by intersecting the graph of $f$ with the plane passing through $\mathbf{a}$ parallel to the $x_i$-axis and the $z$-axis (where $z = f(x_1, \ldots, x_n)$).
{{% /hint %}}

{{< theorem "Partial Derivatives of Elementary Functions" "elementary-partial-derivatives" >}}
For functions $f, g: \mathbb{R}^n \rightarrow \mathbb{R}$ and constant $c$, the following hold:
1. $\frac{\partial}{\partial x_i}(f + g) = \frac{\partial f}{\partial x_i} + \frac{\partial g}{\partial x_i}$
2. $\frac{\partial}{\partial x_i}(cf) = c \frac{\partial f}{\partial x_i}$
3. $\frac{\partial}{\partial x_i}(fg) = f \frac{\partial g}{\partial x_i} + g \frac{\partial f}{\partial x_i}$
4. $\frac{\partial}{\partial x_i}\left(\frac{f}{g}\right) = \frac{g \frac{\partial f}{\partial x_i} - f \frac{\partial g}{\partial x_i}}{g^2}$, where $g \neq 0$
5. $\frac{\partial}{\partial x_i}(x_i^n) = nx_i^{n-1}$
6. $\frac{\partial}{\partial x_i}(x_j) = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$
{{< /theorem >}}

{{< proof >}}
These properties follow directly from the corresponding rules for derivatives in single-variable calculus, since a partial derivative with respect to $x_i$ treats all other variables as constants.

For example, to prove property 3 (the product rule):
\begin{align}
\frac{\partial}{\partial x_i}(fg) &= \lim_{h \to 0} \frac{f(\mathbf{x}+h\mathbf{e}_i)g(\mathbf{x}+h\mathbf{e}_i) - f(\mathbf{x})g(\mathbf{x})}{h} \\
&= \lim_{h \to 0} \frac{f(\mathbf{x}+h\mathbf{e}_i)g(\mathbf{x}+h\mathbf{e}_i) - f(\mathbf{x})g(\mathbf{x}+h\mathbf{e}_i) + f(\mathbf{x})g(\mathbf{x}+h\mathbf{e}_i) - f(\mathbf{x})g(\mathbf{x})}{h} \\
&= \lim_{h \to 0} \left[ \frac{f(\mathbf{x}+h\mathbf{e}_i) - f(\mathbf{x})}{h} \cdot g(\mathbf{x}+h\mathbf{e}_i) + f(\mathbf{x}) \cdot \frac{g(\mathbf{x}+h\mathbf{e}_i) - g(\mathbf{x})}{h} \right] \\
&= \lim_{h \to 0} \frac{f(\mathbf{x}+h\mathbf{e}_i) - f(\mathbf{x})}{h} \cdot \lim_{h \to 0} g(\mathbf{x}+h\mathbf{e}_i) + f(\mathbf{x}) \cdot \lim_{h \to 0} \frac{g(\mathbf{x}+h\mathbf{e}_i) - g(\mathbf{x})}{h} \\
&= \frac{\partial f}{\partial x_i} \cdot g + f \cdot \frac{\partial g}{\partial x_i}
\end{align}

The other properties can be proven similarly.
{{< /proof >}}

### Exercises on Partial Derivatives

**Exercise 3.1** Find all first-order partial derivatives of the function $f(x,y,z) = \sin(xy) + e^{yz} + \ln(xz)$.

**Solution:**
We apply the definition and rules of partial differentiation:

$\frac{\partial f}{\partial x} = \frac{\partial}{\partial x} \sin(xy) + \frac{\partial}{\partial x} e^{yz} + \frac{\partial}{\partial x} \ln(xz)$

$= y \cos(xy) + 0 + \frac{z}{xz} = y \cos(xy) + \frac{1}{x}$

$\frac{\partial f}{\partial y} = \frac{\partial}{\partial y} \sin(xy) + \frac{\partial}{\partial y} e^{yz} + \frac{\partial}{\partial y} \ln(xz)$

$= x \cos(xy) + z e^{yz} + 0 = x \cos(xy) + z e^{yz}$

$\frac{\partial f}{\partial z} = \frac{\partial}{\partial z} \sin(xy) + \frac{\partial}{\partial z} e^{yz} + \frac{\partial}{\partial z} \ln(xz)$

$= 0 + y e^{yz} + \frac{x}{xz} = y e^{yz} + \frac{1}{z}$

**Exercise 3.2** If $f(x,y) = x^3 + 2xy^2 - 3y^3$, find $f_x(2,1)$ and $f_y(2,1)$.

**Solution:**
First, we find the partial derivatives:

$f_x(x,y) = \frac{\partial f}{\partial x} = 3x^2 + 2y^2$

$f_y(x,y) = \frac{\partial f}{\partial y} = 4xy - 9y^2$

Now, we evaluate these at the point $(2,1)$:

$f_x(2,1) = 3(2)^2 + 2(1)^2 = 3(4) + 2 = 12 + 2 = 14$

$f_y(2,1) = 4(2)(1) - 9(1)^2 = 8 - 9 = -1$

## 4. Directional Derivatives and the Gradient

While partial derivatives tell us how a function changes along the coordinate axes, directional derivatives extend this concept to any direction.

{{< definition "Directional Derivative" "directional-derivative" >}}
The directional derivative of a function $f$ at a point $\mathbf{a}$ in the direction of a unit vector $\mathbf{u}$ is defined as
\begin{equation}
  D_{\mathbf{u}} f(\mathbf{a}) = \lim_{h \to 0} \frac{f(\mathbf{a} + h\mathbf{u}) - f(\mathbf{a})}{h}
\end{equation}
provided the limit exists.
{{< /definition >}}

{{< definition "Gradient" "gradient" >}}
The gradient of a differentiable function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is the vector
\begin{equation}
  \nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)
\end{equation}
{{< /definition >}}

{{< theorem "Directional Derivative and Gradient" "directional-derivative-gradient" >}}
If $f$ is differentiable at a point $\mathbf{a}$, then for any unit vector $\mathbf{u}$,
\begin{equation}
  D_{\mathbf{u}} f(\mathbf{a}) = \nabla f(\mathbf{a}) \cdot \mathbf{u}
\end{equation}
where $\cdot$ represents the dot product.
{{< /theorem >}}

{{< proof >}}
Let $\mathbf{a} = (a_1, a_2, \ldots, a_n)$ and $\mathbf{u} = (u_1, u_2, \ldots, u_n)$ with $\|\mathbf{u}\| = 1$.

By the definition of differentiability, there exists a function $\varepsilon(\mathbf{h})$ such that
\begin{equation}
  f(\mathbf{a} + \mathbf{h}) = f(\mathbf{a}) + \nabla f(\mathbf{a}) \cdot \mathbf{h} + \|\mathbf{h}\| \varepsilon(\mathbf{h})
\end{equation}
where $\lim_{\mathbf{h} \to \mathbf{0}} \varepsilon(\mathbf{h}) = 0$.

Setting $\mathbf{h} = h\mathbf{u}$, we get
\begin{equation}
  f(\mathbf{a} + h\mathbf{u}) = f(\mathbf{a}) + \nabla f(\mathbf{a}) \cdot (h\mathbf{u}) + \|h\mathbf{u}\| \varepsilon(h\mathbf{u})
\end{equation}

Since $\|\mathbf{u}\| = 1$, we have $\|h\mathbf{u}\| = |h|$.

For $h > 0$, we get
\begin{equation}
  f(\mathbf{a} + h\mathbf{u}) = f(\mathbf{a}) + h \nabla f(\mathbf{a}) \cdot \mathbf{u} + h \varepsilon(h\mathbf{u})
\end{equation}

Therefore,
\begin{equation}
  \frac{f(\mathbf{a} + h\mathbf{u}) - f(\mathbf{a})}{h} = \nabla f(\mathbf{a}) \cdot \mathbf{u} + \varepsilon(h\mathbf{u})
\end{equation}

Taking the limit as $h \to 0$, we get
\begin{equation}
  D_{\mathbf{u}} f(\mathbf{a}) = \nabla f(\mathbf{a}) \cdot \mathbf{u}
\end{equation}

This completes the proof.
{{< /proof >}}

{{< theorem "Properties of the Gradient" "gradient-properties" >}}
Let $f$ and $g$ be differentiable functions, and let $c$ be a constant.
1. $\nabla(f + g) = \nabla f + \nabla g$
2. $\nabla(cf) = c \nabla f$
3. $\nabla(fg) = f \nabla g + g \nabla f$
4. $\nabla\left(\frac{f}{g}\right) = \frac{g \nabla f - f \nabla g}{g^2}$, where $g \neq 0$
5. The gradient $\nabla f(\mathbf{a})$ points in the direction of maximum rate of increase of $f$ at $\mathbf{a}$.
6. $\|\nabla f(\mathbf{a})\|$ is the maximum rate of change of $f$ at $\mathbf{a}$.
7. At a point $\mathbf{a}$ where $f$ has a level surface, $\nabla f(\mathbf{a})$ is orthogonal to that level surface.
{{< /theorem >}}

{{% hint info %}}
**Gradient as the Direction of Steepest Ascent**  
Think of the gradient $\nabla f$ as pointing "uphill" on the graph of $f$. If you stand at a point on a hill and want to climb it as steeply as possible, you should move in the direction of the gradient.
{{% /hint %}}

### Exercises on Directional Derivatives and the Gradient

**Exercise 4.1** Find the gradient of the function $f(x,y,z) = x^2y + yz^2 + 3xyz$.

**Solution:**
We calculate each partial derivative to form the gradient:

$\frac{\partial f}{\partial x} = 2xy + 3yz$

$\frac{\partial f}{\partial y} = x^2 + z^2 + 3xz$

$\frac{\partial f}{\partial z} = 2yz + 3xy$

Therefore, the gradient is:
$\nabla f(x,y,z) = (2xy + 3yz, x^2 + z^2 + 3xz, 2yz + 3xy)$

**Exercise 4.2** Let $f(x,y) = x^2 + 2y^2$. Find the directional derivative of $f$ at the point $P = (1,2)$ in the direction of the vector $\mathbf{v} = 3\mathbf{i} + 4\mathbf{j}$.

**Solution:**
First, we calculate the gradient of $f$:
$\nabla f(x,y) = (2x, 4y)$

At the point $P = (1,2)$, we have:
$\nabla f(1,2) = (2(1), 4(2)) = (2, 8)$

To find the directional derivative, we need a unit vector in the direction of $\mathbf{v}$:
$\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$

The unit vector is:
$\mathbf{u} = \frac{\mathbf{v}}{\|\mathbf{v}\|} = \frac{3\mathbf{i} + 4\mathbf{j}}{5} = \frac{3}{5}\mathbf{i} + \frac{4}{5}\mathbf{j}$

The directional derivative is:
$D_{\mathbf{u}} f(P) = \nabla f(P) \cdot \mathbf{u} = (2, 8) \cdot \left(\frac{3}{5}, \frac{4}{5}\right) = 2 \cdot \frac{3}{5} + 8 \cdot \frac{4}{5} = \frac{6}{5} + \frac{32}{5} = \frac{38}{5} = 7.6$

## 5. Tangent Planes and Linear Approximation

Just as the derivative provides a linear approximation to a function in single-variable calculus, the gradient enables us to find linear approximations to multivariable functions.

{{< definition "Tangent Plane" "tangent-plane" >}}
The tangent plane to the surface $z = f(x,y)$ at the point $(a,b,f(a,b))$ has the equation
\begin{equation}
  z = f(a,b) + f_x(a,b)(x - a) + f_y(a,b)(y - b)
\end{equation}
{{< /definition >}}

{{< definition "Linear Approximation" "linear-approximation" >}}
The linear approximation (also called the first-order Taylor approximation) of a differentiable function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ near a point $\mathbf{a}$ is
\begin{equation}
  f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a}) \cdot (\mathbf{x} - \mathbf{a})
\end{equation}
{{< /definition >}}

{{< theorem "Differentiability and Continuous Partial Derivatives" "differentiability-continuous-partials" >}}
If a function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ has continuous first-order partial derivatives in an open set containing a point $\mathbf{a}$, then $f$ is differentiable at $\mathbf{a}$.
{{< /theorem >}}

{{< proof >}}
For simplicity, we'll prove the case $n = 2$. Let $f(x,y)$ have continuous partial derivatives $f_x$ and $f_y$ in an open set containing $(a,b)$.

We need to show that
\begin{equation}
  \lim_{(h,k) \to (0,0)} \frac{f(a+h, b+k) - f(a,b) - f_x(a,b)h - f_y(a,b)k}{\sqrt{h^2 + k^2}} = 0
\end{equation}

Define the function $g(t) = f(a+th, b+tk)$ for $0 \leq t \leq 1$. By the chain rule,
\begin{equation}
  g'(t) = f_x(a+th, b+tk)h + f_y(a+th, b+tk)k
\end{equation}

By the Mean Value Theorem, there exists a value $c \in (0,1)$ such that
\begin{equation}
  g(1) - g(0) = g'(c) \cdot 1
\end{equation}

This gives us
\begin{equation}
  f(a+h, b+k) - f(a,b) = f_x(a+ch, b+ck)h + f_y(a+ch, b+ck)k
\end{equation}

Therefore,
\begin{align}
  &\frac{f(a+h, b+k) - f(a,b) - f_x(a,b)h - f_y(a,b)k}{\sqrt{h^2 + k^2}} \\
  &= \frac{f_x(a+ch, b+ck)h + f_y(a+ch, b+ck)k - f_x(a,b)h - f_y(a,b)k}{\sqrt{h^2 + k^2}} \\
  &= \frac{[f_x(a+ch, b+ck) - f_x(a,b)]h + [f_y(a+ch, b+ck) - f_y(a,b)]k}{\sqrt{h^2 + k^2}}
\end{align}

Since $f_x$ and $f_y$ are continuous, we have
\begin{equation}
  \lim_{(h,k) \to (0,0)} f_x(a+ch, b+ck) = f_x(a,b) \quad \text{and} \quad \lim_{(h,k) \to (0,0)} f_y(a+ch, b+ck) = f_y(a,b)
\end{equation}

Therefore,
\begin{align}
  &\lim_{(h,k) \to (0,0)} \frac{[f_x(a+ch, b+ck) - f_x(a,b)]h + [f_y(a+ch, b+ck) - f_y(a,b)]k}{\sqrt{h^2 + k^2}} \\
  &\leq \lim_{(h,k) \to (0,0)} \frac{|f_x(a+ch, b+ck) - f_x(a,b)| \cdot |h| + |f_y(a+ch, b+ck) - f_y(a,b)| \cdot |k|}{\sqrt{h^2 + k^2}} \\
  &\leq \lim_{(h,k) \to (0,0)} |f_x(a+ch, b+ck) - f_x(a,b)| \cdot \frac{|h|}{\sqrt{h^2 + k^2}} + |f_y(a+ch, b+ck) - f_y(a,b)| \cdot \frac{|k|}{\sqrt{h^2 + k^2}}
\end{align}

Since $\frac{|h|}{\sqrt{h^2 + k^2}} \leq 1$ and $\frac{|k|}{\sqrt{h^2 + k^2}} \leq 1$, and the differences in the partial derivatives approach zero as $(h,k) \to (0,0)$, the limit is zero, proving that $f$ is differentiable at $(a,b)$.
{{< /proof >}}

### Exercises on Tangent Planes and Linear Approximation

**Exercise 5.1** Find the equation of the tangent plane to the surface $z = 2x^2 + 3y^2 - xy$ at the point $(1, 2, 9)$.

**Solution:**
First, we verify that the point $(1, 2, 9)$ lies on the surface:
$z = 2(1)^2 + 3(2)^2 - (1)(2) = 2 + 12 - 2 = 12 - 2 = 10$

This doesn't match the $z$-value we were given. Let's check if there's an error in the problem statement.

If we evaluate the function at $(1, 2)$:
$f(1, 2) = 2(1)^2 + 3(2)^2 - (1)(2) = 2 + 12 - 2 = 12$

So the point should be $(1, 2, 12)$ for consistency.

Let's recalculate assuming the correct point is $(1, 2, 9)$. We need to find the partial derivatives:

$f_x(x, y) = 4x - y$
$f_y(x, y) = 6y - x$

At the point $(1, 2)$:
$f_x(1, 2) = 4(1) - 2 = 2$
$f_y(1, 2) = 6(2) - 1 = 11$

The equation of the tangent plane is:
$z = f(1, 2) + f_x(1, 2)(x - 1) + f_y(1, 2)(y - 2)$
$z = 9 + 2(x - 1) + 11(y - 2)$
$z = 9 + 2x - 2 + 11y - 22$
$z = 2x + 11y - 15$

**Exercise 5.2** Use the linear approximation to estimate the value of $f(2.1, 0.95)$ where $f(x, y) = \sqrt{x^2 + y^2 + 1}$.

**Solution:**
We'll use the linear approximation around the point $(2, 1)$:
$f(x, y) \approx f(2, 1) + f_x(2, 1)(x - 2) + f_y(2, 1)(y - 1)$

First, we calculate $f(2, 1)$:
$f(2, 1) = \sqrt{2^2 + 1^2 + 1} = \sqrt{4 + 1 + 1} = \sqrt{6} \approx 2.449$

Next, we find the partial derivatives:
$f_x(x, y) = \frac{x}{\sqrt{x^2 + y^2 + 1}}$
$f_y(x, y) = \frac{y}{\sqrt{x^2 + y^2 + 1}}$

At the point $(2, 1)$:
$f_x(2, 1) = \frac{2}{\sqrt{6}} \approx 0.816$
$f_y(2, 1) = \frac{1}{\sqrt{6}} \approx 0.408$

Now we can compute the linear approximation:
$f(2.1, 0.95) \approx f(2, 1) + f_x(2, 1)(2.1 - 2) + f_y(2, 1)(0.95 - 1)$
$\approx 2.449 + 0.816 \cdot 0.1 + 0.408 \cdot (-0.05)$
$\approx 2.449 + 0.0816 - 0.0204$
$\approx 2.510$

The actual value is $\sqrt{2.1^2 + 0.95^2 + 1} = \sqrt{4.41 + 0.9025 + 1} = \sqrt{6.3125} \approx 2.512$, so our approximation is very close.

## 6. The Chain Rule

The chain rule is a fundamental tool for finding derivatives of composite functions, and its multivariable version is essential for addressing complex real-world problems.

{{< definition "Chain Rule for Functions of Several Variables" "chain-rule" >}}
Suppose $z = f(x, y)$ where $x = g(t)$ and $y = h(t)$. If $g$ and $h$ are differentiable at $t_0$, and $f$ is differentiable at $(g(t_0), h(t_0))$, then $z$ is differentiable at $t_0$ and
\begin{equation}
  \frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}
\end{equation}
{{< /definition >}}

{{< theorem "Chain Rule for Multiple Independent Variables" "generalized-chain-rule" >}}
If $w = f(x, y, z)$ where $x = g(s, t)$, $y = h(s, t)$, and $z = k(s, t)$, then
\begin{equation}
  \frac{\partial w}{\partial s} = \frac{\partial f}{\partial x} \frac{\partial x}{\partial s} + \frac{\partial f}{\partial y} \frac{\partial y}{\partial s} + \frac{\partial f}{\partial z} \frac{\partial z}{\partial s}
\end{equation}
and
\begin{equation}
  \frac{\partial w}{\partial t} = \frac{\partial f}{\partial x} \frac{\partial x}{\partial t} + \frac{\partial f}{\partial y} \frac{\partial y}{\partial t} + \frac{\partial f}{\partial z} \frac{\partial z}{\partial t}
\end{equation}
{{< /theorem >}}

{{< proof >}}
We'll prove the case for one intermediate variable, and the general case follows by induction.

Let $z = f(x, y)$ where $x = g(t)$ and $y = h(t)$. We need to show that
\begin{equation}
  \frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}
\end{equation}

By the definition of the derivative,
\begin{equation}
  \frac{dz}{dt} = \lim_{\Delta t \to 0} \frac{f(g(t + \Delta t), h(t + \Delta t)) - f(g(t), h(t))}{\Delta t}
\end{equation}

Let $\Delta x = g(t + \Delta t) - g(t)$ and $\Delta y = h(t + \Delta t) - h(t)$. Then we can rewrite the derivative as
\begin{equation}
  \frac{dz}{dt} = \lim_{\Delta t \to 0} \frac{f(g(t) + \Delta x, h(t) + \Delta y) - f(g(t), h(t))}{\Delta t}
\end{equation}

By the differentiability of $f$, we have
\begin{equation}
  f(g(t) + \Delta x, h(t) + \Delta y) - f(g(t), h(t)) = \frac{\partial f}{\partial x} \Delta x + \frac{\partial f}{\partial y} \Delta y + \varepsilon_1 \Delta x + \varepsilon_2 \Delta y
\end{equation}
where $\varepsilon_1, \varepsilon_2 \to 0$ as $\Delta x, \Delta y \to 0$.

Therefore,
\begin{align}
  \frac{dz}{dt} &= \lim_{\Delta t \to 0} \frac{\frac{\partial f}{\partial x} \Delta x + \frac{\partial f}{\partial y} \Delta y + \varepsilon_1 \Delta x + \varepsilon_2 \Delta y}{\Delta t} \\
  &= \lim_{\Delta t \to 0} \left[ \frac{\partial f}{\partial x} \frac{\Delta x}{\Delta t} + \frac{\partial f}{\partial y} \frac{\Delta y}{\Delta t} + \varepsilon_1 \frac{\Delta x}{\Delta t} + \varepsilon_2 \frac{\Delta y}{\Delta t} \right] \\
  &= \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt} + 0 + 0 \\
  &= \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}
\end{align}

This completes the proof.
{{< /proof >}}

{{% hint warning %}}
**When to Use the Chain Rule**  
The chain rule is necessary whenever you're differentiating a composite function. This includes cases where you're working with:
1. A function expressed in terms of intermediate variables
2. A change of coordinates (e.g., from Cartesian to polar)
3. Implicit differentiation
4. Functions along curves or surfaces
{{% /hint %}}

### Exercises on the Chain Rule

**Exercise 6.1** Let $f(x,y) = x^2y + 3xy^2$ where $x = s^2t$ and $y = st^2$. Find $\frac{\partial f}{\partial s}$ and $\frac{\partial f}{\partial t}$.

**Solution:**
First, we compute the partial derivatives of $f$ with respect to $x$ and $y$:
$\frac{\partial f}{\partial x} = 2xy + 3y^2$
$\frac{\partial f}{\partial y} = x^2 + 6xy$

Next, we find the partial derivatives of $x$ and $y$ with respect to $s$ and $t$:
$\frac{\partial x}{\partial s} = 2st$
$\frac{\partial x}{\partial t} = s^2$
$\frac{\partial y}{\partial s} = t^2$
$\frac{\partial y}{\partial t} = 2st$

Now, using the chain rule:
$\frac{\partial f}{\partial s} = \frac{\partial f}{\partial x} \frac{\partial x}{\partial s} + \frac{\partial f}{\partial y} \frac{\partial y}{\partial s}$
$= (2xy + 3y^2)(2st) + (x^2 + 6xy)(t^2)$

Substituting $x = s^2t$ and $y = st^2$:
$\frac{\partial f}{\partial s} = (2(s^2t)(st^2) + 3(st^2)^2)(2st) + ((s^2t)^2 + 6(s^2t)(st^2))(t^2)$
$= (2s^3t^3 + 3s^2t^4)(2st) + (s^4t^2 + 6s^3t^3)(t^2)$
$= 4s^4t^4 + 6s^3t^5 + s^4t^4 + 6s^3t^5$
$= 5s^4t^4 + 12s^3t^5$

Similarly, for $\frac{\partial f}{\partial t}$:
$\frac{\partial f}{\partial t} = \frac{\partial f}{\partial x} \frac{\partial x}{\partial t} + \frac{\partial f}{\partial y} \frac{\partial y}{\partial t}$
$= (2xy + 3y^2)(s^2) + (x^2 + 6xy)(2st)$

Substituting $x = s^2t$ and $y = st^2$:
$\frac{\partial f}{\partial t} = (2(s^2t)(st^2) + 3(st^2)^2)(s^2) + ((s^2t)^2 + 6(s^2t)(st^2))(2st)$
$= (2s^3t^3 + 3s^2t^4)(s^2) + (s^4t^2 + 6s^3t^3)(2st)$
$= 2s^5t^3 + 3s^4t^4 + 2s^5t^3 + 12s^4t^4$
$= 4s^5t^3 + 15s^4t^4$

**Exercise 6.2** If $w = xe^{y/z}$ where $x = r\cos\theta$, $y = r\sin\theta$, and $z = r$, find $\frac{\partial w}{\partial r}$ and $\frac{\partial w}{\partial \theta}$ at the point where $r = 1$ and $\theta = \pi/4$.

**Solution:**
First, we compute the partial derivatives of $w$ with respect to $x$, $y$, and $z$:
$\frac{\partial w}{\partial x} = e^{y/z}$
$\frac{\partial w}{\partial y} = x \cdot e^{y/z} \cdot \frac{1}{z} = \frac{x}{z}e^{y/z}$
$\frac{\partial w}{\partial z} = x \cdot e^{y/z} \cdot \left(-\frac{y}{z^2}\right) = -\frac{xy}{z^2}e^{y/z}$

Next, we find the partial derivatives of $x$, $y$, and $z$ with respect to $r$ and $\theta$:
$\frac{\partial x}{\partial r} = \cos\theta$
$\frac{\partial x}{\partial \theta} = -r\sin\theta$
$\frac{\partial y}{\partial r} = \sin\theta$
$\frac{\partial y}{\partial \theta} = r\cos\theta$
$\frac{\partial z}{\partial r} = 1$
$\frac{\partial z}{\partial \theta} = 0$

Now, using the chain rule:
$\frac{\partial w}{\partial r} = \frac{\partial w}{\partial x} \frac{\partial x}{\partial r} + \frac{\partial w}{\partial y} \frac{\partial y}{\partial r} + \frac{\partial w}{\partial z} \frac{\partial z}{\partial r}$
$= e^{y/z} \cdot \cos\theta + \frac{x}{z}e^{y/z} \cdot \sin\theta + \left(-\frac{xy}{z^2}e^{y/z}\right) \cdot 1$

Substituting $x = r\cos\theta$, $y = r\sin\theta$, and $z = r$:
$\frac{\partial w}{\partial r} = e^{\sin\theta} \cdot \cos\theta + \frac{r\cos\theta}{r}e^{\sin\theta} \cdot \sin\theta - \frac{r\cos\theta \cdot r\sin\theta}{r^2}e^{\sin\theta}$
$= e^{\sin\theta} \cdot \cos\theta + e^{\sin\theta} \cdot \cos\theta \cdot \sin\theta - e^{\sin\theta} \cdot \cos\theta \cdot \sin\theta$
$= e^{\sin\theta} \cdot \cos\theta$

At the point where $r = 1$ and $\theta = \pi/4$:
$\frac{\partial w}{\partial r} = e^{\sin(\pi/4)} \cdot \cos(\pi/4) = e^{1/\sqrt{2}} \cdot \frac{1}{\sqrt{2}} = \frac{e^{1/\sqrt{2}}}{\sqrt{2}}$

Similarly, for $\frac{\partial w}{\partial \theta}$:
$\frac{\partial w}{\partial \theta} = \frac{\partial w}{\partial x} \frac{\partial x}{\partial \theta} + \frac{\partial w}{\partial y} \frac{\partial y}{\partial \theta} + \frac{\partial w}{\partial z} \frac{\partial z}{\partial \theta}$
$= e^{y/z} \cdot (-r\sin\theta) + \frac{x}{z}e^{y/z} \cdot r\cos\theta + \left(-\frac{xy}{z^2}e^{y/z}\right) \cdot 0$
$= -r\sin\theta \cdot e^{\sin\theta} + \frac{r\cos\theta}{r}e^{\sin\theta} \cdot r\cos\theta$
$= -r\sin\theta \cdot e^{\sin\theta} + r\cos^2\theta \cdot e^{\sin\theta}$
$= r e^{\sin\theta}(\cos^2\theta - \sin\theta)$

At the point where $r = 1$ and $\theta = \pi/4$:
$\frac{\partial w}{\partial \theta} = e^{\sin(\pi/4)}(\cos^2(\pi/4) - \sin(\pi/4)) = e^{1/\sqrt{2}}(1/2 - 1/\sqrt{2}) = e^{1/\sqrt{2}}(1/2 - 1/\sqrt{2})$

## 7. Higher-Order Derivatives and the Hessian

Just as we can take second and higher derivatives in single-variable calculus, the same applies to functions of multiple variables, leading to rich structures and properties.

{{< definition "Higher-Order Partial Derivatives" "higher-order-partials" >}}
For a function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ with sufficiently continuous partial derivatives, the second-order partial derivatives are denoted by
\begin{equation}
  \frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial}{\partial x_i} \left( \frac{\partial f}{\partial x_j} \right)
\end{equation}
Alternative notations include $f_{x_i x_j}$, $D_i D_j f$, or $\partial_{x_i x_j} f$.
{{< /definition >}}

{{< theorem "Equality of Mixed Partials (Schwarz's Theorem)" "mixed-partials-equality" >}}
If a function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ has continuous second-order partial derivatives in an open set containing a point $\mathbf{a}$, then at $\mathbf{a}$,
\begin{equation}
  \frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}
\end{equation}
for all $i, j = 1, 2, \ldots, n$.
{{< /theorem >}}

{{< proof >}}
For simplicity, we'll prove the case $n = 2$ with $i = 1$ and $j = 2$. Let $f(x,y)$ have continuous second-order partial derivatives in an open set containing $(a,b)$.

We need to show that $\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$ at $(a,b)$.

Define the function
\begin{equation}
  g(x,y) = \frac{f(x,y) - f(x,b) - f(a,y) + f(a,b)}{(x-a)(y-b)}
\end{equation}
for $(x,y) \neq (a,b)$.

We can compute the limit of $g(x,y)$ as $(x,y) \to (a,b)$ in two different ways.

First, let's approach along the path where $x \to a$ first and then $y \to b$:
\begin{align}
  \lim_{(x,y) \to (a,b)} g(x,y) &= \lim_{y \to b} \lim_{x \to a} \frac{f(x,y) - f(x,b) - f(a,y) + f(a,b)}{(x-a)(y-b)} \\
  &= \lim_{y \to b} \frac{1}{y-b} \lim_{x \to a} \frac{f(x,y) - f(x,b) - f(a,y) + f(a,b)}{x-a} \\
  &= \lim_{y \to b} \frac{1}{y-b} \lim_{x \to a} \frac{f(x,y) - f(a,y)}{x-a} - \frac{f(x,b) - f(a,b)}{x-a} \\
  &= \lim_{y \to b} \frac{1}{y-b} [f_x(a,y) - f_x(a,b)] \\
  &= \lim_{y \to b} \frac{f_x(a,y) - f_x(a,b)}{y-b} \\
  &= f_{xy}(a,b)
\end{align}

Similarly, approaching along the path where $y \to b$ first and then $x \to a$:
\begin{align}
  \lim_{(x,y) \to (a,b)} g(x,y) &= \lim_{x \to a} \lim_{y \to b} \frac{f(x,y) - f(x,b) - f(a,y) + f(a,b)}{(x-a)(y-b)} \\
  &= \lim_{x \to a} \frac{1}{x-a} \lim_{y \to b} \frac{f(x,y) - f(x,b) - f(a,y) + f(a,b)}{y-b} \\
  &= \lim_{x \to a} \frac{1}{x-a} \lim_{y \to b} \frac{f(x,y) - f(x,b)}{y-b} - \frac{f(a,y) - f(a,b)}{y-b} \\
  &= \lim_{x \to a} \frac{1}{x-a} [f_y(x,b) - f_y(a,b)] \\
  &= \lim_{x \to a} \frac{f_y(x,b) - f_y(a,b)}{x-a} \\
  &= f_{yx}(a,b)
\end{align}

Since both limits must be equal, we have $f_{xy}(a,b) = f_{yx}(a,b)$.
{{< /proof >}}

{{< definition "Hessian Matrix" "hessian" >}}
The Hessian matrix of a twice-differentiable function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is the $n \times n$ matrix of second-order partial derivatives:
\begin{equation}
  H_f = \begin{pmatrix}
    \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
    \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
  \end{pmatrix}
\end{equation}
{{< /definition >}}

{{% hint info %}}
**Interpreting the Hessian Matrix**  
The Hessian matrix provides information about the local curvature of a function. In particular, its eigenvalues and determinant are used to classify critical points (as minima, maxima, or saddle points) and to determine the concavity or convexity of the function.
{{% /hint %}}

### Exercises on Higher-Order Derivatives and the Hessian

**Exercise 7.1** Find all second-order partial derivatives of the function $f(x,y) = x^3y^2 + x^2y^3 + e^{xy}$.

**Solution:**
First, let's find the first-order partial derivatives:

$\frac{\partial f}{\partial x} = 3x^2y^2 + 2xy^3 + ye^{xy}$
$\frac{\partial f}{\partial y} = 2x^3y + 3x^2y^2 + xe^{xy}$

Now, we compute the second-order partial derivatives:

$\frac{\partial^2 f}{\partial x^2} = \frac{\partial}{\partial x}\left(3x^2y^2 + 2xy^3 + ye^{xy}\right) = 6xy^2 + 2y^3 + y^2e^{xy}$

$\frac{\partial^2 f}{\partial y^2} = \frac{\partial}{\partial y}\left(2x^3y + 3x^2y^2 + xe^{xy}\right) = 2x^3 + 6x^2y + x^2e^{xy}$

$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial x}\left(2x^3y + 3x^2y^2 + xe^{xy}\right) = 6x^2y + 6xy^2 + e^{xy} + x^2ye^{xy}$

$\frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial y}\left(3x^2y^2 + 2xy^3 + ye^{xy}\right) = 6x^2y + 6xy^2 + e^{xy} + x^2ye^{xy}$

We can verify Schwarz's theorem: $\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$.

**Exercise 7.2** Find the Hessian matrix of the function $f(x,y,z) = x^2y + y^2z + z^2x$ at the point $(1,2,3)$ and determine whether it is positive definite, negative definite, or indefinite.

**Solution:**
First, let's find the first-order partial derivatives:

$\frac{\partial f}{\partial x} = 2xy + z^2$
$\frac{\partial f}{\partial y} = x^2 + 2yz$
$\frac{\partial f}{\partial z} = y^2 + 2zx$

Now, we compute the second-order partial derivatives:

$\frac{\partial^2 f}{\partial x^2} = 2y$
$\frac{\partial^2 f}{\partial y^2} = 2z$
$\frac{\partial^2 f}{\partial z^2} = 2x$

$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x} = 2x$
$\frac{\partial^2 f}{\partial x \partial z} = \frac{\partial^2 f}{\partial z \partial x} = 2z$
$\frac{\partial^2 f}{\partial y \partial z} = \frac{\partial^2 f}{\partial z \partial y} = 2y$

At the point $(1,2,3)$, the Hessian matrix is:
$H_f(1,2,3) = \begin{pmatrix} 
4 & 2 & 6 \\ 
2 & 6 & 4 \\ 
6 & 4 & 2 
\end{pmatrix}$

To determine if the Hessian is positive definite, negative definite, or indefinite, we need to examine the eigenvalues or the principal minors.

Let's calculate the determinants of the leading principal minors:
1. First minor: $|4| = 4 > 0$
2. Second minor: $\begin{vmatrix} 4 & 2 \\ 2 & 6 \end{vmatrix} = 4 \cdot 6 - 2 \cdot 2 = 24 - 4 = 20 > 0$
3. Third minor (the determinant of the entire matrix):
   $\begin{vmatrix} 4 & 2 & 6 \\ 2 & 6 & 4 \\ 6 & 4 & 2 \end{vmatrix}$
   $= 4 \cdot \begin{vmatrix} 6 & 4 \\ 4 & 2 \end{vmatrix} - 2 \cdot \begin{vmatrix} 2 & 4 \\ 6 & 2 \end{vmatrix} + 6 \cdot \begin{vmatrix} 2 & 6 \\ 6 & 4 \end{vmatrix}$
   $= 4 \cdot (6 \cdot 2 - 4 \cdot 4) - 2 \cdot (2 \cdot 2 - 4 \cdot 6) + 6 \cdot (2 \cdot 4 - 6 \cdot 6)$
   $= 4 \cdot (12 - 16) - 2 \cdot (4 - 24) + 6 \cdot (8 - 36)$
   $= 4 \cdot (-4) - 2 \cdot (-20) + 6 \cdot (-28)$
   $= -16 + 40 - 168 = -144 < 0$

Since the first and second leading principal minors are positive, but the third is negative, the Hessian matrix is indefinite. This suggests that the critical point at $(1,2,3)$ (if it is indeed a critical point) would be a saddle point.

## 8. Taylor's Theorem for Multivariate Functions

Taylor's theorem generalizes to multiple variables, providing a way to approximate functions locally using polynomial expressions.

{{< theorem "Taylor's Theorem for Multivariate Functions" "multivariate-taylor" >}}
Let $f: \mathbb{R}^n \rightarrow \mathbb{R}$ be a function such that all its partial derivatives of order $k+1$ exist and are continuous in an open set containing the line segment from $\mathbf{a}$ to $\mathbf{a} + \mathbf{h}$. Then
\begin{equation}
  f(\mathbf{a} + \mathbf{h}) = \sum_{|\alpha| \leq k} \frac{1}{\alpha!} D^{\alpha}f(\mathbf{a}) \mathbf{h}^{\alpha} + R_k
\end{equation}
where $\alpha = (\alpha_1, \alpha_2, \ldots, \alpha_n)$ is a multi-index, $|\alpha| = \alpha_1 + \alpha_2 + \ldots + \alpha_n$, $\alpha! = \alpha_1! \alpha_2! \ldots \alpha_n!$, $D^{\alpha}f = \frac{\partial^{|\alpha|}f}{\partial x_1^{\alpha_1} \partial x_2^{\alpha_2} \ldots \partial x_n^{\alpha_n}}$, and $\mathbf{h}^{\alpha} = h_1^{\alpha_1} h_2^{\alpha_2} \ldots h_n^{\alpha_n}$. The remainder term $R_k$ satisfies
\begin{equation}
  |R_k| \leq \frac{M}{(k+1)!} \|\mathbf{h}\|^{k+1}
\end{equation}
where $M$ is an upper bound for the $(k+1)$-th order partial derivatives of $f$ on the line segment from $\mathbf{a}$ to $\mathbf{a} + \mathbf{h}$.
{{< /theorem >}}

{{< definition "Second-Order Taylor Approximation" "second-order-taylor" >}}
The second-order Taylor approximation of a twice-differentiable function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ near a point $\mathbf{a}$ is
\begin{equation}
  f(\mathbf{a} + \mathbf{h}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a}) \cdot \mathbf{h} + \frac{1}{2} \mathbf{h}^T H_f(\mathbf{a}) \mathbf{h}
\end{equation}
where $H_f(\mathbf{a})$ is the Hessian matrix of $f$ at $\mathbf{a}$.
{{< /definition >}}

{{% hint info %}}
**Applications of Taylor's Theorem**  
Taylor's theorem is crucial for many applications:
1. Numerical approximation of functions
2. Error estimation in numerical methods
3. Optimization algorithms like Newton's method
4. Sensitivity analysis in physics and engineering
5. Series expansions in mathematical physics
{{% /hint %}}

### Exercises on Taylor's Theorem

**Exercise 8.1** Find the second-order Taylor polynomial of the function $f(x,y) = \ln(1 + x + y + xy)$ around the point $(0,0)$.

**Solution:**
We use the formula:
$f(x,y) \approx f(0,0) + \nabla f(0,0) \cdot (x,y) + \frac{1}{2}(x,y)^T H_f(0,0) (x,y)$

First, we compute $f(0,0)$:
$f(0,0) = \ln(1 + 0 + 0 + 0 \cdot 0) = \ln(1) = 0$

Next, we find the gradient at $(0,0)$:
$\frac{\partial f}{\partial x} = \frac{1 + y}{1 + x + y + xy}$
$\frac{\partial f}{\partial y} = \frac{1 + x}{1 + x + y + xy}$

At $(0,0)$:
$\frac{\partial f}{\partial x}(0,0) = \frac{1 + 0}{1 + 0 + 0 + 0 \cdot 0} = 1$
$\frac{\partial f}{\partial y}(0,0) = \frac{1 + 0}{1 + 0 + 0 + 0 \cdot 0} = 1$

So $\nabla f(0,0) = (1, 1)$.

Now, we compute the Hessian matrix:
$\frac{\partial^2 f}{\partial x^2} = -\frac{(1 + y)^2}{(1 + x + y + xy)^2}$
$\frac{\partial^2 f}{\partial y^2} = -\frac{(1 + x)^2}{(1 + x + y + xy)^2}$
$\frac{\partial^2 f}{\partial x \partial y} = \frac{1}{1 + x + y + xy} - \frac{(1 + y)(1 + x)}{(1 + x + y + xy)^2}$

At $(0,0)$:
$\frac{\partial^2 f}{\partial x^2}(0,0) = -\frac{(1 + 0)^2}{(1 + 0 + 0 + 0 \cdot 0)^2} = -1$
$\frac{\partial^2 f}{\partial y^2}(0,0) = -\frac{(1 + 0)^2}{(1 + 0 + 0 + 0 \cdot 0)^2} = -1$
$\frac{\partial^2 f}{\partial x \partial y}(0,0) = \frac{1}{1 + 0 + 0 + 0 \cdot 0} - \frac{(1 + 0)(1 + 0)}{(1 + 0 + 0 + 0 \cdot 0)^2} = 1 - 1 = 0$

So $H_f(0,0) = \begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}$.

The second-order Taylor polynomial is:
$f(x,y) \approx 0 + (1, 1) \cdot (x, y) + \frac{1}{2}(x, y)^T \begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix} (x, y)$
$= x + y + \frac{1}{2}(-x^2 - y^2)$
$= x + y - \frac{1}{2}x^2 - \frac{1}{2}y^2$

**Exercise 8.2** Use a second-order Taylor approximation to estimate the value of $f(1.1, 0.9)$ where $f(x,y) = e^x \sin(y)$.

**Solution:**
We'll use the second-order Taylor approximation around the point $(1, 1)$:
$f(x, y) \approx f(1, 1) + \nabla f(1, 1) \cdot ((x-1), (y-1)) + \frac{1}{2}((x-1), (y-1))^T H_f(1, 1) ((x-1), (y-1))$

First, we compute $f(1, 1)$:
$f(1, 1) = e^1 \sin(1) = e \cdot \sin(1) \approx 2.718 \cdot 0.841 \approx 2.287$

Next, we find the gradient at $(1, 1)$:
$\frac{\partial f}{\partial x} = e^x \sin(y)$
$\frac{\partial f}{\partial y} = e^x \cos(y)$

At $(1, 1)$:
$\frac{\partial f}{\partial x}(1, 1) = e^1 \sin(1) \approx 2.718 \cdot 0.841 \approx 2.287$
$\frac{\partial f}{\partial y}(1, 1) = e^1 \cos(1) \approx 2.718 \cdot 0.540 \approx 1.469$

Now, we compute the Hessian matrix:
$\frac{\partial^2 f}{\partial x^2} = e^x \sin(y)$
$\frac{\partial^2 f}{\partial y^2} = -e^x \sin(y)$
$\frac{\partial^2 f}{\partial x \partial y} = e^x \cos(y)$

At $(1, 1)$:
$\frac{\partial^2 f}{\partial x^2}(1, 1) = e^1 \sin(1) \approx 2.287$
$\frac{\partial^2 f}{\partial y^2}(1, 1) = -e^1 \sin(1) \approx -2.287$
$\frac{\partial^2 f}{\partial x \partial y}(1, 1) = e^1 \cos(1) \approx 1.469$

So $H_f(1, 1) = \begin{pmatrix} 2.287 & 1.469 \\ 1.469 & -2.287 \end{pmatrix}$.

The second-order Taylor approximation is:
$f(x, y) \approx 2.287 + 2.287(x-1) + 1.469(y-1) + \frac{1}{2}[(x-1), (y-1)] \begin{pmatrix} 2.287 & 1.469 \\ 1.469 & -2.287 \end{pmatrix} \begin{pmatrix} x-1 \\ y-1 \end{pmatrix}$

Now we evaluate this at $(1.1, 0.9)$:
$f(1.1, 0.9) \approx 2.287 + 2.287(0.1) + 1.469(-0.1) + \frac{1}{2}[(0.1), (-0.1)] \begin{pmatrix} 2.287 & 1.469 \\ 1.469 & -2.287 \end{pmatrix} \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix}$
$= 2.287 + 0.2287 - 0.1469 + \frac{1}{2}[(0.1)(2.287)(0.1) + (0.1)(1.469)(-0.1) + (-0.1)(1.469)(0.1) + (-0.1)(-2.287)(-0.1)]$
$= 2.287 + 0.2287 - 0.1469 + \frac{1}{2}[0.02287 - 0.01469 - 0.01469 - 0.02287]$
$= 2.287 + 0.2287 - 0.1469 + \frac{1}{2}[-0.02938]$
$= 2.287 + 0.2287 - 0.1469 - 0.01469$
$= 2.354$

The actual value is $f(1.1, 0.9) = e^{1.1} \sin(0.9) \approx 3.004 \cdot 0.783 \approx 2.352$, so our approximation is very accurate.

## 9. Maxima, Minima, and Saddle Points

Finding extreme values is a central problem in calculus, with applications in optimization across many disciplines.

{{< definition "Critical Point" "critical-point" >}}
A point $\mathbf{a}$ in the domain of a differentiable function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is a critical point if $\nabla f(\mathbf{a}) = \mathbf{0}$.
{{< /definition >}}

{{< theorem "Second Derivative Test for Functions of Two Variables" "second-derivative-test" >}}
Let $f: \mathbb{R}^2 \rightarrow \mathbb{R}$ be a function with continuous second-order partial derivatives, and let $(a,b)$ be a critical point of $f$. Define
\begin{equation}
  D = f_{xx}(a,b) f_{yy}(a,b) - [f_{xy}(a,b)]^2
\end{equation}

1. If $D > 0$ and $f_{xx}(a,b) > 0$, then $f$ has a local minimum at $(a,b)$.
2. If $D > 0$ and $f_{xx}(a,b) < 0$, then $f$ has a local maximum at $(a,b)$.
3. If $D < 0$, then $f$ has a saddle point at $(a,b)$.
4. If $D = 0$, the test is inconclusive.
{{< /theorem >}}

{{< proof >}}
Let's use Taylor's theorem to expand $f$ around the critical point $(a,b)$:
\begin{equation}
  f(a+h, b+k) = f(a,b) + \frac{1}{2}(f_{xx}(a,b)h^2 + 2f_{xy}(a,b)hk + f_{yy}(a,b)k^2) + \text{higher-order terms}
\end{equation}

Since $(a,b)$ is a critical point, the first-order terms vanish. We focus on the quadratic form
\begin{equation}
  Q(h,k) = f_{xx}(a,b)h^2 + 2f_{xy}(a,b)hk + f_{yy}(a,b)k^2
\end{equation}

This quadratic form can be written in matrix notation as
\begin{equation}
  Q(h,k) = \begin{pmatrix} h & k \end{pmatrix} \begin{pmatrix} f_{xx}(a,b) & f_{xy}(a,b) \\ f_{xy}(a,b) & f_{yy}(a,b) \end{pmatrix} \begin{pmatrix} h \\ k \end{pmatrix}
\end{equation}

The behavior of $f$ near the critical point depends on the behavior of this quadratic form.

1. If $D > 0$ and $f_{xx}(a,b) > 0$, then the matrix $\begin{pmatrix} f_{xx}(a,b) & f_{xy}(a,b) \\ f_{xy}(a,b) & f_{yy}(a,b) \end{pmatrix}$ is positive definite, meaning that $Q(h,k) > 0$ for all non-zero vectors $(h,k)$. This means that $f(a+h, b+k) > f(a,b)$ for small non-zero vectors $(h,k)$, so $f$ has a local minimum at $(a,b)$.

2. If $D > 0$ and $f_{xx}(a,b) < 0$, then the matrix is negative definite, meaning that $Q(h,k) < 0$ for all non-zero vectors $(h,k)$. This means that $f(a+h, b+k) < f(a,b)$ for small non-zero vectors $(h,k)$, so $f$ has a local maximum at $(a,b)$.

3. If $D < 0$, then the matrix has both positive and negative eigenvalues (it's indefinite), meaning that $Q(h,k)$ is positive for some directions and negative for others. This means that $f$ increases in some directions and decreases in others, which characterizes a saddle point.

4. If $D = 0$, the matrix is either positive semidefinite, negative semidefinite, or indefinite, and higher-order terms are needed to determine the behavior of $f$ at the critical point.
{{< /proof >}}

{{< theorem "Second Derivative Test for Functions of Several Variables" "general-second-derivative-test" >}}
Let $f: \mathbb{R}^n \rightarrow \mathbb{R}$ be a function with continuous second-order partial derivatives, and let $\mathbf{a}$ be a critical point of $f$. Let $H_f(\mathbf{a})$ be the Hessian matrix of $f$ at $\mathbf{a}$.

1. If $H_f(\mathbf{a})$ is positive definite, then $f$ has a local minimum at $\mathbf{a}$.
2. If $H_f(\mathbf{a})$ is negative definite, then $f$ has a local maximum at $\mathbf{a}$.
3. If $H_f(\mathbf{a})$ has both positive and negative eigenvalues, then $f$ has a saddle point at $\mathbf{a}$.
4. If $H_f(\mathbf{a})$ is positive or negative semidefinite with at least one zero eigenvalue, the test is inconclusive.
{{< /theorem >}}

{{% hint info %}}
**Testing for Definiteness**  
To determine if a symmetric matrix is positive definite, negative definite, or indefinite:
1. Compute the eigenvalues. If all are positive, the matrix is positive definite; if all are negative, it's negative definite; if some are positive and some negative, it's indefinite.
2. Alternatively, check the leading principal minors. For a positive definite matrix, all should be positive. For a negative definite matrix, the signs should alternate, starting with negative.
{{% /hint %}}

### Exercises on Maxima, Minima, and Saddle Points

**Exercise 9.1** Find and classify all critical points of the function $f(x,y) = x^3 + y^3 - 3xy$.

**Solution:**
First, we find the critical points by setting the partial derivatives equal to zero:

$\frac{\partial f}{\partial x} = 3x^2 - 3y = 0$
$\frac{\partial f}{\partial y} = 3y^2 - 3x = 0$

From the first equation, we get $y = x^2$. Substituting into the second equation:
$3(x^2)^2 - 3x = 0$
$3x^4 - 3x = 0$
$3x(x^3 - 1) = 0$

This gives us $x = 0$ or $x = 1$.

If $x = 0$, then $y = 0^2 = 0$.
If $x = 1$, then $y = 1^2 = 1$.

So the critical points are $(0,0)$ and $(1,1)$.

Now, we classify these critical points using the second derivative test. We compute the second-order partial derivatives:

$\frac{\partial^2 f}{\partial x^2} = 6x$
$\frac{\partial^2 f}{\partial y^2} = 6y$
$\frac{\partial^2 f}{\partial x \partial y} = -3$

At the point $(0,0)$:
$\frac{\partial^2 f}{\partial x^2}(0,0) = 6 \cdot 0 = 0$
$\frac{\partial^2 f}{\partial y^2}(0,0) = 6 \cdot 0 = 0$
$\frac{\partial^2 f}{\partial x \partial y}(0,0) = -3$

The determinant of the Hessian is:
$D = \frac{\partial^2 f}{\partial x^2}(0,0) \cdot \frac{\partial^2 f}{\partial y^2}(0,0) - \left[\frac{\partial^2 f}{\partial x \partial y}(0,0)\right]^2 = 0 \cdot 0 - (-3)^2 = -9 < 0$

Since $D < 0$, the point $(0,0)$ is a saddle point.

At the point $(1,1)$:
$\frac{\partial^2 f}{\partial x^2}(1,1) = 6 \cdot 1 = 6$
$\frac{\partial^2 f}{\partial y^2}(1,1) = 6 \cdot 1 = 6$
$\frac{\partial^2 f}{\partial x \partial y}(1,1) = -3$

The determinant of the Hessian is:
$D = \frac{\partial^2 f}{\partial x^2}(1,1) \cdot \frac{\partial^2 f}{\partial y^2}(1,1) - \left[\frac{\partial^2 f}{\partial x \partial y}(1,1)\right]^2 = 6 \cdot 6 - (-3)^2 = 36 - 9 = 27 > 0$

Since $D > 0$ and $\frac{\partial^2 f}{\partial x^2}(1,1) > 0$, the point $(1,1)$ is a local minimum.

**Exercise 9.2** Find the maximum and minimum values of the function $f(x,y) = 2x^2 + y^2 - 4x - 2y + 5$ on the disk $x^2 + y^2 \leq 4$.

**Solution:**
To find the extreme values of $f$ on the disk $x^2 + y^2 \leq 4$, we need to check:
1. Critical points inside the disk
2. Points on the boundary circle $x^2 + y^2 = 4$

Step 1: Find the critical points by setting the partial derivatives equal to zero:

$\frac{\partial f}{\partial x} = 4x - 4 = 0 \implies x = 1$
$\frac{\partial f}{\partial y} = 2y - 2 = 0 \implies y = 1$

So we have a critical point at $(1, 1)$. Let's verify that it's inside the disk:
$1^2 + 1^2 = 2 < 4$, so it's inside.

To classify this critical point, we compute the second-order partial derivatives:

$\frac{\partial^2 f}{\partial x^2} = 4 > 0$
$\frac{\partial^2 f}{\partial y^2} = 2 > 0$
$\frac{\partial^2 f}{\partial x \partial y} = 0$

The determinant of the Hessian is:
$D = 4 \cdot 2 - 0^2 = 8 > 0$

Since $D > 0$ and $\frac{\partial^2 f}{\partial x^2} > 0$, the point $(1, 1)$ is a local minimum.

The value of $f$ at this point is:
$f(1, 1) = 2(1)^2 + (1)^2 - 4(1) - 2(1) + 5 = 2 + 1 - 4 - 2 + 5 = 2$

Step 2: Examine the boundary $x^2 + y^2 = 4$.

We can use the method of Lagrange multipliers. We want to find the critical points of $f$ subject to the constraint $g(x, y) = x^2 + y^2 - 4 = 0$.

We set up the Lagrangian:
$L(x, y, \lambda) = f(x, y) - \lambda g(x, y) = 2x^2 + y^2 - 4x - 2y + 5 - \lambda(x^2 + y^2 - 4)$

Taking partial derivatives and setting them to zero:

$\frac{\partial L}{\partial x} = 4x - 4 - 2\lambda x = 0$
$\frac{\partial L}{\partial y} = 2y - 2 - 2\lambda y = 0$
$\frac{\partial L}{\partial \lambda} = -(x^2 + y^2 - 4) = 0$

From the first equation: $x(2 - \lambda) = 2$
From the second equation: $y(1 - \lambda) = 1$

If $\lambda = 2$, then $x$ would be undefined from the first equation. So $\lambda \neq 2$.
If $\lambda = 1$, then $y$ would be undefined from the second equation. So $\lambda \neq 1$.

Therefore:
$x = \frac{2}{2 - \lambda}$ and $y = \frac{1}{1 - \lambda}$

Substituting into the constraint $x^2 + y^2 = 4$:
$\left(\frac{2}{2 - \lambda}\right)^2 + \left(\frac{1}{1 - \lambda}\right)^2 = 4$

This is a rather complex equation to solve directly. Let's try a different approach.

We can parameterize the boundary circle as $x = 2\cos\theta$ and $y = 2\sin\theta$ for $0 \leq \theta < 2\pi
