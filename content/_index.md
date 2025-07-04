---
title: Introduction
type: docs
bookToc: false
---

# Numerical Optimization: theory and applications



{{% columns ratio="1:2" %}} <!-- begin columns block -->

{{< center >}}
<div style="margin-left:-100px; ">
{{< figure
  src="./tikZ/main_illustration/main.svg"
  alt="Himmelblau's Function"
  width="400px"
>}}
</div>
{{< /center >}}

<---> <!-- magic separator, between columns -->

## Description 

This page regroups information, lecture documents and practical labs for a course given at doctoral school SIE on numerical optimisation. This version concerns the year 2025. You can fin on the left, all the course materials with solutions to exercises and implementation of programs in python.

{{% /columns %}}

## Course Objectives

This course teaches an overview of modern optimization methods, for applications in inverse problems, machine learning and data science. Alternating between mathematical theory of optimization and practical lab sessions and projects, the course aims at providing doctoral students with the knowledge to solve common problems through numerical optimizations. 

At the end of this course, the students should:

* Be able to recognize convex optimization problems that arise in scientific fields and design appropriate cost functions
* Have an understanding of how such problems are solved, and gain some experience in solving them
* Have knowledge of the underlying tools behind training of machine learning models
* Be able to implement backpropagation and stochastic optimization algorithms for large models



## Program

### Part I - Fundamental Theory (Week 1)

The course will first start with fundamental mathematical concepts of optimization and convex optimization:

* Formulating an optimization problem
* Reminders of linear algebra and formulating a constrained optimization problem
* Reminders on differentiability
* Convexity theory
* Gradient Methods
* Second-order methods

 Session | Duration | Content | Date | Room | Slides |
---------|----------|---------|------|----------|---|
 CM1 | 1.5h | Introduction, Linear algebra and Differentiation reminders, and exercices | 2 June 2025 10am | B-120 | [Slides](./slides/01_introduction/main.pdf) |
 CM2 | 1.5h | Steepest descent algorithm, Newton method and convexity | 2 June 2025 1.15pm | B-120 | [Slides](./slides/02_unconstrained_basics/main.pdf)
 TD1 | 1.5h | Application to linear regression (1/2) | 2 June 2025 3pm | C-213 |
 CM3 | 1.5h | Linesearch algorithms and their convergence | 3 June 2025 10am | B-120 | [Slides](./slides/03_unconstrained_linesearch/main.pdf)
 TD2 | 1.5h | Linesearch in linear regression (2/2)| 3 June 2025 1.15pm | C-213 |
 CM4 | 1.5h | Constrained optimization : linear programming and lagrangian methods | 3 June 2025 3pm | B-120 | [Slides](./slides/04_constrained_optim/main.pdf)

### Part II - Application to Image / Remote Sensing Problems (Week 1)

Then we will apply those concepts in practice in inverse problem solving with examples in image denoising problems in remote sensing:

* Formulating an unconstrained optimization problem and solving it for a practical example
* Numerical implementation in python

This lab will be supervised by [Yassine Mhiri](https://y-mhiri.github.io).

 Session | Duration | Content | Date | Room |
---------|----------|---------|------|------|
 TP | 4h | Implementation of inverse problems for image processing | 4 June 2025 1pm | C-213 |

### Part III - Advanced topics (Week 2)

We then move on to more complicated problems:

* Newton with linesearch and quasi-Newton methods
* Proximal algorithms

| Session | Duration | Content | Date | Room | Slides |
|---------|----------|---------|------|------|------|
| CM5 | 1.5h | Projected Gradient then Newton / Quasi Newton methods | 12 June 2025 10am | B-120 | [Slides 1](./slides/05_constrained_optim_bis/main.pdf) /  [Slides 2](./slides/06_quasi_newton/main.pdf)|
| CM6 | 1.5h | Study in autonomy | 12 June 2025 1.15pm | B-120 |
| TD3 | 1.5h | Study in autonomy | 12 June 2025 3pm | C-213 |

### Part IV - Application to Machine Learning (Week 3)

We finally apply all those concepts with a focus on machine learning training process:

* Stochastic optimization 
* Shallow models: Optimization for linear/logistic regression, support vector machine, perceptron and MLP
* Deep models: Formulating the back propagation algorithm for common layers

For this part, we will rely on Rémi Flamary's slides available at [here](https://remi.flamary.com/cours/intro_ml.html) and [here](https://remi.flamary.com/cours/optim_ds.html).

| Session | Duration | Content | Date | Room |
|---------|----------|---------|------|------|
| Lecture | CM7 | 1.5h | Stochastic optimisation | 12 June 2025 1.15pm | B-120 |
| Lecture | CM8 | 1.5h | Optimization for shallow models : from perceptron to MLP to CNN | 16 June 2025 3pm | B-120 |
| Exercise | TD4 | 1.5h | Implementing backpropagation for neural networks | 17 June 2025 1.15pm | C-213 |
| Lecture | CM9 | 1.5h | Optimization for deep models : Adam and SGD | 17 June 2025 3pm | B-120 |

## Prerequisites

Students should have a decent understanding of basic linear algebra and differentiability of functions over several variables (some reminders will be given at the beginning still). Some capacity to program in Python is desired with knowledge of NumPy.

## Teaching Method

Lectures will alternate between theoretical lectures, mathematical exercises and practical implementation in Python of the algorithms studied. A mini-project in Remote Sensing will also be given to illustrate on a real-world problem.

## Ressources

The course is based on several resources, including:
* *Numerical Optimization* by J. Nocedal and S. Wright, for linesearch and Newton methods
* *Proximal Algorithms* monograph by N. Parikh and S. Boyd, for proximal methods
* *Deep Learning* by I. Goodfellow, Y. Bengio and A. Courville, for backpropagation and stochastic optimization

## Registration

Register through Adum and/or by email to ammar.mian@univ-smb.fr
