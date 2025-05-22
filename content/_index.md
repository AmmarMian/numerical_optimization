---
title: Introduction
type: docs
bookToc: false
---

# Numerical Optimization for Data Science and Machine Learning


{{% columns ratio="1:2" %}} <!-- begin columns block -->

{{< center >}}
<div style="margin-left:-100px; ">
{{< figure
  src="/tikZ/main_illustration/main.svg"
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
* Know common approaches to non-convex optimization problems
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

 Session | Duration | Content | Date | Room |
---------|----------|---------|------|----------|
 CM1 | 1.5h | Introduction, Linear algebra and Differentiation reminders, and exercices | 2 June 2025 10am | B-120 |
 CM2 | 1.5h | Steepest descent algorithm, Newton method and convexity | 2 June 2025 1.15pm | B-120 |
 TD1 | 1.5h | Application to linear regression | 2 June 2025 3pm | C-213 |
 CM3 | 1.5h | Linesearch algorithms and their convergence | 3 June 2025 10am | B-120 |
 CM4 | 1.5h | Constrained optimization : linear programming and lagrangian methods | 3 June 2025 1.15pm | B-120 |
 TD2 | 1.5h | Implementation of Linesearch methods | 3 June 2025 3pm | C-213 |

### Part II - Application to Image / Remote Sensing Problems (Week 1)

Then we will apply those concepts in practice in inverse problem solving with examples in image denoising and pansharpening problems in remote sensing:

* Formulating a cost function with constraints (sparsity, smoothness, etc)
* Numerical implementation in python

 Session | Duration | Content | Date | Room |
---------|----------|---------|------|------|
 TP | 4h | Implementation of inverse problems for image processing | 4 June 2025 1pm | C-213 |

### Part III - Advanced topics (Week 2)

We then move on to more complicated problems:

* Trust regions methods 
* Proximal algorithms

| Session | Duration | Content | Date | Room |
|---------|----------|---------|------|------|
| CM5 | 1.5h | Trust regions methods and their convergence | 12 June 2025 10am | B-120 |
| CM6 | 1.5h | Proximal methods | 12 June 2025 1.15pm | B-120 |
| TD3 | 1.5h | Use of proximal methods in image processing | 12 June 2025 3pm | C-213 |

### Part IV - Application to Machine Learning (Week 3)

We finally apply all those concepts with a focus on machine learning training process:

* Stochastic optimization 
* Shallow models: Optimization for linear/logistic regression, support vector machine, perceptron and MLP
* Deep models: Formulating the back propagation algorithm for common layers

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

## Registration

Register through Adum and/or by email to ammar.mian@univ-smb.fr
