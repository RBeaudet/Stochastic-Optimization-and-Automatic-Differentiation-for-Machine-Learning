# Stochastic-Optimization-and-Automatic-Differentiation-for-Machine-Learning

This repository is our project for the course [SOADML](http://marcocuturi.net/soadml.html) for our last year at ENSAE ParisTech.

We explain and implement methods discussed in the paper [Tracking the gradients using the Hessian: A new look at variance reducing stochastic methods](https://arxiv.org/abs/1710.07462).

## Abstract

<i>From the original paper</i>

Our goal is to improve variance reducing stochastic methods through better control variates. We first propose a modification of SVRG which uses the Hessian to track gradients over time, rather than to recondition, increasing the correlation of the control variates and leading to faster theoretical convergence close to the optimum. We then propose accurate and computationally efficient approximations to the Hessian, both using a diagonal and a low-rank matrix. Finally, we demonstrate the effectiveness of our method on a wide range of problems.
