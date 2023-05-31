---
title: Temporal Difference
author: jh
date: 2023-05-31 20:25:56 +0900
categories: [Machine Learning, Reinforcement Learning]
tags: [ML, RL, Temporal-Difference, Model-Free, MDP, Optimal Policy, State-Value Function, Bellman Equation, Policy Evaluation, Policy Improvement, Policy Control]
math: true
mermaid: true
comments: true
---

## Introduction
[Monte-Carlo Method (MC)](https://friendlyvillain.github.io/posts/monte-carlo-method/)를 통해 [Dynamic Programming (DP)](https://friendlyvillain.github.io/posts/dynamic-programming/)과 달리 Model-Free한 환경에 대해서도 Expereince의 샘플링을 통해 MDP 문제를 해결할 수 있다는 사실을 확인하였다. 

그러나, MC의 경우에는 샘플링한 Episode가 반드시 종료되어야 Model의 업데이트가 가능하다는 한계가 있다. 
작성중