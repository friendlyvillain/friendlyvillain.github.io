---
title: Temporal Difference Example - Frozen Lake
author: jh
date: 2023-05-31 20:25:56 +0900
categories: [Machine Learning, Reinforcement Learning]
tags: [ML, RL, Temporal-Difference, Model-Free, MDP, Optimal Policy, Action-Value Function, Q-function, Bellman Equation, On-policy, Off-policy, Frozen Lake]
math: true
mermaid: true
comments: true
---

## Introduction

본 포스팅에서는 Python 코드를 통해, Temporal Difference (TD) Control 알고리즘을 구현하는 내용을 다룬다. 
[MC 예제](https://friendlyvillain.github.io/posts/mc-example/)에서 다루었던 환경과 동일하게 OpenAI Gym의 Frozen Lake 환경에 대해 앞선 포스팅에서 다룬 On-Policy TD Control 알고리즘인 [**SARSA**](https://friendlyvillain.github.io/posts/temporal-difference/#sarsa-on-policy-td-control-algorithm)와 Off-Policy TD Control 알고리즘인 [**Q-learning**](https://friendlyvillain.github.io/posts/temporal-difference/#q-learning-off-policy-td-control-algorithm)을 적용한다.
코드 구현을 위해 필요한 Python Package는 Gym, Numpy, Matplotlib 이다. 

