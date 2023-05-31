---
title: Temporal Difference
author: jh
date: 2023-05-31 20:25:56 +0900
categories: [Machine Learning, Reinforcement Learning]
tags: [ML, RL, Temporal-Difference, Model-Free, MDP, Optimal Policy, Action-Value Function, Q-function, Bellman Equation, Policy Evaluation, Policy Improvement, Policy Control]
math: true
mermaid: true
comments: true
---

## Introduction

지금까지 MDP 문제를 풀기 위한 방법으로 [Dynamic Programming (DP)](https://friendlyvillain.github.io/posts/dynamic-programming/)과 [Monte-Carlo Method (MC)](https://friendlyvillain.github.io/posts/monte-carlo-method/)에 대해 다루었다. 
DP는 time-step 마다 가치함수를 업데이트 할 수 있지만, Model-Based 한 환경에만 적용할 수 있다는 한계가 있다. 
반면, MC는 Model-Free한 환경에 대해서도 Experience Sampling을 통해 MDP 문제를 풀 수 있지만 Model을 업데이트 하기 위해 Sampling된 Episode가 반드시 종료되어야 한다는 한계가 있다. 
Temporal-Difference (TD)는 DP와 MC의 장점을 모두 결합하여 Model-Free한 환경에 대해서도 Expereience Sampling을 통해 MDP 문제를 풀 수 있고, Episode가 종료되지 않더라도 Model을 업데이트 할 수 있다는 특징이 있다. 
TD Control 알고리즘의 기본적인 형태는 발전된 강화학습 알고리즘인 DQN, Double DQN 등의 형태와 매우 유사하며, Q-value를 어떻게 도출하는냐에 대한 차이만 존재한다. 


## TD Control Algorithm

MC와 마찬가지로 Model-Free한 환경에 적용하기 위해 가치함수로 Action-Value Function을 사용한다. 
