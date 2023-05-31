---
title: Temporal Difference
author: jh
date: 2023-05-31 20:25:56 +0900
categories: [Machine Learning, Reinforcement Learning]
tags: [ML, RL, Temporal-Difference, Model-Free, MDP, Optimal Policy, Action-Value Function, Q-function, Bellman Equation, On-policy, Off-policy]
math: true
mermaid: true
comments: true
---

## Introduction

지금까지 MDP 문제를 풀기 위한 방법으로 [Dynamic Programming (DP)](https://friendlyvillain.github.io/posts/dynamic-programming/)과 [Monte-Carlo Method (MC)](https://friendlyvillain.github.io/posts/monte-carlo-method/)에 대해 다루었다. 
DP는 time-step 마다 가치함수를 업데이트 할 수 있지만, Model-Based 한 환경에만 적용할 수 있다는 한계가 있다. 
반면, MC는 Model-Free한 환경에 대해서도 Experience Sampling을 통해 MDP 문제를 풀 수 있지만 Model을 업데이트 하기 위해 Sampling된 Episode가 반드시 종료되어야 한다는 한계가 있다. 
Temporal-Difference (TD)는 DP와 MC의 장점을 모두 결합하여 Model-Free한 환경에 대해서도 Expereience Sampling을 통해 MDP 문제를 풀 수 있고, Episode가 종료되지 않더라도 Model을 업데이트 할 수 있다는 특징이 있다. 
TD Control 알고리즘의 기본적인 형태는 발전된 강화학습 알고리즘인 DQN, Double DQN 등의 형태와 매우 유사하며, Q-function을 어떻게 도출하는냐에 대한 차이만 존재한다. 


## TD Control Algorithm

MC와 마찬가지로 Model-Free한 환경에 적용하기 위해 가치함수로 Q-function을 사용한다. 
TD는 bootstrap을 통해 Episode가 종료되지 않더라도 Model을 업데이트 할 수 있고, 본 포스팅에서는 가장 기본적인 형태인 매 time-step마다 Model을 업데이트 하는 TD control Algorithm을 다룬다. 
Q-function을 업데이트 하는 방법에 따라 on-policy TD Control Algorithm을 SARSA라 부르고, off-policy TD Control Algorithm을 Q-learning이라 부른다. 

### On-policy vs Off-policy
Agent가 직접 Action을 수행하는 policy를 **Behavior policy**라 하고, 가치를 evaluate하고 improve 하기 위한 policy를 **Target policy**라 한다.
다시 말해, Behavior policy는 MDP 환경과 상호작용하는 정책이며, 해당 환경에서 개선의 대상이 되는 정책을 Target policy라 할 수 있다.
Behavior policy와 Target policy가 동일한 경우 On-policy control이라 부르며, Behavior Policy와 Target Policy가 다른 경우 Off-policy control이라 부른다. 
On-policy control의 경우, Q-function을 업데이트 할 때, Agent가 Next State에서 수행하는 정책이 현재 Policy를 그대로 따른다.
반면, Off-policy control의 경우, Q-function을 업데이트 할 때, Agent가 Next State에서 수행하는 정책이 현재 Policy를 그대로 따르지 않고, 학습시킬 수 있다. 

(작성중)