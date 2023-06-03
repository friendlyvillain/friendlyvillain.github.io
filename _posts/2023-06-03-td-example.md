---
title: Temporal Difference Example - Frozen Lake
author: jh
date: 2023-06-03 14:15:32 +0900
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

## MDP Environment

MC 예제에서 다루었던 Frozen Lake의 MDP Environment와 동일하다. 
State, Action, Reward 와 같은 자세한 설명은 [이 포스팅](https://friendlyvillain.github.io/posts/mc-example/#mdp-environment)에 작성되어 있다.
Reward의 경우 Next State가 Hole에 빠질 경우, MC에서는 $-1$의 Reward를 주었으나 TD에서는 더욱 큰 음의 Reward인 $-5$를 주도록 설정하였다.
MC의 경우에는 Episode가 종료될 때 까지의 누적 Reward를 사용하지만 TD의 경우는 바로 다음 State의 결과만 이용하여 학습하기 때문에 안좋은 (Hole에 빠지는) 행동을 수행하였을 경우 이를 억제하는 효과를 얻을 수 있다. 


## Implementation

