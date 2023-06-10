---
title: Deep Q Network (DQN)
author: jh
date: 2023-06-10 18:43:40 +0900
categories: [Machine Learning, Reinforcement Learning]
tags: [ML, RL, DQN, Replay Memory, Temporal-Difference, Model-Free, MDP, Optimal Policy, Action-Value Function, Q-function, Bellman Equation, Off-policy]
math: true
mermaid: true
comments: true
---

## Introduction

TD Control Algorithm을 통해 Model-Free한 환경에서 매 time-step 마다 Agent가 Model을 업데이트하며 학습할 수 있는 것을 확인하였다. 
그러나 앞서 살펴본 TD Control Algorithm에서는 모든 state와 action에 대한 Q-function (Q-table)을 구해야 한다는 한계가 있다.
따라서, TD Control Algorithm은 **State와 Action이 모두 이산적 (Discrete)**인 경우에만 적용 가능하다. 
본 포스팅에서는 Action은 Discrete 하지만, **State는 연속적 (Continuous)**인 경우에 대해 적용할 수 있는 알고리즘인 Deep Q Network (DQN) 에 대해 다루고, Action까지 Continous한 경우에 대한 알고리즘은 차후 포스팅에서 다룬다.

DQN은 Q-learning과 기계학습에서 신경망 (Neural Network)를 결합한 알고리즘으로 Q-function을 구하기 위해 신경망의 input으로 state를 사용한다는 점을 제외하고 Q-function을 구하는 수식 자체는 Q-learning과 동일하다.
본 포스팅에서는 DQN에 있어서 가장 유명하다고도 볼 수 있는 Nature에 개제된 논문인 "**Human-level control through deep reinforcement learning**"에서 다룬 DQN 알고리즘과 그 원리에 대해 다룬다.
DQN에서는 학습의 효율 개선을 위한 다음 2가지 특징이 추가된 점이 Q-learning에서 가장 크게 달라졌다고 볼 수 있다.

- Target Network
- Replay Buffer

## DQN Algorithm

### Continuous State




## Reference
V. Mnih, K. Kavukcuoglu, D. Silver et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, 2015.