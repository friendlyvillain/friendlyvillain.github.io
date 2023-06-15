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

DQN은 Q-learning과 기계학습에서 신경망 (Neural Network, NN)를 결합한 알고리즘으로 Q-function을 구하기 위해 신경망의 input으로 state를 사용한다는 점을 제외하고 Q-function을 구하는 수식 자체는 Q-learning과 동일하다.
본 포스팅에서는 DQN에 있어서 가장 유명하다고도 볼 수 있는 Nature에 개제된 논문인 "**Human-level control through deep reinforcement learning**"에서 다룬 DQN 알고리즘과 그 원리에 대해 다룬다.
DQN에서는 학습의 효율 개선을 위한 다음 2가지 특징이 추가된 점이 Q-learning에서 가장 크게 달라졌다고 볼 수 있다.

- Target Network
- Replay Buffer

## DQN Algorithm

### Continuous State Background

TD example 포스팅에서 다뤘던 Frozen Lake는 총 16개의 이산적인 State로 구성되어 있고, 문제를 풀기 위해, Terminal을 제외한 State에서 행동 가능한 Action의 조합에 따른 Q-table을 도출하였다.
그러나 Frozen Lake와 같이 State가 Discrete하지 않고, Continuous하거나 혹은 State의 개수가 Discrete 하더라도 그 개수가 매우 많을 경우, Q-table을 이용한 방법은 적용이 불가능하거나 Table에 대한 메모리가 State에 비례하여 기하급수적으로 증가하고, 모든 State에 대한 Q-table을 구하기 위한 시간 또한 크게 증가한다. 
Continuous한 State에 대한 예시로 DQN 구현 예제에서도 다룰 다음과 같은 Cartpole 환경을 생각해보자. 

![cartpole-env](/assets/img/posts/dqn/cartpole_env.png){: width="300" height="300" }
_Cartpole_

Cartpole 환경에 대한 구체적인 정보는 향후 예시에서 다룰 포스팅과 [OpenAI gym 페이지](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)를 참고하고, 본 포스팅에서는 State와 Action에 대해서만 간략하게 소개한다.
Cartpole 환경에서 State는 다음과 같은 총 4개의 성분으로 구성된다. 

- 카트의 위치 (Cart position): $-4 \sim 4$
- 카트의 속도 (Cart velocity): $-\infty \sim \infty$
- 폴의 각도 (Pole angle[radian]): $-0.418 \sim 0.418$ 
- 폴의 각속도 (Pole angular velocity): $-\infty \sim \infty$

반면, Cartpole 환경에서 Action은 Discrete한 2개의 Action (0: Move left, 1: Move right)으로만 구성된다.

위와 같은 Continuous한 성분으로 구성된 State에 대해서는 Q-table 방식을 적용할 수 없다. 
이를 해결하기 위해, 기계학습에서 널리 사용되고 있는 심층 신경망 (Deep Neural Network, DNN) 구조가 도입되었다.
DQN을 구현할 때, 입력과 출력의 차원 (dimension)만 MDP 환경에 맞게 설정하면 DNN의 hidden layer 구조는 자유롭게 설정 가능하다. 
먼저, DQN의 input dimension은 state의 차원과 동일하다.
다음으로, DQN의 output dimension은 MDP의 Action Space 개수와 동일하다. 
DQN의 입출력 차원에 대한 개념은 [다음 절](#action-value-function-q-function-in-dqn)에서 더 자세히 설명한다.

위의 Cartpole 예제에서 DQN input의 차원은 $1 \times 4$의 vector가 되며, output의 차원은 $1 \times 2$의 vector가 된다.
Cartpole 예제에서 state를 4개의 성분으로 구성된 vector로 사용하지 않고, 관찰한 시점에서의 ($3 \times W \times H$) 차원의 RGB 이미지를 state로 구성한다면 DQN의 input 차원 또한 ($3 \times W \times H$) 크기의 데이터를 input으로 받는 Convolutional Neural Network (CNN)의 구조와 동일해진다. (이 경우에도 output 차원은 동일하다.)
따라서, state가 더 복잡한 Matrix 형태, Cubic 형태로 구성된다면 DQN의 input 또한 state의 차원과 동일하게 Matrix 형태, Cubic 형태로 구성하면 된다.

### Action-Value function (Q-function) in DQN

이전 포스팅에서 다룬 Q-table에서는 관찰한 state에서 취한 action에 대한 state-action pair (Q-value)에 대한 값을 업데이트 하였다.
DQN의 경우에는 관찰한 state를 설계한 NN 모델에 입력값으로 넣은 출력 결과를 Q-value 값으로 사용한다.
앞선 절에서 언급하였듯이 DQN의 출력 차원은 Action space의 크기와 동일한 이유는 이는 각각의 결과가 관찰한 state에 대한 Q-value를 의미하기 때문이다.
즉, continuous state의 경우에는 학습 과정에서 모든 state를 관찰할 수 없더라도 충분히 다양한 state에 대해 DQN 모델이 학습되었다면 학습 과정에서 경험하지 못했던 state가 입력으로 들어오더라도 정확한 결과를 도출해낼 수 있다.
결국, continuous state 또한 무수히 많은 discrete state의 집합이기 때문에 다양한 discrete state를 경험하며 DQN이 학습되었다면 마치 supervised learning의 원리와 같이 경험해보지 못한 state에 대해서도 높은 정확도를 보이게 되는 것이다.    

여기까지 설명한 DQN의 입출력 차원을 왜 state의 차원과 action space의 크기로 정의하는지 그 원리를 이해하였다면 (원리를 정화히 이해하지 못하더라도), DQN 모델을 정의하여 학습을 시키면 어느정도 reasonable한 결과를 얻을 수 있을 것이다. 
추가로 DQN의 성능을 향상시키기 위하여 강화학습을 접하였다면 누구나 접했을 정도로 유명한 **Human-level control through deep reinforcement learning**에서 다음과 같은 **Target Network**와 **Replay Buffer** 개념이 도입되었다.
(작성중)

- Target Network

- Replay Buffer

## Reference
V. Mnih, K. Kavukcuoglu, D. Silver et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, 2015.