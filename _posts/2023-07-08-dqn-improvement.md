---
title: DQN improvement - DDQN, DDDQN
author: jh
date: 2023-07-08 19:32:40 +0900
categories: [Machine Learning, Reinforcement Learning]
tags: [ML, RL, DQN, Double DQN, Dueling Architecture, DDQN, DDDQN]
math: true
mermaid: true
comments: true
---

## 1. Introduction

앞선 포스팅에서는 가장 baseline되는 DQN과 관련된 이론을 소개하고, Cartpole을 통한 예제를 적용하여 DQN을 구현하였다. 
Main network와 target network의 분리, replay buffer의 도입으로 DQN의 성능이 향상되었으나 학습 진행 과정에서 불안정성 또한 확인된다. 
본 포스팅에서는 다음 2가지 방법을 통해 DQN 성능을 추가로 향상시키는 방법에 대해 기술한다. 

 - (1) Q-function 목적 함수 변형 [[1]](#1)
 - (2) DQN 구조 변형 [[2]](#2)

(1)의 방법은 Double DQN (DDQN) 과 관련된 내용이며, (2)의 방법은 Dueling architecture 와 관련된 내용이다. 
(1)의 방법과 (2)의 방법은 모두 2016년 Google에서 제안하였으며, 2가지 방법을 동시에 적용한 기법을 Dueling Double DQN (DDDQN)이라 한다. 


## 2. Double DQN

기존 DQN의 target value는 다음과 같이 표현된다. 

$$
Y_{t}^{DQN} = R_{t+1} + \gamma \max_{a'}Q(S_{t+1}, a' | \theta^{-}) 
$$

위의 수식에서 $\max$ operator 가 Q-value를 over-estimation 하는 경향이 있기 때문에, 학습 과정에서 성능이 불안정해지는 원인이 된다. 
이와 같은 문제점을 보완하기 위해 Double DQN (DDQN)이 도입되었고, DDQN에서는 target value를 아래와 같이 수정한다. 

$$
Y_{t}^{DDQN} = R_{t+1} + \gamma Q(S_{t+1}, \text{arg}\max_{a'} Q(S_{t+1}, a'| \theta) | \theta^{-}) 
$$

위의 DDQN에서의 target value 수식을 보면 next state에서 agent가 취하는 next action은 main network의 greedy policy 기반으로 이루어지며, (next state-next action) pair에 대한 Q-value는 target network 기준으로 계산되는 것을 알 수 있다. 
2번의 Q-value 연산이 이루어지므로 Double DQN 이라고 명명되며, 줄여서 DDQN이라 한다. 
위와 같이 기존 DQN의 over-estimation 문제를 해결하는 것만으로도 높은 성능 향상이 이루어졌음을 [[1]](#1)에서 보였다.   


## 3. Dueling Architecture 



## Reference
### [1] 
H. van Hasselt, A. Guez, and D. Silver, "Deep reinforcement learning with double Q-learning," in Proc. 30th AAAI Conf. Artif. Intell. (AAAI), 2016, pp. 2094-2100.

### [2] 
Z. Wang et al., "Dueling network architectures for deep reinforcement learning," in Proc. 33rd Int. Conf. Mach. Learn. (ICML), 2016, pp. 1995-2003.