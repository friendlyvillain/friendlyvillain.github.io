---
title: DQN improvement - DDQN, DDDQN
author: jh
date: 2023-06-10 18:43:40 +0900
categories: [Machine Learning, Reinforcement Learning]
tags: [ML, RL, DQN, Double DQN, Dueling Architecture, DDQN, DDDQN]
math: true
mermaid: true
comments: true
---

## Introduction

앞선 포스팅에서는 가장 baseline되는 DQN과 관련된 이론을 소개하고, Cartpole을 통한 예제를 적용하여 DQN을 구현하였다. 
Main network와 target network의 분리, replay buffer의 도입으로 DQN의 성능이 향상되었으나 학습 진행 과정에서 불안정성 또한 확인된다. 
본 포스팅에서는 다음 2가지 방법을 통해 DQN 성능을 추가로 향상시키는 방법에 대해 기술한다. 

 - (1) Q-function 목적 함수 변형
 - (2) DQN 구조 변형

(1)의 방법은 Double DQN (DDQN) 과 관련된 내용이며, (2)의 방법은 Dueling architecture 와 관련된 내용이다. 
(1)의 방법과 (2)의 방법을 동시에 적용햐면 Dueling Double DQN (DDDQN)이라 한다. 


