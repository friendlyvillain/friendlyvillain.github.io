---
title: DDQN/DDDQN example - Cartpole
author: jh
date: 2023-07-09 21:49:52 +0900
categories: [Machine Learning, Reinforcement Learning]
tags: [ML, RL, DQN, Double DQN, Dueling Architecture, DDQN, DDDQN, Cartpole, Pytorch, Python]
math: true
mermaid: true
comments: true
---

## 1. Introduction

본 포스팅에서는 앞선 포스팅에서 다루었던 DDQN과 DDDQN을 Python 기반으로 [Cartpole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) 예제에 적용하여 성능을 분석한다. 
전체적인 Cartpole 환경은 [DQN 예제](https://friendlyvillain.github.io/posts/dqn-example/) 에서 다루었던 환경과 동일하며, DDQN과 DDDQN이 적용됨에 따라 target을 구하는 코드와 학습 모델을 설계하는 코드를 중심으로 다룬다. 

## 2. DDQN 

