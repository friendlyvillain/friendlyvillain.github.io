---
title: DQN example
author: jh
date: 2023-07-01 19:22:32 +0900
categories: [Machine Learning, Reinforcement Learning]
tags: [ML, RL, DQN, Replay Memory, Temporal-Difference, Model-Free, MDP, Optimal Policy, Action-Value Function, Q-function, Bellman Equation, Off-policy]
math: true
mermaid: true
comments: true
---

## Introduction

본 포스팅에서는 OpenAI Gym의 [Cartpole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) 환경을 예시로 들어서 앞선 포스팅에서 다룬 [DQN 알고리즘](https://friendlyvillain.github.io/posts/deep-q-network/#dqn-algorithm)을 구현하는 코드에 대해 다룬다. 
본 포스팅에서 다룬 코드를 실행시키기 위해 필요한 Python Package는 (gym > 0.21 또는 gymnasium), numpy, matplotlib 이다. 


## Cartpole

Cartpole 환경에서 agent의 목적은 매 time-step 마다 폴대가 쓰러지지 않도록 폴을 왼쪽 또는 오른쪽으로 움직이도록 행동하는 것이다. 
앞선 포스팅에서 다루었듯이 Continuous State 환경으로 구성되어 있고, 2개의 discrete한 action으로 구성되어 있다. 
특히 gym v0.22 부터 제공되는 환경은 이전 버전에서 제공되던 환경과 큰 변화가 생겼으므로 관련된 변화도 같이 설명한다.  

### State
- 카트의 위치 (Cart position): $-4 \sim 4$
- 카트의 속도 (Cart velocity): $-\infty \sim \infty$
- 폴의 각도 (Pole angle[radian]): $-0.418 \sim 0.418$ 
- 폴의 각속도 (Pole angular velocity): $-\infty \sim \infty$

### Action
- 0: 왼쪽으로 이동
- 1: 오른쪽으로 이동 

### Termination 조건 
- 카트 폴 각도의 절대값이 12도를 초과하는 경우
- 카트 폴의 위치의 절대값이 2.4를 초과하는 경우 (edge에 도달)
- 버림(Truncation): time-step이 특정 횟수 이상을 초과하는 경우 (Cartpole-v0의 경우 200, Cartpole-v1의 경우 500)

gym v0.22부터 새롭게 버림 조건이 추가되었으며 버림 조건을 제외하고, termination이 발생한 경우 올바른 action이 수행되지 않아 폴대가 쓰러진 상황을 의미한다. 


## Environment: Cartpole-v1

본 포스팅에서는 gymnasium-0.28.1 에서 제공하는 Cartpole-v1 환경을 가정한다. 
모델은 다음과 같이 생성할 수 있다. 

```python

import gymnasium as gym
env = gym.make('Cartpole-v1')

```

환경 초기화를 위해서는 다음과 같이 코드를 작성한다. 

```python

state, info = env.reset()

```

gym v0.22 부터 info 라는 추가 정보가 반환되도록 변경되었으나, 본 포스팅에서는 info를 사용하지 않으므로 환경 초기화를 다음과 같이 처리한다. 

```python

state, _ = env.reset()

```

다음으로 매 time-step 마다 agent에 의한 action을 입력 받아 다음 상태와 보상을 추출하기 위해 아래과 같이 step method를 호출한다.  

```python

next_state, reward, done, truncated, info = env.step(action)

```

위의 코드는 현재 상태에서 0 또는 1의 action을 수행했을 경우, 다음 상태 (next_state), 보상 (reward), 종료 여부 (done), 버림 조건 여부 (truncated), 추가 정보 (info)를 반환하는 것을 의미한다. 
gym v0.22 부터 버림 조건을 의미하는 truncated 변수가 추가로 리턴되는 것이 이전 버전과 비교하였을 때 유의미한 차이점이다. 
환경 초기화 시와 마찬가지로 info는 본 포스팅에서 사용하지 않으므로 agent가 수행한 action에 대한 다음 상태 정보를 얻기 위해 아래와 같이 처리한다. 

```python

next_state, reward, done, truncated, _ = env.step(action)

```