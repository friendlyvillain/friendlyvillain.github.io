---
title: DQN example
author: jh
date: 2023-07-01 19:22:32 +0900
categories: [Machine Learning, Reinforcement Learning]
tags: [ML, RL, DQN, Replay Memory, Temporal-Difference, Action-Value Function, Q-function, Off-policy, Cartpole, Pytorch]
math: true
mermaid: true
comments: true
---

## Introduction

본 포스팅에서는 OpenAI Gym의 [Cartpole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) 환경을 예시로 들어서 앞선 포스팅에서 다룬 [DQN 알고리즘](https://friendlyvillain.github.io/posts/deep-q-network/#dqn-algorithm)을 구현하는 코드에 대해 다룬다. 
본 포스팅에서 다룬 코드를 실행시키기 위해 필요한 Python Package는 (gym > 0.21 또는 gymnasium), numpy, matplotlib, pytorch 이다. 


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


## Implementation

### Gymnasium Environment: Cartpole-v1

본 포스팅에서 다루는 DQN 예제에서는 gymnasium-0.28.1 에서 제공하는 Cartpole-v1 환경을 가정한다. 
모델은 다음과 같이 생성할 수 있다. 

```python

import gymnasium as gym
env = gym.make('Cartpole-v1')

```

Cartpole 환경 초기화를 위해서는 다음과 같이 코드를 작성한다. 

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

### DQN Model 

1개의 input layer, 2개의 hidden layer, 1개의 output layer로 구성된 간단한 신경망 모델을 고려한다.
본 포스팅에서는 Pytorch를 사용하여 Class 기반으로 신경망을 설계하였고, 1번째 hidden layer를 구성하는 뉴런의 개수와 2번째 hidden layer를 구성하는 뉴런의 개수는 각각 32, 64로 설정한다. 
신경망의 input layer와 output layer는 Class의 매개 변수로 입력된 값에 의해 뉴런의 개수가 결정되도록 구현되었고, Cartpole 환경에서 input layer의 뉴런의 개수는 state 구성 변수 개수(4)가 되고, output layer의 뉴런의 개수는 agent가 취하는 행동의 개수(2) 이다. 
Hidden layer의 activation function은 ReLU를 사용하고, 학습 안정화를 위해 batch normalization을 적용한다. ㅊ


```python

import random
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, num_input_layers, num_output_layers, duel_opt=False, batch_norm=False):
        super().__init__()

        self.num_input_layers = num_input_layers
        self.num_output_layers = num_output_layers
        self.duel_opt = duel_opt
        self.batch_norm = batch_norm
        self.setup_model()
        self.apply(self._init_weights)
        self._init_final_layer()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight.data, 1)
            nn.init.constant_(module.bias.data, 0)

    def _init_final_layer(self):
        nn.init.xavier_uniform_(self.layer3.weight.data)
        if self.duel_opt is True:
            nn.init.xavier_uniform_(self.layer3_val.weight.data)

    def setup_model(self):
        self.layer1 = nn.Sequential(nn.Linear(self.num_input_layers, 32), nn.BatchNorm1d(32), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.layer3 = nn.Linear(64, self.num_output_layers)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def sample_action(self, state, epsilon):
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            out = self.forward(state)
            return out.argmax().item()

    # Select greedy based action 
    def greedy_action(self, state):
        out = self.forward(state)
        return out.argmax().item()
```


