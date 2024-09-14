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

DDQN의 정의에 따라 DQN 예제 포스팅의 [DQN Agent 클래스](https://friendlyvillain.github.io/posts/dqn-example/#3-3-dqn-agent)에서 train method를 다음과 같이 변경한다. 

```python
class DQNAgent():
    ''''
        def __init__():
            ...
        
        ...
    ''''
    
    def train(self, batch_size):
        states, actions, rewards, next_states, done_mask = self.memory.sample(batch_size)

        states = torch.FloatTensor(np.array(states))
        actions = torch.tensor(actions).type(torch.int64)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))\
        done_mask = torch.FloatTensor(done_mask)

        q_out = self.dqn_net(states.to(device).view(batch_size, -1)) # batch_size by num_actions

        q_a = q_out.gather(dim=1, index=actions.to(device).view(-1, 1)) # batch_size by 1 (q-values for selected actions)

        q_prime_target = self.dqn_target(next_states.to(device).view(batch_size, -1)) # batch_size by num_actions

        # DDQN
        q_prime_net = self.dqn_net(next_states.to(device).view(batch_size, -1)) # main network's q-value (s_prime) 
        a_max_prime = q_prime_net.argmax(1) # main network's greedy action
        max_q_prime = q_prime_target.gather(1, a_max_prime.view(-1, 1)) # target network's q-value applying main network's action

        q_target = rewards.to(device) + self.gamma * max_q_prime_ddqn * (1 - done_mask).to(device).view(-1, 1) # DDQN

        loss = self.criterion(q_a, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```