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

[MC 예제](https://friendlyvillain.github.io/posts/mc-example/)에서 다루었던 Frozen Lake의 MDP Environment와 동일하다. 
State, Action, Reward 와 같은 자세한 설명은 [이 포스팅](https://friendlyvillain.github.io/posts/mc-example/#mdp-environment)에 작성되어 있다.
Reward의 경우 Next State가 Hole에 빠질 경우, MC에서는 $-1$의 Reward를 주었으나 TD에서는 더욱 큰 음의 Reward인 $-5$를 주도록 설정하였다.
MC의 경우에는 Episode가 종료될 때 까지의 누적 Reward를 사용하지만 TD의 경우는 바로 다음 State의 결과만 이용하여 학습하기 때문에 안좋은 (Hole에 빠지는) 행동을 수행하였을 경우 이를 억제하는 효과를 얻을 수 있다. 


## Implementation

```python

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

'''

SFFF
FHFH
FFFH
HFFG

action
0: Left
1: Down
2: Right
3: Up

'''

class TD_Agent:
    def __init__(self, env, discount=0.9, learning_rate=0.01, on_policy=True):
        super().__init__()
        self.env = env
        self.on_policy = on_policy
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.state_goal = self.num_states - 1  # 15
        self.state_terminal = list([5, 7, 11, 12, self.state_goal]) # Hole + Goal

        self.discount = discount
        self.learning_rate = learning_rate

        self.action_value_table = np.zeros([self.num_states, self.num_actions], dtype=float)
        self.policy = np.zeros([self.num_states, 1], dtype=int)  # Except terminal state policy

        self._initialize()

    def _initialize(self):
        self.action_value_table = np.random.uniform(0, 1, (self.num_states, self.num_actions))
        self.action_value_table[self.state_terminal] = 0

        self.policy = np.random.randint(self.num_actions, size=(self.num_states, 1))
        self.policy[self.state_terminal] = -1

        print("Initial action-value table: \n {}".format(self.action_value_table))
        print("Initial policy: \n {}".format(self.policy.reshape(4, -1)))

    def get_action(self, state, epsilon):
        coin = random.random()
        if coin < epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            greedy_action = np.argmax(self.action_value_table[state])
            action = greedy_action

        return action

    def update(self, state, action, reward, s_prime, epsilon):
        if self.on_policy: # On policy
            a_prime = self.get_action(s_prime, epsilon) # e-Greedy based action
        else: # Off policy
            a_prime = self.get_action(s_prime, 0) # Greedy action

        q_val = self.action_value_table[state][action]
        q_val_prime = self.action_value_table[s_prime][a_prime]
        self.action_value_table[state][action] = q_val + self.learning_rate * (reward + self.discount * q_val_prime - q_val)

    def set_greedy_policy(self):
        for state in range(self.num_states):
            greedy_action = np.argmax(self.action_value_table[state])
            self.policy[state] = greedy_action
        self.policy[self.state_terminal] = -1


    def policy_mapping(self):
        policy_map = np.empty([self.num_states, 1], dtype=str)

        for s in range(self.num_states):
            if self.policy[s] == 0:
                policy_map[s] = 'L'
            elif self.policy[s] == 1:
                policy_map[s] = 'D'
            elif self.policy[s] == 2:
                policy_map[s] = 'R'
            elif self.policy[s] == 3:
                policy_map[s] = 'U'
            else:
                policy_map[s] = 'T'

        return policy_map.reshape(4, -1)

def run_td(num_episodes = 15000, num_verifications = 5000, on_policy=True):
    env = gym.make('FrozenLake-v1', is_slippery=True)

    result_train = []
    result_test = []

    agent = TD_Agent(env, discount=0.9, learning_rate=0.01)

    for n_epi in range(num_verifications):
        state = env.reset()
        done = False

        while not done:
            action = agent.get_action(state, np.Inf) # Always random action
            s_prime, reward, done, _ = env.step(action)
            state = s_prime

            if done and state == 15:
                result_test.append(1)
            elif done and state != 15:
                result_test.append(0)

    result_test = np.array(result_test)
    print("Initial accuracy: {:.3f}".format(result_test.sum()/num_verifications * 100))

    for n_epi in range(num_episodes):
        epsilon = max(0.01, 0.30 - 0.1 * (n_epi / 200))  # Linear annealing from 30% to 1%

        state = env.reset()
        done = False

        while not done:
            action = agent.get_action(state, epsilon)
            s_prime, reward, done, _ = env.step(action)

            if done:
                if s_prime != 15:
                    reward = -5
                    result_train.append(0)
                else:
                    reward = 1
                    result_train.append(1)
            else:
                reward = 0.0

            agent.update(state, action, reward, s_prime, epsilon)
            state = s_prime

    result_train = np.array(result_train)

    fig, ax = plt.subplots(1,1)
    ax.plot(result_train)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success')
    plt.pause(0.01)

    result_test = []
    agent.set_greedy_policy() # Setup Greedy Policy

    policy_map = agent.policy_mapping()
    print("Final Greedy Policy: \n {}".format(policy_map))

    for n_epi in range(num_verifications):
        state = env.reset()
        done = False

        while not done:
            action = agent.policy[state].item()
            s_prime, reward, done, _ = env.step(action)
            state = s_prime

            if done and state == 15:
                result_test.append(1)
            elif done and state != 15:
                result_test.append(0)

    result_test = np.array(result_test)
    print("Accuracy after TD control: {:.3f}".format(result_test.sum()/num_verifications * 100))

    env.close()

    return agent

if __name__ == "__main__":
    agent_sarsa = run_td(on_policy=True) # on_policy=True; SARSA
    agent_qlearn = run_td(on_policy=False) # on_policy=False; Q-learning
```

### 코드 설명

Frozen Lake 환경에 dynamics를 주기 위해 **is_slippery=True**로 하여, gym 모듈을 사용하여 다음과 같은 MDP 환경을 만든다.

```python
env = gym.make('FrozenLake-v1', is_slippery=True)
```

TD Control Algorithm을 적용하기 위한 TD_Agent 클래스를 정의하고 initial policy와 Q-function을 초기화 한다.
이 때, on-policy와 off-policy control을 적용하는 경우를 구분하기 위해, TD_Agent 클래스의 초기 변수인 on_policy를 설정한다.
on_policy=True일 경우 SARSA를 적용하고, on_policy=False일 경우 Q-learning을 적용한다.
TD_Agent 클래스의 주요 method는 다음과 같다.

- get_action()
인자로 state와 epsilon 값을 받아서 해당 state에서 $ \epsilon-Greedy$ 기반의 Action을 리턴한다.

- update()
Agent의 Q-function을 업데이트하는 method이다. 
On-policy SARSA일 경우에는 next state에 대한 Action을 구하기 위해 get_action()을 호출하여 현재의 Q-function을 이용한다.
Off-policy Q-learning일 경우에는 next state에 대한 Action을 구하기 위해 마찬가지로 get_action()을 호출하지만 $ \boldsymbol{\epsilon=0} $ 으로 설정하여 항상 Greedy Action을 선택하도록 한다.
그 이후, TD Control Algorithm에 따라 Q-function을 업데이트한다. 


run_td() 함수에서 TD_Agent 객체를 만든 이후, episode의 매 time step 마다 get_action() method를 호출하여 action을 취하고, update() method()를 호출하여 agent 객체의 Q-function을 업데이트한다. 


### 코드 실행 결과

MC 예제에서 다루었던 내용과 동일하게, TD Control을 적용하기 전, initial policy 기반으로 Frozen Lake를 수행하면 Goal에 도달할 확률은 0에 가깝다.
총 15000의 Episode 동안, discount factor=0.9, learning rate=0.01로 설정한 이후, epsilon을 $30\% \rightarrow 1\%$로 감소시키면서 TD Control Algorithm을 적용하였다.
SARSA와 Q-learning 모두 Episode의 초반부에는 Goal에 도달하지 못하고 Hole에 빠지는 경우가 많지만, Episode가 진행될 수록 Goal에 도달하는 경우가 많아지며, 각각의 경우, 최종적으로 도출된 Greedy Policy는 다음과 같다. 

- SARSA
```
Final Greedy Policy: 
 [['L' 'U' 'U' 'U']
 ['L' 'T' 'L' 'T']
 ['U' 'D' 'L' 'T']
 ['T' 'R' 'D' 'T']]
```

- Q-learning
```
[['L' 'U' 'U' 'U']
 ['L' 'T' 'R' 'T']
 ['U' 'D' 'L' 'T']
 ['T' 'R' 'D' 'T']]
```

위의 경우 State=6인 경우에 대한 Action만 차이가 있으나 Frozen Lake 환경의 특성상 State=6에서 Left인 경우와 Right인 경우 모두 Hole에 빠질 확률은 동일하며, Hole에 빠지지 않더라도 다음 State로 전이되는 확률 또한 동일하므로 사실상 동일한 transition을 보이는 action이라 볼 수 있다.
SARSA와 Q-learning 모두 얻어진 Greedy Policy에 따라 Frozen Lake를 수행하면 70% 이상의 높은 확률로 Goal에 도달하는 것을 확인할 수 있다.

**연산 환경에 따라 도달되는 Policy와 정확도에는 차이가 있을 수 있으나, MC와 다르게 TD를 적용할 경우 최종적으로 얻어진 Greedy Policy가 높은 정확도를 보이는 경우가 훨씬 많아지게 된다.**


## Conclusion
본 포스팅에서는 Model-Free한 MDP 환경에서 TD Control 에 대한 Pseudo Code를 간단한 Frozen Lake 환경에 대한 Example을 통해 Python을 이용하여 구현하였다.
TD Control은 MC Control과 달리 Episode가 종료되지 않더라도 Q-function을 업데이트 할 수 있다는 장점이 있다. 
Frozen Lake와 같이 간단한 환경에서는 On-policy TD Control (SARSA)과 Off-policy TD Control (Q-learning) 모두 학습된 결과가 비슷하지만 일반적인 경우 Q-learning의 학습 효율이 더욱 좋은 것으로 알려져 있다. 
TD Control의 경우에도 한계가 존재하는데 이는 Discrete한 State에만 적용이 가능한 알고리즘이라는 것이다. 
따라서, State의 개수가 매우 크게 증가하거나 Continuous State 환경인 경우에는 TD Control을 적용할 수 없다는 한계가 있다.
이를 보완하기 위해, TD Control과 Neural Network (NN)을 결합한 DQN (Deep Q Network) 알고리즘이 도입되었다. 
DQN은 TD Control의 형태와 구현이 매우 유사하기 때문에 TD Control을 이해하였다면 쉽게 구현이 가능하다.
다음 포스팅에서는 이러한 Continuous State를 핸들링하기 위한 강화학습 알고리즘에 대해 다룬다.

## Reference
[Frozen Lake Description](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/#frozen-lake)
