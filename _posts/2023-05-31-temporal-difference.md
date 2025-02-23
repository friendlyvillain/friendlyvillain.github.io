---
title: Temporal Difference
author: jh
date: 2023-05-31 20:25:56 +0900
categories: [Machine Learning, Reinforcement Learning]
tags: [ML, RL, Temporal-Difference, Model-Free, MDP, Optimal Policy, Action-Value Function, Q-function, Bellman Equation, On-policy, Off-policy, SARSA, Q-learning]
math: true
mermaid: true
comments: true
---

## 1. Introduction

지금까지 MDP 문제를 풀기 위한 방법으로 [Dynamic Programming (DP)](https://friendlyvillain.github.io/posts/dynamic-programming/)과 [Monte-Carlo Method (MC)](https://friendlyvillain.github.io/posts/monte-carlo-method/)에 대해 다루었다. 
DP는 time-step 마다 가치함수를 업데이트 할 수 있지만, Model-Based 한 환경에만 적용할 수 있다는 한계가 있다. 
반면, MC는 Model-Free한 환경에 대해서도 Experience Sampling을 통해 MDP 문제를 풀 수 있지만 Model을 업데이트 하기 위해 Sampling된 Episode가 반드시 종료되어야 한다는 한계가 있다. 
Temporal-Difference (TD)는 DP와 MC의 장점을 모두 결합하여 Model-Free한 환경에 대해서도 Expereience Sampling을 통해 MDP 문제를 풀 수 있고, Episode가 종료되지 않더라도 Model을 업데이트 할 수 있다는 특징이 있다. 
TD Control 알고리즘의 기본적인 형태와 원리는 DNN과 결합하여 발전된 강화학습 알고리즘인 DQN, Double DQN 등의 형태와 매우 유사하며, Q-function을 어떻게 도출하는냐에 대한 차이만 존재한다. 


## 2. TD Control Algorithm

MC와 마찬가지로 Model-Free한 환경에 적용하기 위해 가치함수로 Q-function을 사용한다. 
TD는 bootstrap을 통해 Episode가 종료되지 않더라도 Model을 업데이트 할 수 있고, 본 포스팅에서는 가장 기본적인 형태인 **매 time-step마다 Model을 업데이트** 하는 TD control Algorithm을 다룬다. 
Q-function을 업데이트 하는 방법에 따라 on-policy TD Control Algorithm을 [**SARSA**](#sarsa-on-policy-td-control-algorithm)라 부르고, off-policy TD Control Algorithm을 [**Q-learning**](#q-learning-off-policy-td-control-algorithm)이라 부른다. 


### 2-1. On-policy vs Off-policy
Agent가 직접 Action을 수행하는 policy를 **Behavior policy**라 하고, 가치를 evaluate하고 improve 하기 위한 policy를 **Target policy**라 한다.
다시 말해, Behavior policy는 MDP 환경과 상호작용하는 정책이며, 해당 환경에서 학습의 대상이 되는 정책을 Target policy라 할 수 있다.
기호로 Behavior Policy와 Target Policy를 다음과 같이 표현한다. 

- Behavior Policy: $ b(a\|s) $
- Target Policy: $ \pi(a\|s) $

Behavior policy와 Target policy가 동일한 경우 ($ b(a|s)=\pi(a|s) $), On-policy control이라 부른다.
반대로 Behavior Policy와 Target Policy가 다른 경우 ($ b(a|s) \neq \pi(a|s) $), Off-policy control이라 부른다. 
On-policy control의 경우, Q-function을 업데이트 할 때, Agent가 Next State에서 수행하는 Policy (Target Policy)는 현재 Policy (Behavior Policy)를 그대로 따른다.
반면, Off-policy control의 경우, Q-function을 업데이트 할 때, Agent가 Next State에서 수행하는 Policy (Target Policy)는 현재 Policy (Behavior Policy)를 그대로 따르지 않는다. (e.g., Greedy Action)
On-policy control의 경우 local optimal에 빠질 가능성이 있는 반면 수렴이 상대적으로 빠르다.
반대로, Off-policy control의 경우 local optimal에 빠지지 않지만 variance가 크기 때문에 수렴이 상대적으로 느리다.
글로 이해가 되지 않는다면 본 포스팅에서 각각의 TD Control Algorithm에 대해 Q-function 업데이트를 위한 수식을 보면 쉽게 이해가 가능할 것이다.


### 2-2. SARSA: On-Policy TD Control Algorithm

SARSA Algorithm은 매 time step $t$ 마다 State ($S_t$)에서 Behavior Policy를 따르는 Action ($A_t$)을 수행한 이후, 그에 따른 Reward ($R_{t+1}$)를 관찰한다.
$S_t$에서 $A_t$를 수행한 이후, 얻어진 Next State ($S_{t+1}$)에서 **Behavior Policy를 따르는** Action ($A_{t+1}$)을 관찰한 이후, 다음 수식을 통해 Q-function을 업데이트 한다. 

$$

Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]

$$

SARSA라는 이름이 붙은 이유는 State-Action-Reward-(Next)State-(Next)Action 을 관찰하여 Q-function을 업데이트 하기 때문이다. 
 
 - Pseudo Code

 SARSA 알고리즘을 Pseudo Code로 나타내면 다음과 같다. 

![sarsa-algorithm](/assets/img/posts/td/sarsa_algo.png){: width="600" height="500" }
_SARSA Algorithm Pseudo Code_

### 2-3. Q-learning: Off-Policy TD Control Algorithm

Q-learning Algorithm은 SARSA와 그 형태가 매우 유사하다. 
SARSA 알고리즘과 마찬가지로 매 time step $t$ 마다 State ($S_t$)에서 Behavior Policy를 따르는 Action ($A_t$)을 수행한 이후, 그에 따른 Reward ($R_{t+1}$)를 관찰한다.
차이점은 $S_t$에서 $A_t$를 수행한 이후, 얻어진 Next State ($S_{t+1}$)에서 Behavior Policy를 따르는 것이 아니라 **Target Policy (Greedy Policy)를 따르는** Action을 관찰한 이후, 다음 수식을 통해 Q-function을 업데이트 한다.

$$

Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'}Q(S_{t+1}, a') - Q(S_t, A_t) \right]

$$

위의 수식에서 확인할 수 있듯이, Q-learning과 SARSA는 $S_{t+1}$ 에서 관찰하는 Action 과 관련된 내용을 제외하면 완벽하게 동일한 것을 알 수 있다. 

 - Pseudo Code

 Q-learning 알고리즘을 Pseudo Code로 나타내면 다음과 같다. 

![q-learning-algorithm](/assets/img/posts/td/q_learning_algo.png){: width="600" height="500" }
_Q-learning Algorithm Pseudo Code_


## 3. Conclusion
TD Contorl은 DP와 MC의 장점을 결합하여 Model Free한 MDP 환경에서 Episode가 종료되지 않더라도 Model을 업데이트 할 수 있다는 특징이 있다. 
TD Control 알고리즘은 향후 발전된 강화학습의 알고리즘의 형태와 원리가 매우 유사하다.
[다음 포스팅](https://friendlyvillain.github.io/posts/td-example/)에서는 Python 기반으로 Frozen Lake 환경에서 TD Control 알고리즘을 구현하는 예제를 다룬다.


## 4. Reference
[Reinforcement Learning: An Introduction](https://incompleteideas.net/book/the-book.html)
