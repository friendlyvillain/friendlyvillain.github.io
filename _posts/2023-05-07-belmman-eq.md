---
title: Bellman Equation & Optimal Policy
author: jh
date: 2023-05-07 14:07:25 +0900
categories: [Machine Learning, Reinforcement Learning]
tags: [ML, RL, MDP, State-Value Function, Action-Value Function, Q-function, Bellman Equation]
math: true
mermaid: true
comments: true
---

## 1. Bellman Equation
특정 시점 $t$에서의 Value Function을 다음 시점 $t+1$에서의 Value Function과의 관계식으로 표현할 수 있는 수식을 Bellman Equation이라 한다. 
앞선 [MDP 포스트](https://friendlyvillain.github.io/posts/mdp/#relationship-between-state-value-function-and-action-value-function) 에서 도출한 State-Value function과 Action-Value function (Q-function) 사이의 다음 관계식을 참고하면 각각의 Value Function 또한 Bellman Equation을 따르는 관계식으로 표현할 수 있다. 


$$ \ v_{\pi}(s) = \sum_{a} \pi(a|s)q_{\pi}(s, a) $$

$$ \ q_{\pi}(s, a) = \sum_{s'}\sum_{r}p(s', r|s, a) \left\{ r + \gamma v_{\pi}(s') \right\}$$


### 1-1. State-Value Function as Bellman Eq

$$ \ v_{\pi}(s) = \sum_{a} \pi(a|s)q_{\pi}(s, a) = \sum_{a} \pi(a|s)\sum_{s'}\sum_{r}p(s', r|s, a) \left\{ r + \gamma v_{\pi}(s') \right\}$$


### 1-2. Action-Value Function as Bellman Eq

$$ \ q_{\pi}(s, a) = \sum_{s'}\sum_{r}p(s', r|s, a) \left\{ r + \gamma v_{\pi}(s') \right\} = \sum_{s'}\sum_{r}p(s', r|s, a) \left\{ r + \gamma \sum_{a'} \pi(a'|s')q_{\pi}(s', a') \right\}$$


## 2. Optimal Policy
State-Value Function 관점에서 최적 Policy는 어떤 State $s$ 부터 누적 보상값을 최대로하는 정책 $\pi^{\*}$ 를 따르는 경우, State-Value Function을 의미한다. Action-Value Function 관점에서 최적 Policy는 어떤 State $s$ 에서 Action $a$를 수행한 이후의 다음 State부터 누적 보상값을 최대로하는 정책 $ \pi^{\*} $ 를 따르는 경우, Action-Value Function을 의미한다. 2개의 Value Function의 차이는 State $s$부터 항상 최적의 Policy를 따르는가 (**Optimal State-Value Function**) 아니면 State $s$에서 한번은 임의의 Action의 수행을 허용하는가 (**Optimal Action-Value Function**)에 있다. 

Agent가 Policy $\pi^{\*}$를 따르는 경우, 오직 최적 action을 수행하므로 $\pi^{\*}$는 아래와 같이 표현할 수 있다.

$$ \pi^{*}(a|s) = 1 \ \text{if } a \ \text{is an } \textbf{optimal } \text{action, } \text{else} \ 0 $$


### 2-1. Optimal State-Value Function


$$ v_{*}(s) \triangleq \mathbb{E}_{\pi^{*}} [G_t | S_t=s] = \max_{\pi}v_{\pi}(s) \ where \ \forall{s} \in \mathcal{S} $$

- Optimal State-Value Function as Bellman Eq


Optimal Policy를 따르는 경우 특정 State 에서 Action은 더이상 확률을 따르지 않고, 항상 해당 State에서 리턴값을 최대로 하는 Action을 수행하므로, $ \pi (a\|s) $ 부분이 대체되어 Optimal State-Value Function은 다음과 같은 Bellman Eq로 표현할 수 있다.

$$ \ v_{*}(s) = \max_{a}\sum_{s'}\sum_{r}p(s', r|s, a) \left\{ r + \gamma v_{*}(s') \right\}$$


### 2-2. Optimal Action-Value Function

$$ q_{*}(s, a) \triangleq \max_{\pi}q_{\pi}(s, a) \ where \ \forall{s} \in \mathcal{S}, \forall{a} \in \mathcal{A} $$

- Optimal Action-Value Function as Bellman Eq

Optimal State-Value Function 의 경우와 유사하게 $ \pi(a'\|s') $ 부분이 대체되어 Optimal Action-Value Function은 다음과 같은 Bellman Eq로 표현할 수 있다.

$$ \ q_{*}(s, a) = \sum_{s'}\sum_{r}p(s', r|s, a) \left\{ r + \gamma \max_{a'}q_{*}(s', a') \right\}$$

### 2-3. Relationship between State-Value Function and Action Value Function

2개의 Value Function 사이의 관계식은 다음과 같이 표현할 수 있다. 

$$ \ v_{*}(s) = \max_{a}q_{*}(s, a) $$

## 3. RL in Bellman Equation

강화학습 문제에서 누적 보상을 최대로하는 Policy를 도출하는 과정은 결국 Bellman Equation을 푸는 과정이라고 할 수 있다. 
이 때, 크게 Agent가 **MDP의 Dynamics**를 알고 있는가 (**Model-based**)에 모르고 있는가 (**Model-free**)에 따라 Dynamic Programming 기법과 Monte Carlo 기법으로 풀이 방법이 나뉜다. 

## 4. Reference
[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
