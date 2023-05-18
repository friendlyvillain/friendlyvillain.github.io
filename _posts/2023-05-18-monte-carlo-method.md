---
title: Monte Carlo Method
author: jh
date: 2023-05-18 18:07:37 +0900
categories: [Machine Learning, Reinforcement Learning]
tags: [ML, RL, Monte Carlo, Model-Free, MDP, Optimal Policy, State-Value Function, Bellman Equation, Policy Evaluation, Policy Improvement, Policy Control]
math: true
mermaid: true
comments: true
---

## Introduction
Dynamic Programming (DP)는 Bellman Eq.를 풀기 위해 **Model의 Dynamics**, 다시 말하면, MDP의 **State transition Probabilty** 를 알고 있어야 적용이 가능한 방법이다.
그러나 대부분 접하게되는 MDP 문제에서는 State Transition Probability를 정확하게 알 수 없다.
현재까지는 State와 Action이 모두 Discrete한 domain의 경우만 다루었지만 State 또는 Action이 Continuous한 domain의 경우에는 당연히 State Transition Probabilty를 알 수 없다.
뿐만 아니라, State와 Action이 모두 Discrete한 domain의 경우에도 바둑이나 블랙잭 처럼 State와 Action의 복잡도에 따라 Dynamics를 사실상 구할 수 없는 경우가 있다. 
단순한 예시로는 앞선 포스팅에서 DP의 예시인 다루었던 Grid World에서 Dynamics를 추가한 경우이다. 
Grid World에서는 Action (상,하,좌,우) 에 따라 다음 State가 Action의 방향으로 이동하도록 Discrete하게 고정되었는데 만약 Action 대로 움직이지 않을 확률이 존재하여 다음 State가 무엇인지 특정할 수 없다면? 
여기에 추가로 State에도 랜덤성을 부여하여 Action에 따라 어떤 방향으로 이동할 경우 1칸 보다 많이 이동할 확률이 존재하여 다음 State가 무엇인지 특정할 수 없다면?
단순하게 Action과 State에 랜덤성을 추가되는 것만으로도 Model의 Dynamics가 매우 복잡해지게 된다. 
Action에만 랜덤성을 대표적인 MDP의 예시는 [OpenAI Gym](https://www.gymlibrary.dev/)에서 제공하는 Frozen Lake의 사례가 있다. 
