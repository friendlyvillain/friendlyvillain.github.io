---
title: Monte Carlo Method Example - Frozen Lake
author: jh
date: 2023-05-24 18:36:29 +0900
categories: [Machine Learning, Reinforcement Learning]
tags: [ML, RL, Monte Carlo, Python, Model-Free, Frozen Lake, MDP, Optimal Policy, Action-Value Function, Bellman Equation, Policy Evaluation, Policy Improvement, Policy Control]
math: true
mermaid: true
comments: true
---

## Introduction

본 포스팅에서는 Python 코드를 통해, Monte Carlo (MC) Method 알고리즘을 구현하는 내용을 다룬다. 
앞선 포스팅에서 다룬 [MC Control](https://friendlyvillain.github.io/posts/monte-carlo-method/#mc-policy-control) 알고리즘을 이용하여 OpenAI Gym의 Frozen Lake 환경에 대한 Value Function과 Optimal Policy를 도출한다.
코드 구현을 위해 필요한 Python Package는 Gym, Numpy, Matplotlib 이다. 


## MDP Environment

[Frozen Lake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/#frozen-lake)의 MDP 환경은 다음과 같은 4X4 Grid World와 유사하며, 총 16개의 Grid 중 4개는 Hole이 있는 형태이다. 

![frozen-lake-env](/assets/img/posts/mc_example/frozen_lake_env.png){: width="300" height="300" }
_Frozen Lake_

위와 같은 MDP 환경에서 Agent의 목표는 Hole (State 5, 7, 11, 12)에 빠지지 않고, 시작 지점 (State 0) 부터 목표 지점 (State 15)까지 가는 것이다. 