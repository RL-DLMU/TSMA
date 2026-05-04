# TSMA

This is a pytorch implementation of TSMA on [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs), the corresponding paper of TSMA is [Guiding Team Objectives with Individual Policies: A Two-Stage Model Aggregation Framework for Partially Cooperative MARL].

## What Problem Does TSMA Solve?

In Partially Cooperative Markov Games (PCMG), each agent receives both **individual rewards** and **team rewards** . This creates a fundamental tension:

| Approach | Limitation |
|----------|-------------|
| **Mixed reward** (weighted sum) | Requires heavy reward tuning, limited generalizability |
| **Policy distillation** (e.g., IRAT) | Data-dependent, cannot capture policy info outside distillation data distribution |

TSMA uses **structure-level model aggregation** to bypass data dependency, achieving stable training and strong cooperative performance.

---

## Method Overview

TSMA is a two-stage model aggregation framework that uses individual policies to guide team policy learning. It is worth noting that: TSMA is a **universal framework** compatible with **all standard Actor-Critic (AC) algorithms** (e.g., MAPPO, MADDPG), requiring no modification to the base algorithm's structure.
The figure below illustrates the training and aggregation pipeline of a TSMA agent.
<img width="1126" height="690" alt="image" src="https://github.com/user-attachments/assets/f50a6d3c-68a7-4039-a399-b5fe0458d2a5" />

### Stage 1: Individual Policy Aggregation

- Each agent trains **two individual policies** on the same individual reward
- **Adaptive weighted aggregation** based on policy loss


### Stage 2: Individual-Team Policy Aggregation

- Aggregated individual policy merges with team policy
- **Annealing weight** gradually shifts importance from individual to team policy

---

## Requirements

- python=3.7
- [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs)
- torch=1.13.1

## Quick Start

Directly run the main.py, then the algrithm will be trained on scenario 'simple_tag' for 100 episodes.

