# TSMA

This is a pytorch implementation of TSMA on [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs), the corresponding paper of TSMA is [Guiding Team Objectives with Individual Policies: A Two-Stage Model Aggregation Framework for Partially Cooperative MARL].

## Requirements

- python=3.7
- [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs)
- torch=1.1.0

## Quick Start

```shell
$ python main.py --scenario-name=simple_tag --evaluate-episodes=100
```

Directly run the main.py, then the algrithm will be tested on scenario 'simple_tag' for 100 episodes, using the pretrained model.

## Note

+ We have train the agent on scenario 'simple_tag', but the model we provide is not the best because we don't want to waste time on training, you can keep training it for better performence.

+ There are 4 agents in simple_tag, including 3 predators and 1 prey. we use MADDPG to train predators to catch the prey. The prey's action can be controlled by you, in our case we set it random. 

+ The default setting of Multi-Agent Particle Environment(MPE) is sparse reward, you can change it to dense reward by replacing 'shape=False' to 'shape=True' in file multiagent-particle-envs/multiagent/scenarios/simple_tag.py/.
