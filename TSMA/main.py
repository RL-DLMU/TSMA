from runner import Runner
from common.arguments import get_args
from common.utils import make_env
from common.utils import make_env
from common.sisl.environment import get_sisl_envs
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':



    args = get_args()
    if args.scenario_name[:6] == 'simple':
        env, args = make_env(args)
    else:
        env, args = get_sisl_envs(args)

    runner = Runner(args, env)
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()
