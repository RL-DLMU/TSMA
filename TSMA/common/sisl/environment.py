from sisl.walker.multi_walker import MultiWalkerEnv
from sisl.pursuit.pursuit_evade import PursuitEvade
# from common.multiagent.sisl.pursuit.waterworld import MAWaterWorld
from sisl.pursuit.pursuit_config import convert_puisuit


def get_sisl_envs(args):
    if args.scenario_name == "MultiWalker":
        env = MultiWalkerEnv()
        args.n_players = env.n_walkers  # 包含敌人的所有玩家个数
        args.n_agents = env.n_walkers - args.num_adversaries  # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
        args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # 每一维代表该agent的obs维度
        action_shape = []
        for content in env.action_space:
            # action_shape.append(content.n)
            action_shape.append(6)
        args.action_shape = action_shape[:args.n_agents]  # 每一维代表该agent的act维度
        args.high_action = 1
        args.low_action = -1
    elif args.scenario_name[:7] == "Pursuit":
        if args.scenario_name == "Pursuit":
            env = PursuitEvade()
        else:
            env = PursuitEvade(x_size=40, y_size=40, large=True)
        args.n_players = env.n_evaders + env.n_pursuers  # 包含敌人的所有玩家个数
        args.n_agents = env.n_pursuers  # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
        args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # 每一维代表该agent的obs维度
        action_shape = []
        for content in env.action_space:
            action_shape.append(content.n)
            # action_shape.append(2)
        args.action_shape = action_shape[:args.n_agents]  # 每一维代表该agent的act维度
        args.high_action = 1
        args.low_action = -1

    # elif args.scenario_name == "Pursuit":
    #     config_dict = convert_puisuit(args)
    #     env = PursuitEvade(x_size=args.x_size, y_size=args.y_size, **config_dict)
    # elif config.env_name == "WaterWorld":
    #     env = MAWaterWorld(n_pursuers=config.n_pursuers, n_evaders=config.n_evaders, n_coop=config.n_coop,
    #                        n_poison=config.n_poison, radius=config.radius, obstacle_radius=config.obstacle_radius,
    #                        obstacle_loc=config.obstacle_loc, ev_speed=config.ev_speed, poison_speed=config.poison_speed,
    #                        n_sensors=config.n_sensors, sensor_range=config.sensor_range, action_scale=config.action_scale,
    #                        poison_reward=config.poison_reward, food_reward=config.food_reward,
    #                        encounter_reward=config.encounter_reward, control_penalty=config.control_penalty,
    #                        reward_mech=config.reward_mech, addid=config.addid, speed_features=config.speed_features,
    #                        max_cycles=config.max_cycles, idv_use_caught_food=config.idv_use_caught_food)
    else:
        print("Can not support the " +
              args.env_name + "environment.")
        raise NotImplementedError
    # env = MultiAgentEnv(world)

    return env, args
