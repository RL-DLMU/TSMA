import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        # set any world properties first
        world.world_length = args.max_episode_len
        world.cur_steps = 0
        world.dim_c = 2
        num_good_agents = 2  #特工，速度更快
        num_adversaries = 5  #对手速度更慢，想打败特工
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.075
            agent.accel = 3.0 if agent.adversary else 3.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
            if i < num_adversaries:
                agent.action_callback = None  #受控的智能体不需要在这儿获得策略
            else:
                agent.action_callback = self.random_policy  #特工用随机策略
        # world.agents[-1].accel = 3.0
        # world.agents[-1].max_speed = 1.0
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2 # 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def random_policy(self, agent, world):
        if agent.movable:
            agent.action.u = (np.random.random(world.dim_p) * 2 - 1)
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity

        agent.action.c = np.zeros(world.dim_c)
        return agent.action

    def reset_world(self, world):
        world.cur_steps = 0
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
        # world.agents[-1].color = np.array([0.25, 0.25, 0.25])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def individual_reward(self, adv, world):
        rew = 0
        r = 0
        agents = self.good_agents(world)
        dists = [np.sqrt(np.sum(np.square(adv.state.p_pos - a.state.p_pos)))
                 for a in agents]
        rew -= min(dists)
        for idx, a in enumerate(agents):
            if self.is_collision(a, adv):
                if idx==1:
                    rew += 5.
                    r = 5.
                else:
                    rew += 5.
                    r = 5.
                # rew += 5.
                # r = 5.
        # def bound(x):
        #     if x < 1.75:
        #         return 0
        #     else:
        #         return min(np.exp(x - 1.75), 3)
        # if world.rew_bound:  #应该是超出边界的惩罚吧
        #     for p in range(world.dim_p):
        #         x = abs(adv.state.p_pos[p])
        #         rew -= bound(x)
        return (rew, r)

    # def team_reward(self, world):
    #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal.state.p_pos)))
    #              for a in world.agents]
    #     dists = np.array(dists)
    #     flag = (dists < 0.5).all()
    #     rew = 10 if flag else -1
    #     return rew
    def team_reward(self, world):
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        rew = 0
        world.cur_steps += 1
        for idx, a in enumerate(agents):
            n = 0
            for adv in adversaries:
                if self.is_collision(a, adv):
                    n += 1
            if n >= 2 :
                if idx==1:
                    rew += 99
                else:
                    rew += 20
        # if world.cur_steps < world.world_length:
        #     rew = 0
        # else:
        #     n_visited = 0
        #     for ld in world.landmarks:
        #         if ld.visited:
        #             n_visited += 1
        #     rew = n_visited * 1.0 / len(world.landmarks)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_good_pos = []
        other_adversaries_pos = []
        other_good_vel = []
        other_adversaries_vel = []
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        for other in agents:
            comm.append(other.state.c)
            other_good_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_good_vel.append(other.state.p_vel)
        for other in adversaries:
            if other is agent: continue
            comm.append(other.state.c)
            other_adversaries_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_adversaries_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_good_pos + other_good_vel + other_adversaries_pos + other_adversaries_vel)

    # def observation(self, agent, world):
    #     # get positions of all entities in this agent's reference frame
    #     entity_pos = []
    #     for entity in world.landmarks:
    #         if not entity.boundary:
    #             entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    #     # communication of all other agents
    #     comm = []
    #     other_pos = []
    #     other_vel = []
    #     for other in world.agents:
    #         if other is agent: continue
    #         comm.append(other.state.c)
    #         other_pos.append(other.state.p_pos - agent.state.p_pos)
    #         if not other.adversary:
    #             other_vel.append(other.state.p_vel)
    #     return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
