from tqdm import tqdm
# from agent import Agent
# from agent_mappo import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import datetime
from collections import deque
import matplotlib.pyplot as plt
from agents.IPPO import IPPOAgent_td
from agents.MAPPO_irat_multiactors_widv import irat_mulAgent_td_widv
from agents.MADDPG_irat_multiactors_widv import irat_mulAgent_pg

# device = torch.device("cuda",1) if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.buffer = Buffer(args)
        self.agents = self._init_agents()
        self.log_file = self.args.log_dir + '/' + args.method + '/' + self.args.scenario_name

        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y_%m_%d-%H_%M_%S")
        file_name = f"{formatted_time}_DTL.log"
        if not os.path.exists(self.log_file):
            os.makedirs(self.log_file)
        self.file_path = os.path.join(self.log_file, file_name)

        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        sub_agents = self.args.n_agents
        for i in range(int(self.args.n_agents / sub_agents)):
            if self.args.method == "irat_multi_widv": # tsma-mappo
                agents.append(irat_mulAgent_td_widv(i, self.args))
            if self.args.method == "pgirat_multi_widv":  # tsma-maddpg
                agents.append(irat_mulAgent_pg(i, self.args))
        return agents

    def run(self):
        # print("MAPPO,团队奖励训练")
        print(info_print)
        e = 0
        returns = []
        returns_i = []
        returns_prey = []
        returns_team = []
        cur_loss_q_list = []
        cur_loss_a_list = []
        drift_idv_list = []
        drift_team_list = []
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0 or done[0]:
                cur_loss_q_sublist = []
                cur_loss_a_sublist = []
                last_observations = self.env.reset()
                lacal_last_observations = last_observations
                buffer_cur = []
                return_i = 0
                return_prey = 0
                return_team = 0
                if self.args.method[:10] == "irat_multi" or self.args.method[:12] == "pgirat_multi":
                    rnn_state = [None for i in range(self.agents[0].idv_num)]
                else:
                    rnn_state = None
                rnn_state_team = None

            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    if self.args.method == "MAPPO":
                        action, act_log_probs, rnn_state, w = agent.get_action(last_observations, rnn_state, test=False)
                    if self.args.method[:4] == "irat":
                        action, act_log_probs, rnn_state, act_log_probs_team, rnn_state_team = agent.get_action(last_observations, rnn_state, rnn_state_team, test=False)
                    if self.args.method[:4] == "ddpg":
                        if rnn_state_team == None:
                            last_rnn_state = torch.zeros([agent.sub_agents, 128])
                        else:
                            last_rnn_state = rnn_state
                        action, action_soft, rnn_state = agent.get_action(last_observations, last_rnn_state, test=False)
                    if self.args.method[:4] == "pgir":
                        if rnn_state_team == None:
                            last_rnn_state = torch.zeros([agent.idv_num, agent.sub_agents, 128]).to(device)
                            last_rnn_team_state = torch.zeros([agent.sub_agents, 128]).to(device)
                        else:
                            last_rnn_state = rnn_state
                            last_rnn_team_state = rnn_state_team
                        action, action_soft, rnn_state, rnn_state_team = agent.get_action(last_observations, last_rnn_state, last_rnn_team_state, test=False)
                    actions.append(action)

            observations, rewards, rewards_prey, rewards_team, done, info = self.env.step(np.array(actions).flatten())
            if len(set(rewards_team)) != 1:
                aaa =0
            local_observations = observations
            return_i += np.mean(rewards)
            # if rewards_prey:
            try:
                if rewards_prey:
                    return_prey += rewards_prey[0]
            except:
                return_prey += rewards_prey
            return_team += rewards_team[0]
            # return_i.append(rewards[0])
            # return_team.append(rewards_team[0])
            if self.args.method == "MAPPO":
                for idx, ag in enumerate(self.agents):
                    buffer_cur.append((last_observations, actions[idx], actions[idx], rewards_team, observations, act_log_probs, w))
            if self.args.method[:4] == "pgir":
                for idx, ag in enumerate(self.agents):
                    ag.replay_buffer.append((lacal_last_observations, rnn_state_team, action_soft, rewards, local_observations, last_rnn_state, rewards_team, last_rnn_team_state, last_observations, observations, rnn_state))
            if self.args.method[:4] == "ddpg":
                for idx, ag in enumerate(self.agents):
                    ag.replay_buffer.append((rnn_state, action_soft, rewards, last_rnn_state, last_observations, observations))
            lacal_last_observations = local_observations
            last_observations = observations

            if (self.args.method[:4] == "pgir" or self.args.method == "ddpg") and time_step > 5000:
                for idx, ag in enumerate(self.agents):
                    cur_loss_q, cur_loss_a = ag.train(buffer_cur)
                    cur_loss_q_sublist.append(cur_loss_q)
                    cur_loss_a_sublist.append(cur_loss_a)



            if (time_step+1) % self.episode_limit == 0 or done[0]:
                e += 1
                for idx, ag in enumerate(self.agents):
                    if self.args.method[:4] != "pgir" and self.args.method != "ddpg":
                        if self.args.method[:4] == "irat":
                            cur_loss_q, cur_loss_a, drift_idv, drift_team = ag.train(buffer_cur)
                            drift_idv_list.append(drift_idv)
                            drift_team_list.append(drift_team)
                        else:
                            cur_loss_q, cur_loss_a = ag.train(buffer_cur)
                            drift_idv_list.append(0)
                            drift_team_list.append(0)
                        cur_loss_q_list.append(cur_loss_q)
                        cur_loss_a_list.append(cur_loss_a)
                    else:
                        drift_idv_list.append(0)
                        drift_team_list.append(0)
                        if cur_loss_q_sublist:
                            cur_loss_q_list.append(sum(cur_loss_q_sublist) / self.episode_limit)
                            cur_loss_a_list.append(sum(cur_loss_a_sublist) / self.episode_limit)
                        else:
                            cur_loss_q_list.append(0)
                            cur_loss_a_list.append(0)
                    returns_i.append(return_i/self.episode_limit)
                    returns_prey.append(return_prey)
                    returns_team.append(return_team)
                if e == 1:
                    self.writeLog(self.args.method, e, returns_i, returns_prey, cur_loss_a_list, returns_team, drift_idv_list, drift_team_list, info_print)
                if e % 50 == 0:
                    self.writeLog(self.args.method, e, returns_i, returns_prey, cur_loss_a_list, returns_team, drift_idv_list, drift_team_list)


    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes

    def writeLog(self, name, step, return_i, return_prey, cur_loss_a_list, return_team, drift_idv_list, drift_team_list, info=None):
        log_handle = open(self.file_path, "a")
        res = ''
        if info != None:
            res += (info + "\n")
        else:
            num = 50
            for i in range(num):
                idx = i-num
                res += str(name) + '\t' + str(step-num+1+i) + '\t' + "%.4f" % return_i[idx] + '\t' + "%.1f" % return_prey[idx] + '\t' + "%.1f" % cur_loss_a_list[idx] + '\t' + "%.1f" % return_team[idx] + '\t' + "%.1f" % drift_idv_list[idx] + '\t' + "%.1f" % drift_team_list[idx] + "\n"
        log_handle.write(res)
        log_handle.close()