import numpy as np
# from pyibl import Agent
import random as random
import itertools
import torch
from math import exp
import math
import sys
import torch.nn.functional as F
from collections import OrderedDict, deque
from torch import nn
from collections import deque
from itertools import count
# from speedyibl import Agent
from torch_geometric.data import Data, Batch

# device = torch.device("cuda",3) if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")


class IPPOAgent_td():

    def __init__(self, agent_id, args):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.ob_length = args.obs_shape[agent_id]
        self.buffer_size = 5000
        self.gamma = 0.9
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.phase = False
        self.one_hot = False
        # self.ob_length = config.dim[0]*config.dim[1]
        # self.ob_length = 3*7
        self.action_space = args.action_shape[agent_id]
        self.eps = 0.2
        self.lmbda = 0.95
        # self.action_space = gym.spaces.Discrete(len(self.world.intersections[0].phases))
        self.sub_agents = args.n_agents
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

        self.actor = Actor(self.ob_length, self.action_space).to(device)
        # if self.load:
        # 	self.load_model(200)
        self.criterion = nn.MSELoss(reduction='mean')
        self.critic_optimizer = torch.optim.Adam(itertools.chain(self.model.parameters()), lr=self.learning_rate,
                                                 eps=1e-5)
        self.actor_optimizer = torch.optim.Adam(itertools.chain(self.actor.parameters()), lr=self.learning_rate,
                                                eps=1e-5)

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def _build_model(self):
        model = ColightNet(self.ob_length, self.action_space, self.sub_agents).to(device)
        return model

    def remember(self, last_obs, last_phase, actions, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))

    def _batchwise(self, samples):
        # load onto tensor

        batch_list = []
        batch_list_p = []
        actions = []
        rewards = []
        act_log_prob = torch.empty((0), dtype=torch.float32).to(device)
        for item in samples:
            dp = item
            act_log_prob = torch.cat((act_log_prob, torch.tensor(dp[5]).to(device)), 0)
            state = torch.tensor(dp[0], dtype=torch.float32).to(device)  # 获取观测信息
            batch_list.append(Data(x=state, edge_index='aaa'))

            state_p = torch.tensor(dp[4], dtype=torch.float32).to(device)  # 下一观测信息
            batch_list_p.append(Data(x=state_p, edge_index='aaa'))
            rewards.append(dp[3])
            actions.append(dp[2])
        batch_t = Batch.from_data_list(batch_list)
        batch_tp = Batch.from_data_list(batch_list_p)
        # TODO reshape slow warning
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(device)
        if self.sub_agents > 1:
            rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
            actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here
        # rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
        # actions = torch.tensor(np.array(actions), dtype=torch.long)
        # actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here

        return batch_t, batch_tp, rewards, actions, act_log_prob

    # def generate_outcomes(self, y, x, holding):
    # 	self.outcomes[(y, x, holding)] = [self.default_utility]*self.c.outputs

    # def move(self, y, x, holding, explore=True):
    # 	# '''
    # 	# Returns an action from the ibl agent instance.
    # 	# '''
    # 	holding = holding*self.goods
    # 	self.t += 1
    # 	# if (s_hash) not in self.options:
    # 	# 	self.generate_options(s_hash)
    # 	if self.episodeCounter > self.__ep:
    # 		self.epsilon = max((self.epsilon * self.c.eps.discount), 0)
    # 	if self.episodeCounter > self.__ep:
    # 		self.__ep += 1
    # 	# if explore and random.random() < self.__epsilon:
    # 	# 	self.drl.action = random.randrange(self.c.outputs)
    # 	# 	options = [{"action": self.drl.action, "state": s_hash}]
    # 	# else:
    # 	if (y, x, holding) not in self.outcomes:
    # 		self.last_action = random.randrange(self.c.outputs)
    # 		self.generate_outcomes(y, x, holding)
    # 	elif explore and random.random() < self.epsilon:
    # 		self.last_action = self.boltzchoose(y, x, holding)
    # 	else:
    # 		self.last_action = self.choose_td(y, x, holding)
    #
    # 	self.option = (y, x, holding, self.last_action)
    #
    # 	self.x = x
    # 	self.y = y
    # 	self.holding = holding
    #
    # 	return self.last_action

    def get_action(self, ob, rnn_state, test=False):
        """
        input are np.array here
        # TODO: support irregular input in the future
        :param ob: [agents, ob_length] -> [batch, agents, ob_length]
        :param phase: [agents] -> [batch, agents]
        :param test: boolean, exploit while training and determined while testing
        :return: [batch, agents] -> action taken by environment
        """
        # if not test:
        #     if np.random.rand() <= self.epsilon:
        #         return self.sample()
        observation = torch.tensor(ob, dtype=torch.float32).to(device)
        # edge = self.edge_idx
        # TODO: not phase not used
        # if rnn_state is not None:
        #     rnn_state = rnn_state.to(device)  # cuda
        action_dist, rnn_state = self.actor(observation, rnn_state)
        action = action_dist.sample()
        action_log_probs = action_dist.log_prob(action).to(device)
        return action.view(-1).cpu().clone().numpy(), action_log_probs.view(
            -1).cpu().clone().detach().numpy(), rnn_state, None

    def train(self, transition_buffer):
        # samples = random.sample(self.replay_buffer, self.batch_size)
        # b_t, b_tp, rewards, actions = self._batchwise(samples)
        action_space = self.action_space
        max_episode_len = len(transition_buffer)
        b_t, b_tp, rewards, actions, act_log_prob = self._batchwise(transition_buffer)
        # act_log_prob = torch.tensor(np.array(act_log_prob), dtype=torch.float32).to(device)
        obs = b_t.x
        obs = obs.view(max_episode_len, self.sub_agents, -1)
        obs_next = b_tp.x
        out_next = self.model(x=b_tp.x, edge_index='aaa', train=False)
        target = rewards.view(-1, 1) + self.gamma * out_next.view(-1, 1)

        out = self.model(x=b_t.x, edge_index='aaa', train=True)
        # td_error = target - out
        # advantage = compute_advantage(self.gamma, self.lmbda, td_error.cpu()).to(device)  # gae
        td_error = (target - out).view(max_episode_len, self.sub_agents, -1)
        td_error = td_error.permute(1, 0, 2)
        adv = torch.empty((0), dtype=torch.float32).to(device)
        for item in td_error:  # 每个交叉口的gae
            advantage = compute_advantage(self.gamma, self.lmbda, item.cpu()).to(device)  # gae
            adv = torch.cat((adv, advantage), 0)
        adv = adv.view(self.sub_agents, max_episode_len, 1)
        advantage = adv.permute(1, 0, 2)
        advantage = advantage.contiguous().view(-1, 1)
        actions = actions.view(max_episode_len, self.sub_agents, 1)
        actions = actions.type(torch.long)

        epochs = 15
        for _ in range(epochs):
            new_probs = torch.empty((0), dtype=torch.float32).to(device)
            rnn_state = None
            for item in obs:
                inp = item.to(device)
                new_act, rnn_state = self.actor(inp, rnn_state)
                new_probs = torch.cat((new_probs, new_act.probs), 0)
            # new_act, rnn_state = self.actor(obs, rnn_state)
            new_probs = torch.clamp(new_probs, min=1e-10, max=1.0)  ### 不为0！！！！
            new_act_p = new_probs.view(max_episode_len, self.sub_agents, action_space)
            new_log_probs = torch.log(torch.gather(new_act_p, dim=2, index=actions)).to(device)
            ratio = torch.exp(new_log_probs.view(-1, 1) - act_log_prob.detach().view(-1, 1))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            # 策略熵
            log_new_act = torch.log(new_act_p)
            # policy_entropy = - torch.sum(new_act_p * log_new_act, dim=2)
            # L1_loss_a = sum(p.abs().sum() for p in self.actor.parameters() if p.requires_grad)
            # L1_loss_c = sum(p.abs().sum() for p in self.model.parameters() if p.requires_grad)
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # -0.01*L1_loss_a
            new_out = self.model(x=b_t.x, edge_index='aaa', train=True)
            critic_loss = torch.mean(F.mse_loss(new_out, target.detach()))

            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            critic_loss.backward()
            actor_loss.backward()
            self.critic_optimizer.step()
            self.actor_optimizer.step()

        torch.cuda.empty_cache()  # 释放显存
        return critic_loss.cpu().clone().detach().numpy(), actor_loss.cpu().clone().detach().numpy()


class ColightNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_node, ):
        super(ColightNet, self).__init__()
        # self.model_dict = kwargs
        self.action_space = 5  # 8
        self.n_node = n_node
        self.features = input_dim  # 12
        self.module_list = nn.ModuleList()
        self.embedding_MLP = Embedding_MLP(self.features)
        for i in range(1):  # 0
            # block = MultiHeadAttModel(d=self.model_dict.get('INPUT_DIM')[i],
            #                           dv=self.model_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[i],
            #                           d_out=self.model_dict.get('OUTPUT_DIM')[i],
            #                           nv=self.model_dict.get('NUM_HEADS')[i],
            #                           suffix=i)
            block = MultiHeadAttModel(d=128,
                                      dv=16,
                                      d_out=128,
                                      nv=4,
                                      n_node=n_node,
                                      suffix=i)
            self.module_list.append(block)
        output_dict = OrderedDict()

        if len([]) != 0:
            # TODO: dubug this branch
            for l_idx, l_size in enumerate(self.model_dict['OUTPUT_LAYERS']):
                name = f'output_{l_idx}'
                if l_idx == 0:
                    h = nn.Linear(block.d_out, l_size)
                else:
                    h = nn.Linear(self.model_dict.get('OUTPUT_LAYERS')[l_idx - 1], l_size)
                output_dict.update({name: h})
                name = f'relu_{l_idx}'
                output_dict.update({name: nn.ReLU})
            out = nn.Linear(self.model_dict['OUTPUT_LAYERS'][-1], self.action_space.n)  # 128, 8
        else:
            out = nn.Linear(block.d_out, 1)  # in:128, out:8
        name = f'output'
        output_dict.update({name: out})  # 一个线性层128，8
        self.output_layer = nn.Sequential(output_dict)
        # self.dense_1 = nn.Linear(n_node*128, 128)
        self.dense_1 = nn.Linear(128, 128)

    def forward(self, x, edge_index, train=True):
        h = self.embedding_MLP.forward(x, train)
        # TODO: implement att
        # for mdl in self.module_list:  # mdl: MultiHeadAttModel()
        #     h = mdl.forward(h, edge_index, train)
        # h = h.view(-1, 2*128)
        # if train:
        #     h = self.output_layer(h)
        # else:
        #     with torch.no_grad():
        #         h = self.output_layer(h)
        h = F.relu(self.dense_1(h))
        h = self.output_layer(h)
        # h = h.repeat(self.n_node, 1)
        # h = h.expand(h.size(0), self.n_node).reshape(-1, 1)
        return h


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Embedding_MLP(nn.Module):
    def __init__(self, in_size, layers=0):  # layers: [128, 128]
        super(Embedding_MLP, self).__init__()
        # constructor_dict = OrderedDict()
        # for l_idx, l_size in enumerate(layers):  # 将列表组成索引0：128，0：128
        #     name = f"node_embedding_{l_idx}"
        #     if l_idx == 0:
        #         h = nn.Linear(in_size, l_size)
        #         constructor_dict.update({name: h})  # [('node_embedding_0', Linear(in_features=12, out_features=128, bias=True))]
        #     else:
        #         h = nn.Linear(layers[l_idx - 1], l_size)  # 128, 128
        #         constructor_dict.update({name: h})
        #     name = f"n_relu_{l_idx}"
        #     constructor_dict.update({name: nn.ReLU()})
        # constructor_dict: OrderedDict([('node_embedding_0', Linear(in_features=12, out_features=128, bias=True)), ('n_relu_0', ReLU()),
        # ('node_embedding_1', Linear(in_features=128, out_features=128, bias=True)), ('n_relu_1', ReLU())])
        self.embedding_node = nn.Sequential(
            layer_init(nn.Linear(in_size, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU()
        )

    def _forward(self, x):
        x = self.embedding_node(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 3, 1, 2, 4)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3], X.shape[4])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2], X.shape[3])
    X = X.permute(0, 2, 3, 1, 4)
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)


class MultiHeadAttModel(nn.Module):
    def __init__(self, d, dv, d_out, nv, n_node, suffix):
        super(MultiHeadAttModel, self).__init__()
        self.d = d
        self.dv = dv
        self.d_out = d_out
        self.nv = nv
        self.suffix = suffix
        self.fcv = nn.Linear(d, dv * nv)
        self.fck = nn.Linear(d, dv * nv)
        self.fcq = nn.Linear(d, dv * nv)
        self.fcout = nn.Linear(dv * nv, d_out)
        self.n_node = n_node

    def _forward(self, obs, adjs):
        obs = torch.reshape(obs, (-1, self.n_node, self.d_out))
        adjs = adjs.repeat(obs.size(0), 1, 1)
        adjs = torch.reshape(adjs, (-1, self.n_node, 5, self.n_node))
        # hi*Wt
        agent_repr = torch.unsqueeze(obs, dim=2)
        agent_repr_head = F.relu(self.fcq(agent_repr))
        agent_repr_head = transpose_qkv(agent_repr_head, self.nv)
        agent_repr_head = torch.unsqueeze(agent_repr_head, dim=2)  # [?, 16, 1, 1, 32]
        shape1 = agent_repr_head.shape
        # print(shape1)

        # hj*Ws
        neighbor_repr = torch.unsqueeze(obs, dim=1)
        neighbor_repr = neighbor_repr.repeat(1, self.n_node, 1, 1)  # [?, 16, 16, 32]
        shape2 = neighbor_repr.shape
        # print(shape2)
        neighbor_repr = torch.matmul(adjs, neighbor_repr)  # [?, 16, 5, 32]
        shape3 = neighbor_repr.shape
        # print(shape3)
        neighbor_repr_head = F.relu(self.fck(neighbor_repr))
        neighbor_repr_head = transpose_qkv(neighbor_repr_head, self.nv)
        neighbor_repr_head = torch.unsqueeze(neighbor_repr_head, dim=2).permute(0, 1, 2, 4, 3)  # [?, 16, 1, 5, 32]
        # print(neighbor_repr_head.shape)

        att = F.softmax(torch.matmul(agent_repr_head, neighbor_repr_head), dim=4)  # [?, 16, 1, 1, 5]
        # print("att:",att.shape)
        att_record = torch.squeeze(att, dim=2)

        neighbor_hidden_repr_head = F.relu(self.fcv(neighbor_repr))
        neighbor_hidden_repr_head = transpose_qkv(neighbor_hidden_repr_head, self.nv)
        neighbor_hidden_repr_head = torch.unsqueeze(neighbor_hidden_repr_head, dim=2)  # [?, 16, 1, 5, 32]
        # print(neighbor_hidden_repr_head.shape)

        out = torch.mean(torch.matmul(att, neighbor_hidden_repr_head), dim=2)  # [?, 16, 1, 32]
        # print(out.shape)
        out_concat = transpose_output(out, self.nv)
        out_concat = torch.squeeze(out_concat, dim=2)  # [?, 16, 32]
        out_concat = F.relu(self.fcout(out_concat))

        return out_concat.reshape(-1, out_concat.shape[2])

    def forward(self, x, edge_index, train=True):
        if train:
            return self._forward(x, edge_index)
        else:
            with torch.no_grad():
                return self._forward(x, edge_index)


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):  # 输入是原始观测  输出策略的概率
        super(Actor, self).__init__()
        self.embedding_MLP = Embedding_MLP(input_dim, layers=1)
        self.rnn_hidden = None
        self.rnn = nn.GRUCell(128, 128)
        # self.dense_1 = nn.Linear(input_dim, 64)
        # self.dense_2 = nn.Linear(64, 32)
        # self.dense_3 = nn.Linear(64, output_dim)

        self.dense_1 = nn.Linear(128, output_dim)  # 加上 std=0.01
        # self.dense_1 = layer_init(nn.Linear(input_dim, 64))
        # self.dense_2 = layer_init(nn.Linear(64,  output_dim), std=0.01)

    def forward(self, x, rnn_state):
        # x = x.view(-1,12)
        x = self.embedding_MLP(x)
        if rnn_state is not None:
            rnn_state = rnn_state.view(-1, 128)
        h = self.rnn(x, rnn_state)
        # x = F.relu( self.dense_1(x))  # 去掉了 , inplace=False
        x = F.relu(self.dense_1(h))
        # x = F.relu(self.dense_3(x))
        # x = torch.tanh(self.dense_1(x))
        # x = torch.tanh(x)
        x = torch.distributions.Categorical(logits=x)
        return x, h


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

# 	def feedback(self, reward, terminal, y, x, holding):
# 		# '''
# 		# Feedback is passed to the deep rl agent instance.
# 		# :param float: Reward received during transition
# 		# :param boolean: Indicates if the transition is terminal
# 		# :param tensor: State/Observation
# 		# '''
# 		if self.c.mamethod == 'leniency':
# 			if (self.y, self.x, self.holding) not in self.Temps:
# 				self.Temps[(self.y, self.x, self.holding)] = np.ones(self.c.outputs)*self.c.len.max
# 			temp_action = self.Temps[(self.y, self.x, self.holding)][self.last_action]
# 			self.leniency = 1 - np.exp(-self.c.len.theta*temp_action)
#
# 		if terminal:
# 			self.episodeCounter += 1
# #
# 		self.respond(None)
#
# 		if y == self.y and x == self.x:
# 			outcome = reward
# 		else:
# 			if self.holding + holding == 1:
# 				self.goods = y*x
# 			if terminal > 0:
# 				best_utility = 0
# 			elif (y, x, holding*self.goods) in self.outcomes:
# 				utilities = self.blend_compute(self.t, y, x, holding*self.goods)
# 				best_utility = max(utilities,key=lambda x:x[0])[0]
# 			else:
# 				best_utility = self.default_utility
#
# 			outcome = reward + self.c.gamma*best_utility - self.outcomes[(self.y, self.x, self.holding)][self.last_action]
#
# 		if self.c.mamethod == 'leniency':
# 			if outcome > 0 or random.random() > self.leniency: # self.drl.replay_memory._episode[-1][7]:
# 				self.outcomes[(self.y, self.x, self.holding)][self.last_action] += self.c.alpha*outcome
#
# 			if (y, x, holding*self.goods) not in self.Temps:
# 				temp_mean = self.c.len.max
# 			else:
# 				temp_mean = np.mean(self.Temps[(y, x, holding*self.goods)])
# 			if terminal:
# 				self.Temps[(self.y, self.x, self.holding)][self.last_action] = self.c.len.delta*self.Temps[(self.y, self.x, self.holding)][self.last_action]
# 			else:
# 				self.Temps[(self.y, self.x, self.holding)][self.last_action] = self.c.len.delta*((1-self.c.len.tau)*self.Temps[(self.y, self.x, self.holding)][self.last_action] + self.c.len.tau*temp_mean)
#
# 		elif self.c.mamethod == 'hysteretic':
# 			if outcome > 0:
# 				self.outcomes[(self.y, self.x, self.holding)][self.last_action] += self.c.alpha*outcome
# 			else:
# 				self.outcomes[(self.y, self.x, self.holding)][self.last_action] += self.c.hys.beta*outcome
# 		else:
# 			self.outcomes[(self.y, self.x, self.holding)][self.last_action] += self.c.alpha*outcome
#
# 	def blend_compute(self, t, y, x, holding):
# 		outcomes = self.outcomes[(y, x, holding)]
# 		blends = []
# 		for a,i in zip(range(self.c.outputs),count()):
# 			o = (y, x, holding, a)
# 			if o in self.instance_history:
# 				p = self.CompProbability(t,o)
# 				result = outcomes[a]*p[0] + p[1]*self.default_utility
# 				blends.append((result,i))
# 			else:
# 				blends.append((self.default_utility,i))
# 		return blends
#
# 	def choose_td(self, y, x, holding):
# 		utilities = self.blend_compute(self.t, y, x, holding)
# 		best_utility = max(utilities,key=lambda x:x[0])[0]
# 		best = random.choice(list(filter(lambda x: x[0]==best_utility,utilities)))[1]
# 		return best
#
# 	def boltzchoose(self, y, x, holding):
# 		utilities = self.blend_compute(self.t, y, x, holding)
# 		actions = []
# 		P = []
# 		for u in utilities:
# 			actions.append(u[1])
# 			P.append(u[0])
# 		P = np.asarray(P)
# 		# print(P)
# 		P = np.exp(P/0.8)
# 		P = P/sum(P)
# 		best = np.random.choice(actions,p=P)
# 		return best
