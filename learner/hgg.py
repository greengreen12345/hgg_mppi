from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env

import copy
import numpy as np
#from envs import make_env
from envs.utils import get_goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils.gcc_utils import gcc_load_lib, c_double, c_int
import torch, hydra

from scripts.reactive_tamp import REACTIVE_TAMP
from src.m3p2i_aip.config.config_store import ExampleConfig
import  learner.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from src.m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes

class TrajectoryPool:
	def __init__(self, args, pool_length):
		self.args = args
		self.length = pool_length

		self.pool = []
		self.pool_init_state = []
		self.counter = 0

	def insert(self, trajectory, init_state):
		if self.counter<self.length:
			self.pool.append(trajectory.copy())
			self.pool_init_state.append(init_state.copy())
		else:
			self.pool[self.counter%self.length] = trajectory.copy()
			self.pool_init_state[self.counter%self.length] = init_state.copy()
		self.counter += 1

	def pad(self):
		if self.counter>=self.length:
			return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
		pool = copy.deepcopy(self.pool)
		pool_init_state = copy.deepcopy(self.pool_init_state)
		while len(pool)<self.length:
			pool += copy.deepcopy(self.pool)
			pool_init_state += copy.deepcopy(self.pool_init_state)
		return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])

class MatchSampler:
	def __init__(self, args, achieved_trajectory_pool, env):
		self.args = args
		print("*************************************************3")
	# 	self.env = wrapper.IsaacGymWrapper(
	# 	"panda",
	# 	"panda_env",
	# 	num_envs=1,
	# 	viewer=False,
	# 	device="cuda:0",
	# 	cube_on_shelf=False,
	# )
		self.env = env
		print("*************************************************4")
		self.env_test = self.env
		print("*************************************************9")
		# self.env = make_env(args)
		# self.env_test = make_env(args)
		self.dim = np.prod(self.env.reset()['achieved_goal'].shape)
		self.delta = 0.05 #self.env.distance_threshold
		self.goal_distance = get_goal_distance(args)

		# self.length = args.episodes
		self.length = 1
		init_goal = self.env.reset()['achieved_goal'].copy()
		self.pool = np.tile(init_goal[np.newaxis,:],[self.length,1])+np.random.normal(0,self.delta,size=(self.length,self.dim))
		self.init_state = self.env.reset()['observation'].copy()

		self.match_lib = gcc_load_lib('learner/cost_flow.c')
		self.achieved_trajectory_pool = achieved_trajectory_pool

		# estimating diameter
		self.max_dis = 0
		for i in range(1000):
			obs = self.env.reset()
			dis = self.goal_distance(obs['achieved_goal'],obs['desired_goal'])
			if dis>self.max_dis: self.max_dis = dis

	def add_noise(self, pre_goal, noise_std=None):
		goal = pre_goal.copy()
		dim = 2 if self.args.env[:5]=='Fetch' else self.dim
		if noise_std is None: noise_std = self.delta
		goal[:dim] += np.random.normal(0, noise_std, size=dim)
		return goal.copy()

	def sample(self, idx):
		return self.pool[idx].copy()
		# if self.args.env[:5]=='Fetch':
		# 	print("length", len(self.pool))
		# 	return self.add_noise(self.pool[idx])
		# else:
		# 	return self.pool[idx].copy()

	def find(self, goal):
		res = np.sqrt(np.sum(np.square(self.pool-goal),axis=1))
		idx = np.argmin(res)
		if test_pool:
			self.args.logger.add_record('Distance/sampler', res[idx])
		return self.pool[idx].copy()

	def update(self, initial_goals, desired_goals):
		if self.achieved_trajectory_pool.counter == 0:
			self.pool = copy.deepcopy(desired_goals)
			return

		achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
		candidate_goals = []
		candidate_edges = []
		candidate_id = []

		agent = self.args.agent
		# achieved_value = []
		# for i in range(len(achieved_pool)):
		# 	obs = [goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for j in
		# 		   range(achieved_pool[i].shape[0])]
		# 	obs = np.array(obs, dtype=np.float32)
		# 	# obs = torch.tensor(obs, dtype=torch.float32).T
		# 	obs = torch.tensor(obs, dtype=torch.float32)
		# 	value = agent.get_q_value(obs)
		# 	value = torch.clamp(value, min=-1.0 / (1.0 - self.args.gamma), max=0.0)
		# 	achieved_value.append(value.cpu().numpy().copy())

		achieved_value = []

		gamma = self.args.gamma
		q_limit = -1.0 / (1.0 - gamma)
		device = next(agent.policy.parameters()).device  # 获取当前模型所在设备

		for i in range(len(achieved_pool)):
			# 拼接 achieved_goal + init_state 得到 obs 向量
			obs = [goal_concat(achieved_pool_init_state[i], achieved_pool[i][j])
				   for j in range(achieved_pool[i].shape[0])]

			obs = torch.tensor(obs, dtype=torch.float32).to(device)
			obs = obs.T

			with torch.no_grad():
				actions, _, _ = agent.policy.act({"states": obs}, role="policy")
				q_values, _, _ = agent.critic.act({"states": obs, "taken_actions": actions}, role="critic")
				value = q_values.view(-1)  # shape: [N]

				value = torch.clamp(value, min=q_limit, max=0.0)

			achieved_value.append(value.cpu().numpy().copy())

		n = 0
		graph_id = {'achieved': [], 'desired': []}
		for _ in achieved_pool:
			n += 1
			graph_id['achieved'].append(n)
		for _ in desired_goals:
			n += 1
			graph_id['desired'].append(n)
		n += 1  # sink
		self.match_lib.clear(n)

		for i in range(len(achieved_pool)):
			self.match_lib.add(0, graph_id['achieved'][i], 1, 0)

		for i in range(len(achieved_pool)):
			for j in range(len(desired_goals)):
				# goal_j = desired_goals[j].squeeze()  # 确保 shape 是 (3,)
				diff = achieved_pool[i].squeeze(1)  - desired_goals[j]  # shape 应该是 (51, 3)
				distances = np.linalg.norm(diff, axis=1)  # shape: (51,)
				q_vals = achieved_value[i]  # shape: (51,)
				scale = self.args.hgg_L / self.max_dis / (1 - self.args.gamma)
				res = distances - q_vals / scale  # shape: (51,)

				# res = np.sqrt(np.sum(np.square(achieved_pool[i] - desired_goals[j]), axis=1)) - achieved_value[i] / (
				# 			self.args.hgg_L / self.max_dis / (1 - self.args.gamma))
				match_dis = np.min(res) + self.goal_distance(achieved_pool[i][0], initial_goals[j]) * self.args.hgg_c
				match_idx = np.argmin(res)

				edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
				candidate_goals.append(achieved_pool[i][match_idx])
				candidate_edges.append(edge)
				candidate_id.append(j)


		for i in range(len(desired_goals)):
			self.match_lib.add(graph_id['desired'][i], n, 1, 0)

		match_count = self.match_lib.cost_flow(0, n)
		print("match_count", match_count)
		print("self.length", self.length)
		assert match_count == self.length
		#print("self.args.hgg_c", self.args.hgg_c)
		#print(f"[调试] match_count = {match_count}, self.length = {self.length}")

		explore_goals = [0] * self.length

		for i in range(len(candidate_goals)):
			if self.match_lib.check_match(candidate_edges[i]) == 1:
				explore_goals[candidate_id[i]] = candidate_goals[i].copy()
		assert len(explore_goals) == self.length
		self.pool = np.array(explore_goals)



		# for i in range(len(candidate_goals)):
		# 	if self.match_lib.check_match(candidate_edges[i]) == 1:
		# 		explore_goals[candidate_id[i]] = candidate_goals[i].copy()
		#
		# for i in range(len(explore_goals)):
		# 	if type(explore_goals[i]) == int and explore_goals[i] == 0:
		# 		fallback = desired_goals[np.random.randint(len(desired_goals))]
		# 		explore_goals[i] = fallback.copy()
		#
		# assert len(explore_goals) == self.length
		# # self.pool = np.array(explore_goals)
		#
		# for i in range(len(explore_goals)):
		# 	g = explore_goals[i]
		# 	if not isinstance(g, np.ndarray):
		# 		g = np.array(g, dtype=np.float32)
		#
		# 	# 自动 squeeze 多余的维度，例如 (1, 3) → (3,)
		# 	if g.ndim == 2 and g.shape[0] == 1:
		# 		g = g.squeeze(0)
		#
		# 	# fallback 替换无效目标（全部为 0 或 shape 错）
		# 	if g.shape != (3,) or np.all(g == 0):
		# 		#print(f"[修复] explore_goals[{i}] 无效或 shape 不一致，正在替换")
		# 		fallback = desired_goals[np.random.randint(len(desired_goals))]
		# 		g = np.array(fallback, dtype=np.float32).squeeze()
		#
		# 	explore_goals[i] = g
		#
		# # 统一 shape 后安全 stack
		# self.pool = np.stack(explore_goals)

	def update1(self, initial_goals, desired_goals):
		if self.achieved_trajectory_pool.counter==0:
			self.pool = copy.deepcopy(desired_goals)
			return

		achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
		candidate_goals = []
		candidate_edges = []
		candidate_id = []

		agent = self.args.agent
		achieved_value = []
		for i in range(len(achieved_pool)):
			obs = [ goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for  j in range(achieved_pool[i].shape[0])]
			# feed_dict = {
			# 	agent.raw_obs_ph: obs
			# }
			# value = agent.sess.run(agent.q_pi, feed_dict)[:,0]
			obs = np.array(obs, dtype=np.float32)  # 先转成 NumPy 数组
			obs = torch.tensor(obs, dtype=torch.float32)  # 再转成 PyTorch 的 Tensor
			obs = obs.T

			value = agent.get_q_value(obs)
			# value = np.clip(value, -1.0/(1.0-self.args.gamma), 0)
			value = torch.clamp(value, min=-1.0 / (1.0 - self.args.gamma), max=0.0)
			# achieved_value.append(value.copy())
			achieved_value.append(value.cpu().numpy().copy())

		n = 0
		graph_id = {'achieved':[],'desired':[]}
		for i in range(len(achieved_pool)):
			n += 1
			graph_id['achieved'].append(n)
		for i in range(len(desired_goals)):
			n += 1
			graph_id['desired'].append(n)
		n += 1
		self.match_lib.clear(n)

		for i in range(len(achieved_pool)):
			self.match_lib.add(0, graph_id['achieved'][i], 1, 0)
		for i in range(len(achieved_pool)):
			for j in range(len(desired_goals)):

				#res = np.sqrt(np.sum(np.square(achieved_pool[i]-desired_goals[j]),axis=1)) - achieved_value[i]/(self.args.hgg_L/self.max_dis/(1-self.args.gamma))
				res = np.sqrt(np.sum(np.square(achieved_pool[i] - desired_goals[j]), axis=2)) \
					  - achieved_value[i] / (self.args.hgg_L / self.max_dis / (1 - self.args.gamma))
				match_dis = np.min(res)+self.goal_distance(achieved_pool[i][0], initial_goals[j])*self.args.hgg_c
				match_idx = int(np.argmin(res))

				if match_idx >= achieved_pool[i].shape[0]:
					#print(f"[警告] match_idx={match_idx} 超过 achieved_pool[{i}].shape[0]={achieved_pool[i].shape[0]}")
					match_idx = achieved_pool[i].shape[0] - 1  # 兜底使用最后一个

				edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
				#print(f"len(res): {len(res)}, achieved_pool[{i}].shape[0]: {achieved_pool[i].shape[0]}")
				candidate_goals.append(achieved_pool[i][match_idx])




				candidate_edges.append(edge)
				candidate_id.append(j)
				#print("goal_distance:", self.goal_distance(achieved_pool[i][0], initial_goals[j]))
				#print("np.min(res):", np.min(res))
		for i in range(len(desired_goals)):
			self.match_lib.add(graph_id['desired'][i], n, 1, 0)

		match_count = self.match_lib.cost_flow(0,n)
		#print("self.args.hgg_c", self.args.hgg_c)
		#print(f"[调试] match_count = {match_count}, self.length = {self.length}")


		assert match_count==self.length

		explore_goals = [0]*self.length
		for i in range(len(candidate_goals)):
			if self.match_lib.check_match(candidate_edges[i])==1:
				explore_goals[candidate_id[i]] = candidate_goals[i].copy()
		assert len(explore_goals)==self.length
		self.pool = np.array(explore_goals)

class HGGLearner:
	def __init__(self, args):
		self.args = args
		# self.env = make_env(args)
		# self.env_test = make_env(args)
		self.goal_distance = get_goal_distance(args)

		self.env_List = []
		# self.env_List = wrapper.IsaacGymWrapper(
		# 	"panda",
		# 	"panda_env",
		# 	num_envs=1,
		# 	viewer=False,
		# 	device="cuda:0",
		# 	cube_on_shelf=False,
		# )
		# for i in range(args.episodes):
		# 	self.env_List.append(make_env(args))

		self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)

		#self.sampler = MatchSampler(args, self.achieved_trajectory_pool, self.env)
		self.sampler = None
		self.reactive_tamp = None
		self.explore_goal = None
		self.achieved_trajectories = None

	def learn(self, args, env, env_test, agent, buffer):
		#print("*************************************************7")
		self.env = env
		self.env_test = env_test

		# for i in range(args.episodes):
		# 	self.env_List.append(env)
		#print("*************************************************5")
		if self.sampler is None:
			self.sampler = MatchSampler(args, self.achieved_trajectory_pool, self.env)
		#print("*************************************************6")
		initial_goals = []
		desired_goals = []

		# for i in range(args.episodes):
		# 	obs = self.env_List[i].reset()
		# 	goal_a = obs['achieved_goal'].copy()
		# 	goal_d = obs['desired_goal'].copy()
		# 	initial_goals.append(goal_a.copy())
		# 	desired_goals.append(goal_d.copy())

		obs = self.env.reset()
		goal_a = obs['achieved_goal'].copy()
		goal_d = obs['desired_goal'].copy()
		initial_goals.append(goal_a.copy())
		desired_goals.append(goal_d.copy())

		self.sampler.update(initial_goals, desired_goals)

		achieved_trajectories = []
		achieved_init_states = []

		args.episodes = 1
		for i in range(args.episodes):
			obs = self.env._get_obs()
			init_state = obs['observation'].copy()
			#print("i", i)
			explore_goal = self.sampler.sample(i)
			self.explore_goal = explore_goal

			# 将 explore_goal 传入 planner
			self.reactive_tamp.set_intermediate_goal(explore_goal.copy())
			self.env.goal = explore_goal.copy()

			obs = self.env._get_obs()
			current = Trajectory(obs)
			trajectory = [obs['achieved_goal'].copy()]

			for timestep in range(args.timesteps):
			# for timestep in range(20):
				action_mppi = bytes_to_torch(self.reactive_tamp.run_tamp(
					torch_to_bytes(self.env._dof_state), torch_to_bytes(self.env._root_state)))

				# print("goal_a",goal_a)
				# print("goal_d", goal_d)

				action_hgg = agent.step(obs, explore=True, goal_based=True)
				action = action_hgg + action_mppi
				# print("action_hgg", action_hgg)
				# print("action_mppi", action_mppi)
				# print("action", action)
				action = action.repeat(200, 1)
				obs, reward, done, info = self.env.step(action)

				trajectory.append(obs['achieved_goal'].copy())
				print("trajectory", trajectory)
				if timestep==args.timesteps-1: done = True
				current.store_step(action, obs, reward, done)
				if done: break

			achieved_trajectories.append(np.array(trajectory))
			self.achieved_trajectories = achieved_trajectories
			#print("achieved_trajectories", achieved_trajectories)
			achieved_init_states.append(init_state)
			buffer.store_trajectory(current)
			agent.normalizer_update(buffer.sample_batch())

			if buffer.steps_counter>=args.warmup:
				for _ in range(args.train_batches):
					info = agent.train(buffer.sample_batch())
					#args.logger.add_dict(info)
				agent.target_update()

		selection_trajectory_idx = {}
		for i in range(self.args.episodes):
			print("goal_distance", self.goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1]))
			if self.goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1])>0.01:

				selection_trajectory_idx[i] = True
				print("selection_trajectory_True")
		for idx in selection_trajectory_idx.keys():
			self.achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy())


		#print("self.achieved_trajectory_pool", self.achieved_trajectory_pool)


