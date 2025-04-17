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
		self.env = env
		self.env_test = self.env
		
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
		
		achieved_value = []

		gamma = self.args.gamma
		q_limit = -1.0 / (1.0 - gamma)
		device = next(agent.policy.parameters()).device  

		for i in range(len(achieved_pool)):
			
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
				
				diff = achieved_pool[i].squeeze(1)  - desired_goals[j]  
				distances = np.linalg.norm(diff, axis=1)  
				q_vals = achieved_value[i]  # shape: (51,)
				scale = self.args.hgg_L / self.max_dis / (1 - self.args.gamma)
				res = distances - q_vals / scale  
		
				match_dis = np.min(res) + self.goal_distance(achieved_pool[i][0], initial_goals[j]) * self.args.hgg_c
				match_idx = np.argmin(res)

				edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
				candidate_goals.append(achieved_pool[i][match_idx])
				candidate_edges.append(edge)
				candidate_id.append(j)

		for i in range(len(desired_goals)):
			self.match_lib.add(graph_id['desired'][i], n, 1, 0)

		match_count = self.match_lib.cost_flow(0, n)
		
		assert match_count == self.length
		

		explore_goals = [0] * self.length

		for i in range(len(candidate_goals)):
			if self.match_lib.check_match(candidate_edges[i]) == 1:
				explore_goals[candidate_id[i]] = candidate_goals[i].copy()
		assert len(explore_goals) == self.length
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
		
		self.env = env
		self.env_test = env_test

		
		if self.sampler is None:
			self.sampler = MatchSampler(args, self.achieved_trajectory_pool, self.env)
		
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
			
			explore_goal = self.sampler.sample(i)
			self.explore_goal = explore_goal
			self.reactive_tamp.set_intermediate_goal(explore_goal.copy())
			self.env.goal = explore_goal.copy()

			obs = self.env._get_obs()
			current = Trajectory(obs)
			trajectory = [obs['achieved_goal'].copy()]

			for timestep in range(args.timesteps):
			
				action_mppi = bytes_to_torch(self.reactive_tamp.run_tamp(
					torch_to_bytes(self.env._dof_state), torch_to_bytes(self.env._root_state)))

				action_hgg = agent.step(obs, explore=True, goal_based=True)
				action = action_hgg + action_mppi
				
				action = action.repeat(200, 1)
				obs, reward, done, info = self.env.step(action)

				trajectory.append(obs['achieved_goal'].copy())
				print("trajectory", trajectory)
				if timestep==args.timesteps-1: done = True
				current.store_step(action, obs, reward, done)
				if done: break

			achieved_trajectories.append(np.array(trajectory))
			self.achieved_trajectories = achieved_trajectories
			
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
			#print("goal_distance", self.goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1]))
			if self.goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1])>0.01:

				selection_trajectory_idx[i] = True
				#print("selection_trajectory_True")
		for idx in selection_trajectory_idx.keys():
			self.achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy())


		


