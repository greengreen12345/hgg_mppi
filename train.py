# import numpy as np
# import time, hydra
# from common import get_args,experiment_setup
#
# @hydra.main(version_base=None, config_path="../src/m3p2i_aip/config", config_name="config_panda")
#
# if __name__=='__main__':
#
# 	args = get_args()
# 	env, env_test, agent, buffer, learner, tester = experiment_setup(args)
#
# 	#args.logger.summary_init(agent.graph, agent.sess)
#
# 	# Progress info
# 	args.logger.add_item('Epoch')
# 	args.logger.add_item('Cycle')
# 	args.logger.add_item('Episodes@green')
# 	args.logger.add_item('Timesteps')
# 	args.logger.add_item('TimeCost(sec)')
#
# 	# Algorithm info
# 	# for key in agent.train_info.keys():
# 	# 	args.logger.add_item(key, 'scalar')
#
# 	# Test info
# 	# for key in tester.info:
# 	# 	args.logger.add_item(key, 'scalar')
#
# 	#args.logger.summary_setup()
#
#
# 	for epoch in range(args.epochs):
# 		print("epoch", epoch, args.epochs)
# 		for cycle in range(args.cycles):
# 			print("cycle", cycle, args.cycles)
# 			args.logger.tabular_clear()
# 			#args.logger.summary_clear()
# 			start_time = time.time()
#
# 			learner.learn(args, env, env_test, agent, buffer)
# 			# tester.cycle_summary()
#
# 			args.logger.add_record('Epoch', str(epoch)+'/'+str(args.epochs))
# 			args.logger.add_record('Cycle', str(cycle)+'/'+str(args.cycles))
# 			args.logger.add_record('Episodes', buffer.counter)
# 			args.logger.add_record('Timesteps', buffer.steps_counter)
# 			args.logger.add_record('TimeCost(sec)', time.time()-start_time)
#
# 			#args.logger.tabular_show(args.tag)
# 			#args.logger.summary_show(buffer.counter)
#
# 		tester.epoch_summary()
#
# 	tester.final_summary()




import numpy as np
import time
import hydra
from omegaconf import DictConfig
from common import get_args, experiment_setup
from scripts.reactive_tamp import REACTIVE_TAMP
from m3p2i_aip.config.config_store import ExampleConfig

@hydra.main(version_base=None, config_path="/home/my/Hindsight-Goal-Generation-master4/Hindsight-Goal-Generation-master/src/m3p2i_aip/config", config_name="config_panda")
def main(cfg: ExampleConfig):  # <--- Hydra 会自动传入 config_panda.yaml 对应的配置
    args = get_args()

    env, env_test, agent, buffer, learner, tester = experiment_setup(args)

    # ✅ 把 hydra 配置传入 learner，或直接设置 reactive_tamp 实例
    learner.reactive_tamp = REACTIVE_TAMP(cfg, env)

    explore_goal_trajectory = []

    for epoch in range(args.epochs):
        print("*************************epoch***********************", epoch, args.epochs)
        for cycle in range(args.cycles):
            print("*********************************cycle*******************************", cycle, args.cycles)
            args.logger.tabular_clear()
            start_time = time.time()

            learner.learn(args, env, env_test, agent, buffer)
            with open("explore_goals.txt", "a") as f:  # "a" 表示追加写入
                f.write(f"Epoch {epoch}, Cycle {cycle}: {learner.explore_goal}\n")
            with open("achieved_trajectories.txt", "a") as f:  # "a" 表示追加写入
                f.write(f"Epoch {epoch}, Cycle {cycle}: {learner.achieved_trajectories}\n")
            # explore_goal_trajectory.append(learner.explore_goal)
            # print("*************************explore_goal_trajectory********************",explore_goal_trajectory)
            # args.logger.add_record('Epoch', f"{epoch}/{args.epochs}")
            # args.logger.add_record('Cycle', f"{cycle}/{args.cycles}")
            # args.logger.add_record('Episodes', buffer.counter)
            # args.logger.add_record('Timesteps', buffer.steps_counter)
            # args.logger.add_record('TimeCost(sec)', time.time() - start_time)

        tester.epoch_summary()

    tester.final_summary()

if __name__ == "__main__":
    main()

