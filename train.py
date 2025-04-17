import numpy as np
import time
import hydra
from omegaconf import DictConfig
from common import get_args, experiment_setup
from scripts.reactive_tamp import REACTIVE_TAMP
from m3p2i_aip.config.config_store import ExampleConfig

@hydra.main(version_base=None, config_path="/home/my/Hindsight-Goal-Generation-master4/Hindsight-Goal-Generation-master/src/m3p2i_aip/config", config_name="config_panda")
def main(cfg: ExampleConfig):  
    args = get_args()

    env, env_test, agent, buffer, learner, tester = experiment_setup(args)

    learner.reactive_tamp = REACTIVE_TAMP(cfg, env)

    explore_goal_trajectory = []

    for epoch in range(args.epochs):
        
        for cycle in range(args.cycles):
            
            args.logger.tabular_clear()
            start_time = time.time()

            learner.learn(args, env, env_test, agent, buffer)
            
            # args.logger.add_record('Epoch', f"{epoch}/{args.epochs}")
            # args.logger.add_record('Cycle', f"{cycle}/{args.cycles}")
            # args.logger.add_record('Episodes', buffer.counter)
            # args.logger.add_record('Timesteps', buffer.steps_counter)
            # args.logger.add_record('TimeCost(sec)', time.time() - start_time)

        tester.epoch_summary()

    tester.final_summary()

if __name__ == "__main__":
    main()

