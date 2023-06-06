
import time
import argparse
import os
import shutil
import json
import glob
import gym
import datetime
import torch
import random
import e2e_nav
import numpy as np
import pybullet as pb
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3, PPO
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="SingleDroneRew-v0")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--clear_results", action="store_true")
    parser.add_argument("--output_folder", type=str, default="results")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--speed_limit", type=float, default=2)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--control_freq_hz", type=float, default=48)
    parser.add_argument("--simulation_freq_hz", type=float, default=240)
    parser.add_argument("--ctrl_effort_coef", type=float, default=0.0)
    parser.add_argument("--total_timesteps", type=int, default=2e6)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_tests", type=int, default=10)
    parser.add_argument("--remarks", type=str, default="")
    parser.add_argument("--distance_threshold", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    aggregate_phy_steps = int(args.simulation_freq_hz/args.control_freq_hz)

    env_kwargs = dict(
        freq=args.simulation_freq_hz,
        aggregate_phy_steps=aggregate_phy_steps,
        gui=args.gui,
        record=args.record,
        speed_limit=args.speed_limit,
        boundary=(10, 10, 3),
        distance_threshold=args.distance_threshold
    )
    env = gym.make(args.env_id, **env_kwargs)
    set_seed_everywhere(args.seed)
    env.seed(args.seed)

    # offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
    #                         net_arch=[512, 512, 256, 128]
    #                         )
    offpolicy_kwargs = None

    if args.test:
        exp_folder = os.path.join(args.output_folder, args.exp_name, args.run_name)
        if args.log:
            logger = Logger(logging_freq_hz=args.control_freq_hz,
                            num_drones=1,
                            output_folder=exp_folder,
                            colab=False
                            )
        model = SAC.load(exp_folder+"/best_model", env=env)
        for i in range(args.num_tests):
            if args.log:
                logger.reset()
            obs = env.reset()
            init_dist = np.linalg.norm(env.goal-env.getDroneState()[:3])
            START = time.time()
            done = False
            episode_reward = 0
            episode_steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                # pos_err = env.goal - env.getDroneState()[:3]
                # vel_cmd = pos_err/np.linalg.norm(pos_err)*2
                # action = np.array([vel_cmd[0], vel_cmd[1], vel_cmd[2]])
                obs, reward, done, info = env.step(action)
                # print("step-{}, pos: {}, reward: {}, done: {}, info: {}".format(episode_steps, env.getDroneState()[:3], reward, done, info))
                episode_reward += reward
                episode_steps += 1
                if args.log:
                    logger.log(drone=0,
                                timestamp=env.step_counter*env.TIMESTEP,
                                state=env.getDroneState()
                                )
                if args.gui:
                    sync(env.step_counter, START, env.TIMESTEP)
            final_dist = np.linalg.norm(env.goal-env.getDroneState()[:3])
            print("[{}] episode steps: {}, episode reward: {:.3f}, initial dist: {}, final dist: {}".format(i, episode_steps, episode_reward, init_dist, final_dist))
            if args.log:
                logger.save(goal=env.goal)
                time.sleep(1)
                if args.plot:
                    logger.plot()

        env.close()
    else:
        run_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run_name = f"seed_{args.seed}_{run_time}"
        exp_folder = os.path.join(args.output_folder, args.exp_name, run_name)
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder, exist_ok=True)
        with open(f"{exp_folder}/args.json", 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)
        eval_env = gym.make(args.env_id, **env_kwargs)

        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=offpolicy_kwargs,
            tensorboard_log=exp_folder,
            verbose=1,
            learning_starts=args.learning_starts,
            learning_rate=3e-4
        )
        model.learn(
            args.total_timesteps,
            eval_env=eval_env,
            eval_freq=2000,
            eval_log_path=exp_folder,
            log_interval=10,
            n_eval_episodes=20
            )
        model.save(exp_folder+"/final_model")
