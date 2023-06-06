import time
import argparse
import gym
from gym.utils import seeding
import torch
import random
from gym.spaces import Box
import numpy as np
import pybullet as pb
from gym_pybullet_drones.envs.single_agent_rl import BaseSingleAgentAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class CustomNavEnv(BaseSingleAgentAviary):
    def __init__(
        self,
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        freq=240,
        aggregate_phy_steps=1,
        gui=False,
        record=False,
        speed_limit=None,
        boundary=(10, 10, 3),
        ctrl_effort_coef=0.1
        ):
        super().__init__(
            drone_model=drone_model,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record
        )
        if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        elif drone_model == DroneModel.HB:
            self.ctrl = SimplePIDControl(drone_model=DroneModel.HB)
        else:
            print("[ERROR] no controller is available for the specified drone_model")
        if speed_limit is None:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)
        else:
            self.SPEED_LIMIT = speed_limit
        XLIM, YLIM, ZLIM = boundary
        self.boudary = ([-XLIM/2., -YLIM/2., 1.0], [XLIM/2., YLIM/2., ZLIM])
        self.goal = None
        self.last_pos = None
        self.collision_ids = [self.PLANE_ID]
        self.ctrl_effort_coef = ctrl_effort_coef
        self.reach_goal = False

    def _actionSpace(self):
        return Box(
            low = np.array([-1., -1., -1., 0.]),
            high = np.array([1., 1., 1., 1.])
        )

    def _observationSpace(self):
        """
        choice1: obs: X, Y, Z (unit vec), q1, q2, q3, q4, VX, VY, VZ
        choice2: obs: X, Y, Z (unit vec)
        choice3: obs: rel pos
        """
        # return Box(
        #     low=np.array([-np.inf, -np.inf, -np.inf, -1, -1, -1, -1, -np.inf, -np.inf, -np.inf]),
        #     high=np.array([np.inf, np.inf, np.inf, 1, 1, 1, 1, np.inf, np.inf, np.inf])
        # )
        # return Box(
        #     low=np.array([-np.inf, -np.inf, -np.inf]),
        #     high=np.array([np.inf, np.inf, np.inf])
        # )
        return Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -1, -1, -1, -1, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, 1, 1, 1, 1, np.inf, np.inf, np.inf])
        )
        # return Box(
        #     low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
        #     high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        # )


    def _preprocessAction(self, action):
        state = self._getDroneStateVector(0)
        if np.linalg.norm(action[:3]) > 0:
            v_unit_vec = action[:3]/np.linalg.norm(action[:3])
        else:
            v_unit_vec = np.zeros(3)
        rpm, _, _ = self.ctrl.computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                             cur_pos=state[0:3],
                                             cur_quat=state[3:7],
                                             cur_vel=state[10:13],
                                             cur_ang_vel=state[13:16],
                                             target_pos=state[0:3], # same as the current position
                                             target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                             target_vel=self.SPEED_LIMIT * np.abs(action[3]) * v_unit_vec # target the desired velocity vector
                                             )
        return rpm
    
    def _computeObs(self):
        state = self._getDroneStateVector(0)
        rel_goal_pos = self.goal - state[:3]
        # if np.linalg.norm(rel_goal_pos) > 0:
        #     goal_unit_vec = rel_goal_pos/np.linalg.norm(rel_goal_pos)
        # else:
        #     goal_unit_vec = np.zeros(3)
        # return np.hstack([goal_unit_vec, state[3:7], state[10:13]]).reshape(10,).astype(np.float32)
        # return goal_unit_vec.reshape(3,).astype(np.float32)
        # return rel_goal_pos.reshape(3,).astype(np.float32)
        return np.hstack([rel_goal_pos, state[3:7], state[10:13]]).reshape(10,).astype(np.float32)
        # return np.hstack([rel_goal_pos, state[10:13]]).reshape(6,).astype(np.float32)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        reward = 0
        # heading alignment reward
        # curr_dir = self.goal-state[:3]
        # unit_curr_dir = curr_dir/np.linalg.norm(curr_dir)
        # curr_vel = state[10:13]
        # if (np.linalg.norm(curr_vel) > 0):
        #     unit_curr_vel = curr_vel/np.linalg.norm(curr_vel)
        # else:
        #     unit_curr_vel = np.zeros(3)
        # reward += np.dot(unit_curr_vel, unit_curr_dir)*0.1
        
        # approaching reward
        # prev_dist = np.linalg.norm(self.goal-self.last_pos)
        curr_dist = np.linalg.norm(self.goal-state[:3])
        # normalized
        # if np.linalg.norm(state[:3]-self.last_pos) > 0:
        #     nav_reward = (prev_dist - curr_dist)/np.linalg.norm(state[:3]-self.last_pos) # reward shaping for approaching the goal
        # else:
        #     nav_reward = 0
        # unnormalized
        # nav_reward = prev_dist - curr_dist
        # reward += nav_reward

        # negative distance scaled
        reward += -curr_dist * 1./48

        # control effort penalty
        # ctrl_effort_penalty = -self.ctrl_effort_coef * np.inner(state[10:13], state[10:13])

        # time step penalty
        # reward += -0.01 # timestep penalty to encourage fast approaching

        # stability reward
        # curr_vel = state[10:13]
        # curr_body_rate = state[13:16]
        # reward += -np.linalg.norm(curr_body_rate)*0.1
        # reward += np.linalg.norm(curr_vel) - np.linalg.norm(curr_body_rate) # encourage high speed but penalize high angular rate for stable flight
        
        if self._check_collision():
            reward += -50 # collision penalty
        if np.linalg.norm(self.goal-state[:3]) < 0.2:
            reward += 50 # sparse reward for reaching the goal

        self.last_pos = np.copy(state[:3])
        return reward

    def _computeInfo(self):
        return {"is_success": int(self.reach_goal)}

    def _computeDone(self):
        if self._check_collision():
            print("Hit obstacles!")
            return True
        if np.linalg.norm(self.goal-self.pos[0]) < 0.2:
            self.reach_goal = True
            print("Successfully reach the goal!")
            return True

    def _check_collision(self):
        for obj_id in self.collision_ids:
            cont_pts = pb.getContactPoints(self.DRONE_IDS[0], obj_id, physicsClientId=self.CLIENT)
            if len(cont_pts) > 0:
                return True
        return False

    def getDroneState(self):
        return self._getDroneStateVector(0)

    def reset(self):
        pb.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Reset initial pos #####################################
        init_xyz = np.random.uniform(*self.boudary)
        # init_xyz = np.array([0., 0., 2.])
        init_rpy = np.zeros(3)
        pb.resetBasePositionAndOrientation(self.DRONE_IDS[0], init_xyz, pb.getQuaternionFromEuler(init_rpy), physicsClientId=self.CLIENT)
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        self.last_pos = np.copy(self.pos[0])
        #### Start video recording #################################
        self._startVideoRecording()
        #### Set goal ##############################################
        self.goal = np.random.uniform(*self.boudary)
        self.reach_goal = False
        # self.goal = np.array([3., 3., 2.])
        # print("[Reset environment] new initial pos: ({:.2f}, {:.2f}, {:.2f}), \
        #             goal: ({:.2f}, {:.2f}, {:.2f})".format(init_xyz[0], init_xyz[1], init_xyz[2], self.goal[0], self.goal[1], self.goal[2]))
        #### Return the initial observation ########################
        return self._computeObs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

def make_custom_env(args):
    AGGR_PHY_STEPS = int(args.simulation_freq_hz/args.control_freq_hz)
    return TimeLimit(
        CustomNavEnv(
            gui=args.gui, 
            freq=args.simulation_freq_hz, 
            speed_limit=args.speed_limit, 
            physics=Physics.PYB, 
            aggregate_phy_steps=AGGR_PHY_STEPS, 
            ctrl_effort_coef=args.ctrl_effort_coef
            ), 
            max_episode_steps=args.max_episode_steps
    )
    

if __name__ == "__main__":
    import os
    import shutil
    from gym.wrappers import TimeLimit
    from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3, PPO
    from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
    import argparse
    import json
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--clear_results", action="store_true")
    parser.add_argument("--output_folder", type=str, default="results")
    parser.add_argument("--exp", type=str, default="nav_vel_ctrl_exp")
    parser.add_argument("--speed_limit", type=float, default=2)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--control_freq_hz", type=float, default=48)
    parser.add_argument("--simulation_freq_hz", type=float, default=240)
    parser.add_argument("--ctrl_effort_coef", type=float, default=0.0)
    parser.add_argument("--total_timesteps", type=int, default=1e6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_tests", type=int, default=10)
    parser.add_argument("--remarks", type=str, default="")
    args = parser.parse_args()

    results_folder = os.path.join(args.output_folder, args.exp)

    logger = None
    if os.path.exists(results_folder) and args.clear_results:
        shutil.rmtree(results_folder)
    
    if args.log:
        logger = Logger(logging_freq_hz=args.control_freq_hz,
                        num_drones=1,
                        output_folder=results_folder,
                        colab=True
                        )

    set_seed_everywhere(args.seed)
    env = make_custom_env(args)
    env.seed(args.seed)
    # offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
    #                         net_arch=[512, 512, 256, 128]
    #                         ) # or None # or dict(net_arch=dict(qf=[256, 128, 64, 32], pi=[256, 128, 64, 32]))
    # offpolicy_kwargs = dict(net_arch=dict(qf=[256, 128, 64, 32], pi=[256, 128, 64, 32]))
    offpolicy_kwargs = None

    onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                           net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]
                           ) # or None

    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=results_folder+"/tb",
        policy_kwargs=offpolicy_kwargs
        )

    # model = DDPG(
    #     "MlpPolicy",
    #     env,
    #     verbose=1, 
    #     policy_kwargs=offpolicy_kwargs,
    #     tensorboard_log=results_folder+"/tb",
    #     )

    # model = TD3(
    #     "MlpPolicy",
    #     env,
    #     verbose=1, 
    #     policy_kwargs=offpolicy_kwargs,
    #     tensorboard_log=results_folder+"/tb",
    #     )

    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1, 
    #     policy_kwargs=onpolicy_kwargs,
    #     tensorboard_log=results_folder+"/tb",
    #     )
    
    if args.train:
        if not os.path.exists(results_folder):
            os.makedirs(results_folder, exist_ok=True)
        with open(os.path.join(results_folder, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)
        eval_env = make_custom_env(args)
        eval_env.seed(args.seed)
        model.learn(
            total_timesteps=args.total_timesteps, 
            log_interval=10, 
            eval_env=eval_env, 
            eval_log_path=results_folder, 
            eval_freq=10000,
            n_eval_episodes=20
            )
        model.save(results_folder+"/final_model.zip")
    else:
        saved_best_models = sorted(glob.glob(f"{results_folder}/*.zip"))
        print(f"load {saved_best_models[-1]}")
        model.load(saved_best_models[-1])
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
                # vel_cmd = pos_err/np.linalg.norm(pos_err)
                # action = np.array([vel_cmd[0], vel_cmd[1], vel_cmd[2], 1])
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_steps += 1
                if args.log:
                    logger.log(drone=0,
                                timestamp=env.step_counter*env.TIMESTEP,
                                state=env.getDroneState()
                                )
                if args.gui:
                    sync(env.step_counter, START, env.TIMESTEP)
                if done:
                    break
            final_dist = np.linalg.norm(env.goal-env.getDroneState()[:3])
            print("episode steps: {}, episode reward: {:.3f}, initial dist: {}, final dist: {}".format(episode_steps, episode_reward, init_dist, final_dist))
            if args.log and env.reach_goal:
            # if args.log:
                logger.save(goal=env.goal)
                time.sleep(0.2)
                if args.plot:
                    logger.plot()

        env.close()

