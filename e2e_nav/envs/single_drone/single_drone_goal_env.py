import gym
from gym.utils import seeding
import numpy as np
import pybullet as pb
from gym_pybullet_drones.envs.single_agent_rl import BaseSingleAgentAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl


MAX_LIN_VEL_XY = 2 
MAX_LIN_VEL_Z = 1
OFFSET = 0.5
INIT_GOAL_DIST_THRESHOLD = 2


class SingleDroneGoalEnv(BaseSingleAgentAviary):
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
        ctrl_effort_coef=0.1,
        distance_threshold=0.2
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
        self.boundary = ([-XLIM/2.+OFFSET, -YLIM/2.+OFFSET, 1.], [XLIM/2.-OFFSET, YLIM/2.-OFFSET, ZLIM-OFFSET])
        self.goal_boundary = ([-XLIM/2.+OFFSET, -YLIM/2.+OFFSET, OFFSET], [XLIM/2.-OFFSET, YLIM/2.-OFFSET, ZLIM-OFFSET])
        self.goal = None
        self.last_pos = None
        self.collision_ids = [self.PLANE_ID]
        self.ctrl_effort_coef = ctrl_effort_coef
        self.state = None
        self.distance_threshold = distance_threshold
        self.crashed = False
        self.reached = False

    def _actionSpace(self):
        # return gym.spaces.Box(
        #     low = np.array([-1., -1., -1., 0.]),
        #     high = np.array([1., 1., 1., 1.])
        # )
        return gym.spaces.Box(
            low = np.array([-MAX_LIN_VEL_XY, -MAX_LIN_VEL_XY, -MAX_LIN_VEL_Z]),
            high = np.array([MAX_LIN_VEL_XY, MAX_LIN_VEL_XY, MAX_LIN_VEL_Z]),
            dtype=np.float32
        )

    def _observationSpace(self):
        return gym.spaces.Dict(
            {
                "desired_goal": gym.spaces.Box(
                    np.array([-np.inf, -np.inf, 0]),
                    np.array([np.inf, np.inf, np.inf]),
                    dtype = np.float32
                ),
                "achieved_goal": gym.spaces.Box(
                    np.array([-np.inf, -np.inf, 0]),
                    np.array([np.inf, np.inf, np.inf]),
                    dtype = np.float32
                    ),
                "observation": gym.spaces.Box(
                    np.array([-np.inf, -np.inf, -np.inf, -1, -1, -1, -1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
                    np.array([np.inf, np.inf, np.inf, 1, 1, 1, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
                    dtype=np.float32
                )
            }
        )

    def _preprocessAction(self, action):
        # state = self._getDroneStateVector(0)
        # if np.linalg.norm(action[:3]) > 0:
        #     v_unit_vec = action[:3]/np.linalg.norm(action[:3])
        # else:
        #     v_unit_vec = np.zeros(3)
        # rpm, _, _ = self.ctrl.computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
        #                                      cur_pos=state[0:3],
        #                                      cur_quat=state[3:7],
        #                                      cur_vel=state[10:13],
        #                                      cur_ang_vel=state[13:16],
        #                                      target_pos=state[0:3], # same as the current position
        #                                      target_rpy=np.array([0,0,state[9]]), # keep current yaw
        #                                      target_vel=self.SPEED_LIMIT * np.abs(action[3]) * v_unit_vec # target the desired velocity vector
        #                                      )
        rpm, _, _ = self.ctrl.computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                            cur_pos=self.state[0:3],
                                            cur_quat=self.state[3:7],
                                            cur_vel=self.state[10:13],
                                            cur_ang_vel=self.state[13:16],
                                            target_pos=self.state[0:3], # same as the current position
                                            target_rpy=np.array([0,0,self.state[9]]), # keep current yaw
                                            target_vel=action # target the desired velocity vector
                                            )
        return rpm
    
    def _computeObs(self):
        self.state = self._getDroneStateVector(0).copy()
        normalized_obs = self._clipAndNormalizeState(self.state)
        normalized_desired_goal = np.divide(self.goal, np.array(self.boundary[1]))
        normalized_achieved_goal = np.copy(normalized_obs[:3])
        
        # normalized_ang_v = self.state[13:16]/np.linalg.norm(self.state[13:16]) if np.linalg.norm(self.state[13:16]) != 0 else self.state[13:16]
        # obs = np.hstack([self.state[:3], self.state[3:7], self.state[10:13], normalized_ang_v]).reshape(13,).astype(np.float32)
        return {
            "desired_goal": normalized_desired_goal,
            "achieved_goal": normalized_achieved_goal,
            "observation": normalized_obs
        }

    def _clipAndNormalizeState(self, state):
        normalized_pos = np.divide(state[:3], np.array(self.boundary[1]))
        
        normalized_vel_xy = state[10:12]/MAX_LIN_VEL_XY
        normalized_vel_z = state[12]/MAX_LIN_VEL_Z

        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]
        
        obs = np.hstack([normalized_pos, state[3:7], normalized_vel_xy, normalized_vel_z, normalized_ang_vel]).reshape(13,).astype(np.float32)
        return obs

    def _computeReward(self):
        self._check_collision()
        if self.crashed:
            return -1
        self._check_success()
        if self.reached:
            return 1
        return 0
        # curr_dist = np.linalg.norm(self.goal-self.state[:3])
        # return -curr_dist * 1./48

    def _check_success(self):
        if np.linalg.norm(self.goal-self.state[:3]) <= self.distance_threshold:
            self.reached = True

    def _computeInfo(self):
        return {"is_success": float(self.reached), "crashed": int(self.crashed)}

    def _computeDone(self):
        if self.crashed or self.reached:
            return True
        else:
            return False

    def _check_collision(self):
        # hit obstacles
        for obj_id in self.collision_ids:
            cont_pts = pb.getContactPoints(self.DRONE_IDS[0], obj_id, physicsClientId=self.CLIENT)
            if len(cont_pts) > 0:
                self.crashed = True
        # hit boundary
        self.crashed = self.crashed or not np.array_equal(
                                                self.pos[0],
                                                np.clip(
                                                    self.pos[0], a_min=self.boundary[0], a_max=self.boundary[1]
                                                )
                                            )

    def getDroneState(self):
        return self._getDroneStateVector(0)

    def compute_reward(self, achieved_goal, goal, info):
        # return (np.linalg.norm(goal-achieved_goal) <= self.distance_threshold).astype(np.float32)
        curr_dist = np.linalg.norm(goal-achieved_goal)
        if curr_dist<= self.distance_threshold:
            return 1
        else:
            # return -curr_dist * 1./48
            return 0

    def _gen_new_goal(self, init_xyz):
        new_goal = np.random.uniform(*self.goal_boundary)
        while np.linalg.norm(new_goal-init_xyz) < INIT_GOAL_DIST_THRESHOLD:
            new_goal = np.random.uniform(*self.goal_boundary)
        return new_goal

    def reset(self):
        pb.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Reset initial pos #####################################
        init_xyz = np.random.uniform(*self.boundary)
        init_rpy = np.zeros(3)
        pb.resetBasePositionAndOrientation(self.DRONE_IDS[0], init_xyz, pb.getQuaternionFromEuler(init_rpy), physicsClientId=self.CLIENT)
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Set goal ##############################################
        self.goal = self._gen_new_goal(init_xyz)
        self.crashed = False
        self.reached = False
        # print("[Reset environment] new initial pos: ({:.2f}, {:.2f}, {:.2f}), \
        #             goal: ({:.2f}, {:.2f}, {:.2f})".format(init_xyz[0], init_xyz[1], init_xyz[2], self.goal[0], self.goal[1], self.goal[2]))
        #### Return the initial observation ########################
        return self._computeObs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]