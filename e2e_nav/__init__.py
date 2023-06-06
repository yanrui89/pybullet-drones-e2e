from gym.envs.registration import register

register(
    id="SingleDroneGoal-v0",
    entry_point="e2e_nav.envs:SingleDroneGoalEnv",
    max_episode_steps=500
)

register(
    id="SingleDrone-v0",
    entry_point="e2e_nav.envs:SingleDroneEnv",
    max_episode_steps=500
)

register(
    id="SingleDroneRel-v0",
    entry_point="e2e_nav.envs:SingleDroneRelEnv",
    max_episode_steps=500
)

register(
    id="SingleDroneRew-v0",
    entry_point="e2e_nav.envs:SingleDroneRewEnv",
    max_episode_steps=500
)

