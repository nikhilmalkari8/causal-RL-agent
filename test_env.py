from envs.causal_gridworld import CausalDoorEnv

env = CausalDoorEnv()
obs, _ = env.reset()
env.render()

done = False
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, done, _, _ = env.step(action)
    env.render()
    if done:
        print(f"Episode finished with reward {reward}")
