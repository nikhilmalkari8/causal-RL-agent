# test_env_rewards.py

from envs.causal_gridworld import CausalDoorEnv  # or your env import

def test_random_agent():
    env = CausalDoorEnv()
    state = env.reset()
    done = False
    total_reward = 0.0
    step_num = 0
    
    while not done and step_num < 50:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
        print(f"Step: {step_num}, Action: {action}, Reward: {reward}, Done: {done}")
        step_num += 1

    
    print(f"Total reward in random episode: {total_reward}")

if __name__ == "__main__":
    test_random_agent()
