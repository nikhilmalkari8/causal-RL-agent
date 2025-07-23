import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CausalDoorEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self):
        super(CausalDoorEnv, self).__init__()
        self.grid_size = 5
        self.max_steps = 30

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Observation: grid as 5x5 integer matrix
        # 0=empty,1=agent,2=switch,3=closed door,4=open door,5=goal
        self.observation_space = spaces.Box(low=0, high=5, shape=(self.grid_size, self.grid_size), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Positions
        self.agent_pos = [0, 0]
        self.switch_pos = [0, 4]
        self.door_pos = [2, 2]
        self.goal_pos = [4, 4]

        self.door_open = False
        self.steps = 0

        self._update_grid()
        return self.grid.copy(), {}

    def _update_grid(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.grid[tuple(self.switch_pos)] = 2
        self.grid[tuple(self.door_pos)] = 4 if self.door_open else 3
        self.grid[tuple(self.goal_pos)] = 5
        self.grid[tuple(self.agent_pos)] = 1

    def step(self, action):
        self.steps += 1
        x, y = self.agent_pos

        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.grid_size - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.grid_size - 1:
            y += 1

        new_pos = [x, y]

        # If the agent moves onto the switch, open the door
        if new_pos == self.switch_pos:
            self.door_open = True

        # If door is closed, block movement into door cell
        if new_pos == self.door_pos and not self.door_open:
            # Don't move, agent stays in place
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        # Check if agent reached goal or max steps reached
        done = self.agent_pos == self.goal_pos or self.steps >= self.max_steps
        reward = 1.0 if self.agent_pos == self.goal_pos else 0.0

        self._update_grid()
        return self.grid.copy(), reward, done, False, {}

    def render(self):
        symbols = {
            0: '.',
            1: 'A',
            2: 'S',
            3: 'D',
            4: 'd',
            5: 'G',
        }
        print("\nGrid:")
        for row in self.grid:
            print(" ".join(symbols[cell] for cell in row))
        print()

    def close(self):
        pass
