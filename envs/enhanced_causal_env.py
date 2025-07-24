import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import random

class ObjectType(Enum):
    EMPTY = 0
    AGENT = 1
    WALL = 2
    SWITCH = 3
    DOOR_CLOSED = 4
    DOOR_OPEN = 5
    KEY = 6
    CHEST_CLOSED = 7
    CHEST_OPEN = 8
    GOAL = 9
    LEVER = 10
    BRIDGE_DOWN = 11
    BRIDGE_UP = 12

@dataclass
class CausalRule:
    """Defines a causal relationship in the environment"""
    trigger_type: ObjectType
    trigger_pos: Tuple[int, int]
    effect_type: ObjectType
    effect_pos: Tuple[int, int]
    new_effect_type: ObjectType
    activation_condition: str = "step_on"
    description: str = ""

class EnhancedCausalEnv(gym.Env):
    """
    Enhanced multi-room environment with configurable causal structures
    """
    
    def __init__(self, 
                 config_name: str = "default",
                 partial_observability: bool = False,
                 observation_radius: int = 3,
                 max_steps: int = 100,
                 intervention_mode: bool = False):
        
        super().__init__()
        
        self.config_name = config_name
        self.partial_observability = partial_observability
        self.observation_radius = observation_radius
        self.max_steps = max_steps
        self.intervention_mode = intervention_mode
        
        # Load environment configuration
        self.config = self._load_config(config_name)
        self.grid_height, self.grid_width = self.config['grid_size']
        
        # Action space: 0=up, 1=down, 2=left, 3=right, 4=interact
        self.action_space = spaces.Discrete(5)
        
        # Observation space
        if partial_observability:
            obs_size = (2 * observation_radius + 1, 2 * observation_radius + 1)
        else:
            obs_size = (self.grid_height, self.grid_width)
        
        self.observation_space = spaces.Box(
            low=0, high=len(ObjectType), 
            shape=obs_size, 
            dtype=np.int8
        )
        
        # Initialize environment
        self._setup_environment()
        self.reset()
    
    def _load_config(self, config_name: str) -> Dict:
        """Load environment configuration"""
        configs = {
            "default": {
                "grid_size": (12, 15),
                "agent_start": (1, 1),
                "objects": [
                    (ObjectType.SWITCH, (2, 8)),
                    (ObjectType.DOOR_CLOSED, (5, 11)),
                    (ObjectType.KEY, (8, 2)),
                    (ObjectType.CHEST_CLOSED, (8, 13)),
                    (ObjectType.LEVER, (9, 8)),
                    (ObjectType.BRIDGE_DOWN, (10, 5)),
                    (ObjectType.GOAL, (10, 13))
                ],
                "walls": [
                    # Room 1 boundaries
                    [(0, 0), (6, 0), (6, 9), (0, 9)],
                    # Room 2 boundaries  
                    [(6, 10), (11, 10), (11, 14), (6, 14)],
                    # Connecting passages
                    [(5, 9), (5, 11)]
                ],
                "causal_rules": [
                    {
                        "trigger": (ObjectType.SWITCH, (2, 8)),
                        "effect": (ObjectType.DOOR_CLOSED, (5, 11)),
                        "new_state": ObjectType.DOOR_OPEN,
                        "description": "Switch opens the door"
                    },
                    {
                        "trigger": (ObjectType.KEY, (8, 2)),
                        "effect": (ObjectType.CHEST_CLOSED, (8, 13)),
                        "new_state": ObjectType.CHEST_OPEN,
                        "description": "Key opens the chest"
                    },
                    {
                        "trigger": (ObjectType.LEVER, (9, 8)),
                        "effect": (ObjectType.BRIDGE_DOWN, (10, 5)),
                        "new_state": ObjectType.BRIDGE_UP,
                        "description": "Lever raises the bridge"
                    }
                ]
            },
            
            "complex": {
                "grid_size": (15, 18),
                "agent_start": (1, 1),
                "objects": [
                    # Room 1: Multiple switches and doors
                    (ObjectType.SWITCH, (2, 3)),
                    (ObjectType.SWITCH, (2, 5)),
                    (ObjectType.DOOR_CLOSED, (7, 4)),
                    (ObjectType.DOOR_CLOSED, (4, 8)),
                    
                    # Room 2: Key-chest mechanics
                    (ObjectType.KEY, (9, 2)),
                    (ObjectType.KEY, (11, 6)),
                    (ObjectType.CHEST_CLOSED, (9, 10)),
                    (ObjectType.CHEST_CLOSED, (13, 3)),
                    
                    # Room 3: Lever-bridge system
                    (ObjectType.LEVER, (5, 12)),
                    (ObjectType.LEVER, (8, 15)),
                    (ObjectType.BRIDGE_DOWN, (10, 13)),
                    (ObjectType.BRIDGE_DOWN, (12, 16)),
                    
                    # Final goal
                    (ObjectType.GOAL, (13, 16))
                ],
                "walls": [
                    # Complex room structure
                    [(0, 0), (8, 0), (8, 9), (0, 9)],
                    [(8, 0), (14, 0), (14, 8), (8, 8)],
                    [(0, 9), (6, 9), (6, 17), (0, 17)],
                    [(6, 9), (14, 9), (14, 17), (6, 17)]
                ],
                "causal_rules": [
                    {
                        "trigger": (ObjectType.SWITCH, (2, 3)),
                        "effect": (ObjectType.DOOR_CLOSED, (7, 4)),
                        "new_state": ObjectType.DOOR_OPEN,
                        "description": "Red switch opens red door"
                    },
                    {
                        "trigger": (ObjectType.SWITCH, (2, 5)),
                        "effect": (ObjectType.DOOR_CLOSED, (4, 8)),
                        "new_state": ObjectType.DOOR_OPEN,
                        "description": "Blue switch opens blue door"
                    }
                    # Add more complex causal chains...
                ]
            },
            
            "intervention_test": {
                # Environment specifically designed for intervention testing
                "grid_size": (10, 10),
                "agent_start": (1, 1),
                "objects": [
                    (ObjectType.SWITCH, (2, 1)),  # Fixed: Match visual position
                    (ObjectType.DOOR_CLOSED, (5, 4)),  # Fixed: Match visual position  
                    (ObjectType.GOAL, (8, 8))
                ],
                "walls": [],
                "causal_rules": [
                    {
                        "trigger": (ObjectType.SWITCH, (2, 1)),  # Fixed: Match object position
                        "effect": (ObjectType.DOOR_CLOSED, (5, 4)),  # Fixed: Match object position
                        "new_state": ObjectType.DOOR_OPEN,
                        "description": "Switch controls door"
                    }
                ]
            }
        }
        
        return configs.get(config_name, configs["default"])
    
    def _setup_environment(self):
        """Setup environment from configuration"""
        self.causal_rules = []
        for rule_config in self.config['causal_rules']:
            rule = CausalRule(
                trigger_type=rule_config['trigger'][0],
                trigger_pos=rule_config['trigger'][1],
                effect_type=rule_config['effect'][0],
                effect_pos=rule_config['effect'][1],
                new_effect_type=rule_config['new_state'],
                description=rule_config['description']
            )
            self.causal_rules.append(rule)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize grid
        self.grid = np.full((self.grid_height, self.grid_width), ObjectType.EMPTY.value, dtype=np.int8)
        
        # Create walls
        self._create_walls()
        
        # Place objects
        for obj_type, pos in self.config['objects']:
            if self._is_valid_position(pos):
                self.grid[pos] = obj_type.value
        
        # Place agent
        self.agent_pos = self.config['agent_start']
        self.grid[self.agent_pos] = ObjectType.AGENT.value
        
        # Reset state
        self.steps = 0
        self.activated_objects = set()
        self.collected_items = set()
        
        return self._get_observation(), {}
    
    def _create_walls(self):
        """Create walls from configuration"""
        # Outer boundaries
        self.grid[0, :] = ObjectType.WALL.value
        self.grid[-1, :] = ObjectType.WALL.value
        self.grid[:, 0] = ObjectType.WALL.value
        self.grid[:, -1] = ObjectType.WALL.value
        
        # Custom walls from config
        for wall_coords in self.config.get('walls', []):
            for i in range(len(wall_coords)):
                start = wall_coords[i]
                end = wall_coords[(i + 1) % len(wall_coords)]
                self._draw_line(start, end, ObjectType.WALL.value)
    
    def _draw_line(self, start: Tuple[int, int], end: Tuple[int, int], value: int):
        """Draw a line of walls between two points"""
        x1, y1 = start
        x2, y2 = end
        
        # Simple line drawing
        if x1 == x2:  # Vertical line
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if self._is_valid_position((x1, y)):
                    self.grid[x1, y] = value
        elif y1 == y2:  # Horizontal line
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if self._is_valid_position((x, y1)):
                    self.grid[x, y1] = value
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds"""
        x, y = pos
        return 0 <= x < self.grid_height and 0 <= y < self.grid_width
    
    def step(self, action: int):
        self.steps += 1
        reward = 0.0
        done = False
        
        # Get new position based on action
        new_pos = self._get_new_position(action)
        
        # Check if movement is valid
        if self._can_move_to(new_pos):
            # Clear agent from old position (but preserve other objects)
            if self.grid[self.agent_pos] == ObjectType.AGENT.value:
                self.grid[self.agent_pos] = ObjectType.EMPTY.value
            
            # Check if new position is the goal BEFORE moving
            if self.grid[new_pos] == ObjectType.GOAL.value:
                reward += 10.0
                done = True
                print(f"🏆 GOAL REACHED! Agent at {new_pos}")
            
            # Move agent
            old_pos = self.agent_pos
            self.agent_pos = new_pos
            
            # Check for object interactions (but don't overwrite goal)
            if not done:  # Only if we haven't reached goal
                reward += self._handle_object_interaction(new_pos)
            
            # Update agent position on grid (preserve goal if we're on it)
            if not done:
                self.grid[self.agent_pos] = ObjectType.AGENT.value
        
        # Check for time limit
        if self.steps >= self.max_steps:
            done = True
        
        # Small negative reward for each step (but not if we reached goal)
        if not done or reward <= 0:
            reward -= 0.01
        
        return self._get_observation(), reward, done, False, self._get_info()
    
    def _get_new_position(self, action: int) -> Tuple[int, int]:
        """Calculate new position based on action"""
        x, y = self.agent_pos
        
        if action == 0:  # Up
            return (x - 1, y)
        elif action == 1:  # Down
            return (x + 1, y)
        elif action == 2:  # Left
            return (x, y - 1)
        elif action == 3:  # Right
            return (x, y + 1)
        elif action == 4:  # Interact (stay in place)
            return (x, y)
        
        return (x, y)
    
    def _can_move_to(self, pos: Tuple[int, int]) -> bool:
        """Check if agent can move to position"""
        if not self._is_valid_position(pos):
            return False
        
        cell_value = self.grid[pos]
        
        # Can't move through walls
        if cell_value == ObjectType.WALL.value:
            return False
        
        # Can't move through closed doors
        if cell_value == ObjectType.DOOR_CLOSED.value:
            return False
        
        # Can't move through down bridges
        if cell_value == ObjectType.BRIDGE_DOWN.value:
            return False
        
        return True
    
    def _handle_object_interaction(self, pos: Tuple[int, int]) -> float:
        """Handle interaction with objects at position"""
        reward = 0.0
        cell_value = self.grid[pos]
        
        # Switch activation
        if cell_value == ObjectType.SWITCH.value:
            reward += self._activate_switch(pos)
        
        # Key collection
        elif cell_value == ObjectType.KEY.value:
            reward += self._collect_key(pos)
        
        # Lever activation
        elif cell_value == ObjectType.LEVER.value:
            reward += self._activate_lever(pos)
        
        # Chest interaction
        elif cell_value == ObjectType.CHEST_CLOSED.value:
            reward += self._interact_with_chest(pos)
        
        return reward
    
    def _activate_switch(self, pos: Tuple[int, int]) -> float:
        """Activate switch and trigger causal effects"""
        if pos in self.activated_objects:
            return 0.0
        
        self.activated_objects.add(pos)
        reward = 0.5  # Small reward for activation
        
        # Apply causal rules
        for rule in self.causal_rules:
            if (rule.trigger_type == ObjectType.SWITCH and 
                rule.trigger_pos == pos):
                self._apply_causal_effect(rule)
                reward += 1.0  # Bonus for triggering causal effect
        
        return reward
    
    def _collect_key(self, pos: Tuple[int, int]) -> float:
        """Collect key"""
        if pos in self.collected_items:
            return 0.0
        
        self.collected_items.add(pos)
        self.grid[pos] = ObjectType.EMPTY.value
        return 0.5
    
    def _activate_lever(self, pos: Tuple[int, int]) -> float:
        """Activate lever and trigger causal effects"""
        if pos in self.activated_objects:
            return 0.0
        
        self.activated_objects.add(pos)
        reward = 0.5
        
        # Apply causal rules
        for rule in self.causal_rules:
            if (rule.trigger_type == ObjectType.LEVER and 
                rule.trigger_pos == pos):
                self._apply_causal_effect(rule)
                reward += 1.0
        
        return reward
    
    def _interact_with_chest(self, pos: Tuple[int, int]) -> float:
        """Interact with chest (requires key)"""
        # Check if we have the required key
        required_key_pos = self._find_key_for_chest(pos)
        
        if required_key_pos and required_key_pos in self.collected_items:
            self.grid[pos] = ObjectType.CHEST_OPEN.value
            return 2.0  # Good reward for successful interaction
        
        return 0.0
    
    def _find_key_for_chest(self, chest_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find which key opens this chest"""
        for rule in self.causal_rules:
            if (rule.effect_type == ObjectType.CHEST_CLOSED and 
                rule.effect_pos == chest_pos and
                rule.trigger_type == ObjectType.KEY):
                return rule.trigger_pos
        return None
    
    def _apply_causal_effect(self, rule: CausalRule):
        """Apply the effect of a causal rule"""
        if self._is_valid_position(rule.effect_pos):
            self.grid[rule.effect_pos] = rule.new_effect_type.value
    
    def _get_observation(self):
        """Get current observation"""
        if self.partial_observability:
            return self._get_partial_observation()
        else:
            return self.grid.copy()
    
    def _get_partial_observation(self):
        """Get partial observation around agent"""
        x, y = self.agent_pos
        r = self.observation_radius
        
        obs = np.full((2*r + 1, 2*r + 1), ObjectType.WALL.value, dtype=np.int8)
        
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                world_x, world_y = x + i, y + j
                if self._is_valid_position((world_x, world_y)):
                    obs[i + r, j + r] = self.grid[world_x, world_y]
        
        return obs
    
    def _get_info(self) -> Dict:
        """Get environment info"""
        return {
            'agent_pos': self.agent_pos,
            'steps': self.steps,
            'activated_objects': len(self.activated_objects),
            'collected_items': len(self.collected_items),
            'causal_rules_triggered': self._count_triggered_rules()
        }
    
    def _count_triggered_rules(self) -> int:
        """Count how many causal rules have been triggered"""
        count = 0
        for rule in self.causal_rules:
            if rule.trigger_pos in self.activated_objects or rule.trigger_pos in self.collected_items:
                count += 1
        return count
    
    def render(self, mode='human'):
        """Render the environment"""
        symbols = {
            ObjectType.EMPTY.value: '.',
            ObjectType.AGENT.value: 'A',
            ObjectType.WALL.value: '#',
            ObjectType.SWITCH.value: 'S',
            ObjectType.DOOR_CLOSED.value: 'D',
            ObjectType.DOOR_OPEN.value: 'd',
            ObjectType.KEY.value: 'K',
            ObjectType.CHEST_CLOSED.value: 'C',
            ObjectType.CHEST_OPEN.value: 'c',
            ObjectType.GOAL.value: 'G',
            ObjectType.LEVER.value: 'L',
            ObjectType.BRIDGE_DOWN.value: 'B',
            ObjectType.BRIDGE_UP.value: 'b'
        }
        
        print("\nEnvironment State:")
        for row in self.grid:
            print(' '.join(symbols.get(cell, '?') for cell in row))
        print(f"Steps: {self.steps}, Activated: {len(self.activated_objects)}, Items: {len(self.collected_items)}")
        print()
    
    def apply_intervention(self, intervention_type: str, **kwargs):
        """Apply intervention for testing causal understanding"""
        if intervention_type == "swap_switch_positions":
            # Swap positions of two switches
            self._swap_objects(ObjectType.SWITCH)
        
        elif intervention_type == "change_causal_rule":
            # Change which switch controls which door
            old_rule = kwargs.get('old_rule')
            new_rule = kwargs.get('new_rule')
            self._update_causal_rule(old_rule, new_rule)
        
        elif intervention_type == "add_obstacle":
            # Add wall to block certain paths
            pos = kwargs.get('position')
            if pos and self._is_valid_position(pos):
                self.grid[pos] = ObjectType.WALL.value
        
        elif intervention_type == "remove_object":
            # Remove an object from the environment
            obj_type = kwargs.get('object_type')
            pos = kwargs.get('position')
            if pos and self._is_valid_position(pos):
                self.grid[pos] = ObjectType.EMPTY.value
    
    def _swap_objects(self, obj_type: ObjectType):
        """Swap positions of objects of given type"""
        positions = []
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.grid[i, j] == obj_type.value:
                    positions.append((i, j))
        
        if len(positions) >= 2:
            # Swap first two found
            pos1, pos2 = positions[0], positions[1]
            self.grid[pos1], self.grid[pos2] = self.grid[pos2], self.grid[pos1]
    
    def _update_causal_rule(self, old_rule_idx: int, new_rule: CausalRule):
        """Update a causal rule"""
        if 0 <= old_rule_idx < len(self.causal_rules):
            self.causal_rules[old_rule_idx] = new_rule
    
    def get_causal_graph(self) -> Dict:
        """Return the causal structure as a graph"""
        graph = {}
        for rule in self.causal_rules:
            trigger = f"{rule.trigger_type.name}_{rule.trigger_pos}"
            effect = f"{rule.effect_type.name}_{rule.effect_pos}"
            graph[trigger] = graph.get(trigger, []) + [effect]
        return graph