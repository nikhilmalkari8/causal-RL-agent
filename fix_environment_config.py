#!/usr/bin/env python3
"""
Fix the environment configuration to require causal reasoning
"""

def create_proper_causal_config():
    """Create a proper causal environment configuration"""
    
    config = {
        "grid_size": (10, 10),
        "agent_start": (1, 1),
        "objects": [
            # Switch in top-left area
            ("SWITCH", (2, 2)),
            # Door that BLOCKS the path to goal
            ("DOOR_CLOSED", (5, 5)),
            # Goal in bottom-right, only accessible through door
            ("GOAL", (8, 8))
        ],
        "walls": [
            # Create walls that force agent to go through the door
            # Vertical wall blocking direct path
            [(4, 4), (4, 9)],  # Left side of door
            [(6, 4), (6, 9)],  # Right side of door
            # Horizontal wall
            [(4, 4), (8, 4)],  # Bottom wall
            [(4, 6), (8, 6)],  # Top wall
        ],
        "causal_rules": [
            {
                "trigger": ("SWITCH", (2, 2)),
                "effect": ("DOOR_CLOSED", (5, 5)),
                "new_state": "DOOR_OPEN",
                "description": "Switch opens the door"
            }
        ]
    }
    
    return config

def patch_environment_file():
    """Patch the enhanced_causal_env.py to use better config"""
    
    env_file = 'envs/enhanced_causal_env.py'
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Create the new intervention_test config
        new_config = '''
            "intervention_test": {
                # Environment specifically designed for intervention testing
                "grid_size": (10, 10),
                "agent_start": (1, 1),
                "objects": [
                    ("SWITCH", (2, 2)),  # Switch in accessible location
                    ("DOOR_CLOSED", (5, 5)),  # Door blocking path
                    ("GOAL", (8, 8))  # Goal behind door
                ],
                "walls": [
                    # Create a corridor that forces going through door
                    # Walls that block alternative paths
                    [(3, 3), (3, 7)],  # Left wall
                    [(7, 3), (7, 7)],  # Right wall  
                    [(3, 3), (7, 3)],  # Bottom wall
                    [(3, 7), (4, 7)],  # Top-left wall
                    [(6, 7), (7, 7)],  # Top-right wall
                    # Door is the only way through at (5,5)
                ],
                "causal_rules": [
                    {
                        "trigger": ("SWITCH", (2, 2)),
                        "effect": ("DOOR_CLOSED", (5, 5)),
                        "new_state": "DOOR_OPEN",
                        "description": "Switch controls door"
                    }
                ]
            }'''
        
        # Find the intervention_test config and replace it
        start_marker = '"intervention_test": {'
        end_marker = '}'
        
        start_idx = content.find(start_marker)
        if start_idx == -1:
            print("❌ Could not find intervention_test config")
            return False
        
        # Find the matching closing brace
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(content[start_idx:]):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = start_idx + i + 1
                    break
        
        # Replace the config
        new_content = content[:start_idx] + new_config + content[end_idx:]
        
        # Backup original
        import shutil
        shutil.copy(env_file, env_file + '.backup')
        
        # Write new content
        with open(env_file, 'w') as f:
            f.write(new_content)
        
        print("✅ Updated environment configuration")
        print(f"   Backup saved as {env_file}.backup")
        return True
        
    except Exception as e:
        print(f"❌ Failed to patch environment: {e}")
        return False

def create_simple_fixed_environment():
    """Create a simple fixed environment file"""
    
    fixed_env_content = '''import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.enhanced_causal_env import EnhancedCausalEnv, ObjectType

class FixedCausalEnv(EnhancedCausalEnv):
    """Fixed version of causal environment that actually requires causal reasoning"""
    
    def __init__(self, **kwargs):
        # Override config to use proper causal setup
        kwargs['config_name'] = 'fixed_causal'
        super().__init__(**kwargs)
    
    def _load_config(self, config_name):
        """Load the fixed configuration"""
        return {
            "grid_size": (10, 10),
            "agent_start": (1, 1),
            "objects": [
                (ObjectType.SWITCH, (2, 2)),
                (ObjectType.DOOR_CLOSED, (5, 5)), 
                (ObjectType.GOAL, (8, 8))
            ],
            "walls": [
                # Create walls that force going through the door
                [(3, 3), (7, 3)],  # Bottom wall
                [(3, 3), (3, 7)],  # Left wall  
                [(7, 3), (7, 7)],  # Right wall
                [(3, 7), (4, 7)],  # Top-left
                [(6, 7), (7, 7)],  # Top-right
                # This creates a room with door at (5,5) as only exit
            ],
            "causal_rules": [
                {
                    "trigger": (ObjectType.SWITCH, (2, 2)),
                    "effect": (ObjectType.DOOR_CLOSED, (5, 5)),
                    "new_state": ObjectType.DOOR_OPEN,
                    "description": "Switch opens door"
                }
            ]
        }
'''
    
    try:
        with open('envs/fixed_causal_env.py', 'w') as f:
            f.write(fixed_env_content)
        
        print("✅ Created fixed_causal_env.py")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create fixed environment: {e}")
        return False

def main():
    """Apply environment fixes"""
    print("🔧 FIXING ENVIRONMENT CONFIGURATION")
    print("=" * 50)
    print("The current environment is too easy (91% random success)")
    print("We need to make it require actual causal reasoning")
    print("")
    
    import os
    
    # Method 1: Try to patch existing file
    print("🔧 Method 1: Patching existing environment...")
    if patch_environment_file():
        print("✅ Environment patched successfully")
    else:
        print("❌ Patching failed, trying alternative method...")
        
        # Method 2: Create new fixed environment
        print("🔧 Method 2: Creating fixed environment file...")
        if create_simple_fixed_environment():
            print("✅ Fixed environment created")
        else:
            print("❌ Both methods failed")
            return False
    
    print("\n🧪 Testing fixed environment...")
    
    # Test the fix
    try:
        if os.path.exists('envs/fixed_causal_env.py'):
            print("Using new fixed environment")
            from envs.fixed_causal_env import FixedCausalEnv
            env = FixedCausalEnv()
        else:
            print("Using patched original environment") 
            from envs.enhanced_causal_env import EnhancedCausalEnv
            env = EnhancedCausalEnv(config_name="intervention_test")
        
        # Quick test
        state, _ = env.reset()
        print(f"✅ Fixed environment loads successfully")
        env.render()
        
        # Test that random agent now fails
        print("\n🎲 Testing random agent on fixed environment...")
        successes = 0
        for episode in range(20):
            state, _ = env.reset()
            for step in range(50):
                action = env.action_space.sample()
                state, reward, done, truncated, _ = env.step(action)
                if done and reward > 0:
                    successes += 1
                    break
                if done or truncated:
                    break
        
        random_success_rate = successes / 20
        print(f"Random agent success rate: {random_success_rate:.1%}")
        
        if random_success_rate < 0.3:
            print("✅ Environment is now properly challenging!")
            return True
        else:
            print(f"⚠️ Still too easy ({random_success_rate:.1%}), may need further adjustments")
            return False
            
    except Exception as e:
        print(f"❌ Error testing fixed environment: {e}")
        return False

if __name__ == "__main__":
    main()
'''