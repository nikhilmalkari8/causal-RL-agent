#!/usr/bin/env python3
"""
Quick fixes to apply to existing files
Run this to patch your existing enhanced_ppo_agent.py
"""

import os
import shutil

def add_missing_method_to_agent():
    """Add the missing select_action_with_hidden method to EnhancedPPOAgent"""
    
    agent_file = 'agents/enhanced_ppo_agent.py'
    
    # Method to add
    method_to_add = '''
    def select_action_with_hidden(self, state_tensor, hidden_state=None, deterministic=False):
        """
        Enhanced select_action method that handles LSTM hidden states
        """
        with torch.no_grad():
            # Handle LSTM models that need hidden state
            if hasattr(self.policy, 'forward'):
                # Check if this is an LSTM that expects hidden state
                forward_args = self.policy.forward.__code__.co_varnames
                if 'hidden' in forward_args or 'hx' in forward_args:
                    # LSTM-style forward
                    outputs = self.policy.forward(state_tensor, hidden_state)
                    if isinstance(outputs, tuple) and len(outputs) == 3:
                        # (action_logits, value, new_hidden)
                        action_logits, value, new_hidden = outputs
                        
                        if deterministic:
                            action = torch.argmax(action_logits, dim=-1)
                            dist = Categorical(logits=action_logits)
                            log_prob = dist.log_prob(action)
                        else:
                            dist = Categorical(logits=action_logits)
                            action = dist.sample()
                            log_prob = dist.log_prob(action)
                        
                        return action.item(), log_prob, value, new_hidden
            
            # Fall back to regular select_action
            action, log_prob, value = self.select_action(state_tensor, deterministic=deterministic)
            return action, log_prob, value, hidden_state
'''
    
    try:
        # Read the original file
        with open(agent_file, 'r') as f:
            content = f.read()
        
        # Check if method already exists
        if 'select_action_with_hidden' in content:
            print("✅ select_action_with_hidden method already exists")
            return True
        
        # Find the class definition and add the method
        # Look for the end of the __init__ method or another method
        insertion_point = content.find('    def select_action(self,')
        
        if insertion_point == -1:
            print("❌ Could not find insertion point in EnhancedPPOAgent")
            return False
        
        # Insert the new method before select_action
        new_content = content[:insertion_point] + method_to_add + '\n' + content[insertion_point:]
        
        # Backup original file
        import shutil
        shutil.copy(agent_file, agent_file + '.backup')
        
        # Write the updated file
        with open(agent_file, 'w') as f:
            f.write(new_content)
        
        print("✅ Added select_action_with_hidden method to EnhancedPPOAgent")
        print(f"   Backup saved as {agent_file}.backup")
        return True
        
    except Exception as e:
        print(f"❌ Failed to patch agent file: {e}")
        return False

def create_requirements_file():
    """Create a requirements.txt file with all necessary dependencies"""
    requirements = [
        "torch>=1.9.0",
        "numpy>=1.20.0", 
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "gymnasium>=0.26.0",
        "pandas>=1.3.0"
    ]
    
    try:
        with open('requirements.txt', 'w') as f:
            for req in requirements:
                f.write(req + '\n')
        
        print("✅ Created requirements.txt")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create requirements.txt: {e}")
        return False

def check_and_fix_imports():
    """Check and fix common import issues"""
    
    files_to_check = [
        'systematic_training.py',
        'environment_test.py', 
        'run_phase1_complete.py'
    ]
    
    for filename in files_to_check:
        if not os.path.exists(filename):
            print(f"⚠️ File {filename} not found")
            continue
            
        try:
            with open(filename, 'r') as f:
                content = f.read()
            
            # Check for common import issues and fix them
            fixes_applied = []
            
            # Fix torch.distributions import
            if 'from torch.distributions import Categorical' not in content and 'Categorical' in content:
                if 'import torch' in content:
                    content = content.replace('import torch', 'import torch\nfrom torch.distributions import Categorical')
                    fixes_applied.append('Added Categorical import')
            
            # Add numpy import if missing
            if 'import numpy as np' not in content and ('np.' in content or 'numpy' in content):
                content = 'import numpy as np\n' + content
                fixes_applied.append('Added numpy import')
            
            # Add os import if missing
            if 'import os' not in content and 'os.' in content:
                content = 'import os\n' + content
                fixes_applied.append('Added os import')
                
            if fixes_applied:
                with open(filename, 'w') as f:
                    f.write(content)
                print(f"✅ Fixed imports in {filename}: {', '.join(fixes_applied)}")
            else:
                print(f"✅ {filename} imports look good")
                
        except Exception as e:
            print(f"❌ Error checking {filename}: {e}")

def main():
    """Apply all fixes"""
    print("🔧 APPLYING QUICK FIXES")
    print("=" * 40)
    
    import os
    
    # Check if we're in the right directory
    if not os.path.exists('agents') or not os.path.exists('models'):
        print("❌ Please run this script from the causal-RL-agent project root directory")
        print("   (The directory containing agents/, models/, envs/ folders)")
        return False
    
    fixes_applied = 0
    
    # Fix 1: Add missing method to agent
    print("\n🔧 Fix 1: Adding missing method to EnhancedPPOAgent...")
    if add_missing_method_to_agent():
        fixes_applied += 1
    
    # Fix 2: Create requirements file
    print("\n🔧 Fix 2: Creating requirements.txt...")
    if create_requirements_file():
        fixes_applied += 1
    
    # Fix 3: Check imports
    print("\n🔧 Fix 3: Checking and fixing imports...")
    check_and_fix_imports()
    fixes_applied += 1
    
    print(f"\n✅ Applied {fixes_applied} fixes")
    print("\n🚀 You can now run:")
    print("   python run_phase1_complete.py")
    
    return True

if __name__ == "__main__":
    main()