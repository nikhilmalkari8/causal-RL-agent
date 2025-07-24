# 🧠 Enhanced Causal Reasoning RL Agent

A transformer-based reinforcement learning agent that learns causal relationships and follows natural language instructions in multi-room gridworld environments.

## 🎯 Project Overview

This project demonstrates causal reasoning in AI by training agents that understand "what causes what" rather than just memorizing action sequences. The agent can:

- **Learn causal relationships** (switch controls door, key opens chest)
- **Follow complex instructions** ("First activate switch, then go to goal")
- **Generalize to new environments** with same causal structure
- **Adapt to interventions** when causal rules change

## 🏗️ Project Structure

```
causal-rl-agent/
├── envs/
│   └── enhanced_causal_env.py      # Multi-room environments with causal structures
├── models/
│   ├── enhanced_transformer_policy.py  # Main transformer agent
│   └── baseline_models.py          # LSTM, CNN, MLP baselines
├── agents/
│   └── enhanced_ppo_agent.py       # PPO training with causal auxiliary loss
├── language/
│   └── instruction_processor.py    # Natural language instruction system
├── evaluation/
│   └── evaluation_framework.py     # Comprehensive evaluation suite
├── scripts/
│   └── train_phase1_complete.py    # Complete training script
└── requirements.txt
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd causal-rl-agent

# Create virtual environment
python -m venv causal-rl
source causal-rl/bin/activate  # On Windows: causal-rl\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Agent

```bash
# Complete Phase 1 training (includes baselines and evaluation)
python scripts/train_phase1_complete.py

# Quick training (fewer episodes)
python scripts/train_phase1_complete.py --max_episodes 500

# Skip training and evaluate pre-trained model
python scripts/train_phase1_complete.py --skip_training --model_path models/best_transformer_agent_0.850.pth
```

### 3. View Results

After training, check these directories:
- `results/` - Evaluation results and reports
- `plots/` - Training progress and comparison plots  
- `models/` - Saved model checkpoints

## 🧪 Evaluation Framework

The agent is evaluated on multiple dimensions:

### **Causal Understanding**
- **Intervention tests**: Performance when causal rules change
- **Counterfactual reasoning**: Understanding "what if" scenarios
- **Rule swapping**: Adaptation to modified environments

### **Generalization**
- **Zero-shot transfer**: Success in unseen environments
- **Compositional reasoning**: Combining known elements in new ways
- **Cross-domain transfer**: Same causal structure, different appearance

### **Language Understanding**
- **Instruction following**: Simple to complex commands
- **Compositional language**: Multi-step instructions
- **Spatial reasoning**: Location-based instructions

## 📊 Expected Results

**Target Performance (Phase 1):**
- **Success Rate**: >80% on training environments
- **Causal Understanding**: >70% on intervention tests
- **Generalization**: >60% retention on transfer tasks
- **Baseline Improvement**: >15% over best non-causal baseline

## 🔧 Customization

### Adding New Environments

```python
# In enhanced_causal_env.py, add to _load_config():
"my_custom_env": {
    "grid_size": (10, 10),
    "objects": [
        (ObjectType.SWITCH, (2, 3)),
        (ObjectType.GOAL, (8, 8))
    ],
    "causal_rules": [
        {
            "trigger": (ObjectType.SWITCH, (2, 3)),
            "effect": (ObjectType.DOOR_CLOSED, (5, 5)),
            "new_state": ObjectType.DOOR_OPEN,
            "description": "Switch opens door"
        }
    ]
}
```

### Adding New Instructions

```python
# In instruction_processor.py, add to _create_instruction_templates():
new_instructions = [
    "Find the golden key in the secret room",
    "Use the magic lever to activate the portal"
]
self.instructions.extend(new_instructions)
```

### Modifying Agent Architecture

```python
# In enhanced_transformer_policy.py, adjust model parameters:
policy = EnhancedTransformerPolicy(
    d_model=512,        # Increase model size
    nhead=16,           # More attention heads
    num_layers=8,       # Deeper network
    # ... other parameters
)
```

## 📈 Phase 1 Success Criteria

✅ **Technical Achievements:**
- Multi-room environment with complex causal chains
- Transformer architecture with causal attention
- Language instruction integration
- Comprehensive evaluation framework

✅ **Performance Targets:**
- >80% success rate on training tasks
- >15% improvement over LSTM baseline
- Successful intervention adaptation
- Multi-step instruction following

✅ **Research Readiness:**
- Clean, modular, extensible codebase
- Systematic evaluation methodology
- Baseline comparisons
- Reproducible results

## 🎓 Next Steps (Phase 2+)

After completing Phase 1, the project can be enhanced with:

1. **Explicit Causal Graph Learning** - Learn causal structure representations
2. **Hierarchical Skill Discovery** - Discover reusable causal skills
3. **Advanced Language Integration** - Complex compositional reasoning
4. **Real-world Applications** - Robotics, scientific discovery

## 🐛 Troubleshooting

**Common Issues:**

1. **CUDA out of memory**: Reduce batch size or model size
2. **Training instability**: Lower learning rate or increase entropy coefficient
3. **Poor generalization**: Increase environment diversity or causal loss weight
4. **Language understanding issues**: Expand instruction dataset or increase language model capacity

**Performance Debugging:**

```bash
# Check if agent is learning causal relationships
python -c "
from envs.enhanced_causal_env import EnhancedCausalEnv
env = EnhancedCausalEnv()
env.render()  # Visualize environment
print('Causal rules:', env.causal_rules)
"

# Test language processing
python -c "
from language.instruction_processor import InstructionDataset
dataset = InstructionDataset()
inst = dataset.get_random_instruction()
print('Instruction:', inst)
print('Tokenized:', dataset.tokenize_instruction(inst))
"
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{causal-rl-agent,
  title={Enhanced Causal Reasoning in Reinforcement Learning with Transformer Architectures},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/causal-rl-agent}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests for any improvements.

---

**Built for PhD applications to Stanford, MIT, Berkeley, and UCL** 🎓