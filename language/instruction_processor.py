import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass
from enum import Enum

class InstructionType(Enum):
    SIMPLE = "simple"           # "Go to the goal"
    SEQUENTIAL = "sequential"   # "First activate switch, then go to goal"  
    CONDITIONAL = "conditional" # "If door is closed, find the switch"
    SPATIAL = "spatial"         # "Go to the red switch in the top room"
    CAUSAL = "causal"          # "Use the lever to raise the bridge"

@dataclass
class ParsedInstruction:
    """Structured representation of a parsed instruction"""
    instruction_type: InstructionType
    main_goal: str
    sub_goals: List[str]
    spatial_references: List[str]
    causal_references: List[Dict[str, str]]
    temporal_order: List[str]
    conditions: List[str]

class InstructionTokenizer:
    """
    Simple tokenizer for natural language instructions
    """
    
    def __init__(self):
        # Build vocabulary with common words for the domain
        self.vocab = {
            '<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3,
            
            # Actions
            'go': 4, 'move': 5, 'walk': 6, 'reach': 7, 'get': 8, 'take': 9,
            'activate': 10, 'press': 11, 'push': 12, 'pull': 13, 'use': 14,
            'open': 15, 'close': 16, 'find': 17, 'collect': 18,
            
            # Objects
            'switch': 20, 'door': 21, 'key': 22, 'chest': 23, 'goal': 24,
            'lever': 25, 'bridge': 26, 'button': 27, 'treasure': 28,
            
            # Spatial terms
            'to': 30, 'at': 31, 'in': 32, 'on': 33, 'near': 34, 'next': 35,
            'left': 36, 'right': 37, 'up': 38, 'down': 39, 'north': 40,
            'south': 41, 'east': 42, 'west': 43, 'room': 44, 'area': 45,
            
            # Temporal/Sequential
            'first': 50, 'then': 51, 'after': 52, 'before': 53, 'finally': 54,
            'next': 55, 'last': 56, 'when': 57, 'while': 58,
            
            # Causal terms
            'because': 60, 'so': 61, 'therefore': 62, 'causes': 63, 'enables': 64,
            'requires': 65, 'needs': 66, 'opens': 67, 'closes': 68, 'controls': 69,
            
            # Conditionals
            'if': 70, 'unless': 71, 'when': 72, 'whenever': 73,
            
            # Colors/Descriptors
            'red': 80, 'blue': 81, 'green': 82, 'yellow': 83, 'black': 84,
            'white': 85, 'small': 86, 'large': 87, 'big': 88, 'tiny': 89,
            
            # Common words
            'the': 90, 'a': 91, 'an': 92, 'and': 93, 'or': 94, 'but': 95,
            'is': 96, 'was': 97, 'are': 98, 'were': 99, 'will': 100,
            'can': 101, 'should': 102, 'must': 103, 'may': 104
        }
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def tokenize(self, instruction: str) -> List[int]:
        """Convert instruction string to token IDs"""
        # Simple preprocessing
        instruction = instruction.lower().strip()
        instruction = re.sub(r'[^\w\s]', '', instruction)  # Remove punctuation
        words = instruction.split()
        
        # Convert to tokens
        tokens = [self.vocab['<START>']]
        for word in words:
            token_id = self.vocab.get(word, self.vocab['<UNK>'])
            tokens.append(token_id)
        tokens.append(self.vocab['<END>'])
        
        return tokens
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert token IDs back to instruction string"""
        words = []
        for token_id in tokens:
            if token_id in [self.vocab['<START>'], self.vocab['<END>'], self.vocab['<PAD>']]:
                continue
            word = self.reverse_vocab.get(token_id, '<UNK>')
            words.append(word)
        return ' '.join(words)
    
    def pad_sequence(self, tokens: List[int], max_length: int) -> List[int]:
        """Pad sequence to max_length"""
        if len(tokens) >= max_length:
            return tokens[:max_length]
        else:
            return tokens + [self.vocab['<PAD>']] * (max_length - len(tokens))

class InstructionParser:
    """
    Parse natural language instructions into structured representations
    """
    
    def __init__(self):
        self.tokenizer = InstructionTokenizer()
        
        # Pattern matching for different instruction types
        self.patterns = {
            'sequential': [
                r'first .+ then .+',
                r'after .+ do .+',
                r'.+ before .+',
                r'.+ and then .+'
            ],
            'conditional': [
                r'if .+ then .+',
                r'when .+ do .+',
                r'unless .+ then .+'
            ],
            'causal': [
                r'.+ (opens|closes|activates|controls) .+',
                r'use .+ to .+',
                r'.+ causes .+',
                r'.+ enables .+'
            ],
            'spatial': [
                r'go to .+ (in|at|near) .+',
                r'find .+ (in|at|near) .+',
                r'.+ (left|right|north|south|up|down) .+'
            ]
        }
    
    def parse(self, instruction: str) -> ParsedInstruction:
        """Parse instruction into structured format"""
        instruction_lower = instruction.lower()
        
        # Determine instruction type
        instruction_type = InstructionType.SIMPLE
        for type_name, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, instruction_lower):
                    instruction_type = InstructionType(type_name)
                    break
            if instruction_type.value != 'simple':
                break
        
        # Extract components
        main_goal = self._extract_main_goal(instruction_lower)
        sub_goals = self._extract_sub_goals(instruction_lower)
        spatial_references = self._extract_spatial_references(instruction_lower)
        causal_references = self._extract_causal_references(instruction_lower)
        temporal_order = self._extract_temporal_order(instruction_lower)
        conditions = self._extract_conditions(instruction_lower)
        
        return ParsedInstruction(
            instruction_type=instruction_type,
            main_goal=main_goal,
            sub_goals=sub_goals,
            spatial_references=spatial_references,
            causal_references=causal_references,
            temporal_order=temporal_order,
            conditions=conditions
        )
    
    def _extract_main_goal(self, instruction: str) -> str:
        """Extract the main goal from instruction"""
        # Look for goal-related keywords
        goal_patterns = [
            r'(reach|get to|go to) .* goal',
            r'(find|collect|get) .* treasure',
            r'(reach|arrive at) .* destination'
        ]
        
        for pattern in goal_patterns:
            match = re.search(pattern, instruction)
            if match:
                return match.group(0)
        
        # Default: return the whole instruction if no specific goal found
        return instruction.split('.')[0]  # First sentence
    
    def _extract_sub_goals(self, instruction: str) -> List[str]:
        """Extract sub-goals from instruction"""
        sub_goals = []
        
        # Look for sequential markers
        sequential_parts = re.split(r'\b(first|then|after|before|next|finally)\b', instruction)
        
        for i, part in enumerate(sequential_parts):
            if part.strip() and part not in ['first', 'then', 'after', 'before', 'next', 'finally']:
                cleaned_part = part.strip(' ,.')
                if cleaned_part:
                    sub_goals.append(cleaned_part)
        
        return sub_goals
    
    def _extract_spatial_references(self, instruction: str) -> List[str]:
        """Extract spatial references"""
        spatial_refs = []
        
        spatial_patterns = [
            r'(in|at|near|next to) the \w+ \w+',
            r'(left|right|north|south|up|down) \w+',
            r'(top|bottom|center|middle) \w+'
        ]
        
        for pattern in spatial_patterns:
            matches = re.findall(pattern, instruction)
            spatial_refs.extend(matches)
        
        return [' '.join(match) if isinstance(match, tuple) else match for match in spatial_refs]
    
    def _extract_causal_references(self, instruction: str) -> List[Dict[str, str]]:
        """Extract causal relationships"""
        causal_refs = []
        
        # Pattern: X causes/opens/controls Y
        causal_patterns = [
            (r'(\w+) (opens|closes|activates|controls) (\w+)', ['cause', 'relation', 'effect']),
            (r'use (\w+) to (\w+)', ['cause', 'effect']),
            (r'(\w+) causes (\w+)', ['cause', 'effect']),
            (r'(\w+) enables (\w+)', ['cause', 'effect'])
        ]
        
        for pattern, labels in causal_patterns:
            matches = re.findall(pattern, instruction)
            for match in matches:
                if len(match) == len(labels):
                    causal_ref = {labels[i]: match[i] for i in range(len(labels))}
                    causal_refs.append(causal_ref)
        
        return causal_refs
    
    def _extract_temporal_order(self, instruction: str) -> List[str]:
        """Extract temporal ordering information"""
        temporal_markers = ['first', 'then', 'after', 'before', 'next', 'finally', 'last']
        
        temporal_order = []
        for marker in temporal_markers:
            if marker in instruction:
                temporal_order.append(marker)
        
        return temporal_order
    
    def _extract_conditions(self, instruction: str) -> List[str]:
        """Extract conditional statements"""
        conditions = []
        
        conditional_patterns = [
            r'if (.+?) then',
            r'when (.+?) do',
            r'unless (.+?) then'
        ]
        
        for pattern in conditional_patterns:
            matches = re.findall(pattern, instruction)
            conditions.extend(matches)
        
        return conditions

class InstructionDataset:
    """
    Dataset of instructions for training and evaluation
    """
    
    def __init__(self):
        self.instructions = []
        self.tokenizer = InstructionTokenizer()
        self.parser = InstructionParser()
        self._create_instruction_templates()
    
    def _create_instruction_templates(self):
        """Create a diverse set of instruction templates"""
        templates = [
            # Simple instructions
            "Go to the goal",
            "Find the treasure",
            "Reach the destination",
            "Navigate to the target",
            
            # Sequential instructions (correct for intervention_test)
            "First activate the switch then go to the goal",
            "Press the switch and then reach the goal",
            "Use the switch before going to the goal",
            "Activate the switch then navigate to the goal",
            "Hit the switch first then go to the target",
            "Find the switch then go to the destination",
            
            # Conditional instructions (correct for environment)
            "If the door is closed then find the switch",
            "When the door blocks your path activate the switch",
            "If you cannot reach the goal find the switch first",
            "When the path is blocked use the switch",
            
            # Causal instructions (matching actual environment)
            "Use the switch to open the door",
            "The switch controls the door",
            "Press the switch to open the path",
            "Activate the switch to access the goal",
            "The switch opens the door to the goal",
            "Hit the switch to clear the path",
            
            # Spatial instructions (correct positions)
            "Go to the switch in the top area",
            "Find the switch near the top of the room",
            "Activate the switch in the upper region",
            "The switch is in the top part of the grid",
            
            # Complex multi-step instructions (environment-specific)
            "First go to the switch in the top area then navigate to the goal in the bottom right",
            "If the door is closed find the switch near the top and activate it before going to the goal",
            "Use the switch in the upper area to open the door then reach the goal",
            "Navigate to the switch then move through the opened door to reach the goal",
            "Activate the switch to open the path then go to the goal in the corner",
            
            # Alternative phrasings
            "Hit the switch then reach the target",
            "Press the button then go to the goal", # Note: calling switch a button
            "Activate the control then navigate to the destination",
            "Trigger the switch then move to the goal",
            "Use the lever then go to the target", # Note: calling switch a lever
        ]
        
        # Store original templates
        self.instructions = templates.copy()
        
        # Add variations with different object names (but same meaning)
        object_synonyms = {
            'switch': ['button', 'lever', 'control', 'trigger'],
            'goal': ['target', 'destination', 'treasure', 'endpoint'],
            'door': ['path', 'passage', 'way', 'route']
        }
        
        variations = []
        for template in templates:
            for original, synonyms in object_synonyms.items():
                if original in template.lower():
                    for synonym in synonyms:
                        variation = template.lower().replace(original, synonym)
                        variations.append(variation.capitalize())
        
        self.instructions.extend(variations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_instructions = []
        for instruction in self.instructions:
            if instruction not in seen:
                seen.add(instruction)
                unique_instructions.append(instruction)
        
        self.instructions = unique_instructions
    
    def get_random_instruction(self) -> str:
        """Get a random instruction from the dataset"""
        return np.random.choice(self.instructions)
    
    def get_instruction_by_type(self, instruction_type: InstructionType) -> str:
        """Get a random instruction of specific type"""
        matching_instructions = []
        
        for instruction in self.instructions:
            parsed = self.parser.parse(instruction)
            if parsed.instruction_type == instruction_type:
                matching_instructions.append(instruction)
        
        if matching_instructions:
            return np.random.choice(matching_instructions)
        else:
            return self.get_random_instruction()
    
    def tokenize_instruction(self, instruction: str, max_length: int = 32) -> torch.Tensor:
        """Tokenize and pad instruction"""
        tokens = self.tokenizer.tokenize(instruction)
        padded_tokens = self.tokenizer.pad_sequence(tokens, max_length)
        return torch.tensor(padded_tokens, dtype=torch.long)
    
    def create_instruction_batch(self, batch_size: int, max_length: int = 32) -> Tuple[List[str], torch.Tensor]:
        """Create a batch of random instructions"""
        instructions = [self.get_random_instruction() for _ in range(batch_size)]
        tokenized = torch.stack([self.tokenize_instruction(inst, max_length) for inst in instructions])
        return instructions, tokenized
    
    def get_all_instructions(self) -> List[str]:
        """Get all instructions in the dataset"""
        return self.instructions.copy()
    
    def add_instruction(self, instruction: str):
        """Add a new instruction to the dataset"""
        self.instructions.append(instruction)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.tokenizer.vocab_size