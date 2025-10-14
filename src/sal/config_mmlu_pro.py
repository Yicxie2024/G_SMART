"""
Extended Config for MMLU-Pro support
This can be imported alongside the original config
"""

from dataclasses import dataclass
from sal.config import Config


@dataclass  
class MMluProConfig(Config):
    """Extended configuration with MMLU-Pro specific settings"""
    
    # Override default dataset settings
    dataset_name: str = "TIGER-Lab/MMLU-Pro"
    dataset_split: str = "test"
    data_name: str = "mmlu_pro"
    
    # MMLU-Pro specific prompt (override if needed)
    system_prompt: str = (
        "Answer the following multiple-choice question with clear reasoning. "
        "Analyze each option carefully and provide your reasoning before selecting. "
        "End your response with: Therefore, the answer is (X) where X is the option letter."
    )
    
    # Adjusted search parameters for multiple choice
    temperature: float = 0.6  # Lower temperature for more focused answers
    n: int = 8  # Fewer samples needed for MC questions
    beam_width: int = 4
    
    # Subject/category filter (optional)
    mmlu_pro_category: str = None  # e.g., "physics", "biology", None for all
    
    def __post_init__(self):
        super().__post_init__()
        
        # Additional MMLU-Pro specific initialization
        if self.data_name != "mmlu_pro":
            self.data_name = "mmlu_pro"


def get_mmlu_pro_config_from_yaml(yaml_path: str) -> MMluProConfig:
    """
    Load MMLU-Pro config from YAML file
    
    Usage:
        config = get_mmlu_pro_config_from_yaml("recipes/Qwen2.5-7B-Instruct/beam_search_smart_cocoa_mmlu_pro.yaml")
    """
    from sal.utils.parser import H4ArgumentParser
    import sys
    
    # Temporarily modify sys.argv to load the yaml
    original_argv = sys.argv.copy()
    sys.argv = ["config_loader", yaml_path]
    
    parser = H4ArgumentParser(MMluProConfig)
    config = parser.parse()
    
    # Restore original argv
    sys.argv = original_argv
    
    return config

