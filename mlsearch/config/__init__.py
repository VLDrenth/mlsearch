"""
Configuration loader for MLSearch model settings.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_model_config() -> Dict[str, str]:
    """
    Load model configuration from YAML file with environment variable overrides.
    
    Returns:
        Dict mapping model types to model names
    """
    # Default fallback configuration
    default_config = {
        "default": "gpt-4o-mini",
        "orchestrator": "gpt-4.1-2025-04-14", 
        "agent": "gpt-4.1-2025-04-14",
        "worker": "gpt-4o-mini",
        "analysis": "gpt-4o-mini",
        "reasoning": "gpt-4o-mini",
        "planning": "gpt-4.1-2025-04-14"
    }
    
    # Try to load from YAML file
    config_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config and 'models' in yaml_config:
                    # Merge with defaults, preferring YAML values
                    config = {**default_config, **yaml_config['models']}
                else:
                    config = default_config
        else:
            config = default_config
    except Exception as e:
        # If there's any error loading YAML, fall back to defaults
        print(f"Warning: Could not load model config from {config_path}: {e}")
        config = default_config
    
    # Apply environment variable overrides
    for model_type in config.keys():
        env_var = f"MLSEARCH_MODEL_{model_type.upper()}"
        env_value = os.getenv(env_var)
        if env_value:
            config[model_type] = env_value
    
    return config


def get_model_name(model_type: str = "default") -> str:
    """
    Get the model name for a specific component type.
    
    Args:
        model_type: Type of model to get (default, orchestrator, agent, etc.)
        
    Returns:
        Model name string
    """
    config = load_model_config()
    return config.get(model_type, config["default"])


# Load config once at module import
_MODEL_CONFIG = load_model_config()


def get_config() -> Dict[str, str]:
    """Get the loaded model configuration."""
    return _MODEL_CONFIG.copy()