"""Configuration management for the data generation pipeline."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

class APIConfig(BaseModel):
    """API configuration for different LLM providers."""
    claude_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("CLAUDE_API_KEY"))
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    
    # Rate limiting
    max_requests_per_minute: int = 50
    max_concurrent_requests: int = 5
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0

class DataConfig(BaseModel):
    """Data processing configuration."""
    # Data paths
    input_data_path: str = "data/input"
    output_data_path: str = "data/output"
    
    # Data splitting
    train_ratio: float = 0.8
    test_ratio: float = 0.2
    random_seed: int = 42
    
    # Supported datasets
    supported_datasets: List[str] = ["gala", "browsecomp", "custom"]
    
    # Shuffling
    shuffle_data: bool = True

class GenerationConfig(BaseModel):
    """Text generation configuration."""
    # Generation parameters
    max_rounds: int = 10
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # Model preferences (in order of preference)
    model_preferences: List[str] = ["claude-3-sonnet-20240229", "gemini-1.5-pro", "gpt-4"]
    
    # Quality control
    min_answer_length: int = 10
    max_answer_length: int = 4000
    
    # Filtering
    filter_duplicates: bool = True
    similarity_threshold: float = 0.85

class PipelineConfig(BaseModel):
    """Main pipeline configuration."""
    api: APIConfig = Field(default_factory=APIConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/pipeline.log"
    
    # Processing
    batch_size: int = 16
    num_workers: int = 4
    
    # Resume capability
    enable_resume: bool = True
    checkpoint_every: int = 100

    @classmethod
    def from_yaml(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)

    def validate_config(self) -> None:
        """Validate configuration settings."""
        # Check API keys
        if not any([self.api.claude_api_key, self.api.gemini_api_key, self.api.openai_api_key]):
            raise ValueError("At least one API key must be provided")
        
        # Check data paths
        if not os.path.exists(self.data.input_data_path):
            raise ValueError(f"Input data path does not exist: {self.data.input_data_path}")
        
        # Check ratios
        if abs(self.data.train_ratio + self.data.test_ratio - 1.0) > 1e-6:
            raise ValueError("Train and test ratios must sum to 1.0")
        
        # Check generation parameters
        if self.generation.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        
        if not (0 <= self.generation.temperature <= 2.0):
            raise ValueError("temperature must be between 0 and 2.0")

# Default configuration instance
default_config = PipelineConfig()