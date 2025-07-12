#!/usr/bin/env python3
"""
Main entry point for the SFT data generation pipeline.

This script provides a command-line interface for generating training data
for supervised fine-tuning using various LLM APIs.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig
from pipeline import DataGenerationPipeline
from utils import (
    create_directory_structure,
    load_sample_data,
    validate_api_keys,
    format_execution_time
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SFT Data Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with single dataset
    python main.py --dataset-path data/input/dataset.json --output-dir data/output

    # Multiple datasets
    python main.py --dataset-path data/input/gala.json data/input/browsecomp.json --output-dir data/output

    # With configuration file
    python main.py --config configs/pipeline.yaml --dataset-path data/input/dataset.json

    # Resume from checkpoint
    python main.py --resume checkpoints/train_checkpoint.json --dataset-path data/input/dataset.json

    # Create sample data for testing
    python main.py --create-sample-data
        """
    )
    
    # Dataset options
    parser.add_argument(
        "--dataset-path",
        type=str,
        nargs="+",
        help="Path(s) to input dataset files"
    )
    
    parser.add_argument(
        "--dataset-type",
        type=str,
        nargs="+",
        choices=["auto", "json", "jsonl", "csv", "gala", "browsecomp"],
        default=["auto"],
        help="Dataset type(s) corresponding to dataset paths"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output",
        help="Output directory for generated data"
    )
    
    # Configuration options
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--create-sample-config",
        action="store_true",
        help="Create sample configuration file"
    )
    
    # API configuration
    parser.add_argument(
        "--claude-api-key",
        type=str,
        help="Claude API key"
    )
    
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        help="Gemini API key"
    )
    
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum conversation rounds per sample"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for text generation"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing"
    )
    
    # Data splitting
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Resume and checkpointing
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint file"
    )
    
    parser.add_argument(
        "--enable-resume",
        action="store_true",
        default=True,
        help="Enable checkpoint saving for resume capability"
    )
    
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N samples"
    )
    
    # Utility options
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample data for testing"
    )
    
    parser.add_argument(
        "--setup-directories",
        action="store_true",
        help="Create directory structure"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run - validate setup without generating data"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()

def load_config(args) -> PipelineConfig:
    """Load configuration from file or command line arguments."""
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = PipelineConfig.from_yaml(args.config)
    else:
        print("Using default configuration")
        config = PipelineConfig()
    
    # Override with command line arguments
    if args.claude_api_key:
        config.api.claude_api_key = args.claude_api_key
    if args.gemini_api_key:
        config.api.gemini_api_key = args.gemini_api_key
    if args.openai_api_key:
        config.api.openai_api_key = args.openai_api_key
    
    if args.max_rounds:
        config.generation.max_rounds = args.max_rounds
    if args.temperature:
        config.generation.temperature = args.temperature
    if args.batch_size:
        config.batch_size = args.batch_size
    
    if args.train_ratio:
        config.data.train_ratio = args.train_ratio
        config.data.test_ratio = 1.0 - args.train_ratio
    
    if args.random_seed:
        config.data.random_seed = args.random_seed
    
    if args.output_dir:
        config.data.output_data_path = args.output_dir
    
    if args.enable_resume:
        config.enable_resume = args.enable_resume
    if args.checkpoint_every:
        config.checkpoint_every = args.checkpoint_every
    
    if args.log_level:
        config.log_level = args.log_level
    
    return config

def validate_setup(config: PipelineConfig, args) -> bool:
    """Validate the setup before running pipeline."""
    print("Validating setup...")
    
    # Validate API keys
    api_validation = validate_api_keys(config)
    if not any(api_validation.values()):
        print("❌ No valid API keys found!")
        print("Please provide at least one API key:")
        print("  - Claude: --claude-api-key or CLAUDE_API_KEY")
        print("  - Gemini: --gemini-api-key or GEMINI_API_KEY")
        print("  - OpenAI: --openai-api-key or OPENAI_API_KEY")
        return False
    
    print("✅ API keys validation:")
    for api, valid in api_validation.items():
        status = "✅" if valid else "❌"
        print(f"  {status} {api.upper()}")
    
    # Validate dataset paths
    if args.dataset_path:
        print("✅ Dataset paths validation:")
        for path in args.dataset_path:
            if not Path(path).exists():
                print(f"❌ Dataset file not found: {path}")
                return False
            else:
                print(f"  ✅ {path}")
    
    # Validate configuration
    try:
        config.validate_config()
        print("✅ Configuration validation passed")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    
    return True

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Handle utility commands
    if args.create_sample_config:
        config_path = "configs/sample_config.yaml"
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        config = PipelineConfig()
        config.to_yaml(config_path)
        print(f"✅ Created sample configuration at {config_path}")
        return
    
    if args.setup_directories:
        create_directory_structure(".")
        print("✅ Directory structure created")
        return
    
    if args.create_sample_data:
        sample_path = "data/input/sample_data.json"
        sample_data = load_sample_data(sample_path)
        print(f"✅ Created sample data at {sample_path}")
        print(f"   Contains {len(sample_data)} samples")
        return
    
    # Load configuration
    try:
        config = load_config(args)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Validate configuration only
    if args.validate_config:
        if validate_setup(config, args):
            print("✅ All validations passed")
            return
        else:
            print("❌ Validation failed")
            sys.exit(1)
    
    # Validate required arguments
    if not args.dataset_path:
        print("❌ No dataset path provided")
        print("Use --dataset-path to specify input dataset(s)")
        sys.exit(1)
    
    # Validate setup
    if not validate_setup(config, args):
        sys.exit(1)
    
    # Create output directory
    Path(config.data.output_data_path).mkdir(parents=True, exist_ok=True)
    
    # Dry run
    if args.dry_run:
        print("✅ Dry run completed successfully")
        print("Setup is valid. You can now run the pipeline without --dry-run")
        return
    
    # Initialize pipeline
    try:
        pipeline = DataGenerationPipeline(config)
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Run pipeline
    try:
        print("\n🚀 Starting data generation pipeline...")
        print(f"📊 Input datasets: {args.dataset_path}")
        print(f"📁 Output directory: {config.data.output_data_path}")
        print(f"🔧 Configuration: {config.generation.max_rounds} max rounds, batch size {config.batch_size}")
        
        # Resume from checkpoint if specified
        if args.resume:
            pipeline.resume_pipeline(args.resume)
        
        # Run the pipeline
        dataset_types = args.dataset_type if len(args.dataset_type) == len(args.dataset_path) else None
        summary = pipeline.run_pipeline(args.dataset_path, dataset_types)
        
        # Print summary
        print("\n✅ Pipeline completed successfully!")
        print(f"📈 Generated {summary['data_summary']['successful_train_samples']} training samples")
        print(f"📈 Generated {summary['data_summary']['successful_test_samples']} test samples")
        print(f"⏱️  Total execution time: {format_execution_time(summary['execution_summary']['total_execution_time'])}")
        
        print("\n📁 Output files:")
        for file_type, path in summary['output_files'].items():
            print(f"  📄 {file_type}: {path}")
        
        print(f"\n📊 Pipeline summary saved to: {Path(config.data.output_data_path) / 'pipeline_summary.json'}")
        
    except KeyboardInterrupt:
        print("\n⏸️  Pipeline interrupted by user")
        print("💾 Progress has been saved and can be resumed later")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()