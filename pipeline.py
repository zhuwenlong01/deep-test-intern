"""Main data generation pipeline for SFT training data."""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from tqdm import tqdm
import hashlib

from config import PipelineConfig
from data_splitter import DataSplitter
from api_client import MultiAPIClient, ConversationManager
from utils import (
    save_checkpoint, 
    load_checkpoint, 
    filter_duplicates,
    validate_generated_data,
    create_sft_dataset_format
)

@dataclass
class GenerationResult:
    """Result of data generation for a single sample."""
    original_data: Dict[str, Any]
    generated_conversations: List[Dict[str, Any]]
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    api_client_used: str = ""

class DataGenerationPipeline:
    """Main pipeline for generating SFT training data."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.data_splitter = DataSplitter(config)
        self.api_client = MultiAPIClient(config)
        self.conversation_manager = ConversationManager(self.api_client, config)
        
        # Pipeline state
        self.current_step = 0
        self.total_steps = 0
        self.processed_samples = 0
        self.failed_samples = 0
        self.checkpoint_data = {}
        
        # Results storage
        self.generation_results: List[GenerationResult] = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_path = Path(self.config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            rotation="50 MB",
            retention="10 days",
            level=self.config.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
    
    def run_pipeline(self, dataset_paths: List[str], dataset_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the complete data generation pipeline."""
        logger.info("Starting data generation pipeline")
        start_time = time.time()
        
        try:
            # Step 1: Load and split data
            logger.info("Step 1: Loading and splitting data")
            train_data, test_data = self._load_and_split_data(dataset_paths, dataset_types)
            
            # Step 2: Generate training data
            logger.info("Step 2: Generating training data")
            train_results = asyncio.run(self._generate_data_batch(train_data, "train"))
            
            # Step 3: Generate test data (optional, smaller batch)
            logger.info("Step 3: Generating test data")
            test_sample_size = min(len(test_data), len(train_data) // 10)  # 10% of training data
            test_sample = test_data[:test_sample_size]
            test_results = asyncio.run(self._generate_data_batch(test_sample, "test"))
            
            # Step 4: Process and format results
            logger.info("Step 4: Processing and formatting results")
            formatted_data = self._format_results(train_results, test_results)
            
            # Step 5: Save results
            logger.info("Step 5: Saving results")
            output_paths = self._save_results(formatted_data)
            
            # Step 6: Generate summary
            summary = self._generate_summary(formatted_data, start_time, output_paths)
            
            logger.info("Pipeline completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _load_and_split_data(self, dataset_paths: List[str], dataset_types: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load and split data into train and test sets."""
        if len(dataset_paths) == 1:
            # Single dataset
            data = self.data_splitter.load_dataset(dataset_paths[0], dataset_types[0] if dataset_types else "auto")
            train_data, test_data = self.data_splitter.split_data(data)
        else:
            # Multiple datasets
            train_data, test_data = self.data_splitter.process_multiple_datasets(dataset_paths, dataset_types)
        
        logger.info(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
        return train_data, test_data
    
    async def _generate_data_batch(self, data: List[Dict[str, Any]], split_name: str) -> List[GenerationResult]:
        """Generate data for a batch of samples."""
        logger.info(f"Generating data for {len(data)} samples in {split_name} split")
        
        # Check for existing checkpoint
        checkpoint_path = f"checkpoints/{split_name}_checkpoint.json"
        if self.config.enable_resume and Path(checkpoint_path).exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint_data = load_checkpoint(checkpoint_path)
            start_index = checkpoint_data.get("processed_samples", 0)
        else:
            start_index = 0
        
        results = []
        batch_size = self.config.batch_size
        
        # Process data in batches
        for i in range(start_index, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch_results = await self._process_batch(batch_data, i)
            results.extend(batch_results)
            
            # Save checkpoint
            if self.config.enable_resume and i % self.config.checkpoint_every == 0:
                checkpoint_data = {
                    "processed_samples": i + len(batch_data),
                    "total_samples": len(data),
                    "timestamp": time.time()
                }
                save_checkpoint(checkpoint_data, checkpoint_path)
                logger.info(f"Saved checkpoint: {i + len(batch_data)}/{len(data)} samples processed")
        
        return results
    
    async def _process_batch(self, batch_data: List[Dict[str, Any]], batch_index: int) -> List[GenerationResult]:
        """Process a batch of data samples."""
        logger.info(f"Processing batch {batch_index // self.config.batch_size + 1}")
        
        tasks = []
        for sample in batch_data:
            task = self._process_single_sample(sample)
            tasks.append(task)
        
        # Process samples concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process sample {batch_index + i}: {result}")
                processed_results.append(GenerationResult(
                    original_data=batch_data[i],
                    generated_conversations=[],
                    success=False,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_sample(self, sample: Dict[str, Any]) -> GenerationResult:
        """Process a single data sample."""
        start_time = time.time()
        
        try:
            # Generate prompt from sample
            prompt = self._create_prompt_from_sample(sample)
            
            # Run conversation
            conversation_history = await self.conversation_manager.run_conversation(prompt, sample)
            
            # Validate generated data
            if not self._validate_conversation(conversation_history):
                return GenerationResult(
                    original_data=sample,
                    generated_conversations=[],
                    success=False,
                    error_message="Generated conversation failed validation"
                )
            
            processing_time = time.time() - start_time
            
            return GenerationResult(
                original_data=sample,
                generated_conversations=conversation_history,
                success=True,
                processing_time=processing_time,
                api_client_used=conversation_history[-1].get("client", "unknown") if conversation_history else "none"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing sample: {e}")
            return GenerationResult(
                original_data=sample,
                generated_conversations=[],
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _create_prompt_from_sample(self, sample: Dict[str, Any]) -> str:
        """Create a prompt from a data sample."""
        # This is a flexible prompt generation that can handle various data formats
        
        # Common prompt templates
        if "question" in sample and "answer" in sample:
            # Q&A format
            return f"问题：{sample['question']}\n请提供详细的回答。"
        
        elif "instruction" in sample:
            # Instruction format
            return f"指令：{sample['instruction']}\n请按照指令执行并提供详细的回答。"
        
        elif "prompt" in sample:
            # Direct prompt format
            return sample["prompt"]
        
        elif "text" in sample:
            # Text completion format
            return f"请基于以下内容进行扩展和详细说明：\n{sample['text']}"
        
        elif "context" in sample:
            # Context-based format
            context = sample["context"]
            if "question" in sample:
                return f"基于以下背景信息：{context}\n问题：{sample['question']}\n请提供详细的回答。"
            else:
                return f"基于以下背景信息进行分析和说明：\n{context}"
        
        else:
            # Generic format - try to extract meaningful content
            content_fields = ["content", "input", "description", "task", "query"]
            for field in content_fields:
                if field in sample:
                    return f"请对以下内容进行详细分析和说明：\n{sample[field]}"
            
            # Last resort - use the entire sample as context
            return f"请基于以下信息进行分析和详细说明：\n{json.dumps(sample, ensure_ascii=False, indent=2)}"
    
    def _validate_conversation(self, conversation_history: List[Dict[str, Any]]) -> bool:
        """Validate generated conversation."""
        if not conversation_history:
            return False
        
        # Check minimum conversation length
        if len(conversation_history) < 1:
            return False
        
        # Check response quality
        for turn in conversation_history:
            response = turn.get("response", "")
            if len(response.strip()) < self.config.generation.min_answer_length:
                return False
            
            if len(response) > self.config.generation.max_answer_length:
                return False
        
        return True
    
    def _format_results(self, train_results: List[GenerationResult], test_results: List[GenerationResult]) -> Dict[str, Any]:
        """Format results for output."""
        formatted_data = {
            "train": [],
            "test": [],
            "metadata": {
                "total_train_samples": len(train_results),
                "successful_train_samples": sum(1 for r in train_results if r.success),
                "total_test_samples": len(test_results),
                "successful_test_samples": sum(1 for r in test_results if r.success),
                "generation_config": self.config.generation.model_dump(),
                "timestamp": time.time()
            }
        }
        
        # Process training results
        for result in train_results:
            if result.success:
                formatted_sample = self._format_single_result(result)
                formatted_data["train"].append(formatted_sample)
        
        # Process test results
        for result in test_results:
            if result.success:
                formatted_sample = self._format_single_result(result)
                formatted_data["test"].append(formatted_sample)
        
        # Apply deduplication if enabled
        if self.config.generation.filter_duplicates:
            formatted_data["train"] = filter_duplicates(
                formatted_data["train"], 
                similarity_threshold=self.config.generation.similarity_threshold
            )
            formatted_data["test"] = filter_duplicates(
                formatted_data["test"], 
                similarity_threshold=self.config.generation.similarity_threshold
            )
        
        return formatted_data
    
    def _format_single_result(self, result: GenerationResult) -> Dict[str, Any]:
        """Format a single generation result."""
        # Create SFT training format
        conversations = []
        
        for turn in result.generated_conversations:
            conversation = {
                "instruction": turn.get("prompt", ""),
                "output": turn.get("response", ""),
                "input": "",  # For instruction-following format
                "metadata": {
                    "round": turn.get("round", 1),
                    "api_client": turn.get("client", "unknown"),
                    "timestamp": turn.get("timestamp", time.time())
                }
            }
            conversations.append(conversation)
        
        return {
            "conversations": conversations,
            "original_data": result.original_data,
            "generation_metadata": {
                "success": result.success,
                "processing_time": result.processing_time,
                "api_client_used": result.api_client_used,
                "num_turns": len(result.generated_conversations)
            }
        }
    
    def _save_results(self, formatted_data: Dict[str, Any]) -> Dict[str, str]:
        """Save generated results to files."""
        output_dir = Path(self.config.data.output_data_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = {}
        
        # Save training data
        train_path = output_dir / "sft_train_data.jsonl"
        self._save_jsonl(formatted_data["train"], train_path)
        output_paths["train"] = str(train_path)
        
        # Save test data
        test_path = output_dir / "sft_test_data.jsonl"
        self._save_jsonl(formatted_data["test"], test_path)
        output_paths["test"] = str(test_path)
        
        # Save metadata
        metadata_path = output_dir / "generation_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data["metadata"], f, ensure_ascii=False, indent=2)
        output_paths["metadata"] = str(metadata_path)
        
        # Save in different formats for compatibility
        
        # Alpaca format
        alpaca_train_path = output_dir / "alpaca_train_data.json"
        alpaca_train_data = create_sft_dataset_format(formatted_data["train"], format_type="alpaca")
        with open(alpaca_train_path, 'w', encoding='utf-8') as f:
            json.dump(alpaca_train_data, f, ensure_ascii=False, indent=2)
        output_paths["alpaca_train"] = str(alpaca_train_path)
        
        # ShareGPT format
        sharegpt_train_path = output_dir / "sharegpt_train_data.json"
        sharegpt_train_data = create_sft_dataset_format(formatted_data["train"], format_type="sharegpt")
        with open(sharegpt_train_path, 'w', encoding='utf-8') as f:
            json.dump(sharegpt_train_data, f, ensure_ascii=False, indent=2)
        output_paths["sharegpt_train"] = str(sharegpt_train_path)
        
        return output_paths
    
    def _save_jsonl(self, data: List[Dict[str, Any]], file_path: Path):
        """Save data in JSONL format."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    
    def _generate_summary(self, formatted_data: Dict[str, Any], start_time: float, output_paths: Dict[str, str]) -> Dict[str, Any]:
        """Generate pipeline execution summary."""
        end_time = time.time()
        execution_time = end_time - start_time
        
        summary = {
            "execution_summary": {
                "total_execution_time": execution_time,
                "start_time": start_time,
                "end_time": end_time,
                "config": self.config.model_dump()
            },
            "data_summary": formatted_data["metadata"],
            "output_files": output_paths,
            "api_statistics": self.api_client.get_client_statistics()
        }
        
        # Save summary
        summary_path = Path(self.config.data.output_data_path) / "pipeline_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Pipeline completed in {execution_time:.2f} seconds")
        logger.info(f"Generated {summary['data_summary']['successful_train_samples']} training samples")
        logger.info(f"Generated {summary['data_summary']['successful_test_samples']} test samples")
        
        return summary
    
    def resume_pipeline(self, checkpoint_path: str) -> Dict[str, Any]:
        """Resume pipeline from checkpoint."""
        logger.info(f"Resuming pipeline from checkpoint: {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint_data = load_checkpoint(checkpoint_path)
        
        # Restore pipeline state
        self.processed_samples = checkpoint_data.get("processed_samples", 0)
        self.checkpoint_data = checkpoint_data
        
        logger.info(f"Resumed from checkpoint: {self.processed_samples} samples already processed")
        
        return checkpoint_data
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        return {
            "processed_samples": self.processed_samples,
            "failed_samples": self.failed_samples,
            "total_results": len(self.generation_results),
            "successful_results": sum(1 for r in self.generation_results if r.success),
            "api_statistics": self.api_client.get_client_statistics()
        }