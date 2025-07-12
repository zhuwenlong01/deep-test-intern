"""Utility functions for the data generation pipeline."""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import difflib
from loguru import logger

def save_checkpoint(checkpoint_data: Dict[str, Any], checkpoint_path: str) -> None:
    """Save checkpoint data to file."""
    checkpoint_path_obj = Path(checkpoint_path)
    checkpoint_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(checkpoint_path_obj, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint data from file."""
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using difflib."""
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    text1 = text1.strip().lower()
    text2 = text2.strip().lower()
    
    # Calculate similarity ratio
    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    return similarity

def filter_duplicates(data: List[Dict[str, Any]], similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
    """Filter out duplicate or highly similar samples."""
    if not data:
        return data
    
    filtered_data = []
    seen_texts = []
    
    for sample in data:
        # Extract text content for comparison
        text_content = extract_text_content(sample)
        
        # Check for duplicates
        is_duplicate = False
        for seen_text in seen_texts:
            similarity = calculate_text_similarity(text_content, seen_text)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_data.append(sample)
            seen_texts.append(text_content)
    
    logger.info(f"Filtered {len(data) - len(filtered_data)} duplicate samples")
    return filtered_data

def extract_text_content(sample: Dict[str, Any]) -> str:
    """Extract text content from a sample for similarity comparison."""
    # Try to extract meaningful text from different formats
    
    if "conversations" in sample:
        # Extract from conversations
        texts = []
        for conv in sample["conversations"]:
            if "instruction" in conv:
                texts.append(conv["instruction"])
            if "output" in conv:
                texts.append(conv["output"])
        return " ".join(texts)
    
    elif "instruction" in sample and "output" in sample:
        return f"{sample['instruction']} {sample['output']}"
    
    elif "question" in sample and "answer" in sample:
        return f"{sample['question']} {sample['answer']}"
    
    elif "prompt" in sample and "response" in sample:
        return f"{sample['prompt']} {sample['response']}"
    
    else:
        # Try to extract any text content
        text_parts = []
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 10:
                text_parts.append(value)
        return " ".join(text_parts)

def calculate_content_hash(content: str) -> str:
    """Calculate hash of content for deduplication."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def validate_generated_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate generated data quality."""
    valid_data = []
    
    for sample in data:
        if is_valid_sample(sample):
            valid_data.append(sample)
        else:
            logger.warning("Invalid sample found and removed")
    
    return valid_data

def is_valid_sample(sample: Dict[str, Any]) -> bool:
    """Check if a sample is valid."""
    # Check basic structure
    if not isinstance(sample, dict):
        return False
    
    # Check for required fields based on format
    if "conversations" in sample:
        conversations = sample["conversations"]
        if not isinstance(conversations, list) or len(conversations) == 0:
            return False
        
        for conv in conversations:
            if not isinstance(conv, dict):
                return False
            if "instruction" not in conv or "output" not in conv:
                return False
            if not conv["instruction"].strip() or not conv["output"].strip():
                return False
    
    elif "instruction" in sample and "output" in sample:
        if not sample["instruction"].strip() or not sample["output"].strip():
            return False
    
    else:
        # Check for any meaningful content
        text_content = extract_text_content(sample)
        if len(text_content.strip()) < 10:
            return False
    
    return True

def create_sft_dataset_format(data: List[Dict[str, Any]], format_type: str = "alpaca") -> List[Dict[str, Any]]:
    """Convert data to different SFT dataset formats."""
    formatted_data = []
    
    for sample in data:
        if format_type == "alpaca":
            formatted_sample = convert_to_alpaca_format(sample)
        elif format_type == "sharegpt":
            formatted_sample = convert_to_sharegpt_format(sample)
        elif format_type == "chatglm":
            formatted_sample = convert_to_chatglm_format(sample)
        else:
            formatted_sample = sample
        
        if formatted_sample:
            formatted_data.append(formatted_sample)
    
    return formatted_data

def convert_to_alpaca_format(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert sample to Alpaca format."""
    if "conversations" in sample and sample["conversations"]:
        # Use the first conversation
        conv = sample["conversations"][0]
        return {
            "instruction": conv.get("instruction", ""),
            "input": conv.get("input", ""),
            "output": conv.get("output", "")
        }
    
    elif "instruction" in sample and "output" in sample:
        return {
            "instruction": sample["instruction"],
            "input": sample.get("input", ""),
            "output": sample["output"]
        }
    
    else:
        # Try to extract from original data
        original_data = sample.get("original_data", {})
        if "question" in original_data and "answer" in original_data:
            return {
                "instruction": original_data["question"],
                "input": "",
                "output": original_data["answer"]
            }
    
    return None

def convert_to_sharegpt_format(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert sample to ShareGPT format."""
    conversations = []
    
    if "conversations" in sample and sample["conversations"]:
        for conv in sample["conversations"]:
            conversations.append({
                "from": "human",
                "value": conv.get("instruction", "")
            })
            conversations.append({
                "from": "gpt",
                "value": conv.get("output", "")
            })
    
    elif "instruction" in sample and "output" in sample:
        conversations.append({
            "from": "human",
            "value": sample["instruction"]
        })
        conversations.append({
            "from": "gpt",
            "value": sample["output"]
        })
    
    else:
        original_data = sample.get("original_data", {})
        if "question" in original_data and "answer" in original_data:
            conversations.append({
                "from": "human",
                "value": original_data["question"]
            })
            conversations.append({
                "from": "gpt",
                "value": original_data["answer"]
            })
    
    if conversations:
        return {
            "conversations": conversations,
            "id": sample.get("id", "")
        }
    
    return None

def convert_to_chatglm_format(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert sample to ChatGLM format."""
    if "conversations" in sample and sample["conversations"]:
        conv = sample["conversations"][0]
        return {
            "prompt": conv.get("instruction", ""),
            "response": conv.get("output", ""),
            "history": []
        }
    
    elif "instruction" in sample and "output" in sample:
        return {
            "prompt": sample["instruction"],
            "response": sample["output"],
            "history": []
        }
    
    else:
        original_data = sample.get("original_data", {})
        if "question" in original_data and "answer" in original_data:
            return {
                "prompt": original_data["question"],
                "response": original_data["answer"],
                "history": []
            }
    
    return None

def create_directory_structure(base_path: str) -> None:
    """Create necessary directory structure."""
    directories = [
        "data/input",
        "data/output",
        "logs",
        "checkpoints",
        "configs"
    ]
    
    base_path_obj = Path(base_path)
    
    for directory in directories:
        dir_path = base_path_obj / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def load_sample_data(sample_path: str) -> List[Dict[str, Any]]:
    """Load sample data for testing."""
    sample_data = [
        {
            "question": "什么是人工智能？",
            "answer": "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。",
            "category": "AI"
        },
        {
            "question": "如何学习机器学习？",
            "answer": "学习机器学习需要掌握数学基础、编程技能和实践经验。",
            "category": "ML"
        },
        {
            "question": "深度学习和机器学习有什么区别？",
            "answer": "深度学习是机器学习的一个子集，使用多层神经网络来学习复杂的模式。",
            "category": "DL"
        }
    ]
    
    # Save sample data
    sample_path_obj = Path(sample_path)
    sample_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(sample_path_obj, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created sample data at {sample_path}")
    return sample_data

def get_file_stats(file_path: str) -> Dict[str, Any]:
    """Get file statistics."""
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        return {"exists": False}
    
    stat = file_path_obj.stat()
    
    return {
        "exists": True,
        "size": stat.st_size,
        "size_mb": stat.st_size / (1024 * 1024),
        "modified_time": stat.st_mtime,
        "is_file": file_path_obj.is_file(),
        "is_dir": file_path_obj.is_dir()
    }

def format_execution_time(seconds: float) -> str:
    """Format execution time in a human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)} minutes {remaining_seconds:.2f} seconds"
    else:
        hours = seconds // 3600
        remaining_seconds = seconds % 3600
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60
        return f"{int(hours)} hours {int(minutes)} minutes {seconds:.2f} seconds"

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - 3] + "..."

def count_tokens_estimate(text: str) -> int:
    """Estimate token count (rough approximation)."""
    # Simple estimation: 1 token ≈ 4 characters for English, 1 token ≈ 1.5 characters for Chinese
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    english_chars = len(text) - chinese_chars
    
    estimated_tokens = chinese_chars / 1.5 + english_chars / 4
    return int(estimated_tokens)

def validate_api_keys(config) -> Dict[str, bool]:
    """Validate API keys availability."""
    validation_results = {
        "claude": bool(config.api.claude_api_key),
        "gemini": bool(config.api.gemini_api_key),
        "openai": bool(config.api.openai_api_key)
    }
    
    return validation_results

def create_sample_config(config_path: str) -> None:
    """Create a sample configuration file."""
    from config import PipelineConfig
    
    config = PipelineConfig()
    config.to_yaml(config_path)
    logger.info(f"Created sample configuration at {config_path}")

def merge_datasets(dataset_paths: List[str], output_path: str) -> None:
    """Merge multiple datasets into one."""
    merged_data = []
    
    for path in dataset_paths:
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.jsonl'):
                for line in f:
                    merged_data.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    merged_data.append(data)
    
    # Save merged data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Merged {len(merged_data)} samples to {output_path}")

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    # Remove common artifacts
    text = text.replace('\r', '').replace('\n\n\n', '\n\n')
    
    return text.strip()