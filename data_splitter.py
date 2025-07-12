"""Data splitting module for processing datasets like gala, browsecomp, etc."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
import numpy as np
from loguru import logger
import random

class DataSplitter:
    """Data splitter for various dataset formats."""
    
    def __init__(self, config):
        """Initialize data splitter with configuration."""
        self.config = config
        self.random_seed = config.data.random_seed
        self.train_ratio = config.data.train_ratio
        self.test_ratio = config.data.test_ratio
        self.shuffle_data = config.data.shuffle_data
        
        # Set random seeds for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
    
    def load_dataset(self, dataset_path: str, dataset_type: str = "auto") -> List[Dict[str, Any]]:
        """Load dataset from various formats."""
        dataset_path_obj = Path(dataset_path)
        
        if dataset_type == "auto":
            dataset_type = self._detect_dataset_type(dataset_path_obj)
        
        logger.info(f"Loading dataset from {dataset_path} with type {dataset_type}")
        
        if dataset_type == "jsonl":
            return self._load_jsonl(dataset_path_obj)
        elif dataset_type == "json":
            return self._load_json(dataset_path_obj)
        elif dataset_type == "csv":
            return self._load_csv(dataset_path_obj)
        elif dataset_type == "gala":
            return self._load_gala_format(dataset_path_obj)
        elif dataset_type == "browsecomp":
            return self._load_browsecomp_format(dataset_path_obj)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def _detect_dataset_type(self, dataset_path: Path) -> str:
        """Detect dataset type from file extension and content."""
        suffix = dataset_path.suffix.lower()
        
        if suffix == ".jsonl":
            return "jsonl"
        elif suffix == ".json":
            return "json"
        elif suffix == ".csv":
            return "csv"
        else:
            # Try to detect by content
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('{') and first_line.endswith('}'):
                        return "jsonl"
                    elif first_line.startswith('[') or first_line.startswith('{'):
                        return "json"
                    else:
                        return "csv"
            except Exception:
                return "json"  # Default fallback
    
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL format dataset."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON format dataset."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure data is a list
        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
            else:
                data = [data]
        
        return data
    
    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load CSV format dataset."""
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def _load_gala_format(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load GALA dataset format."""
        # GALA specific loading logic
        # This is a placeholder - adjust based on actual GALA format
        logger.info("Loading GALA format dataset")
        return self._load_json(file_path)
    
    def _load_browsecomp_format(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load BrowseComp dataset format."""
        # BrowseComp specific loading logic
        # This is a placeholder - adjust based on actual BrowseComp format
        logger.info("Loading BrowseComp format dataset")
        return self._load_json(file_path)
    
    def split_data(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into train and test sets."""
        if self.shuffle_data:
            random.shuffle(data)
        
        # Calculate split sizes
        total_size = len(data)
        train_size = int(total_size * self.train_ratio)
        
        # Split data
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        logger.info(f"Split data: {len(train_data)} train, {len(test_data)} test samples")
        
        return train_data, test_data
    
    def stratified_split(self, data: List[Dict[str, Any]], stratify_key: str = "category") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Perform stratified splitting based on a key."""
        if stratify_key not in data[0]:
            logger.warning(f"Stratify key '{stratify_key}' not found, using random split")
            return self.split_data(data)
        
        # Extract stratification labels
        labels = [item.get(stratify_key, "unknown") for item in data]
        
        # Perform stratified split
        train_data, test_data = train_test_split(
            data,
            test_size=self.test_ratio,
            random_state=self.random_seed,
            stratify=labels,
            shuffle=self.shuffle_data
        )
        
        logger.info(f"Stratified split: {len(train_data)} train, {len(test_data)} test samples")
        
        return train_data, test_data
    
    def save_split_data(self, train_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]], 
                       output_dir: str, format: str = "jsonl") -> None:
        """Save split data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            self._save_jsonl(train_data, output_path / "train.jsonl")
            self._save_jsonl(test_data, output_path / "test.jsonl")
        elif format == "json":
            self._save_json(train_data, output_path / "train.json")
            self._save_json(test_data, output_path / "test.json")
        elif format == "csv":
            self._save_csv(train_data, output_path / "train.csv")
            self._save_csv(test_data, output_path / "test.csv")
        
        logger.info(f"Saved split data to {output_dir}")
    
    def _save_jsonl(self, data: List[Dict[str, Any]], file_path: Path) -> None:
        """Save data in JSONL format."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    
    def _save_json(self, data: List[Dict[str, Any]], file_path: Path) -> None:
        """Save data in JSON format."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _save_csv(self, data: List[Dict[str, Any]], file_path: Path) -> None:
        """Save data in CSV format."""
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    
    def get_split_statistics(self, train_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the data split."""
        stats = {
            "total_samples": len(train_data) + len(test_data),
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "train_ratio": len(train_data) / (len(train_data) + len(test_data)),
            "test_ratio": len(test_data) / (len(train_data) + len(test_data)),
        }
        
        # Add field statistics if available
        if train_data and isinstance(train_data[0], dict):
            train_keys = set(train_data[0].keys())
            test_keys = set(test_data[0].keys())
            stats["common_fields"] = list(train_keys.intersection(test_keys))
            stats["train_only_fields"] = list(train_keys - test_keys)
            stats["test_only_fields"] = list(test_keys - train_keys)
        
        return stats
    
    def process_multiple_datasets(self, dataset_paths: List[str], dataset_types: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process and combine multiple datasets."""
        all_data = []
        
        if dataset_types is None:
            dataset_types = ["auto"] * len(dataset_paths)
        
        for path, dtype in zip(dataset_paths, dataset_types):
            data = self.load_dataset(path, dtype)
            # Add source information
            for item in data:
                item['_source'] = path
            all_data.extend(data)
        
        logger.info(f"Loaded {len(all_data)} samples from {len(dataset_paths)} datasets")
        
        return self.split_data(all_data)