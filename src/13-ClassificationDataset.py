# PyTorch Dataset Module 
#------------------------
# Enhanced Dataset class for classification tasks (refusal & jailbreak).
# 
# IMPROVEMENTS:
# - Full Config/Utils integration
# - Support for prompt+response concatenation
# - Confidence scores as sample weights
# - Data validation
# - Tokenization caching
# - Dataset statistics
# All imports are in 01-Imports.py
###############################################################################


class ClassificationDataset(Dataset):
    """Enhanced PyTorch Dataset for classification tasks (refusal & jailbreak)."""
    
    def __init__(self, 
                 texts: List[str], 
                 labels: List[int],
                 tokenizer: RobertaTokenizer,
                 prompts: List[str] = None,
                 confidences: List[float] = None,
                 max_length: int = None,
                 use_cache: bool = True,
                 validate_labels: bool = True,
                 task_type: str = 'refusal'):
        """
        Initialize enhanced dataset.
        
        Args:
            texts: List of response texts
            labels: List of integer labels
            tokenizer: HuggingFace tokenizer
            prompts: Optional list of prompts to concatenate with responses
            confidences: Optional confidence scores for weighted sampling
            max_length: Maximum sequence length (uses config if None)
            use_cache: Whether to cache tokenized inputs
            validate_labels: Whether to validate label values
            task_type: 'refusal' or 'jailbreak' for validation
        """
        # Use config values - NO HARDCODING!
        self.max_length = max_length or MODEL_CONFIG['max_length']
        self.verbose = EXPERIMENT_CONFIG.get('verbose', True)
        
        # Core data
        self.texts = texts
        self.labels = labels
        self.prompts = prompts
        self.confidences = confidences if confidences is not None else [1.0] * len(texts)
        self.tokenizer = tokenizer
        self.task_type = task_type
        
        # Validation
        assert len(texts) == len(labels), "Texts and labels must have same length"
        if prompts is not None:
            assert len(prompts) == len(texts), "Prompts must match texts length"
        if confidences is not None:
            assert len(confidences) == len(texts), "Confidences must match texts length"
        
        # Validate labels if requested
        if validate_labels:
            self._validate_labels()
        
        # Caching
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        
        # Statistics
        self.stats = self._compute_statistics()
        
        if self.verbose:
            self._print_dataset_info()
    
    def _validate_labels(self):
        """Validate label values based on task type."""
        unique_labels = set(self.labels)
        
        if self.task_type == 'refusal':
            # Refusal: 0, 1, 2, -1 (error)
            valid_labels = {0, 1, 2, -1}
            invalid = unique_labels - valid_labels
            if invalid:
                raise ValueError(f"Invalid refusal labels found: {invalid}. "
                               f"Valid labels are {valid_labels}")
        
        elif self.task_type == 'jailbreak':
            # Jailbreak: 0, 1, -1 (error)
            valid_labels = {0, 1, -1}
            invalid = unique_labels - valid_labels
            if invalid:
                raise ValueError(f"Invalid jailbreak labels found: {invalid}. "
                               f"Valid labels are {valid_labels}")
        
        # Check for error labels
        error_count = sum(1 for label in self.labels if label == -1)
        if error_count > 0:
            print(f"⚠️  Warning: {error_count} error labels (-1) found in dataset")
    
    def _compute_statistics(self) -> Dict:
        """Compute dataset statistics."""
        # Label distribution
        label_counts = {}
        for label in self.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Text length statistics
        text_lengths = [len(text.split()) for text in self.texts]
        
        # Confidence statistics if available
        conf_stats = {}
        if self.confidences is not None:
            conf_stats = {
                'mean': np.mean(self.confidences),
                'std': np.std(self.confidences),
                'min': np.min(self.confidences),
                'max': np.max(self.confidences)
            }
        
        return {
            'size': len(self.texts),
            'label_distribution': label_counts,
            'text_length': {
                'mean': np.mean(text_lengths),
                'std': np.std(text_lengths),
                'min': np.min(text_lengths),
                'max': np.max(text_lengths)
            },
            'confidence': conf_stats,
            'has_prompts': self.prompts is not None
        }
    
    def _print_dataset_info(self):
        """Print dataset information."""
        print_banner(f"{self.task_type.upper()} DATASET INFO", char="─")
        print(f"  Size: {self.stats['size']:,} samples")
        print(f"  Task: {self.task_type}")
        print(f"  Max length: {self.max_length}")
        print(f"  Cache enabled: {self.use_cache}")
        
        # Label distribution
        print(f"\n  Label distribution:")
        for label, count in sorted(self.stats['label_distribution'].items()):
            pct = safe_divide(count, self.stats['size'], 0) * 100
            if self.task_type == 'refusal' and label != -1:
                label_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else f'Label_{label}'
            elif self.task_type == 'jailbreak':
                label_name = {0: 'Failed', 1: 'Succeeded', -1: 'Error'}.get(label, f'Label_{label}')
            else:
                label_name = f'Label_{label}'
            print(f"    {label_name}: {count:,} ({pct:.1f}%)")
        
        # Text length stats
        print(f"\n  Text length (words):")
        print(f"    Mean: {self.stats['text_length']['mean']:.1f}")
        print(f"    Std: {self.stats['text_length']['std']:.1f}")
        print(f"    Range: [{self.stats['text_length']['min']:.0f}, {self.stats['text_length']['max']:.0f}]")
        
        # Confidence stats if available
        if self.stats['confidence']:
            print(f"\n  Confidence scores:")
            print(f"    Mean: {self.stats['confidence']['mean']:.1f}")
            print(f"    Std: {self.stats['confidence']['std']:.1f}")
            print(f"    Range: [{self.stats['confidence']['min']:.1f}, {self.stats['confidence']['max']:.1f}]")
        
        print("─" * 60)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index with optional caching.
        
        Args:
            idx: Index
        
        Returns:
            Dictionary with keys: input_ids, attention_mask, label, confidence
        """
        # Check cache first
        if self.use_cache and idx in self.cache:
            return self.cache[idx]
        
        # Prepare text
        text = str(self.texts[idx])
        
        # Optionally concatenate prompt and response
        if self.prompts is not None:
            prompt = str(self.prompts[idx])
            # Use special tokens to separate prompt and response
            text = f"{prompt} [SEP] {text}"
        
        label = self.labels[idx]
        confidence = self.confidences[idx] if self.confidences is not None else 1.0
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'confidence': torch.tensor(confidence, dtype=torch.float32)
        }
        
        # Cache if enabled
        if self.use_cache:
            self.cache[idx] = item
        
        return item
    
    def get_sample_weights(self) -> np.ndarray:
        """
        Get sample weights for weighted sampling based on confidence.
        
        Returns:
            Array of sample weights
        """
        # Inverse confidence weighting - lower confidence gets higher weight
        # This helps the model learn from uncertain cases
        weights = 1.0 / (np.array(self.confidences) + 1e-6)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        return weights
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.
        
        Returns:
            Tensor of class weights
        """
        # Count samples per class
        class_counts = {}
        for label in self.labels:
            if label != -1:  # Exclude error labels
                class_counts[label] = class_counts.get(label, 0) + 1
        
        # Calculate weights (inverse frequency)
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        
        weights = []
        for label in sorted(class_counts.keys()):
            weight = total / (num_classes * class_counts[label])
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(train_dataset: Dataset, 
                      val_dataset: Dataset,
                      test_dataset: Dataset, 
                      batch_size: int = None,
                      num_workers: int = None,
                      use_weighted_sampling: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders with optional weighted sampling.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size (uses config if None)
        num_workers: Number of workers (uses config if None)
        use_weighted_sampling: Whether to use confidence-based weighted sampling
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Use config values - NO HARDCODING!
    batch_size = batch_size or TRAINING_CONFIG['batch_size']
    num_workers = num_workers or TRAINING_CONFIG['num_workers']
    pin_memory = TRAINING_CONFIG['pin_memory']
    
    # Create weighted sampler if requested
    train_sampler = None
    if use_weighted_sampling and hasattr(train_dataset, 'get_sample_weights'):
        weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    if EXPERIMENT_CONFIG.get('verbose', True):
        print_banner("DATALOADERS CREATED", char="─")
        print(f"  Batch size: {batch_size}")
        print(f"  Num workers: {num_workers}")
        print(f"  Pin memory: {pin_memory}")
        print(f"  Weighted sampling: {use_weighted_sampling}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print("─" * 60)
    
    return train_loader, val_loader, test_loader


def create_jailbreak_filtered_dataset(df: pd.DataFrame,
                                     tokenizer: RobertaTokenizer,
                                     text_column: str = 'response',
                                     label_column: str = 'jailbreak_success') -> Dataset:
    """
    Create dataset filtered for jailbreak attempts only.
    
    IMPORTANT: Uses the is_jailbreak_attempt field to filter!
    
    Args:
        df: DataFrame with labeled data
        tokenizer: Tokenizer to use
        text_column: Column containing text
        label_column: Column containing labels
    
    Returns:
        ClassificationDataset for jailbreak classification
    """
    # Filter for jailbreak attempts only
    if 'is_jailbreak_attempt' in df.columns:
        jailbreak_df = df[df['is_jailbreak_attempt'] == 1].copy()
        if EXPERIMENT_CONFIG.get('verbose', True):
            print(f"Filtered to {len(jailbreak_df):,} jailbreak attempts "
                  f"(from {len(df):,} total samples)")
    else:
        print("⚠️  Warning: is_jailbreak_attempt column not found, using all samples")
        jailbreak_df = df.copy()
    
    # Extract data
    texts = jailbreak_df[text_column].tolist()
    labels = jailbreak_df[label_column].tolist()
    
    # Optional: include prompts for context
    prompts = None
    if 'prompt' in jailbreak_df.columns:
        prompts = jailbreak_df['prompt'].tolist()
    
    # Optional: include confidence scores
    confidences = None
    if 'jailbreak_confidence' in jailbreak_df.columns:
        confidences = jailbreak_df['jailbreak_confidence'].tolist()
    
    return ClassificationDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        prompts=prompts,
        confidences=confidences,
        task_type='jailbreak'
    )


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
