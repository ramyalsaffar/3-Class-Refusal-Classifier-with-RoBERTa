"""
PyTorch Dataset Module

Custom Dataset class for refusal classification.
"""

import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from typing import List, Dict


class RefusalDataset(Dataset):
    """PyTorch Dataset for refusal classification."""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer: RobertaTokenizer, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            texts: List of response texts
            labels: List of integer labels (0, 1, 2)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        assert len(texts) == len(labels), "Texts and labels must have same length"
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Index
        
        Returns:
            Dictionary with keys: input_ids, attention_mask, label
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(train_dataset: Dataset, val_dataset: Dataset, 
                      test_dataset: Dataset, batch_size: int = 16,
                      num_workers: int = 0):
    """
    Create PyTorch DataLoaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    from transformers import RobertaTokenizer
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    texts = [
        "I cannot help with that request.",
        "Here's how to solve it: ...",
        "I can provide general info, but consult an expert."
    ]
    labels = [1, 0, 2]
    
    dataset = RefusalDataset(texts, labels, tokenizer)
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Label: {sample['label']}")
