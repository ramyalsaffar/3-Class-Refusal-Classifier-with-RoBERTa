# PyTorch Dataset Module
#-----------------------
# Custom Dataset class for refusal classification.
# All imports are in 00-Imports.py
###############################################################################


class RefusalDataset(Dataset):
    """PyTorch Dataset for refusal classification."""

    def __init__(self, texts: List[str], labels: List[int],
                 tokenizer: RobertaTokenizer, max_length: int = None):
        """
        Initialize dataset.

        Args:
            texts: List of response texts
            labels: List of integer labels (0, 1, 2)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length (uses config if None)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length or MODEL_CONFIG['max_length']

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
                      test_dataset: Dataset, batch_size: int = None,
                      num_workers: int = None):
    """
    Create PyTorch DataLoaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size (uses config if None)
        num_workers: Number of workers (uses config if None)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    batch_size = batch_size or TRAINING_CONFIG['batch_size']
    num_workers = num_workers or TRAINING_CONFIG['num_workers']
    pin_memory = TRAINING_CONFIG['pin_memory']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
