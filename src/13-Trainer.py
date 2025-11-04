# Trainer Module
#---------------
# Training loop with progress tracking, early stopping, and checkpointing.
# Includes weighted loss calculation for class imbalance.
# All imports are in 01-Imports.py
###############################################################################


# =============================================================================
# WEIGHTED LOSS FUNCTIONS (for class imbalance)
# =============================================================================

def calculate_class_weights(class_counts: List[int], device: torch.device) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset.

    Formula: weight_i = total_samples / (num_classes * count_i)

    Args:
        class_counts: List of sample counts per class [count_class0, count_class1, count_class2]
        device: torch device

    Returns:
        torch.Tensor of class weights

    Example:
        class_counts = [3000, 2385, 615]  # [no_refusal, hard_refusal, soft_refusal]
        weights = calculate_class_weights(class_counts, device)
        # Result: [0.67, 0.84, 3.25]

    Raises:
        ValueError: If any class has zero samples
    """
    # Check for zero counts (would cause division by zero)
    if any(count == 0 for count in class_counts):
        raise ValueError(
            f"Class counts contain zeros: {class_counts}. "
            f"All classes must have at least one sample for weighted loss calculation."
        )

    total_samples = sum(class_counts)
    num_classes = len(class_counts)

    # Safe division - all counts verified to be > 0
    weights = [total_samples / (num_classes * count) for count in class_counts]

    return torch.FloatTensor(weights).to(device)


def get_weighted_criterion(class_counts: List[int], device: torch.device,
                          class_names: List[str] = None) -> nn.CrossEntropyLoss:
    """
    Get weighted CrossEntropyLoss criterion.

    Generic function that works for any number of classes (binary, 3-class, multi-class).

    Args:
        class_counts: List of sample counts per class
        device: torch device
        class_names: Optional list of class names for display (default: Class 0, Class 1, ...)

    Returns:
        nn.CrossEntropyLoss with class weights
    """
    class_weights = calculate_class_weights(class_counts, device)

    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Generic printing for any number of classes
    for i, weight in enumerate(class_weights):
        class_label = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
        print(f"  {class_label}: {weight:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    return criterion


# =============================================================================
# TRAINER CLASS
# =============================================================================

class Trainer:
    """Train RoBERTa classifier with progress tracking."""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 scheduler, device):
        """
        Initialize trainer.

        Generic trainer that works for any classification model.

        Args:
            model: Classification model (RefusalClassifier or JailbreakDetector)
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: torch device
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Configuration from config
        self.epochs = TRAINING_CONFIG['epochs']
        self.gradient_clip = TRAINING_CONFIG['gradient_clip']
        self.early_stopping_patience = TRAINING_CONFIG['early_stopping_patience']
        self.save_best_only = TRAINING_CONFIG['save_best_only']

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_accuracy': []
        }

        # Early stopping tracking
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def train_epoch(self) -> float:
        """
        Train one epoch with progress bar.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        with tqdm(total=num_batches, desc="Training", leave=False) as pbar:
            for batch_idx, batch in enumerate(self.train_loader):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                # Update weights
                self.optimizer.step()
                self.scheduler.step()

                # Track loss
                total_loss += loss.item()

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(self) -> Tuple[float, float, float]:
        """
        Validate with metrics calculation.

        Returns:
            Tuple of (avg_loss, f1_score, accuracy)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        num_batches = len(self.val_loader)

        with torch.no_grad():
            with tqdm(total=num_batches, desc="Validation", leave=False) as pbar:
                for batch in self.val_loader:
                    # Move to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)

                    # Forward pass
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)

                    # Get predictions
                    preds = torch.argmax(logits, dim=1)

                    # Track metrics
                    total_loss += loss.item()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    # Update progress bar
                    pbar.update(1)

        # Calculate metrics
        avg_loss = total_loss / num_batches
        f1 = f1_score(all_labels, all_preds, average='macro')
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, f1, accuracy

    def train(self, model_save_path: str = None, resume_from_checkpoint: str = None):
        """
        Full training with early stopping and optional checkpoint resumption.

        Args:
            model_save_path: Path to save best model (uses default if None)
            resume_from_checkpoint: Path to checkpoint to resume training from (optional)

        Example:
            # Start new training
            trainer.train()

            # Resume from checkpoint if training was interrupted
            trainer.train(resume_from_checkpoint='models/experiment_best.pt')
        """
        if model_save_path is None:
            model_save_path = os.path.join(models_path, f"{EXPERIMENT_CONFIG['experiment_name']}_best.pt")

        # Resume from checkpoint if specified
        start_epoch = 1
        if resume_from_checkpoint:
            if os.path.exists(resume_from_checkpoint):
                print(f"\n🔄 Resuming training from checkpoint: {resume_from_checkpoint}")
                self.load_checkpoint(resume_from_checkpoint)
                start_epoch = self.best_epoch + 1
                print(f"   Resuming from epoch {start_epoch}/{self.epochs}")
                print(f"   Previous best F1: {self.best_val_f1:.4f}\n")
            else:
                print(f"\n⚠️  Checkpoint not found: {resume_from_checkpoint}")
                print("   Starting training from scratch\n")

        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
        print(f"Epochs: {self.epochs} (starting from epoch {start_epoch})")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print(f"Device: {self.device}")
        print(f"Trainable parameters: {count_parameters(self.model):,}")
        print("="*60 + "\n")

        for epoch in range(start_epoch, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            print("-" * 40)

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_f1, val_accuracy = self.validate_epoch()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            self.history['val_accuracy'].append(val_accuracy)

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val F1: {val_f1:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")

            # Check for improvement
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                self.patience_counter = 0

                # Save best model
                if self.save_best_only or epoch == self.epochs:
                    self.save_checkpoint(model_save_path, epoch=epoch)
                    print(f"✓ Saved best model (F1: {val_f1:.4f})")
            else:
                self.patience_counter += 1
                print(f"No improvement ({self.patience_counter}/{self.early_stopping_patience})")

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                print(f"Best epoch: {self.best_epoch} (F1: {self.best_val_f1:.4f})")
                break

        # Training complete
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best Validation F1: {self.best_val_f1:.4f} (Epoch {self.best_epoch})")
        print(f"Model saved to: {model_save_path}")
        print("="*60 + "\n")

        return self.history

    def save_checkpoint(self, path: str, epoch: int = None):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch (optional, uses best_epoch if not provided)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch,
            'epoch': epoch if epoch is not None else self.best_epoch,
            'history': self.history
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_f1 = checkpoint['best_val_f1']
        self.best_epoch = checkpoint['best_epoch']
        self.history = checkpoint['history']
        print(f"✓ Loaded checkpoint from {path}")
        print(f"  Best Val F1: {self.best_val_f1:.4f}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
