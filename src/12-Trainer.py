# Trainer Module
#---------------
# Training loop with progress tracking, early stopping, and checkpointing.
# All imports are in 00-Imports.py
###############################################################################


class Trainer:
    """Train RoBERTa classifier with progress tracking."""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 scheduler, device):
        """
        Initialize trainer.

        Args:
            model: RefusalClassifier model
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

    def train(self, model_save_path: str = None):
        """
        Full training with early stopping.

        Args:
            model_save_path: Path to save best model (uses default if None)
        """
        if model_save_path is None:
            model_save_path = os.path.join(models_path, f"{EXPERIMENT_CONFIG['experiment_name']}_best.pt")

        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
        print(f"Epochs: {self.epochs}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print(f"Device: {self.device}")
        print(f"Trainable parameters: {count_parameters(self.model):,}")
        print("="*60 + "\n")

        for epoch in range(1, self.epochs + 1):
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
                    self.save_checkpoint(model_save_path)
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

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch,
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
