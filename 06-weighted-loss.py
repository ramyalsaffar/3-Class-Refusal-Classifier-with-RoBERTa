"""
Weighted Loss Module

Handles class imbalance with weighted CrossEntropyLoss.
"""

import torch
import torch.nn as nn
from typing import List


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
    """
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    
    weights = [total_samples / (num_classes * count) for count in class_counts]
    
    return torch.FloatTensor(weights).to(device)


def get_weighted_criterion(class_counts: List[int], device: torch.device) -> nn.CrossEntropyLoss:
    """
    Get weighted CrossEntropyLoss criterion.
    
    Args:
        class_counts: List of sample counts per class
        device: torch device
    
    Returns:
        nn.CrossEntropyLoss with class weights
    """
    class_weights = calculate_class_weights(class_counts, device)
    
    print(f"Class weights: {class_weights.cpu().numpy()}")
    print(f"  Class 0 (No Refusal): {class_weights[0]:.3f}")
    print(f"  Class 1 (Hard Refusal): {class_weights[1]:.3f}")
    print(f"  Class 2 (Soft Refusal): {class_weights[2]:.3f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    return criterion


if __name__ == "__main__":
    # Test weight calculation
    
    # Example class distribution
    class_counts = [3000, 2385, 615]  # [no_refusal, hard_refusal, soft_refusal]
    device = torch.device('cpu')
    
    print("Class distribution:")
    print(f"  Class 0 (No Refusal): {class_counts[0]} ({class_counts[0]/sum(class_counts)*100:.1f}%)")
    print(f"  Class 1 (Hard Refusal): {class_counts[1]} ({class_counts[1]/sum(class_counts)*100:.1f}%)")
    print(f"  Class 2 (Soft Refusal): {class_counts[2]} ({class_counts[2]/sum(class_counts)*100:.1f}%)")
    print(f"  Total: {sum(class_counts)}\n")
    
    criterion = get_weighted_criterion(class_counts, device)
    
    # Test criterion
    batch_size = 8
    num_classes = 3
    
    logits = torch.randn(batch_size, num_classes)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 2, 2])  # Mix of labels
    
    loss = criterion(logits, labels)
    print(f"\nTest loss: {loss.item():.4f}")
    
    # Compare with unweighted loss
    unweighted_criterion = nn.CrossEntropyLoss()
    unweighted_loss = unweighted_criterion(logits, labels)
    print(f"Unweighted loss: {unweighted_loss.item():.4f}")
    print(f"Difference: {abs(loss.item() - unweighted_loss.item()):.4f}")
