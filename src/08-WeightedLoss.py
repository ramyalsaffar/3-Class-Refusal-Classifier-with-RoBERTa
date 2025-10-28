# Weighted Loss Module
#---------------------
# Handles class imbalance with weighted CrossEntropyLoss.
# All imports are in 00-Imports.py
###############################################################################


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


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
