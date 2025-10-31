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


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
