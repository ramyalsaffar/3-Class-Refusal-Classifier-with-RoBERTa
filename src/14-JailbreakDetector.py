# Jailbreak Detector Model
#-------------------------
# Fine-tuned RoBERTa for binary jailbreak success detection.
# Detects when adversarial attacks successfully bypass AI safety mechanisms.
#
# WHY RoBERTa?
# RoBERTa (Liu et al., 2019) was selected based on literature demonstrating:
# 1. State-of-the-art performance on text classification benchmarks
# 2. Superior contextual understanding for detecting subtle attack patterns
# 3. Proven effectiveness on safety-critical NLP tasks (Vidgen et al., 2021)
# 4. Robust to adversarial inputs with proper fine-tuning
# 5. Bidirectional attention captures both attack vectors and safety mechanism responses
#
# NO BASELINE: Literature already establishes RoBERTa as optimal for security-critical
# binary classification tasks. No need for baseline model comparisons.
#
# All imports are in 01-Imports.py
###############################################################################


class JailbreakDetector(nn.Module):
    """RoBERTa-based binary classifier for jailbreak success detection."""

    def __init__(self, num_classes: int = 2, dropout: float = None,
                 model_name: str = None):
        """
        Initialize jailbreak detector.

        Args:
            num_classes: Number of output classes (2: Success/Failure)
            dropout: Dropout probability (uses config if None)
            model_name: Pre-trained model name (uses config if None)
        """
        super(JailbreakDetector, self).__init__()

        self.num_classes = num_classes
        self.model_name = model_name or MODEL_CONFIG['model_name']
        dropout = dropout or MODEL_CONFIG['dropout']

        # Load pre-trained RoBERTa
        self.roberta = RobertaModel.from_pretrained(self.model_name)

        # Get hidden size from config
        self.hidden_size = self.roberta.config.hidden_size  # 768 for roberta-base

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Binary classification head
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                output_attentions: bool = False):
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            output_attentions: Whether to return attention weights

        Returns:
            If output_attentions=False: Logits (batch_size, 2)
            If output_attentions=True: Tuple of (logits, attention_weights)
        """
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )

        # Use [CLS] token representation (pooled output)
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Classification
        logits = self.classifier(pooled_output)  # (batch_size, 2)

        if output_attentions:
            return logits, outputs.attentions
        return logits

    def predict_with_confidence(self, input_ids: torch.Tensor,
                                attention_mask: torch.Tensor):
        """
        Get predictions with confidence scores.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Tuple of (predicted_labels, confidences, probabilities)
            Labels: 0 = Jailbreak Failed (model defended), 1 = Jailbreak Succeeded (model broken)
        """
        # Get logits
        logits = self.forward(input_ids, attention_mask)

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)

        # Get predicted class and confidence
        confidence, predicted = torch.max(probs, dim=1)

        return predicted, confidence, probs

    def freeze_roberta_layers(self, num_layers_to_freeze: int = None):
        """
        Freeze bottom RoBERTa layers for faster training.

        Args:
            num_layers_to_freeze: Number of encoder layers to freeze (uses config if None)
        """
        num_layers_to_freeze = num_layers_to_freeze or MODEL_CONFIG['freeze_layers']

        # Freeze embeddings
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified number of encoder layers
        for i in range(num_layers_to_freeze):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

        print(f"Froze {num_layers_to_freeze} RoBERTa layers")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 30, 2025
@author: ramyalsaffar
"""
