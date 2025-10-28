"""
RoBERTa Classifier Model

Fine-tuned RoBERTa for 3-class refusal classification.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig


class RefusalClassifier(nn.Module):
    """RoBERTa-based classifier for refusal detection."""
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.1, 
                 model_name: str = 'roberta-base'):
        """
        Initialize classifier.
        
        Args:
            num_classes: Number of output classes (3)
            dropout: Dropout probability
            model_name: Pre-trained model name
        """
        super(RefusalClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pre-trained RoBERTa
        self.roberta = RobertaModel.from_pretrained(model_name)
        
        # Get hidden size from config
        self.hidden_size = self.roberta.config.hidden_size  # 768 for roberta-base
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
        
        Returns:
            Logits (batch_size, num_classes)
        """
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (pooled output)
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)  # (batch_size, num_classes)
        
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
        """
        # Get logits
        logits = self.forward(input_ids, attention_mask)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Get predicted class and confidence
        confidence, predicted = torch.max(probs, dim=1)
        
        return predicted, confidence, probs
    
    def freeze_roberta_layers(self, num_layers_to_freeze: int = 6):
        """
        Freeze bottom RoBERTa layers for faster training.
        
        Args:
            num_layers_to_freeze: Number of encoder layers to freeze
        """
        # Freeze embeddings
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze specified number of encoder layers
        for i in range(num_layers_to_freeze):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False
        
        print(f"Froze {num_layers_to_freeze} RoBERTa layers")
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        print("Unfroze all parameters")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = RefusalClassifier(num_classes=3, dropout=0.1)
    
    print(f"Model: {model.model_name}")
    print(f"Hidden size: {model.hidden_size}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    seq_length = 128
    
    input_ids = torch.randint(0, 50000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    
    logits = model(input_ids, attention_mask)
    print(f"\nLogits shape: {logits.shape}")  # (4, 3)
    
    predicted, confidence, probs = model.predict_with_confidence(input_ids, attention_mask)
    print(f"Predictions: {predicted}")
    print(f"Confidences: {confidence}")
    print(f"Probabilities shape: {probs.shape}")
