# models/wav2vec_model.py

import torch
import torch.nn as nn
from transformers import Wav2VecForCTC, Wav2Vec2Config, Wav2Vec2Processor
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class XLSRConfig:
    """Configuration for XLSR Wav2Vec Model"""
    pretrained_model_name: str = "facebook/wav2vec2-large-xlsr-53"
    freeze_feature_encoder: bool = True
    vocab_size: int = 184  # Adjusted for Tagalog + English characters
    hidden_dropout: float = 0.1
    activation_dropout: float = 0.1
    attention_dropout: float = 0.1
    feat_proj_dropout: float = 0.0
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    layerdrop: float = 0.1
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = True
    gradient_checkpointing: bool = True  # Enable for memory efficiency
    pad_token_id: int = 0

class TaglishXLSR(nn.Module):
    def __init__(self, config: XLSRConfig):
        super().__init__()
        
        self.config = config
        
        # Initialize XLSR Wav2Vec model
        self.wav2vec = Wav2VecForCTC.from_pretrained(
            config.pretrained_model_name,
            vocab_size=config.vocab_size,
            hidden_dropout=config.hidden_dropout,
            activation_dropout=config.activation_dropout,
            attention_dropout=config.attention_dropout,
            feat_proj_dropout=config.feat_proj_dropout,
            mask_time_prob=config.mask_time_prob,
            layerdrop=config.layerdrop,
            ctc_loss_reduction=config.ctc_loss_reduction,
            ctc_zero_infinity=config.ctc_zero_infinity,
            gradient_checkpointing=config.gradient_checkpointing,
            pad_token_id=config.pad_token_id
        )
        
        if config.freeze_feature_encoder:
            self._freeze_feature_encoder()
    
    def _freeze_feature_encoder(self):
        """Freeze the feature encoder layers"""
        for param in self.wav2vec.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            input_values: Tensor of shape (batch_size, sequence_length)
            attention_mask: Optional mask of shape (batch_size, sequence_length)
            labels: Optional tensor of shape (batch_size, sequence_length) for CTC loss
            
        Returns:
            tuple: (logits, loss if labels provided else None)
        """
        outputs = self.wav2vec(
            input_values,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs.logits, outputs.loss if labels is not None else None
    
    def save_pretrained(self, save_directory: str):
        """Save the model to a directory"""
        self.wav2vec.save_pretrained(save_directory)
        
    @classmethod
    def from_pretrained(cls, load_directory: str, config: XLSRConfig):
        """Load a pretrained model from a directory"""
        model = cls(config)
        model.wav2vec = Wav2VecForCTC.from_pretrained(load_directory)
        return model