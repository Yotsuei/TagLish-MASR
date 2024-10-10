import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import Dict, Optional, Tuple
import yaml
from pathlib import Path

class TagLishWav2Vec2(nn.Module):
    def __init__(self, config_path: str):
        """
        Initialize the TagLish Wav2Vec2 model.
        
        Args:
            config_path: Path to the configuration file
        """
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Wav2Vec2 model and processor
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.config['wav2vec2']['model_name']
        )
        self.processor = Wav2Vec2Processor.from_pretrained(
            self.config['wav2vec2']['model_name']
        )
        
        # Freeze feature encoder if specified
        if self.config['wav2vec2']['freeze_feature_encoder']:
            self._freeze_feature_encoder()
        
        # Set dropout values
        self._set_dropout_values()
        
    def _freeze_feature_encoder(self):
        """Freeze the feature encoder layers of Wav2Vec2."""
        for param in self.model.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
    
    def _set_dropout_values(self):
        """Set dropout values according to configuration."""
        self.model.config.hidden_dropout = self.config['wav2vec2']['hidden_dropout']
        self.model.config.attention_dropout = self.config['wav2vec2']['attention_dropout']
        self.model.config.feat_proj_dropout = self.config['wav2vec2']['feat_proj_dropout']
        self.model.config.mask_time_prob = self.config['wav2vec2']['mask_time_prob']
        self.model.config.layerdrop = self.config['wav2vec2']['layerdrop']
    
    def forward(self, 
                batch: Dict[str, torch.Tensor],
                return_logits: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            batch: Dictionary containing:
                - waveforms: Tensor of shape (batch_size, sequence_length)
                - attention_mask: Optional mask for padding
            return_logits: Whether to return logits instead of loss
            
        Returns:
            Dictionary containing model outputs
        """
        # Get inputs
        input_values = batch['waveforms']
        attention_mask = batch.get('attention_mask', None)
        
        # Process through Wav2Vec2
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        result = {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states
        }
        
        if return_logits:
            return result
        
        # If labels are provided, compute loss
        if 'labels' in batch:
            loss = outputs.loss
            result['loss'] = loss
            
        return result
    
    def prepare_input(self, 
                     waveforms: torch.Tensor,
                     sample_rate: int = 16000) -> Dict[str, torch.Tensor]:
        """
        Prepare input for the model.
        
        Args:
            waveforms: Input audio waveforms
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary containing processed inputs
        """
        # Process inputs using Wav2Vec2 processor
        inputs = self.processor(
            waveforms,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        return inputs
    
    def decode(self, 
               logits: torch.Tensor,
               skip_special_tokens: bool = True) -> list:
        """
        Decode model predictions to text.
        
        Args:
            logits: Model output logits
            skip_special_tokens: Whether to remove special tokens from output
            
        Returns:
            List of decoded predictions
        """
        # Get predictions
        pred_ids = torch.argmax(logits, dim=-1)
        
        # Decode predictions to text
        predictions = self.processor.batch_decode(
            pred_ids,
            skip_special_tokens=skip_special_tokens
        )
        
        return predictions
    
    def save_pretrained(self, save_path: str):
        """
        Save the model and processor.
        
        Args:
            save_path: Directory to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        
        # Save processor
        self.processor.save_pretrained(save_path)
    
    @classmethod
    def from_pretrained(cls, 
                       model_path: str,
                       config_path: str) -> 'TagLishWav2Vec2':
        """
        Load a pretrained model.
        
        Args:
            model_path: Path to the pretrained model
            config_path: Path to the configuration file
            
        Returns:
            Loaded TagLishWav2Vec2 model
        """
        instance = cls(config_path)
        instance.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        instance.processor = Wav2Vec2Processor.from_pretrained(model_path)
        return instance