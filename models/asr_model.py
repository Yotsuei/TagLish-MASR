# models/asr_model.py
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from config.config import Config

class TaglishASR(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(Config.MODEL_NAME)
        self.model = Wav2Vec2ForCTC.from_pretrained(Config.MODEL_NAME)
        
        # Modify the model for Taglish vocabulary
        # This should be updated based on your specific vocabulary
        self.model.freeze_feature_encoder()
    
    def forward(self, input_values):
        return self.model(input_values)
    
    def prepare_input(self, audio):
        inputs = self.processor(
            audio,
            sampling_rate=Config.SAMPLING_RATE,
            return_tensors="pt",
            padding=True
        )
        return inputs.input_values.to(Config.DEVICE)