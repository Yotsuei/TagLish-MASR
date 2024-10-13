import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config

class Wav2Vec2Model:
    def __init__(self, model_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize Wav2Vec2 model for fine-tuning.
        
        :param model_dir: Directory where the pretrained Wav2Vec2 model is stored.
        :param device: Device to use for training ('cuda' or 'cpu').
        """
        self.device = device
        
        # Load Wav2Vec2 model configuration
        config = Wav2Vec2Config.from_pretrained(model_dir)
        
        # Load the pre-trained Wav2Vec2 model (for CTC, since we're doing speech recognition)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir, config=config)
        self.model.to(self.device)

    def forward(self, input_values):
        """
        Forward pass for the Wav2Vec2 model.

        :param input_values: Tensor of preprocessed audio data (waveforms).
        :return: logits from the model for speech recognition.
        """
        # Move input to the appropriate device
        input_values = input_values.to(self.device)
        
        # Forward pass through the Wav2Vec2 model
        outputs = self.model(input_values)
        logits = outputs.logits
        
        return logits

    def save_model(self, save_path='models/checkpoints/wav2vec2_finetuned'):
        """
        Save the fine-tuned model to a specified path.
        
        :param save_path: Directory to save the model. Defaults to models/checkpoints/wav2vec2_finetuned.
        """
        self.model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
