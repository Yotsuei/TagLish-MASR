import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from ..utils.config_utils import get_config_value
import os

class Wav2Vec2Model:
    """
    Class to load, fine-tune, and infer using Wav2Vec2 for speech recognition.
    """

    def __init__(self, model_dir="models/pretrained/wav2vec2-large-xlsr-53", tokenizer_path=None, device=None):
        """
        Initialize the Wav2Vec2 model and custom tokenizer (if specified).
        Load model from the local directory.
        :param model_dir: The directory containing the locally saved Wav2Vec2 model.
        :param tokenizer_path: Path to the custom tokenizer (vocabulary) for TagLish.
        :param device: Device to run the model on (CPU or GPU).
        """
        # Load custom tokenizer or processor (you must create this separately)
        if tokenizer_path:
            self.processor = Wav2Vec2Processor.from_pretrained(tokenizer_path)
        else:
            raise ValueError("A custom tokenizer must be provided!")

        # Load Wav2Vec2 model for CTC (speech-to-text) from local safetensors
        model_path = model_dir
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path, from_safetensors=True)

        # Determine device (GPU if available)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print(f"Model loaded from {model_dir} on {self.device}")

    def fine_tune(self, train_dataset, val_dataset, epochs=5, batch_size=4):
        """
        Fine-tune the Wav2Vec2 model on the given dataset.
        :param train_dataset: Training dataset.
        :param val_dataset: Validation dataset.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        """
        # Placeholder for fine-tuning logic
        pass

    def transcribe(self, audio):
        """
        Transcribe a single audio input using the Wav2Vec2 model.
        :param audio: Audio data to be transcribed.
        """
        inputs = self.processor(audio, return_tensors="pt", padding=True).input_values.to(self.device)
        logits = self.model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription
