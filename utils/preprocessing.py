# utils/preprocessing.py
import torch
import torchaudio
import numpy as np
import json
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from config.config import Config

@dataclass
class AudioTranscriptionItem:
    audio_path: str
    transcription: str
    speaker_id: str = None

class AudioPreprocessor:
    def __init__(self):
        self.sample_rate = Config.SAMPLING_RATE
        
    def process_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Process audio waveform for model input."""
        # Convert to mono if stereo
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        
        # Resample if necessary
        if self.sample_rate != Config.SAMPLING_RATE:
            resampler = torchaudio.transforms.Resample(
                self.sample_rate, Config.SAMPLING_RATE
            )
            waveform = resampler(waveform)
        
        # Normalize audio
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)
        
        return waveform

    @staticmethod
    def prepare_vocabulary(transcriptions: List[str], vocab_path: str):
        """Create vocabulary from transcriptions for Wav2Vec2 tokenizer."""
        # Get unique characters from all transcriptions
        all_text = " ".join(transcriptions)
        vocab = sorted(set(all_text))
        
        # Create vocabulary dictionary
        vocab_dict = {v: k for k, v in enumerate(vocab)}
        
        # Add special tokens
        vocab_dict["<pad>"] = len(vocab_dict)
        vocab_dict["<unk>"] = len(vocab_dict)
        vocab_dict["|"] = len(vocab_dict)  # Token for word separation
        
        # Save vocabulary
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "w") as f:
            json.dump(vocab_dict, f)
        
        return vocab_dict

class TaglishDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[AudioTranscriptionItem], processor, max_length: int = None):
        self.items = items
        self.processor = processor
        self.max_length = max_length
        self.audio_preprocessor = AudioPreprocessor()
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(item.audio_path)
        waveform = self.audio_preprocessor.process_audio(waveform)
        
        # Prepare input values
        inputs = self.processor(
            waveform,
            sampling_rate=Config.SAMPLING_RATE,
            return_tensors="pt",
            padding=True
        )
        
        # Prepare labels
        with self.processor.as_target_processor():
            labels = self.processor(item.transcription).input_ids
        
        return {
            "input_values": inputs.input_values.squeeze(),
            "labels": torch.tensor(labels),
            "speaker_id": item.speaker_id
        }
