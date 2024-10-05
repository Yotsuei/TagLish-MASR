# src/config.py

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class TrainingConfig:
    """Configuration for training the TagLish ASR model"""
    learning_rate: float = 3e-4  # Adjusted for XLSR
    batch_size: int = 4
    max_epochs: int = 50
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    weight_decay: float = 0.005
    logging_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 1000
    output_dir: str = "models/wav2vec_finetuned"
    fp16: bool = True  # Enable mixed precision training
    
@dataclass
class DataConfig:
    """Configuration for data processing"""
    sample_rate: int = 16000  # Required for XLSR-53
    max_duration_seconds: float = 30.0
    min_duration_seconds: float = 1.0
    audio_column_name: str = "audio"
    text_column_name: str = "text"
    preprocessing_num_workers: int = 4
    
@dataclass
class TokenizerConfig:
    """Configuration for tokenizer"""
    vocab_file: str = "vocab.json"
    # Special tokens
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    silence_token: str = "|"
    # Characters for Tagalog-English
    english_chars: List[str] = None
    tagalog_chars: List[str] = None
    
    def __post_init__(self):
        # Initialize character sets if not provided
        if self.english_chars is None:
            self.english_chars = list("abcdefghijklmnopqrstuvwxyz")
        if self.tagalog_chars is None:
            self.tagalog_chars = list("ñ")  # Add more Tagalog-specific characters if needed