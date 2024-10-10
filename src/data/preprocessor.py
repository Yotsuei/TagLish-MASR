import torch
import torchaudio
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Tuple, Optional
from speechbrain.pretrained import EncoderClassifier
from torch.nn.utils.rnn import pad_sequence

class AudioPreprocessor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sample_rate = self.config['data']['sample_rate']
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                 self.config['device']['use_cuda'] else 'cpu')
        
        # Initialize x-vector model for speaker diarization
        self.x_vector_model = EncoderClassifier.from_hparams(
            source=self.config['diarization']['embedding_model'],
            savedir="models/pretrained/x_vector"
        )
        self.x_vector_model.to(self.device)

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load and resample audio file."""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform, self.sample_rate

    def process_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply audio preprocessing steps."""
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normalize audio if configured
        if self.config['audio']['normalize_audio']:
            waveform = (waveform - waveform.mean()) / waveform.std()
        
        # Trim silence if configured
        if self.config['audio']['trim_silence']:
            waveform = self._trim_silence(waveform)
        
        return waveform

    def _trim_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """Remove silence from the audio using energy-based VAD."""
        threshold = self.config['audio']['silence_threshold']
        energy = torch.abs(waveform)
        masks = energy > threshold
        
        # Find start and end of speech
        start = masks.argmax()
        end = len(masks) - masks.flip(1).argmax()
        
        return waveform[:, start:end]

    def extract_speaker_embeddings(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract x-vector embeddings for speaker diarization."""
        segment_length = int(self.config['diarization']['segment_length'] * self.sample_rate)
        overlap = int(self.config['diarization']['overlap'] * self.sample_rate)
        
        # Split audio into overlapping segments
        segments = []
        for start in range(0, waveform.shape[1] - segment_length + 1, segment_length - overlap):
            end = start + segment_length
            segment = waveform[:, start:end]
            segments.append(segment)
        
        # Pad last segment if necessary
        if waveform.shape[1] % segment_length != 0:
            last_segment = waveform[:, -(segment_length):]
            segments.append(last_segment)
        
        # Convert segments to batch
        segments_batch = pad_sequence(segments, batch_first=True)
        
        # Extract x-vector embeddings
        with torch.no_grad():
            embeddings = self.x_vector_model.encode_batch(segments_batch)
        
        return embeddings

    def save_processed_audio(self, 
                           waveform: torch.Tensor, 
                           embeddings: torch.Tensor,
                           output_path: Path,
                           features_path: Path):
        """Save processed audio and features."""
        # Save processed waveform
        torchaudio.save(output_path, waveform, self.sample_rate)
        
        # Save speaker embeddings
        torch.save(embeddings, features_path)

    def process_dataset(self, input_dir: str, output_dir: str, features_dir: str):
        """Process entire dataset."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        features_path = Path(features_dir)
        
        # Create output directories if they don't exist
        output_path.mkdir(parents=True, exist_ok=True)
        features_path.mkdir(parents=True, exist_ok=True)
        
        for audio_file in input_path.glob('*.wav'):
            try:
                # Load and process audio
                waveform, _ = self.load_audio(str(audio_file))
                processed_waveform = self.process_audio(waveform)
                
                # Extract speaker embeddings
                embeddings = self.extract_speaker_embeddings(processed_waveform)
                
                # Save processed data
                output_file = output_path / audio_file.name
                features_file = features_path / f"{audio_file.stem}_xvector.pt"
                
                self.save_processed_audio(processed_waveform, 
                                       embeddings,
                                       output_file, 
                                       features_file)
                
                logging.info(f"Successfully processed {audio_file.name}")
                
            except Exception as e:
                logging.error(f"Error processing {audio_file.name}: {str(e)}")