import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
import yaml
from typing import Optional, Tuple, Dict
import numpy as np

class TagLishDataset(Dataset):
    def __init__(self, 
                 config_path: str,
                 split: str = 'train',
                 transform: Optional[torch.nn.Module] = None):
        """
        Args:
            config_path: Path to config.yaml
            split: One of 'train', 'val', or 'test'
            transform: Optional transform to be applied to the audio
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.split = split
        self.transform = transform
        self.sample_rate = self.config['data']['sample_rate']
        
        # Set up data directories
        self.processed_dir = Path(self.config['data']['processed_audio_dir'])
        self.features_dir = Path(self.config['data']['features_dir'])
        
        # Get file lists
        self.audio_files = sorted(list(self.processed_dir.glob('*.wav')))
        self.feature_files = sorted(list(self.features_dir.glob('*_xvector.pt')))
        
        # Split dataset
        total_size = len(self.audio_files)
        indices = np.arange(total_size)
        np.random.seed(self.config['device']['seed'])
        np.random.shuffle(indices)
        
        train_size = int(total_size * self.config['data']['train_split'])
        val_size = int(total_size * self.config['data']['val_split'])
        
        if split == 'train':
            self.indices = indices[:train_size]
        elif split == 'val':
            self.indices = indices[train_size:train_size + val_size]
        else:  # test
            self.indices = indices[train_size + val_size:]
            
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get audio and speaker embedding for a given index."""
        # Get actual index from split indices
        actual_idx = self.indices[idx]
        
        # Load audio
        audio_path = self.audio_files[actual_idx]
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Ensure correct sample rate
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # Load speaker embeddings
        feature_path = self.feature_files[actual_idx]
        speaker_embedding = torch.load(feature_path)
        
        # Apply transform if specified
        if self.transform is not None:
            waveform = self.transform(waveform)
        
        return {
            'waveform': waveform,
            'speaker_embedding': speaker_embedding,
            'audio_path': str(audio_path)
        }
    
    @staticmethod
    def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        # Get max audio length in batch
        max_length = max(x['waveform'].shape[1] for x in batch)
        
        # Pad audio to max length
        waveforms = []
        speaker_embeddings = []
        audio_paths = []
        
        for sample in batch:
            waveform = sample['waveform']
            padding_length = max_length - waveform.shape[1]
            
            if padding_length > 0:
                waveform = torch.nn.functional.pad(waveform, (0, padding_length))
            
            waveforms.append(waveform)
            speaker_embeddings.append(sample['speaker_embedding'])
            audio_paths.append(sample['audio_path'])
        
        return {
            'waveforms': torch.stack(waveforms),
            'speaker_embeddings': torch.stack(speaker_embeddings),
            'audio_paths': audio_paths
        }