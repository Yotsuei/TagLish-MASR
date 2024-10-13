# dataset.py

import os
from torch.utils.data import Dataset
from src.utils.audio_utils import load_audio, audio_to_tensor, normalize_audio

class SpeechDataset(Dataset):
    def __init__(self, data_dir, file_list, transform=None):
        """
        Custom dataset for loading audio data.

        :param data_dir: Directory where audio files are located.
        :param file_list: List of audio file names to be used in the dataset.
        :param transform: Optional transforms to be applied on the audio (e.g., augmentation).
        """
        self.data_dir = data_dir
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.data_dir, self.file_list[idx])
        
        # Load audio using the utility function
        audio, sr = load_audio(audio_path)
        
        # Convert to tensor and normalize
        audio_tensor = audio_to_tensor(audio)
        normalized_audio = normalize_audio(audio_tensor)

        # Apply any transformations (if provided)
        if self.transform:
            normalized_audio = self.transform(normalized_audio)

        return normalized_audio, sr
