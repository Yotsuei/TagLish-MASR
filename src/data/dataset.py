import os
import torch
from torch.utils.data import Dataset
import torchaudio

class SpeechDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.data_dir = data_dir
        self.audio_files = [f for f in os.listdir(data_dir) if f.endswith(('.m4a', '.wav'))]
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.data_dir, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(audio_path)

        sample = {
            'waveform': waveform,
            'sample_rate': sample_rate,
            'file_name': self.audio_files[idx]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

# Example usage
if __name__ == "__main__":
    dataset = SpeechDataset(data_dir="data/processed/train")
    sample = dataset[0]
    print(f"Sample waveform shape: {sample['waveform'].shape}, file name: {sample['file_name']}")
