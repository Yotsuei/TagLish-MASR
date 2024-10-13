import os
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from preprocessor import AudioPreprocessor

class TagLishDataset(Dataset):
    def __init__(self, data_dir, split="train", sample_rate=16000, normalize=True, test_size=0.2, random_seed=42):
        """
        Initializes the dataset by splitting the data into train and test sets, applying preprocessing.
        :param data_dir: Directory where the raw audio data is stored.
        :param split: "train" or "test" to determine the dataset split.
        :param sample_rate: Target sample rate for audio (Wav2Vec2 expects 16000 Hz).
        :param normalize: Whether to normalize the audio waveform.
        :param test_size: Proportion of data to be used for the test set.
        :param random_seed: Seed for reproducibility in train-test split.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate, normalize=normalize)

        # Load all audio files (.wav format)
        all_audio_files = list(self.data_dir.glob("**/*.wav"))

        # Perform train-test split
        train_files, test_files = train_test_split(
            all_audio_files, test_size=test_size, random_state=random_seed
        )

        # Set dataset based on the split
        if self.split == "train":
            self.audio_files = train_files
        else:
            self.audio_files = test_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Retrieves an audio sample and applies preprocessing (resampling, normalization).
        :param idx: Index of the sample.
        :return: A dictionary containing the audio file path and the processed audio tensor.
        """
        audio_path = self.audio_files[idx]

        # Load the audio file using torchaudio (as Wav2Vec2 expects raw waveforms)
        waveform, sample_rate = torchaudio.load(audio_path)

        # Preprocess the audio (resampling, normalization)
        processed_waveform = self.preprocessor.process(waveform, sample_rate)

        return {
            "audio_path": audio_path,
            "waveform": processed_waveform,
            "sample_rate": self.preprocessor.sample_rate,
        }
