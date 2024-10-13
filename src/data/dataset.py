# src/data/dataset.py

import os
import random
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from ..utils.config_utils import get_config_value

# Function to load audio dataset paths
def load_data(data_dir):
    """Load audio file paths and labels from a given directory."""
    audio_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):  # Ensure only .wav files are loaded
                audio_files.append(os.path.join(root, file))
    print(f"Loaded {len(audio_files)} audio files from {data_dir}")
    return audio_files

# Function to split the dataset into train and test sets
def train_test_split(data_dir, test_size=0.2, random_seed=42):
    """Split the dataset into training and testing sets."""
    audio_files = load_data(data_dir)
    
    # Randomize the dataset to ensure a good distribution
    random.seed(random_seed)
    random.shuffle(audio_files)
    
    # Use sklearn's train_test_split for easy handling
    train_files, test_files = sklearn_train_test_split(
        audio_files, test_size=test_size, random_state=random_seed
    )
    
    return train_files, test_files

# Example usage
if __name__ == "__main__":
    data_dir = get_config_value("paths.raw_data")
    train_files, test_files = train_test_split(data_dir)
    print(f"Train files: {len(train_files)} | Test files: {len(test_files)}")
    print("Train files:", train_files)
    print("Test files:", test_files)

