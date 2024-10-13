# prepare_data.py

import os
import random
from shutil import copyfile
from src.utils.audio_utils import load_audio  # Using for audio validation or preview if needed

def train_test_split(data_dir, train_dir, test_dir, split_ratio=0.8, seed=42):
    """
    Split the dataset into training and testing sets based on the given ratio.

    :param data_dir: Directory with the original audio files.
    :param train_dir: Directory where the training split will be saved.
    :param test_dir: Directory where the test split will be saved.
    :param split_ratio: Ratio of data to be used for training (e.g., 0.8 means 80% training).
    :param seed: Random seed for reproducibility.
    """
    random.seed(seed)
    
    # Get list of all files in the data directory
    all_files = os.listdir(data_dir)
    
    # Shuffle and split files
    random.shuffle(all_files)
    split_idx = int(len(all_files) * split_ratio)
    
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy files to respective directories
    for file in train_files:
        copyfile(os.path.join(data_dir, file), os.path.join(train_dir, file))

    for file in test_files:
        copyfile(os.path.join(data_dir, file), os.path.join(test_dir, file))

    print(f"Training data size: {len(train_files)}")
    print(f"Test data size: {len(test_files)}")

# Example usage: split data and prepare directories
data_dir = "data/raw"
train_dir = "data/processed/train"
test_dir = "data/processed/test"

train_test_split(data_dir, train_dir, test_dir)
