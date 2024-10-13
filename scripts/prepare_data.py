import os
import torchaudio
import numpy as np
from sklearn.model_selection import train_test_split

# Directory paths
RAW_DATA_DIR = 'data/raw/'
PROCESSED_DATA_DIR = 'data/processed/'
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, 'train/')
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, 'test/')

# Sample rate to which audio files will be resampled (Wav2Vec2 model expects 16kHz)
TARGET_SAMPLE_RATE = 16000

def load_audio(file_path):
    """Load an audio file."""
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def resample_audio(waveform, orig_sample_rate, target_sample_rate=TARGET_SAMPLE_RATE):
    """Resample audio to target sample rate."""
    if orig_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    return waveform

def normalize_audio(waveform):
    """Normalize audio to be between -1 and 1."""
    return waveform / waveform.abs().max()

def prepare_and_save(file_name, waveform, sample_rate, save_dir):
    """Process and save audio data."""
    # Resample the audio
    waveform = resample_audio(waveform, sample_rate)
    # Normalize the audio
    waveform = normalize_audio(waveform)
    
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the processed audio
    save_path = os.path.join(save_dir, file_name)
    torchaudio.save(save_path, waveform, TARGET_SAMPLE_RATE)
    print(f"Saved processed audio: {save_path}")

def split_data(file_names):
    """Split the data into train and test sets."""
    train_files, test_files = train_test_split(file_names, test_size=0.2, random_state=42)
    return train_files, test_files

def prepare_data():
    """Main function to prepare data."""
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    
    # List all .wav audio files in the raw data directory
    audio_files = [file for file in os.listdir(RAW_DATA_DIR) if file.endswith(('.mp3', '.flac', '.ogg', '.wav'))]
    
    # Split the data into train and test sets
    train_files, test_files = split_data(audio_files)
    
    # Process and save the train data
    print("Processing train data...")
    for file_name in train_files:
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        waveform, sample_rate = load_audio(file_path)
        prepare_and_save(file_name, waveform, sample_rate, TRAIN_DIR)

    # Process and save the test data
    print("Processing test data...")
    for file_name in test_files:
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        waveform, sample_rate = load_audio(file_path)
        prepare_and_save(file_name, waveform, sample_rate, TEST_DIR)

    print("Data preparation and splitting completed!")

if __name__ == "__main__":
    prepare_data()
