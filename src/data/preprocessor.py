# src/data/preprocessor.py

import os
from .dataset import load_data
from ..utils.audio_utils import preprocess_audio, save_audio
from ..utils.config_utils import get_config_value
from ..utils.audio_utils import plot_audio_waveform


# Function to preprocess an individual audio file
def preprocess_audio_file(file_path, output_dir):
    """Preprocess a single audio file and save the output."""
    try:
        # Process the audio file
        audio, sample_rate = preprocess_audio(file_path)
        
        # Save the processed audio in the output directory
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        save_audio(audio, sample_rate, output_path)
        print(f"Processed and saved: {output_path}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Function to preprocess a dataset of audio files one by one
def preprocess_dataset(data_dir, output_dir):
    """Preprocess the entire dataset by resampling, applying VAD, and normalizing audio."""
    audio_files = load_data(data_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loaded {len(audio_files)} audio files from {data_dir}.")
    for file_path in audio_files:
        print(f"Processing file: {file_path}")
        preprocess_audio_file(file_path, output_dir)
        

# Example usage
if __name__ == "__main__":
    raw_data_dir = get_config_value("paths.raw_data")
    processed_data_dir = get_config_value("paths.processed_data")
    
    preprocess_dataset(raw_data_dir, processed_data_dir)
