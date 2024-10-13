import os
import shutil
from pathlib import Path

def prepare_data(data_src_dir, data_dst_dir):
    """
    Prepares the data directory structure and ensures raw audio files are in place.
    :param data_src_dir: Source directory containing the raw audio files.
    :param data_dst_dir: Destination directory where the audio data will be organized.
    """
    data_src_dir = Path(data_src_dir)
    data_dst_dir = Path(data_dst_dir)
    
    # Ensure destination directories exist
    raw_data_dir = data_dst_dir / 'raw'
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Move or copy audio files into the raw folder
    audio_files = list(data_src_dir.glob('**/*.wav'))  # Assuming .wav format
    for audio_file in audio_files:
        shutil.copy(audio_file, raw_data_dir / audio_file.name)

    print(f"Copied {len(audio_files)} audio files to {raw_data_dir}")

if __name__ == "__main__":
    # Define the source and destination directories
    data_src_dir = "data"
    data_dst_dir = "data"
    
    # Prepare the data by organizing it in the correct directory structure
    prepare_data(data_src_dir, data_dst_dir)
