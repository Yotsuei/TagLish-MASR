# utils/audio_utils.py
import torchaudio
import torch
import numpy as np
from config.config import Config

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != Config.SAMPLING_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, Config.SAMPLING_RATE)
        waveform = resampler(waveform)
    
    return waveform.squeeze()

def prepare_dataset(audio_paths, transcriptions):
    dataset = []
    for audio_path, transcription in zip(audio_paths, transcriptions):
        audio = load_audio(audio_path)
        dataset.append({
            "audio": audio,
            "transcription": transcription
        })
    return dataset