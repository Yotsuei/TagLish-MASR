# audio_utils.py

import librosa
import torch

def load_audio(file_path, sampling_rate=16000):
    """
    Load an audio file and resample it to the target sampling rate.

    :param file_path: Path to the audio file.
    :param sampling_rate: Target sampling rate. Default is 16kHz, which is compatible with Wav2Vec2.
    :return: A tuple of (audio data, sampling rate).
    """
    audio, sr = librosa.load(file_path, sr=sampling_rate)
    return audio, sr

def audio_to_tensor(audio):
    """
    Convert a numpy audio array to a PyTorch tensor.
    
    :param audio: Numpy array containing the audio data.
    :return: PyTorch tensor of the audio data.
    """
    return torch.tensor(audio)

def normalize_audio(audio_tensor):
    """
    Normalize the audio tensor to a range between -1 and 1.

    :param audio_tensor: PyTorch tensor of the audio data.
    :return: Normalized audio tensor.
    """
    return audio_tensor / torch.max(torch.abs(audio_tensor))
