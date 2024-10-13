# src/utils/audio_utils.py

import librosa
import numpy as np
import webrtcvad
import soundfile as sf
import matplotlib.pyplot as plt


# Function to resample audio to a target sample rate
def resample_audio(audio_path, target_sample_rate=16000):
    """Load and resample audio to target sample rate."""
    audio, sample_rate = librosa.load(audio_path, sr=None)  # Load with original sample rate
    if sample_rate != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate, res_type='kaiser_best')
        print(f"Resampled audio from {sample_rate} Hz to {target_sample_rate} Hz.")
    else:
        print(f"No resampling needed. Audio is already at {target_sample_rate} Hz.")
    return audio, target_sample_rate


# Function to apply voice activity detection (VAD)
def preprocess_audio(audio_path, target_sample_rate=16000, vad_enabled=True, normalize=True):
    """Full preprocessing pipeline to resample, apply VAD, and normalize."""
    audio, sample_rate = resample_audio(audio_path, target_sample_rate)
    print(f"Loaded audio with sample rate: {sample_rate} and duration: {len(audio) / sample_rate} seconds")

    # Check if audio is empty
    if len(audio) == 0:
        print(f"Audio data is empty for: {audio_path}")
        return audio, sample_rate
    
    # Plot waveform to verify audio content
    plot_audio_waveform(audio, sample_rate, title=f"Waveform of {audio_path}")
    
    if vad_enabled:
        print("Applying VAD...")
        audio = apply_vad(audio, sample_rate, aggressiveness=0)
        if len(audio) == 0:
            print(f"No speech detected in: {audio_path}")
            return audio, sample_rate

    if normalize:
        audio = normalize_audio(audio)
    
    return audio, sample_rate

def apply_vad(audio, sample_rate, aggressiveness=0):
    """Basic Voice Activity Detection using energy threshold."""
    frame_length = 1024  # Length of each frame
    hop_length = 512     # Hop length between frames
    energy_threshold = 0.6  # Adjust based on testing

    # Compute short-time energy
    energy = np.array([
        np.sum(np.square(audio[i:i + frame_length]))
        for i in range(0, len(audio), hop_length)
        if i + frame_length <= len(audio)
    ])

    # Detect speech frames
    speech_frames = energy > energy_threshold
    if not np.any(speech_frames):
        print("No speech frames detected based on energy threshold.")
        return np.array([])

    # Reconstruct audio from detected speech frames
    detected_audio = []
    for i in range(len(speech_frames)):
        if speech_frames[i]:
            start = i * hop_length
            end = start + frame_length
            detected_audio.extend(audio[start:end])

    return np.array(detected_audio)

# Generator to split audio into frames
def frame_generator(audio, sample_rate, frame_duration_ms):
    """Generate audio frames for VAD."""
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    for start in range(0, len(audio), frame_size):
        frame = audio[start:start + frame_size]
        if len(frame) == frame_size:  # Only yield full frames
            yield frame

# Function to normalize audio amplitude
def normalize_audio(audio):
    """Normalize audio to the range [-1, 1]."""
    if len(audio) == 0:
        return audio
    max_abs_value = np.max(np.abs(audio))
    if max_abs_value > 0:
        normalized_audio = audio / max_abs_value
        return normalized_audio
    return audio

# Function to save the processed audio
def save_audio(audio, sample_rate, file_path):
    """Save the processed audio back to disk."""
    sf.write(file_path, audio, sample_rate)

def plot_audio_waveform(audio, sample_rate, title='Audio Waveform'):
    """Plot the waveform of the audio signal."""
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(audio)) / sample_rate, audio)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.xlim(0, len(audio) / sample_rate)
    plt.grid()
    plt.show()
