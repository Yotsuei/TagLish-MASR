import torchaudio
import os

class AudioPreprocessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def extract_features(self, waveform):
        """
        Extract features from waveform. 
        For now, this is a placeholder. You can integrate MFCCs, Mel-spectrogram, etc. if needed.
        """
        # Example: Mel-spectrogram extraction
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, 
            n_mels=128
        )
        return mel_spectrogram(waveform)
    
    def preprocess(self, audio_path):
        """
        Load audio, resample if needed, and extract features.
        """
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary (resample is already done in prepare_data.py)
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        features = self.extract_features(waveform)
        return features

# Example usage
if __name__ == "__main__":
    preprocessor = AudioPreprocessor(sample_rate=16000)
    audio_path = "path_to_audio_file.wav"
    features = preprocessor.preprocess(audio_path)
    print("Extracted features:", features.shape)
