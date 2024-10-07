# models/diarization.py
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from config.config import Config

class SpeakerDiarization:
    def __init__(self):
        self.device = Config.DEVICE
        # Load pretrained x-vector model
        self.xvector_extractor = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="models/pretrained_models/xvector"
        )
    
    def extract_xvectors(self, audio_signal):
        # Move to appropriate device
        audio_signal = audio_signal.to(self.device)
        
        # Extract x-vectors
        embeddings = self.xvector_extractor.encode_batch(audio_signal)
        return embeddings