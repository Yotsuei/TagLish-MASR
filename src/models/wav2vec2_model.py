import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import yaml
from sklearn.cluster import KMeans
from speechbrain.pretrained import EncoderClassifier
import torch.nn.functional as F

class XVectorDiarizer(nn.Module):
    def __init__(self, config_path: str):
        """
        Initialize the X-Vector based speaker diarizer.
        
        Args:
            config_path: Path to the configuration file
        """
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize x-vector model
        self.x_vector_model = EncoderClassifier.from_hparams(
            source=self.config['diarization']['embedding_model'],
            savedir="models/pretrained/x_vector"
        )
        
        # Set clustering parameters
        self.min_speakers = self.config['diarization']['min_speakers']
        self.max_speakers = self.config['diarization']['max_speakers']
        self.segment_length = self.config['diarization']['segment_length']
        self.overlap = self.config['diarization']['overlap']
    
    def extract_embeddings(self, 
                          waveform: torch.Tensor,
                          sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract x-vector embeddings from audio segments.
        
        Args:
            waveform: Input audio waveform
            sample_rate: Audio sample rate
            
        Returns:
            Tensor of x-vector embeddings
        """
        # Calculate segment and step sizes in samples
        segment_size = int(self.segment_length * sample_rate)
        step_size = int((self.segment_length - self.overlap) * sample_rate)
        
        # Extract segments
        segments = []
        for start in range(0, waveform.shape[1] - segment_size + 1, step_size):
            end = start + segment_size
            segment = waveform[:, start:end]
            segments.append(segment)
        
        # Convert segments to batch
        segments = torch.stack(segments)
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = self.x_vector_model.encode_batch(segments)
        
        return embeddings
    
    def cluster_speakers(self, 
                        embeddings: torch.Tensor,
                        num_speakers: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Cluster embeddings to identify speakers.
        
        Args:
            embeddings: X-vector embeddings
            num_speakers: Optional number of speakers (if known)
            
        Returns:
            Tuple of speaker labels and clustering score
        """
        # Convert embeddings to numpy
        embeddings_np = embeddings.cpu().numpy()
        
        if num_speakers is None:
            # Try different numbers of speakers
            best_score = -float('inf')
            best_labels = None
            
            for n_speakers in range(self.min_speakers, self.max_speakers + 1):
                kmeans = KMeans(n_clusters=n_speakers, random_state=42)
                labels = kmeans.fit_predict(embeddings_np)
                score = kmeans.score(embeddings_np)
                
                if score > best_score:
                    best_score = score
                    best_labels = labels
            
            return best_labels, best_score
        else:
            # Use specified number of speakers
            kmeans = KMeans(n_clusters=num_speakers, random_state=42)
            labels = kmeans.fit_predict(embeddings_np)
            score = kmeans.score(embeddings_np)
            
            return labels, score
    
    def forward(self, 
                batch: Dict[str, torch.Tensor],
                num_speakers: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the diarizer.
        
        Args:
            batch: Dictionary containing:
                - waveforms: Input audio waveforms
                - speaker_embeddings: Optional pre-computed embeddings
            num_speakers: Optional number of speakers
            
        Returns:
            Dictionary containing diarization outputs
        """
        # Get embeddings
        if 'speaker_embeddings' in batch:
            embeddings = batch['speaker_embeddings']
        else:
            embeddings = self.extract_embeddings(batch['waveforms'])
        
        # Perform clustering
        labels, score = self.cluster_speakers(embeddings, num_speakers)
        
        # Convert labels to tensor
        labels_tensor = torch.from_numpy(labels).to(embeddings.device)
        
        return {
            'embeddings': embeddings,
            'speaker_labels': labels_tensor,
            'diarization_score': torch.tensor(score)
        }
    
    def save_pretrained(self, save_path: str):
        """
        Save the diarizer model.
        
        Args:
            save_path: Directory to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save x-vector model
        self.x_vector_model.save_pretrained(save_path)
    
    @classmethod
    def from_pretrained(cls, 
                       model_path: str,
                       config_path: str) -> 'XVectorDiarizer':
        """
        Load a pretrained diarizer.
        
        Args:
            model_path: Path to the pretrained model
            config_path: Path to the configuration file
            
        Returns:
            Loaded XVectorDiarizer model
        """
        instance = cls(config_path)
        instance.x_vector_model = EncoderClassifier.from_hparams(
            source=model_path,
            savedir="models/pretrained/x_vector"
        )
        return instance