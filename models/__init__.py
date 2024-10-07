# models/__init__.py
from .asr_model import TaglishASR
from .diarization import SpeakerDiarization

__all__ = [
    'TaglishASR',
    'SpeakerDiarization'
]

# Script to prepare dataset and update vocabulary
def prepare_taglish_dataset(data_dir: str, vocab_path: str) -> Tuple[List[AudioTranscriptionItem], List[AudioTranscriptionItem]]:
    """
    Prepares the Taglish dataset and creates vocabulary.
    
    Directory structure should be:
    data_dir/
    ├── audio/
    │   ├── speaker1/
    │   │   ├── recording1.wav
    │   │   └── recording2.wav
    │   └── speaker2/
    │       ├── recording1.wav
    │       └── recording2.wav
    └── transcriptions/
        └── transcriptions.json
    
    transcriptions.json format:
    {
        "speaker1/recording1.wav": {
            "transcription": "text here",
            "speaker_id": "speaker1"
        },
        ...
    }
    """
    # Load transcriptions
    with open(os.path.join(data_dir, "transcriptions", "transcriptions.json"), "r") as f:
        transcriptions_dict = json.load(f)
    
    # Create dataset items
    items = []
    for audio_path, data in transcriptions_dict.items():
        full_audio_path = os.path.join(data_dir, "audio", audio_path)
        items.append(AudioTranscriptionItem(
            audio_path=full_audio_path,
            transcription=data["transcription"],
            speaker_id=data["speaker_id"]
        ))
    
    # Prepare vocabulary
    all_transcriptions = [item.transcription for item in items]
    vocab_dict = AudioPreprocessor.prepare_vocabulary(all_transcriptions, vocab_path)
    
    # Split into train and validation
    np.random.seed(Config.SEED)
    np.random.shuffle(items)
    split_idx = int(len(items) * Config.TRAIN_SPLIT)
    train_items = items[:split_idx]
    val_items = items[split_idx:]
    
    return train_items, val_items