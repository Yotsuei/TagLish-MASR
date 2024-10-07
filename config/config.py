# config/config.py
class Config:
    # Wav2Vec2 Configuration
    MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    MAX_INPUT_LENGTH_IN_SEC = 30.0
    
    # Audio Configuration
    SAMPLING_RATE = 16000
    
    # Training Configuration
    TRAIN_SPLIT = 0.8
    EVAL_SPLIT = 0.2
    SEED = 42
    
    # Paths
    DATA_DIR = "data"
    MODEL_DIR = "models"
    
    # Device Configuration
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")