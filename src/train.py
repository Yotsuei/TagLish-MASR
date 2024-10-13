# train.py
import torch
from src.models.wav2vec2_model import Wav2Vec2Model
from src.training.trainer import Trainer
from src.training.validator import Validator
from src.data.dataset import load_data
from src.utils.config_utils import load_config

def main():
    config = load_config("config/config.yaml")
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_dataloader, val_dataloader = load_data(config)

    # Initialize model
    wav2vec_model = Wav2Vec2Model(config)
    wav2vec_model.to(device)

    # Training
    trainer = Trainer(wav2vec_model, train_dataloader, device, config)
    trainer.train()

    # Validation
    validator = Validator(wav2vec_model, val_dataloader, device)
    validator.validate()

if __name__ == "__main__":
    main()
