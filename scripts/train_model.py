# scripts/train_model.py

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.train import TaglishTrainer
from models.wav2vec_model import XLSRConfig
from src.config import TrainingConfig, DataConfig, TokenizerConfig
from torch.utils.data import DataLoader
# Import your data loading functionality
# from src.data_loading import create_dataloaders

def main():
    # Initialize configurations
    model_config = XLSRConfig(
        freeze_feature_encoder=True,
        gradient_checkpointing=True
    )
    
    training_config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=4,
        max_epochs=50
    )
    
    data_config = DataConfig(
        sample_rate=16000,
        max_duration_seconds=30.0
    )
    
    tokenizer_config = TokenizerConfig()
    
    # Initialize trainer
    trainer = TaglishTrainer(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        tokenizer_config=tokenizer_config
    )
    
    # Create dataloaders
    # train_dataloader, eval_dataloader = create_dataloaders(...)
    
    # Start training
    # trainer.train(train_dataloader, eval_dataloader)

if __name__ == "__main__":
    main()