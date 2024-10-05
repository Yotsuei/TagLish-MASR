# src/train.py

import torch
from pathlib import Path
from transformers import Wav2Vec2Processor
from models.wav2vec_model import TaglishXLSR, XLSRConfig
from src.config import TrainingConfig, DataConfig, TokenizerConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import logging
from tqdm import tqdm

class TaglishTrainer:
    def __init__(
        self,
        model_config: XLSRConfig = None,
        training_config: TrainingConfig = None,
        data_config: DataConfig = None,
        tokenizer_config: TokenizerConfig = None
    ):
        # Initialize configurations with defaults if not provided
        self.model_config = model_config or XLSRConfig()
        self.training_config = training_config or TrainingConfig()
        self.data_config = data_config or DataConfig()
        self.tokenizer_config = tokenizer_config or TokenizerConfig()
        
        # Initialize model and move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler()
        
        # Initialize processor
        self.processor = self._initialize_processor()
        
        # Setup logging
        self._setup_logging()

    def _initialize_model(self) -> TaglishXLSR:
        """Initialize the XLSR model"""
        model = TaglishXLSR(self.model_config)
        model.to(self.device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        return model
    
    def _initialize_optimizer(self) -> AdamW:
        """Initialize the optimizer"""
        return AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
    
    def _initialize_scheduler(self) -> LinearLR:
        """Initialize the learning rate scheduler"""
        return LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.training_config.warmup_steps
        )
    
    def _initialize_processor(self) -> Wav2Vec2Processor:
        """Initialize the XLSR processor"""
        return Wav2Vec2Processor.from_pretrained(self.model_config.pretrained_model_name)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.training_config.output_dir}/training.log"),
                logging.StreamHandler()
            ]
        )

    def train(self, train_dataloader, eval_dataloader=None):
        """Main training loop"""
        self.model.train()
        
        for epoch in range(self.training_config.max_epochs):
            logging.info(f"Starting epoch {epoch + 1}")
            self._train_epoch(train_dataloader)
            
            if eval_dataloader:
                self._evaluate(eval_dataloader)
            
            # Save checkpoint
            if (epoch + 1) % self.training_config.save_steps == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, dataloader):
        """Training logic for one epoch"""
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step logic here
            pass

    def _evaluate(self, dataloader):
        """Evaluation logic"""
        self.model.eval()
        # Evaluation logic here
        self.model.train()

    def _save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.training_config.output_dir) / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(checkpoint_dir))
        self.processor.save_pretrained(str(checkpoint_dir))