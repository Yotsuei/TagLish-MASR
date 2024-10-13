# trainer.py
import torch
from tqdm import tqdm
from transformers import AdamW

class Trainer:
    def __init__(self, model, train_dataloader, device, config):
        self.model = model
        self.train_dataloader = train_dataloader
        self.device = device
        self.config = config
        self.optimizer = AdamW(self.model.parameters(), lr=config['learning_rate'])

    def train(self):
        self.model.train()
        epochs = self.config['epochs']
        checkpoint_interval = self.config['checkpoint_interval']

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0

            # Training loop
            for batch in tqdm(self.train_dataloader):
                inputs = batch['input_values'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_dataloader)
            print(f"Average loss for epoch {epoch+1}: {avg_loss}")

            # Save checkpoint at intervals
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        save_path = f"models/checkpoints/wav2vec2_finetuned_epoch_{epoch+1}"
        self.model.save_pretrained(save_path)
        print(f"Model checkpoint saved at {save_path}")
