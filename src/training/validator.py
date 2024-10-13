# validator.py
import torch
from tqdm import tqdm

class Validator:
    def __init__(self, model, val_dataloader, device):
        self.model = model
        self.val_dataloader = val_dataloader
        self.device = device

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader):
                inputs = batch['input_values'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

        avg_val_loss = total_loss / len(self.val_dataloader)
        print(f"Validation loss: {avg_val_loss}")
        return avg_val_loss
