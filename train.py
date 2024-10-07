# train.py
import torch
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from models import prepare_taglish_dataset
from models.asr_model import TaglishASR
from utils.audio_utils import prepare_dataset
from config.config import Config
from utils.preprocessing import TaglishDataset

# Updated train.py with validation
def train():
    # Initialize model and processor
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        "vocab.json",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=Config.SAMPLING_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    
    # Prepare datasets
    train_items, val_items = prepare_taglish_dataset("data", "vocab.json")
    train_dataset = TaglishDataset(train_items, processor)
    val_dataset = TaglishDataset(val_items, processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    model = TaglishASR().to(Config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    for epoch in range(Config.NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        
        for batch in train_progress:
            optimizer.zero_grad()
            
            input_values = batch["input_values"].to(Config.DEVICE)
            labels = batch["labels"].to(Config.DEVICE)
            
            outputs = model(input_values, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_progress.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
        
        with torch.no_grad():
            for batch in val_progress:
                input_values = batch["input_values"].to(Config.DEVICE)
                labels = batch["labels"].to(Config.DEVICE)
                
                outputs = model(input_values, labels=labels)
                loss = outputs.loss
                
                val_loss += loss.item()
                val_progress.set_postfix({"loss": loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f"{Config.MODEL_DIR}/best_model.pt")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f"{Config.MODEL_DIR}/checkpoint_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()