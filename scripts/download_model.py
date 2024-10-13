# scripts/download_model.py

import os
from transformers import AutoModelForPreTraining

# Directory to store the pretrained model
model_dir = "models/pretrained/wav2vec2-large-xlsr-53/"

# Create the directory if it does not exist
os.makedirs(model_dir, exist_ok=True)

# Download and save the Wav2Vec2 model
model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# Save model weights to the specified directory
model.save_pretrained(model_dir)

print(f"Model saved locally at: {model_dir}")
