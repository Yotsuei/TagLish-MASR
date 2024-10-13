import os
from transformers import AutoModelForPreTraining

def download_wav2vec2_model(model_name, save_directory):
    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    print(f"Downloading {model_name}...")
    
    # Load the processor and model from Hugging Face
    model = AutoModelForPreTraining.from_pretrained(model_name)
    
    # Save the processor and model locally
    model.save_pretrained(save_directory)
    
    print(f"{model_name} has been downloaded and saved to {save_directory}")

if __name__ == "__main__":
    # Specify model name and save path
    model_name = "facebook/wav2vec2-large-xlsr-53"
    save_path = "models/pretrained/wav2vec2-large-xlsr-53/"
    
    download_wav2vec2_model(model_name, save_path)
