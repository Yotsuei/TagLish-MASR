# TagLish: Fine-Tuning Wav2Vec Model for Code-Switched Speech Recognition with Multi-Speaker Identification using Speaker Diarization

This project focuses on fine-tuning a Wav2Vec2 model for Tagalog-English (TagLish) code-switched speech recognition and integrating x-vector speaker diarization for multi-speaker identification. The model leverages the self-supervised learning capabilities of the Wav2Vec2 architecture and incorporates speaker diarization to recognize and distinguish between multiple speakers in an audio stream.

## Project Structure

```
TagLish-MASR/
├── config/
│   └── config.yaml                  # Configuration file for model, training, data paths, etc.
├── data/
│   ├── raw/                         # Raw audio data
│   ├── processed/                   # Processed audio data (features, resampled, etc.)
│   └── features/                    # Extracted features from the data
├── models/
│   ├── checkpoints/                 # Checkpoints saved during training
│   └── pretrained/                  # Pretrained models (Wav2Vec2, x-vector)
├── src/
│   ├── data/                        # Data handling scripts
│   │   ├── __init__.py
│   │   ├── preprocessor.py          # Preprocessing audio files
│   │   └── dataset.py               # Dataset and dataloaders
│   ├── models/                      # Model architectures
│   │   ├── __init__.py
│   │   ├── wav2vec2_model.py        # Wav2Vec2-based speech recognition model
│   │   └── speaker_diarization.py   # X-vector-based speaker diarization model
│   ├── training/                    # Training and validation logic
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training loop
│   │   └── validator.py             # Validation loop
│   ├── utils/                       # Utility functions
│   │   ├── __init__.py
│   │   └── audio_utils.py           # Audio processing utilities
│   └── train.py                     # Main training script
├── requirements.txt                 # Required dependencies
└── README.md                        # This README file
```

## Setup

### 1. Install Dependencies

Ensure you have Python 3.12.6 installed, and then install the required dependencies. Run the following command to install all necessary packages:

```bash
pip install -r requirements.txt
```

### 2. Configure Settings

Modify `config/config.yaml` to set up your training parameters, data paths, and model configurations. Some key parameters you might want to update:

- **data paths**: Set the paths to your raw and processed data.
- **training**: Modify parameters such as `num_epochs`, `batch_size`, and `learning_rate`.
- **model**: Specify whether to load pretrained Wav2Vec2 and x-vector models.

### 3. Data Preparation

Place your raw audio files in the `data/raw/` folder. The audio files should be in WAV format. The `preprocessor.py` script handles resampling and normalization of the audio files.

### 4. Run the Training

To start training the model, run the `train.py` script:

```bash
python src/train.py
```

This will:
- Preprocess the data.
- Fine-tune the Wav2Vec2 model for speech recognition on code-switched Tagalog-English audio.
- Train the x-vector speaker diarization model to distinguish between multiple speakers in the audio data.
- Save model checkpoints in the `models/checkpoints/` directory after each epoch.

### 5. Monitoring the Training

During training, you’ll see the following outputs in the console:
- **Training Loss**: Shows the training progress.
- **Validation Loss**: Shows how well the model performs on the validation set after each epoch.
- **Checkpointing**: Model checkpoints will be saved after every epoch in `models/checkpoints/`.

## Important Scripts

- **config/config.yaml**: Configuration file that holds settings for training parameters, paths to data, model configurations, and device settings (CPU/GPU).
  
- **src/train.py**: Main script to train the models. It handles loading the data, instantiating models, training for multiple epochs, validating, and saving checkpoints.

- **src/models/wav2vec2_model.py**: The Wav2Vec2-based model for speech recognition, fine-tuned on code-switched Tagalog-English speech data.

- **src/models/speaker_diarization.py**: The x-vector model for multi-speaker identification using speaker diarization.

- **src/data/dataset.py**: Contains logic for loading and batching the audio data for training.

- **src/training/trainer.py**: Contains the training loop for the models, including loss calculation and backpropagation.

- **src/training/validator.py**: Responsible for running validation after each epoch and calculating the validation loss.

- **src/utils/audio_utils.py**: Utility functions for audio preprocessing, including loading, resampling, and normalization.

## Future Improvements

- **Hyperparameter Tuning**: Fine-tune hyperparameters like learning rate, batch size, and epochs based on the model's performance on validation data.
- **Additional Data**: Add more diverse audio data to improve model generalization, especially for the speaker diarization task.
- **Evaluation**: After training, evaluate the model on a held-out test set to assess its real-world performance.

## Requirements

- Python 3.12.6
- PyTorch
- Torchaudio
- HuggingFace Transformers
- Librosa
- NumPy
- Scikit-learn
- PyYAML

Refer to `requirements.txt` for the full list of dependencies.