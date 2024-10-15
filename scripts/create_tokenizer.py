# scripts/create_tokenizer.py

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from src.utils.config_utils import load_config
import os

def train_tokenizer(corpus_file, vocab_size=32, output_dir="tokenizer"):
    """
    Train a character-level tokenizer on the TagLish dataset.
    
    :param corpus_file: Path to a text file containing the TagLish corpus.
    :param vocab_size: Vocabulary size for the tokenizer.
    :param output_dir: Directory to save the trained tokenizer.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define a tokenizer with a WordLevel model
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    
    # Pre-tokenize by splitting words into characters
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Train the tokenizer on the corpus
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size, 
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    # Train on the corpus
    tokenizer.train([corpus_file], trainer)
    
    # Post-process for adding special tokens to input/output (optional)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    
    # Save tokenizer
    tokenizer.save(output_dir + "/tokenizer.json")
    
    print(f"Tokenizer trained and saved to {output_dir}/tokenizer.json")

if __name__ == "__main__":
    # Load the config file
    config = load_config("config/config.yaml")
    
    # Get tokenizer settings from the config file
    corpus_path = config['tokenizer']['corpus_file']
    output_dir = config['tokenizer']['output_dir']
    vocab_size = config['tokenizer']['vocab_size']

    # Train tokenizer using config values
    train_tokenizer(corpus_path, vocab_size, output_dir)
