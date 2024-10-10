#!/usr/bin/env python3
import logging
import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path to enable imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessor import AudioPreprocessor

def setup_logging():
    """Set up logging configuration"""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'preprocessing.log'),
            logging.StreamHandler()
        ]
    )

def get_project_paths():
    """Get default paths relative to project root"""
    return {
        'config': project_root / 'config' / 'config.yaml',
        'input_dir': project_root / 'data' / 'raw',
        'output_dir': project_root / 'data' / 'processed',
        'features_dir': project_root / 'data' / 'features'
    }

def main():
    # Get default paths
    default_paths = get_project_paths()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess audio files for TagLish-MASR')
    parser.add_argument('--config', type=str, default=str(default_paths['config']),
                      help='Path to configuration file')
    parser.add_argument('--input-dir', type=str, default=str(default_paths['input_dir']),
                      help='Directory containing raw audio files')
    parser.add_argument('--output-dir', type=str, default=str(default_paths['output_dir']),
                      help='Directory to save processed audio files')
    parser.add_argument('--features-dir', type=str, default=str(default_paths['features_dir']),
                      help='Directory to save extracted features')
    
    args = parser.parse_args()

    # Set up logging
    setup_logging()
    logging.info("Starting preprocessing pipeline...")

    try:
        # Create necessary directories if they don't exist
        for dir_path in [args.output_dir, args.features_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Initialize preprocessor
        preprocessor = AudioPreprocessor(args.config)
        
        # Process dataset
        logging.info(f"Processing audio files from {args.input_dir}")
        preprocessor.process_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            features_dir=args.features_dir
        )
        
        logging.info("Preprocessing completed successfully!")

    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()