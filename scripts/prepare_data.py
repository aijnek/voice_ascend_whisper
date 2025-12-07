"""Data preparation script for Whisper fine-tuning."""

import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from voice_ascend_whisper.data.processor import create_whisper_processor


def load_config(config_path: str = "configs/data_config.yaml") -> dict:
    """Load data configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["dataset"]


def load_local_common_voice(data_dir: str, language: str) -> DatasetDict:
    """
    Load Common Voice dataset from local directory.

    Args:
        data_dir: Path to the extracted Common Voice directory
        language: Language code (e.g., 'hi', 'en', 'ja')

    Returns:
        DatasetDict with train, dev, and test splits
    """
    data_path = Path(data_dir)

    # Find the Common Voice corpus directory
    cv_dirs = list(data_path.glob("cv-corpus-*"))
    if not cv_dirs:
        raise ValueError(f"No Common Voice corpus directory found in {data_dir}")

    # Use the first (or only) corpus directory found
    cv_dir = cv_dirs[0] / language

    if not cv_dir.exists():
        raise ValueError(
            f"Language directory '{language}' not found in {cv_dirs[0]}. "
            f"Available languages: {[d.name for d in cv_dirs[0].iterdir() if d.is_dir()]}"
        )

    clips_dir = cv_dir / "clips"

    print(f"Loading from: {cv_dir}")
    print(f"Clips directory: {clips_dir}")

    datasets = {}
    split_files = {
        "train": cv_dir / "train.tsv",
        "validation": cv_dir / "dev.tsv",
        "test": cv_dir / "test.tsv",
    }

    for split_name, tsv_file in split_files.items():
        if not tsv_file.exists():
            print(f"Warning: {tsv_file} not found, skipping {split_name} split")
            continue

        # Load TSV file
        df = pd.read_csv(tsv_file, sep="\t")

        # Create full paths to audio files
        df["audio_path"] = df["path"].apply(lambda x: str(clips_dir / x))

        # Create dataset
        dataset = Dataset.from_dict({
            "audio": df["audio_path"].tolist(),
            "sentence": df["sentence"].tolist(),
        })

        datasets[split_name] = dataset

    return DatasetDict(datasets)


def prepare_dataset_fn(processor, config):
    """
    Create preprocessing function for dataset.

    Args:
        processor: WhisperProcessor for feature extraction and tokenization
        config: Dataset configuration dictionary

    Returns:
        Function that preprocesses dataset examples
    """
    target_sr = config["sampling_rate"]

    def prepare(batch):
        """Preprocess audio and text."""
        # Load audio file directly using soundfile
        audio_path = batch["audio"]
        audio_array, sampling_rate = sf.read(audio_path)

        # Resample if necessary
        if sampling_rate != target_sr:
            audio_array = librosa.resample(
                audio_array, orig_sr=sampling_rate, target_sr=target_sr
            )

        # Extract mel-spectrogram features
        batch["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=target_sr
        ).input_features[0]

        # Tokenize text transcription
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids

        return batch

    return prepare


def main():
    """Main data preparation pipeline."""
    # Load configuration
    print("=" * 70)
    print("WHISPER DATA PREPARATION")
    print("=" * 70)

    config = load_config()
    print(f"\nData directory: {config['data_dir']}")
    print(f"Language: {config['language']}")
    print(f"Sampling rate: {config['sampling_rate']} Hz")
    print(f"Cache directory: {config['cache_dir']}")

    # Create cache directory
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if preprocessed data already exists
    output_dir = cache_dir / "preprocessed"

    if output_dir.exists() and (output_dir / "dataset_dict.json").exists():
        print("\n" + "=" * 70)
        print("LOADING PREPROCESSED DATASET")
        print("=" * 70)
        print(f"Found existing preprocessed data at: {output_dir}")
        print("Loading from cache...")

        from datasets import load_from_disk
        common_voice = load_from_disk(str(output_dir))

        print("Preprocessed data loaded successfully!")
        print(f"Splits: {list(common_voice.keys())}")
        for split_name, split_data in common_voice.items():
            print(f"{split_name}: {len(split_data)} examples")
    else:
        # Load local dataset
        print("\n" + "=" * 70)
        print("LOADING DATASET")
        print("=" * 70)

        common_voice = load_local_common_voice(
            config["data_dir"],
            config["language"]
        )

        print(f"\nDataset loaded successfully!")
        print(f"Splits: {list(common_voice.keys())}")

        # Print dataset info
        for split_name, split_data in common_voice.items():
            print(f"{split_name}: {len(split_data)} examples")

        # Create processor
        print("\n" + "=" * 70)
        print("CREATING PROCESSOR")
        print("=" * 70)

        processor = create_whisper_processor(
            model_name="openai/whisper-small",
            language="Hindi",
            task="transcribe",
        )

        # Prepare preprocessing function
        prepare_fn = prepare_dataset_fn(processor, config)

        # Preprocess dataset
        print("\n" + "=" * 70)
        print("PREPROCESSING DATASET")
        print("=" * 70)
        print("Extracting features and tokenizing text...")

        # Get columns to remove (keep only input_features and labels)
        remove_columns = list(common_voice["train"].features.keys())

        common_voice = common_voice.map(
            prepare_fn,
            remove_columns=remove_columns,
            num_proc=config.get("num_proc", 4),
            desc="Preprocessing dataset",
        )

        print("\nPreprocessing completed!")

        # Save preprocessed dataset
        print("\n" + "=" * 70)
        print("SAVING PREPROCESSED DATASET")
        print("=" * 70)

        output_dir.mkdir(exist_ok=True)

        print(f"Saving to: {output_dir}")
        common_voice.save_to_disk(str(output_dir))

        print("Dataset saved successfully!")

    # Print statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    for split_name, split_data in common_voice.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Examples: {len(split_data)}")
        print(f"  Features: {list(split_data.features.keys())}")

        # Sample one example
        if len(split_data) > 0:
            sample = split_data[0]
            input_features = np.array(sample['input_features'])
            print(f"  Input features shape: {input_features.shape}")
            print(f"  Labels length: {len(sample['labels'])}")

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 70)
    print(f"\nPreprocessed data saved to: {output_dir}")
    print("\nNext step: Run training script")
    print("  uv run python scripts/train.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
