"""Inference script for transcribing audio with fine-tuned Whisper model."""

import argparse
import time
from pathlib import Path

import librosa
import torch
from transformers import WhisperProcessor

from voice_ascend_whisper.models.lora_whisper import load_lora_whisper
from voice_ascend_whisper.utils.device import get_device


def load_audio(audio_path: str, sampling_rate: int = 16000):
    """
    Load and preprocess audio file.

    Args:
        audio_path: Path to audio file
        sampling_rate: Target sampling rate (Whisper uses 16kHz)

    Returns:
        Audio array resampled to target sampling rate
    """
    print(f"Loading audio from: {audio_path}")

    # Load audio
    audio, sr = librosa.load(audio_path, sr=sampling_rate)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    duration = len(audio) / sampling_rate
    print(f"Audio duration: {duration:.2f} seconds")

    return audio


def transcribe(
    audio_path: str,
    model_path: str = "models/final",
    base_model: str = "openai/whisper-small",
):
    """
    Transcribe audio file using fine-tuned Whisper model.

    Args:
        audio_path: Path to audio file
        model_path: Path to fine-tuned LoRA model directory
        base_model: Base Whisper model name

    Returns:
        Transcription text
    """
    # Setup device
    device = get_device()

    # Load model
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = load_lora_whisper(
        base_model_name=base_model,
        adapter_path=str(model_path),
        device=device,
    )

    model.eval()

    # Load processor
    print("\nLoading processor...")
    processor = WhisperProcessor.from_pretrained(str(model_path))

    # Load audio
    print("\n" + "=" * 70)
    print("LOADING AUDIO")
    print("=" * 70)

    audio = load_audio(audio_path)

    # Extract features
    print("\n" + "=" * 70)
    print("EXTRACTING FEATURES")
    print("=" * 70)

    input_features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features

    input_features = input_features.to(device)

    print(f"Input features shape: {input_features.shape}")

    # Generate transcription
    print("\n" + "=" * 70)
    print("GENERATING TRANSCRIPTION")
    print("=" * 70)

    start_time = time.time()

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            max_length=225,
        )

    inference_time = time.time() - start_time

    # Decode
    transcription = processor.tokenizer.decode(
        generated_ids[0], skip_special_tokens=True
    )

    print(f"\nInference time: {inference_time:.2f} seconds")

    return transcription


def main():
    """Main inference pipeline."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio with fine-tuned Whisper model"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file to transcribe",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/final",
        help="Path to fine-tuned LoRA model directory",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="openai/whisper-small",
        help="Base Whisper model name",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("WHISPER INFERENCE")
    print("=" * 70)

    # Check audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Run transcription
    transcription = transcribe(
        audio_path=str(audio_path),
        model_path=args.model,
        base_model=args.base_model,
    )

    # Print result
    print("\n" + "=" * 70)
    print("TRANSCRIPTION RESULT")
    print("=" * 70)
    print(f"\n{transcription}\n")
    print("=" * 70)


if __name__ == "__main__":
    main()
