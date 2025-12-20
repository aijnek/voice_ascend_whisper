"""Evaluation script for fine-tuned Whisper model."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from tqdm import tqdm
from transformers import WhisperProcessor

from transformers import WhisperForConditionalGeneration
from finetune_whisper.models.lora_whisper import load_lora_whisper
from finetune_whisper.utils.device import get_device, clear_mps_cache
from finetune_whisper.utils.metrics import (
    compute_wer_from_texts,
    compute_detailed_metrics,
)


def load_config(config_path: str = "configs/data_config.yaml") -> dict:
    """Load data configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["dataset"]


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Whisper model")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to fine-tuned LoRA model directory (None = use base model only)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="openai/whisper-small",
        help="Base Whisper model name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (test/validation)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None = all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation_results.json",
        help="Output file for evaluation results",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("WHISPER MODEL EVALUATION")
    print("=" * 70)

    # Setup device
    print("\nSetting up device...")
    device = get_device()

    # Load model
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    if args.model is None:
        # Use base model without LoRA
        print(f"Using base model: {args.base_model}")
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
        if device:
            model = model.to(device)
        processor = WhisperProcessor.from_pretrained(args.base_model)
        model_path = None
    else:
        # Use fine-tuned LoRA model
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading model from: {model_path}")

        model = load_lora_whisper(
            base_model_name=args.base_model,
            adapter_path=str(model_path),
            device=device,
        )

        # Load processor from base model (checkpoints don't have processor files)
        print("\nLoading processor...")
        processor = WhisperProcessor.from_pretrained(args.base_model)

    model.eval()

    # Load dataset
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    config = load_config()
    cache_dir = Path(config["cache_dir"])
    dataset_path = cache_dir / "preprocessed"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {dataset_path}. "
            "Please run scripts/prepare_data.py first."
        )

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(str(dataset_path))

    if args.split not in dataset:
        raise ValueError(
            f"Split '{args.split}' not found in dataset. "
            f"Available splits: {list(dataset.keys())}"
        )

    eval_dataset = dataset[args.split]
    print(f"\nEvaluating on {args.split} split: {len(eval_dataset)} examples")

    # Limit samples if specified
    if args.max_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_samples, len(eval_dataset))))
        print(f"Limited to {len(eval_dataset)} samples")

    # Run evaluation
    print("\n" + "=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)

    predictions = []
    references = []

    print(f"Processing {len(eval_dataset)} examples...")

    with torch.no_grad():
        for i in tqdm(range(0, len(eval_dataset), args.batch_size)):
            batch = eval_dataset[i : i + args.batch_size]

            # Get input features
            if isinstance(batch["input_features"], list):
                input_features = torch.tensor(batch["input_features"]).to(device)
            else:
                input_features = torch.tensor([batch["input_features"]]).to(device)

            # Generate predictions
            generated_ids = model.generate(
                input_features,
                max_length=225,
            )

            # Decode predictions
            transcriptions = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            predictions.extend(transcriptions)

            # Decode references
            if isinstance(batch["labels"], list):
                labels_batch = batch["labels"]
            else:
                labels_batch = [batch["labels"]]

            for labels in labels_batch:
                # Convert to list if tensor
                if isinstance(labels, torch.Tensor):
                    labels = labels.tolist()

                # Remove -100 padding tokens
                labels = [l for l in labels if l != -100]

                # Decode
                reference = processor.tokenizer.decode(labels, skip_special_tokens=True)
                references.append(reference)

            # Clear cache periodically
            if i % (args.batch_size * 10) == 0:
                clear_mps_cache()

    # Compute metrics
    print("\n" + "=" * 70)
    print("COMPUTING METRICS")
    print("=" * 70)

    wer = compute_wer_from_texts(predictions, references)
    detailed_metrics = compute_detailed_metrics(predictions, references)

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nWord Error Rate (WER): {wer:.2f}%")
    print(f"\nDetailed Metrics:")
    print(f"  Match Error Rate (MER): {detailed_metrics['mer']:.2f}%")
    print(f"  Word Information Lost (WIL): {detailed_metrics['wil']:.2f}%")
    print(f"  Word Information Preserved (WIP): {detailed_metrics['wip']:.2f}%")
    print(f"\nError Breakdown:")
    print(f"  Substitutions: {detailed_metrics['substitutions']}")
    print(f"  Deletions: {detailed_metrics['deletions']}")
    print(f"  Insertions: {detailed_metrics['insertions']}")
    print(f"  Hits: {detailed_metrics['hits']}")

    # Show sample predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    num_samples = min(5, len(predictions))
    for i in range(num_samples):
        print(f"\nExample {i + 1}:")
        print(f"  Reference: {references[i]}")
        print(f"  Prediction: {predictions[i]}")

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "model_path": str(model_path) if model_path else None,
        "base_model": args.base_model,
        "is_finetuned": model_path is not None,
        "split": args.split,
        "num_samples": len(predictions),
        "wer": wer,
        "detailed_metrics": detailed_metrics,
        "sample_predictions": [
            {"reference": ref, "prediction": pred}
            for ref, pred in zip(references[:10], predictions[:10])
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
