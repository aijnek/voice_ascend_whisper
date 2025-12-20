"""Training script for Whisper fine-tuning with LoRA."""

from pathlib import Path

import yaml
import torch
from datasets import load_from_disk
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from finetune_whisper.models.lora_whisper import create_lora_whisper
from finetune_whisper.data.processor import create_whisper_processor
from finetune_whisper.data.collator import DataCollatorSpeechSeq2SeqWithPadding
from finetune_whisper.utils.device import print_device_info, clear_mps_cache
from finetune_whisper.utils.metrics import create_compute_metrics


def load_configs():
    """Load all configuration files."""
    configs = {}

    # Load data config
    with open("configs/data_config.yaml", "r") as f:
        configs["data"] = yaml.safe_load(f)["dataset"]

    # Load LoRA config
    with open("configs/lora_config.yaml", "r") as f:
        configs["lora"] = yaml.safe_load(f)["lora"]

    # Load training config
    with open("configs/training_config.yaml", "r") as f:
        configs["training"] = yaml.safe_load(f)["training"]

    return configs


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("WHISPER FINE-TUNING WITH LORA")
    print("=" * 70)

    # Load configurations
    print("\nLoading configurations...")
    configs = load_configs()

    # Setup device
    print("\n" + "=" * 70)
    print("DEVICE SETUP")
    print("=" * 70)
    device = print_device_info()

    # Load preprocessed dataset
    print("\n" + "=" * 70)
    print("LOADING PREPROCESSED DATASET")
    print("=" * 70)

    cache_dir = Path(configs["data"]["cache_dir"])
    dataset_path = cache_dir / "preprocessed"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {dataset_path}. "
            "Please run scripts/prepare_data.py first."
        )

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(str(dataset_path))

    print(f"\nDataset loaded successfully!")
    for split_name, split_data in dataset.items():
        print(f"{split_name}: {len(split_data)} examples")

    # Limit eval dataset size for quick testing
    max_eval_samples = configs["training"].get("max_eval_samples", -1)
    if max_eval_samples > 0:
        print(f"\nLimiting eval dataset to {max_eval_samples} samples for quick testing")
        if "validation" in dataset:
            dataset["validation"] = dataset["validation"].select(
                range(min(max_eval_samples, len(dataset["validation"])))
            )
            print(f"Validation dataset reduced to: {len(dataset['validation'])} examples")
        elif "test" in dataset:
            dataset["test"] = dataset["test"].select(
                range(min(max_eval_samples, len(dataset["test"])))
            )
            print(f"Test dataset reduced to: {len(dataset['test'])} examples")

    # Create processor
    print("\n" + "=" * 70)
    print("CREATING PROCESSOR")
    print("=" * 70)

    processor = create_whisper_processor(
        model_name=configs["training"]["model_name"],
        language=configs["training"]["language"],
        task=configs["training"]["task"],
    )

    # Create LoRA model
    print("\n" + "=" * 70)
    print("CREATING LORA MODEL")
    print("=" * 70)

    # Get continue_from_checkpoint parameter (if provided)
    continue_checkpoint = configs["training"].get("continue_from_checkpoint")

    model = create_lora_whisper(
        base_model_name=configs["training"]["model_name"],
        lora_config=configs["lora"],
        device=device,
        continue_from_checkpoint=continue_checkpoint,
    )

    # Set language and task for generation
    # Use language and task instead of forced_decoder_ids (deprecated)
    model.generation_config.language = configs["training"]["language"].lower()
    model.generation_config.task = configs["training"]["task"]
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Create data collator
    print("\n" + "=" * 70)
    print("CREATING DATA COLLATOR")
    print("=" * 70)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    print("Data collator created successfully!")

    # Create compute metrics function
    print("\n" + "=" * 70)
    print("SETTING UP METRICS")
    print("=" * 70)

    compute_metrics = create_compute_metrics(processor.tokenizer)
    print("Metrics setup complete!")

    # Setup training arguments
    print("\n" + "=" * 70)
    print("CONFIGURING TRAINING ARGUMENTS")
    print("=" * 70)

    training_config = configs["training"]

    training_args = Seq2SeqTrainingArguments(
        output_dir=training_config["output_dir"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        num_train_epochs=training_config["num_train_epochs"],
        max_steps=training_config["max_steps"],
        learning_rate=training_config["learning_rate"],
        warmup_steps=training_config["warmup_steps"],
        weight_decay=training_config["weight_decay"],
        max_grad_norm=training_config["max_grad_norm"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        fp16=training_config["fp16"],
        bf16=training_config["bf16"],
        eval_strategy=training_config["eval_strategy"],
        eval_steps=training_config["eval_steps"],
        save_strategy=training_config["save_strategy"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        predict_with_generate=training_config["predict_with_generate"],
        generation_max_length=training_config["generation_max_length"],
        logging_steps=training_config["logging_steps"],
        logging_first_step=training_config["logging_first_step"],
        report_to=training_config["report_to"],
        dataloader_num_workers=training_config["dataloader_num_workers"],
        remove_unused_columns=training_config["remove_unused_columns"],
        label_names=["labels"],  # Required for PEFT compatibility
        push_to_hub=False,
    )

    print("\nTraining configuration:")
    print(f"  Batch size: {training_config['per_device_train_batch_size']}")
    print(f"  Gradient accumulation: {training_config['gradient_accumulation_steps']}")
    print(
        f"  Effective batch size: {training_config['per_device_train_batch_size'] * training_config['gradient_accumulation_steps']}"
    )
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Epochs: {training_config['num_train_epochs']}")
    print(f"  Gradient checkpointing: {training_config['gradient_checkpointing']}")

    # Create trainer
    print("\n" + "=" * 70)
    print("CREATING TRAINER")
    print("=" * 70)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("test")),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.tokenizer,
    )

    print("Trainer created successfully!")

    # Start training
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print("\nThis may take several hours depending on your hardware...")
    print("Monitor progress with TensorBoard:")
    print(f"  tensorboard --logdir {training_config['output_dir']}")
    print("=" * 70 + "\n")

    try:
        trainer.train()

        # Clear MPS cache after training
        clear_mps_cache()

        # Save best model (permanent, won't be deleted by save_total_limit)
        print("\n" + "=" * 70)
        print("SAVING BEST MODEL")
        print("=" * 70)

        best_model_dir = Path("models/best")
        best_model_dir.mkdir(parents=True, exist_ok=True)

        # Best model is automatically loaded if load_best_model_at_end=True
        print(f"Saving best model (by {training_config.get('metric_for_best_model', 'wer')})")
        print(f"Saving to: {best_model_dir}")
        model.save_pretrained(str(best_model_dir))
        processor.save_pretrained(str(best_model_dir))
        print("Best model saved successfully!")

        # Print training summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"\nBest model (by {training_config.get('metric_for_best_model', 'wer')}) saved to: {best_model_dir}")
        print(f"Recent checkpoints: {training_config['output_dir']}")
        print(f"\nTo use the best model for inference or evaluation:")
        print(f"  Model path: {best_model_dir}")
        print("\nNext steps:")
        print("  1. Evaluate: uv run python scripts/evaluate.py")
        print("  2. Inference: uv run python scripts/inference.py --audio <path>")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR DURING TRAINING")
        print("=" * 70)
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("  - Reduce batch_size in configs/training_config.yaml")
        print("  - Enable gradient_checkpointing")
        print("  - Check MPS device availability")
        print("  - Monitor memory usage with Activity Monitor")
        print("=" * 70)
        raise


if __name__ == "__main__":
    main()
