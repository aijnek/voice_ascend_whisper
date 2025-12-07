"""Metrics computation for Whisper evaluation."""

import evaluate
from transformers import WhisperTokenizer


def create_compute_metrics(tokenizer: WhisperTokenizer):
    """
    Create compute_metrics function for Seq2SeqTrainer.

    Args:
        tokenizer: WhisperTokenizer for decoding predictions

    Returns:
        Function that computes WER metric
    """
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        """
        Compute Word Error Rate (WER) metric.

        Args:
            pred: Predictions from model containing pred.predictions and pred.label_ids

        Returns:
            Dictionary with 'wer' metric
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id (can't decode -100)
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute WER
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    return compute_metrics


def compute_wer_from_texts(predictions: list[str], references: list[str]) -> float:
    """
    Compute WER from lists of prediction and reference texts.

    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions

    Returns:
        WER as percentage (0-100)
    """
    metric = evaluate.load("wer")
    wer = 100 * metric.compute(predictions=predictions, references=references)
    return wer


def compute_detailed_metrics(predictions: list[str], references: list[str]) -> dict:
    """
    Compute detailed error metrics using jiwer.

    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions

    Returns:
        Dictionary with detailed metrics (WER, substitutions, deletions, insertions)
    """
    import jiwer

    # Compute all measures
    measures = jiwer.compute_measures(references, predictions)

    return {
        "wer": measures["wer"] * 100,  # Convert to percentage
        "mer": measures["mer"] * 100,  # Match Error Rate
        "wil": measures["wil"] * 100,  # Word Information Lost
        "wip": measures["wip"] * 100,  # Word Information Preserved
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
        "hits": measures["hits"],
    }
