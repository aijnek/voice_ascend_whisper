"""Service for managing audio files."""

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import librosa
import soundfile as sf

from webapp.config import Settings


class AudioService:
    """Service for audio file operations."""

    @staticmethod
    def save_audio(
        base64_audio: str,
        text_id: int,
        settings: Settings,
        recording_id: Optional[int] = None,
    ) -> tuple[Path, float, int]:
        """Save Base64-encoded WAV audio to file.

        Args:
            base64_audio: Base64-encoded WAV audio data
            text_id: Associated text ID
            settings: Application settings
            recording_id: Recording ID (if available)

        Returns:
            Tuple of (file_path, duration, file_size)

        Raises:
            ValueError: If audio data is invalid or too long
        """
        # Decode Base64 audio
        try:
            audio_bytes = base64.b64decode(base64_audio)
        except Exception as e:
            raise ValueError(f"Invalid Base64 audio data: {e}")

        # Load audio using soundfile
        try:
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        except Exception as e:
            raise ValueError(f"Failed to read audio data: {e}")

        # Resample to target sample rate if needed
        target_sample_rate = settings.TARGET_SAMPLE_RATE
        if sample_rate != target_sample_rate:
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=target_sample_rate,
            )
            sample_rate = target_sample_rate

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = librosa.to_mono(audio_data.T)

        # Calculate duration
        duration = len(audio_data) / sample_rate

        # Validate duration
        max_duration = settings.MAX_AUDIO_DURATION
        if duration > max_duration:
            raise ValueError(
                f"Audio duration ({duration:.2f}s) exceeds maximum ({max_duration}s)"
            )

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if recording_id:
            filename = f"rec_{recording_id}_{timestamp}.wav"
        else:
            filename = f"rec_text{text_id}_{timestamp}.wav"

        # Ensure audio directory exists
        audio_dir = settings.WEBAPP_AUDIO_DIR
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Save to file
        file_path = audio_dir / filename
        sf.write(file_path, audio_data, sample_rate)

        # Get file size
        file_size = file_path.stat().st_size

        return file_path, duration, file_size

    @staticmethod
    def delete_audio(file_path: Path) -> bool:
        """Delete an audio file.

        Args:
            file_path: Path to audio file

        Returns:
            True if deleted, False if file not found
        """
        try:
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def get_audio_info(file_path: Path) -> Optional[dict]:
        """Get audio file information.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with audio info (duration, sample_rate, channels) or None
        """
        if not file_path.exists():
            return None

        try:
            audio_data, sample_rate = sf.read(file_path)
            duration = len(audio_data) / sample_rate
            channels = 1 if len(audio_data.shape) == 1 else audio_data.shape[1]

            return {
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "file_size": file_path.stat().st_size,
            }
        except Exception:
            return None

    @staticmethod
    def validate_audio(file_path: Path, settings: Settings) -> tuple[bool, str]:
        """Validate audio file quality.

        Args:
            file_path: Path to audio file
            settings: Application settings

        Returns:
            Tuple of (is_valid, message)
        """
        info = AudioService.get_audio_info(file_path)
        if not info:
            return False, "Cannot read audio file"

        # Check duration
        if info["duration"] < settings.MIN_AUDIO_DURATION:
            return False, f"Audio too short (< {settings.MIN_AUDIO_DURATION}s)"

        if info["duration"] > settings.MAX_AUDIO_DURATION:
            return False, f"Audio too long (> {settings.MAX_AUDIO_DURATION}s)"

        # Check sample rate
        if info["sample_rate"] != settings.TARGET_SAMPLE_RATE:
            return False, f"Invalid sample rate (expected {settings.TARGET_SAMPLE_RATE}Hz)"

        # Check channels
        if info["channels"] != 1:
            return False, "Audio must be mono"

        return True, "Valid audio file"

    @staticmethod
    def get_relative_path(file_path: Path, settings: Settings) -> str:
        """Get relative path from WEBAPP_DATA_DIR.

        Args:
            file_path: Absolute file path
            settings: Application settings

        Returns:
            Relative path string
        """
        try:
            return str(file_path.relative_to(settings.WEBAPP_DATA_DIR))
        except ValueError:
            return str(file_path)
