"""Service for managing Recording entries."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlmodel import Session, select

from webapp.config import Settings
from webapp.models.recording import Recording, RecordingCreate, RecordingUpdate
from webapp.services.audio_service import AudioService


class RecordingService:
    """Service for Recording CRUD operations."""

    @staticmethod
    def create_recording(
        session: Session,
        recording_data: RecordingCreate,
        base64_audio: str,
        settings: Settings,
    ) -> Recording:
        """Create a new recording with audio file.

        Args:
            session: Database session
            recording_data: Recording creation data
            base64_audio: Base64-encoded WAV audio data
            settings: Application settings

        Returns:
            Created Recording instance

        Raises:
            ValueError: If audio data is invalid
        """
        # Save audio file
        file_path, duration, file_size = AudioService.save_audio(
            base64_audio=base64_audio,
            text_id=recording_data.text_id,
            settings=settings,
        )

        # Get relative path
        relative_path = AudioService.get_relative_path(file_path, settings)

        # Create recording in database
        db_recording = Recording(
            text_id=recording_data.text_id,
            filename=file_path.name,
            file_path=relative_path,
            file_size=file_size,
            duration=duration,
            sample_rate=settings.TARGET_SAMPLE_RATE,
            channels=1,
            format="wav",
            is_validated=False,
            notes=recording_data.notes,
        )

        session.add(db_recording)
        session.commit()
        session.refresh(db_recording)

        # Update filename with recording ID
        new_file_path, duration, file_size = AudioService.save_audio(
            base64_audio=base64_audio,
            text_id=recording_data.text_id,
            settings=settings,
            recording_id=db_recording.id,
        )

        # Delete old file
        AudioService.delete_audio(file_path)

        # Update recording with new filename
        db_recording.filename = new_file_path.name
        db_recording.file_path = AudioService.get_relative_path(new_file_path, settings)
        session.add(db_recording)
        session.commit()
        session.refresh(db_recording)

        return db_recording

    @staticmethod
    def get_recording(session: Session, recording_id: int) -> Optional[Recording]:
        """Get a recording by ID.

        Args:
            session: Database session
            recording_id: Recording ID

        Returns:
            Recording instance if found, None otherwise
        """
        return session.get(Recording, recording_id)

    @staticmethod
    def get_recordings(
        session: Session,
        skip: int = 0,
        limit: int = 100,
        text_id: Optional[int] = None,
        validated_only: bool = False,
    ) -> list[Recording]:
        """Get list of recordings with optional filters.

        Args:
            session: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            text_id: Filter by text ID
            validated_only: Only return validated recordings

        Returns:
            List of Recording instances
        """
        statement = select(Recording)

        if text_id:
            statement = statement.where(Recording.text_id == text_id)
        if validated_only:
            statement = statement.where(Recording.is_validated == True)

        statement = statement.offset(skip).limit(limit).order_by(Recording.created_at.desc())
        return list(session.exec(statement).all())

    @staticmethod
    def update_recording(
        session: Session,
        recording_id: int,
        recording_data: RecordingUpdate,
    ) -> Optional[Recording]:
        """Update a recording entry.

        Args:
            session: Database session
            recording_id: Recording ID
            recording_data: Update data

        Returns:
            Updated Recording instance if found, None otherwise
        """
        db_recording = session.get(Recording, recording_id)
        if not db_recording:
            return None

        update_dict = recording_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(db_recording, key, value)

        db_recording.updated_at = datetime.now()
        session.add(db_recording)
        session.commit()
        session.refresh(db_recording)
        return db_recording

    @staticmethod
    def delete_recording(
        session: Session,
        recording_id: int,
        settings: Settings,
    ) -> bool:
        """Delete a recording and its audio file.

        Args:
            session: Database session
            recording_id: Recording ID
            settings: Application settings

        Returns:
            True if deleted, False if not found
        """
        db_recording = session.get(Recording, recording_id)
        if not db_recording:
            return False

        # Delete audio file
        file_path = settings.WEBAPP_DATA_DIR / db_recording.file_path
        AudioService.delete_audio(file_path)

        # Delete database entry
        session.delete(db_recording)
        session.commit()
        return True

    @staticmethod
    def validate_recording(
        session: Session,
        recording_id: int,
        is_valid: bool,
        notes: Optional[str] = None,
    ) -> Optional[Recording]:
        """Manually validate/invalidate a recording.

        Args:
            session: Database session
            recording_id: Recording ID
            is_valid: Whether recording is valid
            notes: Optional validation notes

        Returns:
            Updated Recording instance if found, None otherwise
        """
        db_recording = session.get(Recording, recording_id)
        if not db_recording:
            return None

        db_recording.is_validated = is_valid
        if notes:
            db_recording.notes = notes
        db_recording.updated_at = datetime.now()

        session.add(db_recording)
        session.commit()
        session.refresh(db_recording)
        return db_recording

    @staticmethod
    def count_recordings(
        session: Session,
        text_id: Optional[int] = None,
        validated_only: bool = False,
    ) -> int:
        """Count total number of recordings.

        Args:
            session: Database session
            text_id: Optional text ID filter
            validated_only: Only count validated recordings

        Returns:
            Total count of recordings
        """
        statement = select(Recording)

        if text_id:
            statement = statement.where(Recording.text_id == text_id)
        if validated_only:
            statement = statement.where(Recording.is_validated == True)

        return len(list(session.exec(statement).all()))

    @staticmethod
    def get_recording_file_path(recording: Recording, settings: Settings) -> Path:
        """Get absolute file path for a recording.

        Args:
            recording: Recording instance
            settings: Application settings

        Returns:
            Absolute path to recording file
        """
        return settings.WEBAPP_DATA_DIR / recording.file_path
