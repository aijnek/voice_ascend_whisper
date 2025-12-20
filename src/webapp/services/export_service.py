"""Service for exporting datasets to Common Voice format."""

import random
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlmodel import Session, select

from finetune_whisper.data.formats import (
    create_common_voice_tsv,
    validate_common_voice_format,
)
from webapp.config import Settings
from webapp.models.dataset import DatasetExport, DatasetExportCreate
from webapp.models.recording import Recording


class ExportService:
    """Service for dataset export operations."""

    @staticmethod
    def create_export(
        session: Session,
        export_data: DatasetExportCreate,
        settings: Settings,
    ) -> DatasetExport:
        """Create a new dataset export.

        Args:
            session: Database session
            export_data: Export configuration data
            settings: Application settings

        Returns:
            Created DatasetExport instance with status='pending'
        """
        # Create export directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = export_data.name.replace(" ", "_").lower()
        export_dir_name = f"{export_name}_{timestamp}"
        export_path = settings.WEBAPP_EXPORTS_DIR / export_dir_name

        # Get relative path
        relative_path = export_path.relative_to(settings.WEBAPP_DATA_DIR)

        # Create database entry
        db_export = DatasetExport(
            name=export_data.name,
            description=export_data.description,
            export_path=str(relative_path),
            total_recordings=0,
            train_count=0,
            dev_count=0,
            test_count=0,
            train_ratio=export_data.train_ratio,
            dev_ratio=export_data.dev_ratio,
            test_ratio=export_data.test_ratio,
            split_strategy=export_data.split_strategy,
            min_duration=export_data.min_duration,
            max_duration=export_data.max_duration,
            validated_only=export_data.validated_only,
            status="pending",
        )

        session.add(db_export)
        session.commit()
        session.refresh(db_export)

        return db_export

    @staticmethod
    def execute_export(
        session: Session,
        export_id: int,
        settings: Settings,
    ) -> DatasetExport:
        """Execute the dataset export process.

        Args:
            session: Database session
            export_id: Export ID
            settings: Application settings

        Returns:
            Updated DatasetExport instance

        Raises:
            ValueError: If export not found or invalid configuration
        """
        db_export = session.get(DatasetExport, export_id)
        if not db_export:
            raise ValueError(f"Export {export_id} not found")

        try:
            # Update status
            db_export.status = "processing"
            session.add(db_export)
            session.commit()

            # Get export directory
            export_path = settings.WEBAPP_DATA_DIR / db_export.export_path
            export_path.mkdir(parents=True, exist_ok=True)

            # Create clips directory
            clips_dir = export_path / "clips"
            clips_dir.mkdir(exist_ok=True)

            # Query recordings with filters
            recordings = ExportService._query_recordings(session, db_export)

            if len(recordings) == 0:
                raise ValueError("No recordings found matching the filter criteria")

            # Split recordings into train/dev/test
            train_recs, dev_recs, test_recs = ExportService._split_recordings(
                recordings,
                db_export.train_ratio,
                db_export.dev_ratio,
                db_export.test_ratio,
                db_export.split_strategy,
            )

            # Create TSV files for each split
            ExportService._create_tsv_for_split(train_recs, export_path / "train.tsv", clips_dir, settings)
            ExportService._create_tsv_for_split(dev_recs, export_path / "dev.tsv", clips_dir, settings)
            ExportService._create_tsv_for_split(test_recs, export_path / "test.tsv", clips_dir, settings)

            # Validate dataset format
            is_valid, errors = validate_common_voice_format(export_path)
            if not is_valid:
                raise ValueError(f"Dataset validation failed: {errors}")

            # Update export statistics
            db_export.total_recordings = len(recordings)
            db_export.train_count = len(train_recs)
            db_export.dev_count = len(dev_recs)
            db_export.test_count = len(test_recs)
            db_export.status = "completed"
            db_export.completed_at = datetime.now()
            db_export.error_message = None

            session.add(db_export)
            session.commit()

            # Update 'latest' symlink
            ExportService._update_latest_symlink(export_path, settings)

            session.refresh(db_export)
            return db_export

        except Exception as e:
            # Update status on error
            db_export.status = "failed"
            db_export.error_message = str(e)
            session.add(db_export)
            session.commit()
            session.refresh(db_export)
            raise

    @staticmethod
    def _query_recordings(
        session: Session,
        export: DatasetExport,
    ) -> list[Recording]:
        """Query recordings based on export filters.

        Args:
            session: Database session
            export: DatasetExport instance with filter criteria

        Returns:
            List of Recording instances
        """
        statement = select(Recording)

        # Filter by validation status
        if export.validated_only:
            statement = statement.where(Recording.is_validated == True)

        # Filter by duration
        if export.min_duration is not None:
            statement = statement.where(Recording.duration >= export.min_duration)
        if export.max_duration is not None:
            statement = statement.where(Recording.duration <= export.max_duration)

        # Order by creation time for chronological split
        statement = statement.order_by(Recording.created_at.asc())

        return list(session.exec(statement).all())

    @staticmethod
    def _split_recordings(
        recordings: list[Recording],
        train_ratio: float,
        dev_ratio: float,
        test_ratio: float,
        strategy: str,
    ) -> tuple[list[Recording], list[Recording], list[Recording]]:
        """Split recordings into train/dev/test sets.

        Args:
            recordings: List of recordings
            train_ratio: Training set ratio (e.g., 80.0)
            dev_ratio: Dev set ratio (e.g., 10.0)
            test_ratio: Test set ratio (e.g., 10.0)
            strategy: Split strategy ('random' or 'chronological')

        Returns:
            Tuple of (train_recordings, dev_recordings, test_recordings)
        """
        total = len(recordings)

        # Calculate split sizes
        train_size = int(total * train_ratio / 100)
        dev_size = int(total * dev_ratio / 100)
        test_size = total - train_size - dev_size  # Remainder goes to test

        # Shuffle if random strategy
        if strategy == "random":
            recordings_copy = recordings.copy()
            random.shuffle(recordings_copy)
        else:
            # Chronological: already ordered by created_at
            recordings_copy = recordings

        # Split
        train_recs = recordings_copy[:train_size]
        dev_recs = recordings_copy[train_size : train_size + dev_size]
        test_recs = recordings_copy[train_size + dev_size :]

        return train_recs, dev_recs, test_recs

    @staticmethod
    def _create_tsv_for_split(
        recordings: list[Recording],
        output_path: Path,
        clips_dir: Path,
        settings: Settings,
    ) -> None:
        """Create TSV file for a split using Common Voice format.

        Args:
            recordings: List of recordings
            output_path: Output TSV file path
            clips_dir: Directory to copy audio clips
            settings: Application settings
        """
        # Convert recordings to format expected by create_common_voice_tsv
        recordings_data = []
        for rec in recordings:
            audio_path = settings.WEBAPP_DATA_DIR / rec.file_path
            recordings_data.append(
                {
                    "audio_path": str(audio_path),
                    "sentence": rec.text.content if rec.text else "",
                    "locale": rec.text.language if rec.text else "ja",
                    "client_id": "webapp_user",
                }
            )

        # Create TSV using formats.py utility
        create_common_voice_tsv(
            recordings=recordings_data,
            output_path=output_path,
            clips_dir=clips_dir,
        )

    @staticmethod
    def _update_latest_symlink(export_path: Path, settings: Settings) -> None:
        """Update 'latest' symlink to point to the new export.

        Args:
            export_path: Path to the new export directory
            settings: Application settings
        """
        latest_link = settings.WEBAPP_EXPORTS_DIR / "latest"

        # Remove existing symlink if it exists
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        # Create new symlink
        latest_link.symlink_to(export_path.name)

    @staticmethod
    def get_export(session: Session, export_id: int) -> Optional[DatasetExport]:
        """Get export by ID.

        Args:
            session: Database session
            export_id: Export ID

        Returns:
            DatasetExport instance if found, None otherwise
        """
        return session.get(DatasetExport, export_id)

    @staticmethod
    def get_exports(
        session: Session,
        skip: int = 0,
        limit: int = 100,
    ) -> list[DatasetExport]:
        """Get list of exports.

        Args:
            session: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of DatasetExport instances
        """
        statement = (
            select(DatasetExport)
            .offset(skip)
            .limit(limit)
            .order_by(DatasetExport.created_at.desc())
        )
        return list(session.exec(statement).all())

    @staticmethod
    def delete_export(session: Session, export_id: int, settings: Settings) -> bool:
        """Delete an export and its files.

        Args:
            session: Database session
            export_id: Export ID
            settings: Application settings

        Returns:
            True if deleted, False if not found
        """
        db_export = session.get(DatasetExport, export_id)
        if not db_export:
            return False

        # Delete export directory
        export_path = settings.WEBAPP_DATA_DIR / db_export.export_path
        if export_path.exists():
            import shutil

            shutil.rmtree(export_path)

        # Delete database record
        session.delete(db_export)
        session.commit()
        return True
