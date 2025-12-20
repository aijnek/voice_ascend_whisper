"""Service for managing Text entries."""

from datetime import datetime
from typing import Optional

from sqlmodel import Session, select, func

from webapp.models.text import Text, TextCreate, TextUpdate
from webapp.models.recording import Recording


class TextService:
    """Service for Text CRUD operations."""

    @staticmethod
    def create_text(session: Session, text_data: TextCreate) -> Text:
        """Create a new text entry.

        Args:
            session: Database session
            text_data: Text creation data

        Returns:
            Created Text instance
        """
        db_text = Text.model_validate(text_data)
        session.add(db_text)
        session.commit()
        session.refresh(db_text)
        return db_text

    @staticmethod
    def get_text(session: Session, text_id: int) -> Optional[Text]:
        """Get a text entry by ID.

        Args:
            session: Database session
            text_id: Text ID

        Returns:
            Text instance if found, None otherwise
        """
        return session.get(Text, text_id)

    @staticmethod
    def get_texts(
        session: Session,
        skip: int = 0,
        limit: int = 100,
        language: Optional[str] = None,
        source: Optional[str] = None,
    ) -> list[Text]:
        """Get list of text entries with optional filters.

        Args:
            session: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            language: Filter by language code
            source: Filter by source type

        Returns:
            List of Text instances
        """
        statement = select(Text)

        if language:
            statement = statement.where(Text.language == language)
        if source:
            statement = statement.where(Text.source == source)

        statement = statement.offset(skip).limit(limit).order_by(Text.created_at.desc())
        return list(session.exec(statement).all())

    @staticmethod
    def update_text(
        session: Session, text_id: int, text_data: TextUpdate
    ) -> Optional[Text]:
        """Update a text entry.

        Args:
            session: Database session
            text_id: Text ID
            text_data: Update data

        Returns:
            Updated Text instance if found, None otherwise
        """
        db_text = session.get(Text, text_id)
        if not db_text:
            return None

        update_dict = text_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(db_text, key, value)

        db_text.updated_at = datetime.now()
        session.add(db_text)
        session.commit()
        session.refresh(db_text)
        return db_text

    @staticmethod
    def delete_text(session: Session, text_id: int) -> bool:
        """Delete a text entry.

        Args:
            session: Database session
            text_id: Text ID

        Returns:
            True if deleted, False if not found
        """
        db_text = session.get(Text, text_id)
        if not db_text:
            return False

        session.delete(db_text)
        session.commit()
        return True

    @staticmethod
    def count_texts(session: Session, language: Optional[str] = None) -> int:
        """Count total number of texts.

        Args:
            session: Database session
            language: Optional language filter

        Returns:
            Total count of texts
        """
        statement = select(Text)
        if language:
            statement = statement.where(Text.language == language)

        return len(list(session.exec(statement).all()))

    @staticmethod
    def bulk_create_texts(
        session: Session, texts_data: list[TextCreate]
    ) -> list[Text]:
        """Bulk create text entries.

        Args:
            session: Database session
            texts_data: List of text creation data

        Returns:
            List of created Text instances
        """
        db_texts = [Text.model_validate(text_data) for text_data in texts_data]
        session.add_all(db_texts)
        session.commit()
        for db_text in db_texts:
            session.refresh(db_text)
        return db_texts

    @staticmethod
    def get_texts_without_recordings(
        session: Session,
        skip: int = 0,
        limit: int = 100,
        language: Optional[str] = None,
    ) -> list[Text]:
        """Get texts that don't have any recordings yet.

        Args:
            session: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            language: Filter by language code

        Returns:
            List of Text instances without recordings
        """
        # Subquery to get text IDs that have recordings
        recorded_text_ids = select(Recording.text_id).distinct()

        # Main query to get texts without recordings
        statement = select(Text).where(Text.id.not_in(recorded_text_ids))

        if language:
            statement = statement.where(Text.language == language)

        statement = statement.offset(skip).limit(limit).order_by(Text.created_at.desc())
        return list(session.exec(statement).all())
