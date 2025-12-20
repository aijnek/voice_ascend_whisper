"""データベース設定とセッション管理"""
from sqlmodel import SQLModel, create_engine, Session
from webapp.config import settings


# SQLiteエンジンを作成
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    connect_args={
        "check_same_thread": False,  # SQLite specific
    },
    # Ensure UTF-8 encoding
    pool_pre_ping=True,
)


def create_db_and_tables():
    """全てのテーブルを作成"""
    SQLModel.metadata.create_all(engine)


def get_session():
    """データベースセッション dependency"""
    with Session(engine) as session:
        yield session
