"""Drop and recreate sentiment tables with correct constraints — bypasses ALTER TABLE timeout."""
import asyncio
import asyncpg
from pathlib import Path
from dotenv import load_dotenv
import sys

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

try:
    from app.core.config import settings
    DATABASE_URL = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
except Exception as e:
    print(f"Error loading config: {e}")
    sys.exit(1)


async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    await conn.execute("SET statement_timeout = 0")
    print("✅ Connected")

    # Drop if exist
    await conn.execute("DROP TABLE IF EXISTS daily_sentiment CASCADE")
    await conn.execute("DROP TABLE IF EXISTS sentiment_snapshots CASCADE")
    await conn.execute("DROP TABLE IF EXISTS sentiment_articles CASCADE")
    print("✅ Dropped tables")

    # Recreate sentiment_articles (url unique)
    await conn.execute("""
        CREATE TABLE sentiment_articles (
            id              SERIAL PRIMARY KEY,
            ticker          VARCHAR(10) NOT NULL,
            title           TEXT,
            description     TEXT,
            url             VARCHAR(500) NOT NULL UNIQUE,
            source          VARCHAR(255),
            published_at    TIMESTAMPTZ,
            sentiment_label VARCHAR(20),
            positive_score  FLOAT,
            negative_score  FLOAT,
            neutral_score   FLOAT,
            compound_score  FLOAT,
            ingested_at     TIMESTAMPTZ DEFAULT NOW(),
            created_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    await conn.execute("CREATE INDEX idx_sentiment_articles_ticker ON sentiment_articles(ticker)")
    print("✅ Created sentiment_articles")

    # Recreate sentiment_snapshots (ticker unique)
    await conn.execute("""
        CREATE TABLE sentiment_snapshots (
            id                 SERIAL PRIMARY KEY,
            ticker             VARCHAR(10) NOT NULL UNIQUE,
            avg_positive       FLOAT,
            avg_negative       FLOAT,
            avg_neutral        FLOAT,
            avg_compound       FLOAT,
            dominant_sentiment VARCHAR(20),
            article_count      INTEGER,
            computed_at        TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    await conn.execute("CREATE INDEX idx_sentiment_snapshots_ticker ON sentiment_snapshots(ticker)")
    print("✅ Created sentiment_snapshots")

    # Recreate daily_sentiment (ticker+date unique)
    await conn.execute("""
        CREATE TABLE daily_sentiment (
            id            SERIAL PRIMARY KEY,
            ticker        VARCHAR(10) NOT NULL,
            date          DATE NOT NULL,
            avg_positive  FLOAT DEFAULT 0,
            avg_negative  FLOAT DEFAULT 0,
            avg_neutral   FLOAT DEFAULT 1,
            avg_compound  FLOAT DEFAULT 0,
            article_count INTEGER DEFAULT 0,
            computed_at   TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (ticker, date)
        )
    """)
    await conn.execute("CREATE INDEX idx_daily_sentiment_ticker_date ON daily_sentiment(ticker, date)")
    print("✅ Created daily_sentiment")

    await conn.close()
    print("✅ All sentiment tables recreated with constraints")


if __name__ == "__main__":
    asyncio.run(main())