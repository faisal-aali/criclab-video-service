from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import get_settings

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        # tz_aware: BSON stores UTC milliseconds, and without this the driver
        # hands back *naive* datetimes. Those serialise without an offset, so a
        # browser parses them as local time — a timestamp created seconds ago
        # renders as hours old, and any server-side comparison against an aware
        # "now" is wrong by the machine's UTC offset.
        _client = AsyncIOMotorClient(
            get_settings().mongodb_uri,
            tz_aware=True,
            serverSelectionTimeoutMS=8000,
        )
    return _client


def get_db() -> AsyncIOMotorDatabase:
    return get_client()[get_settings().mongodb_db]


async def ping_mongo() -> bool:
    try:
        await get_client().admin.command("ping")
        return True
    except Exception:
        return False


async def close_mongo() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
