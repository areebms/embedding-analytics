import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache as fastapi_cache_decorator
from redis import asyncio as aioredis

REDIS_URL = os.environ.get("REDIS_URL")


def cache(**kwargs) -> Callable[[Any], Any]:
    if REDIS_URL:
        return fastapi_cache_decorator(**kwargs)
    return lambda f: f


@asynccontextmanager
async def lifespan(app: FastAPI):
    if REDIS_URL:
        redis: aioredis.Redis = aioredis.from_url(REDIS_URL)
        await cast(Awaitable[Any], redis.ping())
        FastAPICache.init(RedisBackend(redis), prefix=os.environ["REDIS_PREFIX"])
        try:
            yield
        finally:
            await redis.aclose()
    else:
        yield
