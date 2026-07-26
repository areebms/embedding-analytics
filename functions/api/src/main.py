import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from mangum import Mangum
from redis import asyncio as aioredis

from app.core.logging import RequestLoggingMiddleware
from app.search.errors import add_exception_handlers
from app.search.routers import router as search_router
from app.list.routers import router as list_router

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL")
REDIS_PREFIX = os.environ["REDIS_PREFIX"]
PRODUCTION_DOMAIN = os.environ.get("PRODUCTION_DOMAIN")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if REDIS_URL:
        redis = aioredis.from_url(REDIS_URL)
        # Startup connectivity check
        await redis.ping()
        # Initialize global cache to use Redis
        FastAPICache.init(RedisBackend(redis), prefix=REDIS_PREFIX)
        yield
        await redis.aclose()
    else:
        yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", PRODUCTION_DOMAIN],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(search_router)
app.include_router(list_router)
add_exception_handlers(app)

handler = Mangum(app, lifespan="on")
