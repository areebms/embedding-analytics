import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from mangum import Mangum
from redis import asyncio as aioredis

from routers import router

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
        await redis.close()
    else:
        yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()

    # Log similarity queries before the cache layer intercepts them,
    # so we see every request regardless of cache hit/miss.
    if request.method == "POST" and request.url.path.startswith("/similarity/"):
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes)
            book_id = request.url.path.removeprefix("/similarity/")
            logger.info("similarity book_id=%s query=%r", book_id, body.get("query"))
        except Exception:
            pass

        # Reconstruct the request so the endpoint can still read the body.
        async def receive():
            return {"type": "http.request", "body": body_bytes}

        request = Request(request.scope, receive)

    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s status=%d duration_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", PRODUCTION_DOMAIN],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(router)


handler = Mangum(app, lifespan="on")
