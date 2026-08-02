import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from app.core.dependencies import lifespan
from app.core.logging import RequestLoggingMiddleware
from app.search.errors import add_exception_handlers
from app.search.routers import router as search_router
from app.list.routers import router as list_router

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

PRODUCTION_DOMAIN = os.environ.get("PRODUCTION_DOMAIN")

app = FastAPI(lifespan=lifespan)

app.add_middleware(RequestLoggingMiddleware)

allow_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
if PRODUCTION_DOMAIN:
    allow_origins.append(PRODUCTION_DOMAIN)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(search_router)
app.include_router(list_router)
add_exception_handlers(app)

handler = Mangum(app, lifespan="on")
