import hashlib
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

import redis as redis_lib
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel

from src.agent.agent import build_agent
from src.agent.tools import calculate, get_current_datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)

_agent = None
_redis: redis_lib.Redis = None
CACHE_TTL = int(os.environ.get("CACHE_TTL", 3600))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _redis
    tools = [get_current_datetime, calculate]
    logger.info("Registered tools (%d):", len(tools))
    for t in tools:
        logger.info("  - %s: %s", t.name, t.description)
        logger.info("    schema: %s", t.args_schema.model_json_schema() if t.args_schema else "none")

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    logger.info("Connecting to Redis...")
    _redis = redis_lib.Redis.from_url(redis_url, decode_responses=True)
    logger.info("Redis ready.")

    conn_string = os.environ.get(
        "POSTGRES_URL",
        "postgresql://dbuser:ZXg3h339k2nXry@localhost:5432/succotash",
    )
    logger.info("Connecting to Postgres checkpointer...")
    async with AsyncPostgresSaver.from_conn_string(conn_string) as checkpointer:
        await checkpointer.setup()
        logger.info("Checkpointer ready.")
        logger.info("Building agent...")
        _agent = build_agent(tools=tools, checkpointer=checkpointer)
        logger.info("Agent ready.")
        yield


app = FastAPI(lifespan=lifespan)


def _cache_key(thread_id: str, query: str) -> str:
    digest = hashlib.sha256(f"{thread_id}:{query}".encode()).hexdigest()
    return f"query_cache:{digest}"


class QueryRequest(BaseModel):
    query: str
    thread_id: str = ""


class QueryResponse(BaseModel):
    response: str
    thread_id: str
    cached: bool = False



@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    logger.info("Received query on thread '%s': %s", thread_id, request.query)

    cache_key = _cache_key(thread_id, request.query)
    cached = _redis.get(cache_key)
    if cached:
        logger.info("Cache hit for thread '%s'", thread_id)
        return QueryResponse(response=cached, thread_id=thread_id, cached=True)

    config = {"configurable": {"thread_id": thread_id}}
    result = await _agent.ainvoke(
        {"messages": [HumanMessage(content=request.query)]},
        config=config,
    )
    response = result["messages"][-1].content

    _redis.setex(cache_key, CACHE_TTL, response)
    logger.info("Cached response for thread '%s' (ttl=%ds)", thread_id, CACHE_TTL)

    return QueryResponse(response=response, thread_id=thread_id, cached=False)
