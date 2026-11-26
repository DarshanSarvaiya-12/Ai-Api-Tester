"""Ai Api Tester - small forwarding proxy.

Passes requests to OpenAI-compatible providers that block browser calls (CORS).
Rules: allowed hosts only, allowed paths only, no logging of keys or messages,
size/time/rate limits, HTTPS only.
"""
import os
import time
from collections import defaultdict, deque
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Edit these in Render > Environment (comma separated)
ALLOWED_HOSTS = {
    h.strip().lower()
    for h in os.getenv("ALLOWED_HOSTS", "models.jarvislabs.net").split(",")
    if h.strip()
}
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))

ALLOWED_PATH_ENDINGS = ("/chat/completions", "/models")
MAX_BODY_BYTES = 1_000_000  # 1 MB

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["authorization", "content-type", "x-target-url"],
    max_age=600,
)

_hits = defaultdict(deque)


def _too_many(ip: str) -> bool:
    now = time.monotonic()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        return True
    q.append(now)
    if len(_hits) > 5000:  # keep memory small
        _hits.clear()
    return False


def _err(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message}}, status_code=status)


@app.get("/")
async def health():
    return {"ok": True, "name": "ai-api-tester-proxy"}


@app.api_route("/forward", methods=["GET", "POST"])
async def forward(request: Request):
    ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )
    if _too_many(ip):
        return _err(429, "Too many requests. Wait one minute and try again.")

    target = request.headers.get("x-target-url", "")
    try:
        u = urlparse(target)
        host = (u.hostname or "").lower()
        port = u.port
    except ValueError:
        return _err(400, "Bad target URL.")

    if u.scheme != "https" or u.username or u.password or u.query or u.fragment:
        return _err(400, "Target must be a plain https URL.")
    if port not in (None, 443):
        return _err(400, "Target port is not allowed.")
    if host not in ALLOWED_HOSTS:
        return _err(403, "This provider host is not allowed on the proxy. Add it to ALLOWED_HOSTS.")
    if not u.path.rstrip("/").endswith(ALLOWED_PATH_ENDINGS):
        return _err(403, "This path is not allowed on the proxy.")

    body = None
    if request.method == "POST":
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            return _err(413, "Message is too large.")
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return _err(413, "Message is too large.")

    headers = {}
    for name in ("authorization", "content-type", "accept"):
        if name in request.headers:
            headers[name] = request.headers[name]

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=120, write=30, pool=10),
        follow_redirects=False,
    )
    try:
        req = client.build_request(request.method, target, headers=headers, content=body)
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError:
        await client.aclose()
        return _err(502, "Proxy could not reach the provider.")

    async def stream():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        except httpx.HTTPError:
            return
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        stream(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )