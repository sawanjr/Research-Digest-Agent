from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi import Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from research_agent.api.routes import router
from research_agent.config import configure_logging


configure_logging()

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="Semantic Research Digest Agent", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)
