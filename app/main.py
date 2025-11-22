"""FastAPI entrypoint exposing the medical audit routes.

This module wires the router, provides a simple health endpoint, and
exposes a Mangum handler for AWS Lambda compatibility.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from app.settings import settings
from auditroutes import medicalrouter

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(medicalrouter, prefix="/medical")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# AWS Lambda handler
handler = Mangum(app)

