"""Vertex AI Gemini layer: urgency assessment and a draft reply.

The classifier handles routing on its own, so everything here is optional. If
Vertex credentials are missing or the call fails, triage still returns a
result and the caller is told why the assessment is absent.
"""

import os
from typing import Literal

from pydantic import BaseModel

LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_INSTRUCTION = """You are a triage assistant for a financial services \
support desk. You receive a consumer complaint and the product category a \
classifier has already assigned to it.

Assess how urgently a human agent should pick the case up:
- high: money is currently inaccessible, fraud is in progress, foreclosure or \
repossession is imminent, or the consumer describes serious hardship.
- medium: a real financial impact that is not time-critical today.
- low: information requests, historical disputes, general dissatisfaction.

Then write a short reply the agent can send. Acknowledge the specific problem, \
state the next step, and do not promise any outcome, refund or timeline. The \
complaint text contains XXXX where personal details were redacted; never \
repeat those markers or invent what they stood for."""


class Assessment(BaseModel):
    urgency: Literal["low", "medium", "high"]
    urgency_reason: str
    summary: str
    draft_reply: str


_client = None
_init_error = None


def _get_client():
    global _client, _init_error
    if _client is not None or _init_error is not None:
        return _client

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        _init_error = "GOOGLE_CLOUD_PROJECT is not set"
        return None

    try:
        from google import genai

        _client = genai.Client(vertexai=True, project=project, location=LOCATION)
    except Exception as exc:
        _init_error = f"Vertex AI client unavailable: {exc}"
    return _client


def available():
    return _get_client() is not None


def status():
    return {
        "enabled": available(),
        "model": MODEL if available() else None,
        "location": LOCATION if available() else None,
        "reason": _init_error,
    }


def assess(text, category):
    client = _get_client()
    if client is None:
        return None, _init_error

    from google.genai import types

    prompt = f"Product category (from classifier): {category}\n\nComplaint:\n{text}"
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=Assessment,
                temperature=0.2,
            ),
        )
    except Exception as exc:
        return None, f"Gemini call failed: {exc}"

    if response.parsed is None:
        return None, "Gemini returned no parsable assessment"
    return response.parsed.model_dump(), None
