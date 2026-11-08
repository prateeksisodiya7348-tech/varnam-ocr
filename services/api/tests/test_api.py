"""API contract tests.

Everything here goes through the real HTTP stack against the fake engine, so
the tests cover routing, validation, persistence and serialization together -
the places bugs actually live.
"""

from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image


def _png(size: tuple[int, int] = (800, 400), bars: int = 3) -> bytes:
    """Render a synthetic page of solid text bars as PNG."""
    image = Image.new("L", size, color=255)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    for index in range(bars):
        top = 40 + index * 90
        draw.rectangle([60, top, size[0] - 60, top + 28], fill=0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_root_reports_service_identity(client: TestClient) -> None:
    """The root endpoint is a cheap way to confirm the deploy is live."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "Varnam OCR"


def test_liveness_and_readiness(client: TestClient) -> None:
    """Both probes answer, and readiness reflects a loaded engine."""
    assert client.get("/health/live").status_code == 200
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["model_loaded"] is True


def test_ocr_returns_structured_lines(client: TestClient) -> None:
    """A valid image yields text, boxes and a confidence for every line."""
    response = client.post(
        "/v1/ocr",
        files={"file": ("page.png", _png(), "image/png")},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == "succeeded"
    assert body["engine"] == "fake"
    assert body["text"]
    assert len(body["lines"]) == 3
    assert 0.0 <= body["mean_confidence"] <= 1.0
    assert body["devanagari_ratio"] > 0.5
    assert body["page"]["width"] == 800

    first = body["lines"][0]
    assert first["text"]
    # Boxes must be inside the original image bounds.
    assert 0 <= first["bbox"]["x0"] < first["bbox"]["x1"] <= 800
    assert 0 <= first["bbox"]["y0"] < first["bbox"]["y1"] <= 400


def test_ocr_sets_tracing_headers(client: TestClient) -> None:
    """Every response carries a request id and a timing header."""
    response = client.post("/v1/ocr", files={"file": ("page.png", _png(), "image/png")})
    assert response.headers["x-request-id"]
    assert float(response.headers["x-response-time-ms"]) >= 0


def test_ocr_echoes_client_request_id(client: TestClient) -> None:
    """A client-supplied correlation id is preserved end to end."""
    response = client.post(
        "/v1/ocr",
        files={"file": ("page.png", _png(), "image/png")},
        headers={"x-request-id": "abc123"},
    )
    assert response.headers["x-request-id"] == "abc123"


def test_ocr_rejects_unsupported_type(client: TestClient) -> None:
    """A text file returns a typed 415 envelope, never a traceback."""
    response = client.post("/v1/ocr", files={"file": ("notes.txt", b"just text", "text/plain")})
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media"


def test_ocr_rejects_oversized_upload(client: TestClient) -> None:
    """The byte cap is enforced server-side, not just in the UI."""
    big = b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024)
    response = client.post("/v1/ocr", files={"file": ("huge.jpg", big, "image/jpeg")})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_ocr_text_export(client: TestClient) -> None:
    """`?export=txt` returns plain text instead of JSON."""
    response = client.post(
        "/v1/ocr?export=txt",
        files={"file": ("page.png", _png(), "image/png")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_ocr_hocr_export(client: TestClient) -> None:
    """`?export=hocr` returns valid-looking hOCR with confidence metadata."""
    response = client.post(
        "/v1/ocr?export=hocr",
        files={"file": ("page.png", _png(), "image/png")},
    )
    assert response.status_code == 200
    body = response.text
    assert 'class="ocr_page"' in body
    assert "x_wconf" in body


def test_feedback_round_trip(client: TestClient) -> None:
    """A correction is stored and acknowledged with its sample id."""
    ocr = client.post("/v1/ocr", files={"file": ("page.png", _png(), "image/png")}).json()
    response = client.post(
        "/v1/feedback",
        json={
            "job_id": ocr["job_id"],
            "line_index": 0,
            "corrected_text": "  संशोधित   पंक्ति  ",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["stored"] is True
    assert body["sample_id"] > 0


def test_feedback_rejects_unknown_job(client: TestClient) -> None:
    """Unknown jobs return 404 rather than creating orphan samples."""
    response = client.post(
        "/v1/feedback",
        json={"job_id": "does-not-exist", "line_index": 0, "corrected_text": "क"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_feedback_rejects_out_of_range_line(client: TestClient) -> None:
    """An out-of-range line index is rejected so the training set stays aligned."""
    ocr = client.post("/v1/ocr", files={"file": ("page.png", _png(), "image/png")}).json()
    response = client.post(
        "/v1/feedback",
        json={"job_id": ocr["job_id"], "line_index": 999, "corrected_text": "क"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_upload"


def test_job_lookup_returns_not_found(client: TestClient) -> None:
    """Polling an unknown id is a clean 404."""
    response = client.get("/v1/jobs/missing-id")
    assert response.status_code == 404


def test_metrics_exposes_histogram(client: TestClient) -> None:
    """The Prometheus endpoint emits the latency histogram after traffic."""
    client.post("/v1/ocr", files={"file": ("page.png", _png(), "image/png")})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "ocr_latency_ms_bucket" in response.text
    assert "ocr_requests_total" in response.text


def test_openapi_schema_is_valid(client: TestClient) -> None:
    """The OpenAPI document must generate, because the web client is built from it.

    The schema lives under /api so it cannot collide with the frontend's own
    /docs page when the built SPA is mounted at "/".
    """
    schema = client.get("/api/openapi.json").json()
    assert "/v1/ocr" in schema["paths"]
    assert "/v1/ocr/pdf" in schema["paths"]
