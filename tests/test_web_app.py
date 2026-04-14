"""
Tests for web_app.py Flask application.

The model file may not exist in the test environment, so we patch the
module-level model loading to avoid a SystemExit during import.
"""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app():
    """
    Import web_app with the model loading patched out.

    Patches applied before import so the module-level code never tries
    to read output/best_model.pt from disk.
    """
    import importlib
    import sys

    # Remove any cached module so we get a clean import each time.
    sys.modules.pop("web_app", None)

    mock_model = MagicMock()
    mock_checkpoint = MagicMock()
    mock_checkpoint.classes = ["Healthy", "Rust", "Blight"]

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch(
            "plant_diseases.model.PlantDiseaseModel.from_checkpoint",
            return_value=(mock_model, mock_checkpoint),
        ),
        patch("plant_diseases.device.select_device", return_value="cpu"),
        patch("plant_diseases.transforms.build_val_transforms", return_value=MagicMock()),
    ):
        import web_app  # noqa: PLC0415

    return web_app.app, mock_model, mock_checkpoint


# ---------------------------------------------------------------------------
# Fixtures / setup
# ---------------------------------------------------------------------------

import pytest


@pytest.fixture()
def client():
    """Return a Flask test client with the model mocked out."""
    app, mock_model, mock_checkpoint = _make_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, mock_model, mock_checkpoint


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_index_route_is_registered(client):
    """GET / route should be registered on the Flask app."""
    test_client, _, _ = client
    app = test_client.application
    # Verify the route exists by checking the URL map.
    rules = [rule.rule for rule in app.url_map.iter_rules()]
    assert "/" in rules


def test_get_index_returns_200_when_template_exists(client):
    """GET / should return HTTP 200 when the template is present."""
    from unittest.mock import patch as _patch

    test_client, _, _ = client

    # Mock render_template so the test doesn't require the actual template file.
    with _patch("web_app.render_template", return_value="<html>ok</html>") as mock_render:
        response = test_client.get("/")

    assert response.status_code == 200
    mock_render.assert_called_once_with("index.html")


def test_classify_without_file_returns_400(client):
    """POST /classify with no file should return 400."""
    test_client, _, _ = client
    response = test_client.post("/classify")
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data


def test_classify_with_empty_filename_returns_400(client):
    """POST /classify with an empty filename should return 400."""
    test_client, _, _ = client
    # Simulate sending a file field with no file selected (empty filename).
    response = test_client.post(
        "/classify",
        data={"image": (io.BytesIO(b""), "")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data


def test_classify_with_non_image_returns_400(client):
    """POST /classify with a non-image file should return 400."""
    test_client, _, _ = client
    response = test_client.post(
        "/classify",
        data={"image": (io.BytesIO(b"this is not an image"), "test.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data


def test_classify_with_valid_image_returns_label_and_confidence(client):
    """POST /classify with a valid image should return label and confidence."""
    import sys
    sys.modules.pop("web_app", None)

    import torch

    mock_model = MagicMock()
    mock_checkpoint = MagicMock()
    mock_checkpoint.classes = ["Healthy", "Rust", "Blight"]

    # Simulate model returning logits that softmax to a clear winner.
    logits = torch.tensor([[10.0, 1.0, 1.0]])
    mock_model.return_value = logits

    mock_preprocess = MagicMock()
    mock_preprocess.return_value = torch.zeros(3, 224, 224)

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch(
            "plant_diseases.model.PlantDiseaseModel.from_checkpoint",
            return_value=(mock_model, mock_checkpoint),
        ),
        patch("plant_diseases.device.select_device", return_value=torch.device("cpu")),
        patch(
            "plant_diseases.transforms.build_val_transforms",
            return_value=mock_preprocess,
        ),
    ):
        import web_app  # noqa: PLC0415

    web_app.app.config["TESTING"] = True

    # Create a minimal 1x1 red PNG in memory.
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.new("RGB", (4, 4), color=(255, 0, 0)).save(buf, format="PNG")
    buf.seek(0)

    with web_app.app.test_client() as c:
        response = c.post(
            "/classify",
            data={"image": (buf, "leaf.png")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    data = json.loads(response.data)
    assert "label" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
