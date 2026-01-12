import asyncio
import base64
import io
import json

import pytest
import typer
from PIL import Image

from buse import utils


def test_output_data_json(capsys):
    utils.state.format = utils.OutputFormat.json
    utils.output_data({"a": 1})
    out = capsys.readouterr().out.strip()
    assert json.loads(out) == {"a": 1}


def test_output_data_toon(monkeypatch, capsys):
    utils.state.format = utils.OutputFormat.toon
    monkeypatch.setattr(utils.toon, "encode", lambda data: "ENCODED")
    utils.output_data({"a": 1})
    out = capsys.readouterr().out.strip()
    assert out == "ENCODED"
    utils.state.format = utils.OutputFormat.json


def test_handle_errors_ok():
    @utils.handle_errors
    async def ok():
        return "ok"

    assert asyncio.run(ok()) == "ok"


def test_handle_errors_exception(capsys):
    utils.state.format = utils.OutputFormat.json

    @utils.handle_errors
    async def boom():
        raise RuntimeError("fail")

    with pytest.raises(typer.Exit):
        asyncio.run(boom())
    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert data["success"] is False
    assert data["action"] == "boom"
    assert data["error"] == "fail"
    assert data["error_details"]["stage"] == "unhandled"
    assert data["error_details"]["type"] == "RuntimeError"


def test_handle_errors_typer_exit():
    @utils.handle_errors
    async def boom():
        raise typer.Exit(code=2)

    with pytest.raises(typer.Exit):
        asyncio.run(boom())


def _png_base64(mode="RGBA", size=(10, 10)):
    color = (255, 0, 0, 128) if mode == "RGBA" else 128
    img = Image.new(mode, size, color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_output_data_flattens_bbox(capsys):
    utils.state.format = utils.OutputFormat.json
    utils.output_data({"bbox": [1, 2, 3, 4]})
    out = json.loads(capsys.readouterr().out)
    assert out["bbox"] == "1,2,3,4"


def test_downscale_image_resizes_and_converts():
    original = _png_base64()
    result = utils.downscale_image(original, max_width=5, quality=60)
    decoded = base64.b64decode(result)
    img = Image.open(io.BytesIO(decoded))
    assert img.mode == "RGB"
    assert img.width == 5
    assert img.height == 5
