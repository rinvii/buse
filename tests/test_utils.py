import asyncio
import json

import pytest
import typer

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


def test_handle_errors_ok():
    @utils.handle_errors
    async def ok():
        return "ok"

    assert asyncio.run(ok()) == "ok"


def test_handle_errors_exception(capsys):
    @utils.handle_errors
    async def boom():
        raise RuntimeError("fail")

    with pytest.raises(typer.Exit):
        asyncio.run(boom())
    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert data["error"] == "fail"


def test_handle_errors_typer_exit():
    @utils.handle_errors
    async def boom():
        raise typer.Exit(code=2)

    with pytest.raises(typer.Exit):
        asyncio.run(boom())
