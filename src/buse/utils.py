import functools
import json
import traceback
from enum import Enum
from typing import Any

import toon_format as toon
import typer
from pydantic import BaseModel


class OutputFormat(str, Enum):
	json = "json"
	toon = "toon"


class GlobalState:
	format: OutputFormat = OutputFormat.json
	profile: bool = False


state = GlobalState()


def _serialize(obj: Any) -> Any:
	if isinstance(obj, BaseModel):
		return obj.model_dump()
	if isinstance(obj, dict):
		return {k: _serialize(v) for k, v in obj.items()}
	if isinstance(obj, list):
		return [_serialize(i) for i in obj]
	return obj


def output_data(data: Any):
	"""Helper to output data in the selected format."""
	dumped = _serialize(data)

	if state.format == OutputFormat.toon:
		print(toon.encode(dumped))
	else:
		print(json.dumps(dumped, indent=2))


def handle_errors(func):
	@functools.wraps(func)
	async def wrapper(*args, **kwargs):
		try:
			return await func(*args, **kwargs)
		except typer.Exit:
			raise
		except Exception as e:
			error_data = {"error": str(e), "type": e.__class__.__name__, "status": 500, "traceback": traceback.format_exc().splitlines()[-3:]}
			print(json.dumps(error_data, indent=2))
			raise typer.Exit(code=1)

	return wrapper
