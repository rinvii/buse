import functools
import json
import traceback
import base64
import io
from enum import Enum
from typing import Any, Optional

import toon_format as toon
import typer
from pydantic import BaseModel
from PIL import Image
from .models import ActionResult


class OutputFormat(str, Enum):
    json = "json"
    toon = "toon"


class GlobalState:
    format: OutputFormat = OutputFormat.json
    profile: bool = False


state = GlobalState()


def _serialize(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()

    if isinstance(obj, dict):
        if "bbox" in obj and isinstance(obj["bbox"], list):
            obj = dict(obj)
            obj["bbox"] = ",".join(map(str, obj["bbox"]))
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


def handle_errors(func=None, *, action: Optional[str] = None):
    if func is None:
        return lambda f: handle_errors(f, action=action)

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except typer.Exit:
            raise
        except Exception as e:
            error_details = {
                "type": e.__class__.__name__,
                "stage": "unhandled",
                "traceback": traceback.format_exc().splitlines()[-3:],
            }
            action_name = action or func.__name__
            payload = ActionResult(
                success=False,
                action=action_name,
                message=None,
                error=str(e),
                error_details=error_details,
                extracted_content=None,
                profile={} if state.profile else None,
            )
            output_data(payload)
            raise typer.Exit(code=1)

    return wrapper


def downscale_image(
    base64_str: str, max_width: int | None = None, quality: int = 75
) -> str:
    """Converts a base64 image to compressed JPEG, optionally resizing it."""
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    if max_width and img.width > max_width:
        ratio = max_width / float(img.width)
        new_height = int(float(img.height) * ratio)
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
