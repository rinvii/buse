import base64
import os
import math
import httpx
from typing import Optional
from .models import VisualElement, VisualAnalysis, ViewportInfo


class VisionClient:
    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url or os.getenv("BUSE_OMNIPARSER_URL") or ""
        if self.server_url:
            self.server_url = f"{self.server_url.rstrip('/')}/parse/"

    async def analyze(
        self, base64_image: str, viewport: ViewportInfo
    ) -> tuple[VisualAnalysis, str]:
        if not self.server_url:
            raise RuntimeError(
                "OmniParser server URL is not set. Set BUSE_OMNIPARSER_URL."
            )
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {"base64_image": base64_image}
            try:
                response = await client.post(self.server_url, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                error_msg = f"OmniParser server returned {e.response.status_code}: {e.response.text}"
                raise RuntimeError(error_msg) from e
            except Exception as e:
                raise RuntimeError(f"Failed to connect to OmniParser: {e}") from e

            try:
                data = response.json()
            except ValueError as e:
                snippet = response.text[:200] if isinstance(response.text, str) else ""
                raise RuntimeError(
                    f"OmniParser returned invalid JSON: {e}. Response: {snippet}"
                ) from e
            if not isinstance(data, dict):
                raise RuntimeError("OmniParser response must be a JSON object.")

            parsed_content = data.get("parsed_content_list", [])
            if not isinstance(parsed_content, list):
                raise RuntimeError("OmniParser response missing parsed_content_list.")
            som_image_base64 = data.get("som_image_base64", "")
            if not isinstance(som_image_base64, str):
                som_image_base64 = ""
            elements = []

            width = viewport.width
            height = viewport.height
            scale = 100.0

            def _truncate(value: float) -> float:
                return math.trunc(value * scale) / scale

            for i, item in enumerate(parsed_content):
                if not isinstance(item, dict):
                    continue
                bbox = item.get("bbox")
                if (
                    not isinstance(bbox, (list, tuple))
                    or len(bbox) != 4
                    or not all(isinstance(v, (int, float)) for v in bbox)
                ):
                    continue

                bbox_px = [
                    bbox[0] * width,
                    bbox[1] * height,
                    bbox[2] * width,
                    bbox[3] * height,
                ]
                css_center_x = (bbox_px[0] + bbox_px[2]) / 2
                css_center_y = (bbox_px[1] + bbox_px[3]) / 2

                bbox_px = [_truncate(value) for value in bbox_px]
                css_center_x = _truncate(css_center_x)
                css_center_y = _truncate(css_center_y)

                el_type = item.get("type", "unknown")
                el_content = item.get("content", "")
                el_interactivity = item.get("interactivity", False)

                elements.append(
                    VisualElement(
                        index=i,
                        type=el_type,
                        content=el_content,
                        interactivity=el_interactivity,
                        center_x=css_center_x,
                        center_y=css_center_y,
                        bbox=bbox_px,
                    )
                )

            return VisualAnalysis(elements=elements), som_image_base64

    @staticmethod
    def save_som_image(base64_data: str, output_path: str):
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(base64_data))
