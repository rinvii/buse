from typing import List, Optional, Any
from pydantic import BaseModel


class TabInfo(BaseModel):
    id: str
    title: str
    url: str


class ViewportInfo(BaseModel):
    width: int
    height: int
    device_pixel_ratio: float


class VisualElement(BaseModel):
    index: int
    type: str
    content: str
    interactivity: bool
    center_x: float
    center_y: float
    bbox: list[float]


class VisualAnalysis(BaseModel):
    elements: list[VisualElement]
    som_image_path: Optional[str] = None


class Observation(BaseModel):
    session_id: str
    url: str
    title: str
    observed_at: Optional[float] = None
    visual_analysis: Optional[VisualAnalysis] = None
    tabs: List[TabInfo]
    viewport: Optional[ViewportInfo] = None
    screenshot_path: Optional[str] = None
    dom_minified: str
    semantic_snapshot: Optional[str] = None
    semantic_truncated: Optional[bool] = None
    diagnostics: Optional[dict[str, Any]] = None
    som_labels: Optional[int] = None
    som_labels_skipped: Optional[int] = None


class ActionResult(BaseModel):
    success: bool
    action: str
    message: Optional[str] = None
    error: Optional[str] = None
    error_details: Optional[dict[str, Any]] = None
    new_url: Optional[str] = None
    extracted_content: Optional[Any] = None
    profile: Optional[dict[str, float]] = None
