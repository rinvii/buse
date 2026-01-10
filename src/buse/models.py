from typing import List, Optional, Any
from pydantic import BaseModel


class TabInfo(BaseModel):
    id: str
    title: str
    url: str


class Observation(BaseModel):
    session_id: str
    url: str
    title: str
    tabs: List[TabInfo]
    screenshot_path: Optional[str] = None
    dom_minified: str
    # We can add more fields like 'inputs' state if needed later


class ActionResult(BaseModel):
    success: bool
    action: str
    message: Optional[str] = None
    error: Optional[str] = None
    new_url: Optional[str] = None
    extracted_content: Optional[Any] = None
    profile: Optional[dict[str, float]] = None
