from pydantic import BaseModel, Field
from typing import List, Dict, Any

class RecommendRequest(BaseModel):
    user_id: int
    topk: int = Field(10, ge=1, le=50)
    alpha: float = Field(0.5, ge=0.0, le=1.0)

class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, Any]]
