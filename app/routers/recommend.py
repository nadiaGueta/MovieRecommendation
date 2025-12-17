from fastapi import APIRouter, HTTPException
from app.schemas.recommend import RecommendRequest, RecommendResponse
from app.core.state import MODELS
from app.core.recommenders import hybrid_recommend_for_user

router = APIRouter(prefix="/recommend", tags=["recommendation"])

@router.post("/", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if not MODELS["ready"]:
        raise HTTPException(status_code=503, detail="Models not ready yet")

    recs = hybrid_recommend_for_user(
        req.user_id,
        MODELS["clf"],
        MODELS["enc"],
        MODELS["svd"],
        MODELS["ratings"],
        MODELS["movies"],
        MODELS["seen_dict"],
        topk=req.topk,
        alpha=req.alpha,
    )
    return {"user_id": req.user_id, "recommendations": recs}
