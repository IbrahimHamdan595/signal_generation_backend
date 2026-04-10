from fastapi import APIRouter, HTTPException, Depends
from app.core.security import get_current_active_user
from app.db.database import get_db
from app.services.job_service import JobService

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/{job_id}")
async def get_job(
    job_id: int,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = JobService(pool)
    job = await svc.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/latest/{job_type}")
async def get_latest_job(
    job_type: str,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = JobService(pool)
    job = await svc.get_latest(job_type)
    if not job:
        raise HTTPException(404, f"No jobs found of type '{job_type}'")
    return job
