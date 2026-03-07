from pydantic import BaseModel


# TODO C1: flesh out full schemas reusing app/backend/src/models.py


class HealthResponse(BaseModel):
    status: str
