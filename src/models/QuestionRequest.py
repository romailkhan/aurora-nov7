from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    """Request model for the /ask endpoint."""
    
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language question about member data",
        examples=["When is Layla planning her trip to London?"]
    )