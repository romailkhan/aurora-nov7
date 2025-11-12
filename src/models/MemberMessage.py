from pydantic import BaseModel

class MemberMessage(BaseModel):
    """Model for a member message from the Aurora API."""
    
    id: str
    user_id: str
    user_name: str
    timestamp: str
    message: str
    
    def to_document_text(self) -> str:
        """Convert message to text format for embedding."""
        return f"User: {self.user_name}\nDate: {self.timestamp}\nMessage: {self.message}"
    
    def to_metadata(self) -> dict:
        """Convert message to metadata for ChromaDB."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "timestamp": self.timestamp,
            "message": self.message
        }