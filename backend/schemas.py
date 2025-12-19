from pydantic import BaseModel

class CaptionResponse(BaseModel):
    filename: str
    caption: str
    strategy: str = "beam"