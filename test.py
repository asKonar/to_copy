import asyncio
from typing import List
from fastapi import FastAPI, Request
from pydantic import BaseModel, ValidationError, field_validator
import json
from llama_cpp import Llama
from sse_starlette import EventSourceResponse


app = FastAPI()

model_file_path = ""

# Load the AI model
llm = Llama(
    model_path=model_file_path,
    chat_format="llama-2",
    n_ctx=8192,
    stop=["</s>", "[/INST]"],
    verbose=False
    )


class jsonDataValidate(BaseModel):
    """
    Pydantic validation class for OpenAI API like messages.
    """
    messages: List[dict]

    @field_validator('messages')    # Pydantic decorator function
    def validate_items(cls, values):
        """
        Validate the 'messages' field in a Pydantic model.

        Args:
            cls: The Pydantic model class.
            values: The value of the 'messages' field to be validated.

        Raises:
            ValueError: If the 'messages' field is empty, or if any item in the list is not a dictionary, or if an item does not contain the 'role' or 'content' keys.
            ValueError: If an item's 'role' value is not one of the accepted values ('system', 'user', or 'assistant').

        Returns:
            The validated 'messages' field.
        """
        if not values:
            raise ValueError("'messages' cannot be empty.")
        for item in values:
            if not isinstance(item, dict):
                raise ValueError("Each item in 'messages' must be a dictionary.")
            if "role" not in item or "content" not in item:
                raise ValueError("Each 'messages' item must contain 'role' and 'content' keys.")
            accepted_roles = ["system", "user", "assistant"]
            if item["role"] not in accepted_roles:
                raise ValueError(f"""Invalid role value: '{item["role"]}'. Accepted values are {accepted_roles}.""")
        return values



@app.post("/llama/")
async def llama(request: Request, data:jsonDataValidate):
    stream = llm.create_chat_completion(
        messages=data.messages,
        max_tokens=128,
        stream=True
    )
    async def async_generator():
        for item in stream:
            yield item
    async def server_sent_events():
        # Main async loop
        async for item in async_generator():
            if await request.is_disconnected():
                break
            yield json.dumps(item)
    return EventSourceResponse(server_sent_events(), media_type="application/x-ndjson")