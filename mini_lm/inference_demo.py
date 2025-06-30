#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FastAPI server for SmolLM model with OpenAI-like API endpoints.
Supports both chat completions and text completions.
"""

import time
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import GenerationConfig
from smol_model import SmolLMModel
from train_tokenizer import QwenTokenizer

# Initialize FastAPI app
app = FastAPI(
    title="SmolLM API",
    description="API for SmolLM language model with OpenAI-compatible endpoints",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender (system/user/assistant)")
    content: str = Field(..., description="The content of the message")

class ChatCompletionRequest(BaseModel):
    messages: List[Message] = Field(..., description="The messages to generate a response for")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling probability")
    max_tokens: Optional[int] = Field(200, description="Maximum number of tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")

class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to generate completions for")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling probability")
    max_tokens: Optional[int] = Field(200, description="Maximum number of tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

def load_model_and_tokenizer(model_path: str, tokenizer_path: str):
    """Initialize model and tokenizer."""
    global model, tokenizer, device
    
    print("Loading model and tokenizer...")
    model = SmolLMModel.from_pretrained(model_path)
    tokenizer = QwenTokenizer.from_pretrained(tokenizer_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")

def generate_text(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 200,
) -> str:
    """Generate text using the model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    generation_config = GenerationConfig(
        max_length=max_tokens + input_ids.shape[1],  # Account for prompt tokens
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    # Remove the prompt from the generated text
    response_text = generated_text[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
    
    return response_text.strip()

def format_chat_prompt(messages: List[Message]) -> str:
    """Format chat messages into a single prompt string."""
    formatted_prompt = ""
    for msg in messages:
        if msg.role == "system":
            formatted_prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            formatted_prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            formatted_prompt += f"Assistant: {msg.content}\n"
    formatted_prompt += "Assistant:"
    return formatted_prompt

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest) -> Dict[str, Any]:
    """OpenAI-style chat completion endpoint."""
    try:
        prompt = format_chat_prompt(request.messages)
        
        start_time = time.time()
        response_text = generate_text(
            prompt=prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        end_time = time.time()
        
        return {
            "id": f"smollm-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "smollm",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(tokenizer.encode(prompt)),
                "completion_tokens": len(tokenizer.encode(response_text)),
                "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(response_text))
            },
            "latency": end_time - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest) -> Dict[str, Any]:
    """OpenAI-style completion endpoint."""
    try:
        start_time = time.time()
        response_text = generate_text(
            prompt=request.prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        end_time = time.time()
        
        return {
            "id": f"smollm-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "smollm",
            "choices": [{
                "text": response_text,
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(tokenizer.encode(request.prompt)),
                "completion_tokens": len(tokenizer.encode(response_text)),
                "total_tokens": len(tokenizer.encode(request.prompt)) + len(tokenizer.encode(response_text))
            },
            "latency": end_time - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

def main():
    # Configuration - update these paths
    model_path = "path/to/your/trained/model"
    tokenizer_path = "path/to/your/tokenizer"
    
    # Load model and tokenizer
    load_model_and_tokenizer(model_path, tokenizer_path)
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main() 