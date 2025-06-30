#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vLLM server for SmolLM model with OpenAI-compatible API endpoints.
Provides high-performance inference with continuous batching and PagedAttention.
"""

from typing import List, Optional, Dict, Any, Union
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from transformers import PreTrainedTokenizer

# Initialize FastAPI app
app = FastAPI(
    title="SmolLM vLLM API",
    description="High-performance API for SmolLM using vLLM backend",
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

# Global variables
llm: Optional[LLM] = None
tokenizer: Optional[PreTrainedTokenizer] = None

def init_vllm(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_num_batched_tokens: int = 4096,
    max_num_seqs: int = 256,
    quantization: Optional[str] = None,
):
    """Initialize vLLM engine with the specified parameters."""
    global llm

    # Configure vLLM engine arguments
    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        quantization=quantization,
        trust_remote_code=True,  # Required for custom models
    )
    
    # Initialize vLLM engine
    llm = LLM(engine_args=engine_args)
    print(f"vLLM initialized with model from {model_path}")
    print(f"Using tensor parallelism: {tensor_parallel_size}")
    if quantization:
        print(f"Quantization: {quantization}")

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
    """OpenAI-style chat completion endpoint using vLLM."""
    try:
        prompt = format_chat_prompt(request.messages)
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        
        # Generate response
        start_time = time.time()
        outputs = await llm.generate(prompt, sampling_params)
        end_time = time.time()
        
        # Get the generated text
        generated_text = outputs[0].outputs[0].text
        
        # Count tokens (using vLLM's internal tokenizer)
        prompt_tokens = len(llm.get_tokenizer().encode(prompt))
        completion_tokens = len(llm.get_tokenizer().encode(generated_text))
        
        return {
            "id": f"smollm-vllm-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "smollm-vllm",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text,
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "latency": end_time - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest) -> Dict[str, Any]:
    """OpenAI-style completion endpoint using vLLM."""
    try:
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        
        # Generate response
        start_time = time.time()
        outputs = await llm.generate(request.prompt, sampling_params)
        end_time = time.time()
        
        # Get the generated text
        generated_text = outputs[0].outputs[0].text
        
        # Count tokens
        prompt_tokens = len(llm.get_tokenizer().encode(request.prompt))
        completion_tokens = len(llm.get_tokenizer().encode(generated_text))
        
        return {
            "id": f"smollm-vllm-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "smollm-vllm",
            "choices": [{
                "text": generated_text,
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "latency": end_time - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "backend": "vllm"
    }

def main():
    import uvicorn
    
    # vLLM configuration
    model_path = "path/to/your/trained/model"
    tensor_parallel_size = 1  # Increase for multi-GPU
    gpu_memory_utilization = 0.9
    quantization = None  # Options: 'int8', 'int4' for lower memory usage
    
    # Initialize vLLM
    init_vllm(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        quantization=quantization,
    )
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main() 