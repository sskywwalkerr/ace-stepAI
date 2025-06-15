from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

model_name = "ACE-Step/ACE-Step-v1-3.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


class InferenceRequest(BaseModel):
    prompt: str
    max_length: int = 50


@app.post("/generate")
async def generate_music(request: InferenceRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=request.max_length,
                num_return_sequences=1
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_music": generated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))