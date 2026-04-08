"""FastAPI server exposing PromptShield as an OpenEnv environment."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from promptshield.env import PromptShieldEnv
from promptshield.models import PromptShieldAction

app = FastAPI(title="PromptShield OpenEnv")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = PromptShieldEnv()


class ResetRequest(BaseModel):
    task_name: Optional[str] = "task_easy"


class StepRequest(BaseModel):
    classification: str
    confidence: float
    reasoning: str
    sanitized_prompt: Optional[str] = None


@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    global env
    try:
        task_name = body.task_name or "task_easy"
        env = PromptShieldEnv(task_name=task_name)
        obs = env.reset()
        return {
            "observation": obs.model_dump(),
            "task_name": task_name,
            "status": "ok",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(body: StepRequest):
    global env
    try:
        action = PromptShieldAction(
            classification=body.classification,
            confidence=body.confidence,
            reasoning=body.reasoning,
            sanitized_prompt=body.sanitized_prompt,
        )
        next_obs, reward_value, done, info = env.step(action)
        return {
            "observation": next_obs.model_dump(),
            "reward": reward_value,
            "done": done,
            "info": info,
            "state": env.state().model_dump(),
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    return env.state().model_dump()


@app.get("/health")
def health():
    return {"status": "healthy", "environment": "promptshield"}


@app.get("/")
def root():
    return {
        "name": "PromptShield",
        "description": "Prompt Injection Detector OpenEnv Environment",
        "tasks": ["task_easy", "task_medium", "task_hard"],
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
