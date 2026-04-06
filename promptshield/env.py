"""
Core OpenEnv environment class for PromptShield.
Implements reset(), step(), and state() per the OpenEnv interface.
"""

import random

from promptshield.models import (
    PromptShieldObservation,
    PromptShieldAction,
    PromptShieldState,
)
from promptshield.grader import PromptShieldGrader, calculate_episode_score
from promptshield.dataset import load_dataset


class PromptShieldEnv:
    def __init__(self, task_name: str = "task_easy", max_steps: int = 10):
        self.task_name = task_name
        self.max_steps = max_steps
        self._dataset = load_dataset(task_name)
        self._grader = PromptShieldGrader()

        self.current_step = 0
        self.done = False
        self.rewards = []
        self.history = []
        self.current_prompt_index = 0
        self._shuffled_dataset = list(self._dataset)

    def reset(self) -> PromptShieldObservation:
        self.current_step = 0
        self.done = False
        self.rewards = []
        self.history = []

        rng = random.Random(42)
        self._shuffled_dataset = list(self._dataset)
        rng.shuffle(self._shuffled_dataset)
        self.current_prompt_index = 0

        current = self._shuffled_dataset[0]
        return PromptShieldObservation(
            prompt_text=current["prompt_text"],
            context=current["context"],
            step_number=1,
            history=[],
            task_name=self.task_name,
            total_steps=self.max_steps,
        )

    def step(self, action: PromptShieldAction):
        if self.done:
            raise RuntimeError("Episode is already done. Call reset() to start a new episode.")

        current = self._shuffled_dataset[self.current_prompt_index % len(self._shuffled_dataset)]
        reward = self._grader.grade(action, current, self.task_name)

        self.rewards.append(reward.value)
        self.current_step += 1

        self.history.append({
            "step": self.current_step,
            "classification": action.classification,
            "reward": reward.value,
        })

        if self.current_step >= self.max_steps:
            self.done = True

        self.current_prompt_index += 1
        next_index = self.current_prompt_index % len(self._shuffled_dataset)
        next_prompt = self._shuffled_dataset[next_index]

        next_obs = PromptShieldObservation(
            prompt_text=next_prompt["prompt_text"],
            context=next_prompt["context"],
            step_number=self.current_step + 1,
            history=list(self.history),
            task_name=self.task_name,
            total_steps=self.max_steps,
        )

        info = {
            "reward_breakdown": reward.breakdown,
            "is_correct": reward.is_correct,
        }

        return next_obs, reward.value, self.done, info

    def state(self) -> PromptShieldState:
        score = calculate_episode_score(self.rewards, self.max_steps)
        return PromptShieldState(
            task_name=self.task_name,
            current_step=self.current_step,
            total_steps=self.max_steps,
            done=self.done,
            cumulative_reward=sum(self.rewards),
            score=score,
        )
