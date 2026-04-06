"""Quick validation script to test all 3 tasks produce valid scores."""

from promptshield.env import PromptShieldEnv
from promptshield.models import PromptShieldAction

TASKS = ["task_easy", "task_medium", "task_hard"]
STEPS = 5

def run_task_test(task_name: str) -> float:
    env = PromptShieldEnv(task_name=task_name, max_steps=STEPS)
    env.reset()
    for _ in range(STEPS):
        action = PromptShieldAction(
            classification="injection",
            confidence=0.85,
            reasoning="This prompt contains ignore and override patterns typical of direct injection attacks",
            sanitized_prompt="Please help me with a legitimate question instead",
        )
        _, _, done, _ = env.step(action)
        if done:
            break
    state = env.state()
    score = state.score
    assert 0.0 <= score <= 1.0, f"Score {score} out of range for {task_name}"
    return score

results = {}
all_passed = True
for task in TASKS:
    try:
        score = run_task_test(task)
        results[task] = score
        print(f"[PASS] {task}: score={score:.3f}")
    except Exception as e:
        print(f"[FAIL] {task}: {e}")
        all_passed = False

print()
print("Summary:")
for task, score in results.items():
    print(f"  {task}: {score:.3f}")
print()
print("All grader scores in valid range:", all_passed)
