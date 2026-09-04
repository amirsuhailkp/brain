import logging
logging.basicConfig(level=logging.INFO)

from brain import Brain, Goal, JsonlMemory
from tests.mock_llm import BisectingGuesser
from toy_envs.guess_number import GuessNumberAdapter

project = GuessNumberAdapter(low=1, high=100, seed=42)
brain_mem = JsonlMemory("brain_memory.jsonl")
proj_mem = JsonlMemory("guess_number.jsonl")
brain = Brain(BisectingGuesser(), project, brain_mem, proj_mem, max_steps=12)

state = brain.run(Goal(description="Find the hidden number"))
print("stop_reason:", state.stop_reason)
print("steps taken:", state.step)
print("quality report:", state.quality_report)