#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from crewai_vision_tool.crew import CrewaiVisionTool

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# ============================================================================
# FIX FOR LITELLM INDEXERROR BUG 
# ============================================================================
# If you get "IndexError: list index out of range" when using Ollama with CrewAI:
#
# 1. Find the LiteLLM factory.py file:
#    find . -name "factory.py" -path "*/litellm/litellm_core_utils/prompt_templates/*"
#
# 2. Open that file and find line ~232-234 that looks like:
#    msg_i += 1
#    
#    tool_calls = messages[msg_i].get("tool_calls")
#
# 3. Change it to:
#    msg_i += 1
#    if msg_i == len(messages):
#        break
#    tool_calls = messages[msg_i].get("tool_calls")
#
# 4. Save the file - that's it! The IndexError is fixed.
#
# This is the community-approved fix from GitHub issue discussions.
# ============================================================================

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    image_path = input("Enter the location/path of the image here: ")

    inputs = {
        'image_path': image_path
    }
    
    try:
        result = CrewaiVisionTool().crew().kickoff(inputs=inputs)
        print("Crew execution completed successfully!")
        print("Result:", result)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        CrewaiVisionTool().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        CrewaiVisionTool().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        CrewaiVisionTool().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    run()