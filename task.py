
"""
HackerRank Task Definition

This file contains:
- PROMPT: The challenge prompt given to the agent
- BROKEN_PIPELINE: Broken pytorch pipeline which our RL will fix it 
- EXPECTED_ANSWER: Accuracy must be more than 0.85 we can have another metric too
- TOOLS: The tool definitions available to the agent
- TOOL_HANDLERS: The tool handler functions
- grading_func: Function that validates the agent's answer
- save_pipeline: this function will save successful pipeline which has more then 0.85 accuracy
- get_accuracy_from_code: this function will get run code locally for best code then provide us accuracy
"""

import os
import re
from logger import logger

from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, TypedDict

from anthropic.types import ToolUnionParam


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to emit output; stdout will be captured and returned.
    """
    try:
        namespace = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}


# Tool definitions for Anthropic API
TOOLS: list[ToolUnionParam] = [
    {
        "name": "python_expression",
        "description": "Evaluates a Python expression",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Will be passed to exec(). Use print() to output something. Returns stdout. ",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "submit_answer",
        "description": "Submit the final answer",
        "input_schema": {
            "type": "object",
            "properties": {"answer": {"description": "The final answer to submit"}},
            "required": ["answer"],
        },
    },
]

# Tool handlers mapping
TOOL_HANDLERS: dict[str, Callable[..., Any]] = {
    "python_expression": python_expression_tool,
    "submit_answer": submit_answer_tool,
}

#Broken pytorch pipeline
BROKEN_PIPELINE = """
                import torch
                import torch.nn as nn
                from torch.utils.data import DataLoader, TensorDataset
                import pandas as pd

                # Load dataset
                df = pd.read_csv("heart_disease.csv")
                X = df.drop("target", axis=1)
                y = df["target"]

                # ❌ PROBLEM 1: Missing normalization
                # ❌ PROBLEM 2: Incorrect loss (should be BCEWithLogitsLoss)
                # ❌ PROBLEM 3: No train-test split
                # ❌ PROBLEM 4: Missing activation

                dataset = TensorDataset(torch.tensor(X.values, dtype=torch.float32),
                                        torch.tensor(y.values, dtype=torch.float32))

                loader = DataLoader(dataset, batch_size=16, shuffle=True)

                class Model(nn.Module):
                    def __init__(self):
                        super().__init__()
                        # Input features size needs to be determined correctly
                        self.fc = nn.Linear(X.shape[1], 1)
                    def forward(self, x):
                        return self.fc(x)

                model = Model()
                criterion = nn.CrossEntropyLoss()    # WRONG for binary task
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                for epoch in range(5):
                    for xb, yb in loader:
                        optimizer.zero_grad()
                        preds = model(xb)
                        loss = criterion(preds, yb)
                        loss.backward()
                        optimizer.step()

                print("Training complete!")
                """


#expected answer for model result
EXPECTED_ANSWER = 0.85


# The challenge prompt
PROMPT = """You are an expert ML engineer working on a binary classification task. You are given a broken PyTorch training pipeline and a real dataset file named 'heart_disease.csv'.

        Your task is to generate code so it trains correctly and **achieves a test accuracy of 0.50 (50%) or higher
        if you reach accuracy of 0.50 (50%) use `submit_answer` tool**.

        **INSTRUCTIONS:**
        - You must use the 'heart_disease.csv' dataset. do not perform EDA just load data
        - handle missing values, numeric with median and categorical with "missing"
        - **normalization (StandardScaler or similar)**.
        - A **train-test split** (e.g., 80/20).
        - You have to generate a pyTorch model, make sure you add 3 hidden layers and in between add Relu activation
        - The correct loss function for binary classification, which is **BCEWithLogitsLoss**.
        - **Accuracy calculation** on the test set.
        - Your corrected code must print the final test accuracy using this *exact* format: `Final Accuracy: <float>` (e.g., Final Accuracy: 0.856).
        - I need a full pipeline in one single file, which can load data, explore data, handle missing values,
        then do **normalization, then create pyTorch model.

        **FINAL STEP:** Once your code is complete and correct, you MUST submit the *entire* corrected Python code block (including all imports and logic) as the argument to the `submit_answer` tool. Do NOT wrap the code in ```python fences in your submitted answer.

        Broken code:
        {BROKEN_PIPELINE}"""


# Grading function - validates the agent's submitted answer
def grading_func(result: Any) -> bool:
    """
    Validates the agent's answer.

    Args:
        result: The value returned from run_agent_loop (typically the submitted answer)

    Returns:
        True if the answer is correct, False otherwise
    """
    expected_answer = 0.85
    return result >= expected_answer


#To save ML model pipeline to local with accuracy
def save_run_pipeline(code_string: str, run_id: int, accuracy: float, is_new_best: bool = False):
    """
    Save the provided pipeline code to a uniquely named file and log the result

    :param code_string: The pipeline code to be saved to file
    :type code_string: str
    :param run_id: Unique identifier for the pipeline run
    :type run_id: int
    :param accuracy: Accuracy score associated with this run
    :type accuracy: float
    :param is_new_best: Whether this run achieved the best accuracy so far
    :type is_new_best: bool
    """
    directory = "pytorch_pipeline_code_with_accuracy"

    # Joins the directory and the formatted filename string
    full_path = os.path.join(
        directory,
        f"pipeline_run_{run_id}_acc_{accuracy:.3f}.py"
    )

    try:
        # Simple file writing
        with open(full_path, 'w') as f:
            f.write(code_string)
        
        status_tag = "NEW BEST" if is_new_best else f"RUN {run_id}"
        logger.info(f"\n--- {status_tag} PIPELINE SAVED: {full_path} (Accuracy: {accuracy:.3f}) ---")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")


#this function will run code for ML model and give accuracy to check 
def get_accuracy_from_code(code_string: str) -> tuple[float, str | None]:
    """
    Re-executes the submitted code string to capture the Final Accuracy output.
    
    Returns: A tuple of (accuracy: float, error_message: str | None).
    error_message will be set if an execution error occurs or if the required 
    output format is missing.
    """
    if not code_string:
        return 0.0, "Code string was empty."
        
    try:
        # Use the same execution logic as python_expression_tool
        namespace = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            # We must use 'exec' to run the submitted pipeline code
            exec(code_string, namespace, namespace)
        
        output_string = stdout.getvalue()
        
        # Regex to find "Final Accuracy: X.XXX..."
        match = re.search(r"Final Accuracy:\s*(\d+\.\d+)", output_string)
        
        if match:
            accuracy_raw = float(match.group(1))
            # SUCCESS: Found the expected output
            return round(accuracy_raw, 3), None 
        
        # FAILURE REASON 3: Code ran but did not print the required output
        return 0.0, "Missing Output Format: Code executed successfully but did not print 'Final Accuracy: X.XXX'."
        
    except Exception as e:
        # FAILURE REASON 2: Code execution crashed (SyntaxError, ImportError, etc.)
        # Truncate the error message to keep logs readable
        error_detail = str(e).replace('\n', ' / ')
        logger.error(f"Code Execution Error: {type(e).__name__}: {error_detail[:150]}...")
        return 0.0, f"Code Execution Error: {type(e).__name__}: {error_detail[:150]}..."
