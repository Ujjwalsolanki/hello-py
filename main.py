import asyncio
import json
from collections.abc import Callable
import os
from typing import Any, Awaitable
from logger import logger

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

from dotenv import load_dotenv
load_dotenv()

from task import EXPECTED_ANSWER, PROMPT, get_accuracy_from_code, grading_func, python_expression_tool, save_run_pipeline, submit_answer_tool

MAX_TOKENS = 4096

async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 1,
    model: str = "claude-haiku-4-5",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.

    Args:
        prompt: The initial prompt for the agent
        tools: List of tool definitions for Anthropic API
        tool_handlers: Dictionary mapping tool names to their handler functions
        max_steps: Maximum number of steps before stopping (default 5)
        model: The Anthropic model to use
        verbose: Whether to print detailed output (default True)

    Returns:
        The submitted answer if submit_answer was called, otherwise None
    """
    print("*"*50)
    client = AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    print("*"*50)
    print("client created")
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            logger.info(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )
        logger.info(f"model initialized for step {step}")

        # --- TOKEN TRACKING LOGIC ---
        # FIX: Initialize total_tokens_used before the loop
        total_tokens_used = 0
        
        if response.usage:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            step_tokens = input_tokens + output_tokens
            total_tokens_used += step_tokens
            
            if verbose:
                logger.info(f"Token Usage (Step {step+1}): Input={input_tokens}, Output={output_tokens}, Total={step_tokens}")
        # ----------------------------

        assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"], (
            f"unsupported stop_reason {response.stop_reason}"
        )

        if response.stop_reason == "max_tokens":
            logger.error(
                f"Model reached max_tokens limit {MAX_TOKENS}. Increase MAX_TOKENS, "
                "simplify your task, or update the code to provide "
                "a message back to the model when it exceeds MAX_TOKENS."
            )

        # Track if we need to continue
        has_tool_use = False
        tool_results = []
        submitted_answer = None

        # Process the response
        for content in response.content:
            if content.type == "text":
                if verbose:
                    logger.info(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        logger.info(f"Using tool: {tool_name}")

                    # Extract arguments based on tool
                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    # Call the appropriate tool handler
                    if tool_name == "python_expression":
                        assert (
                            isinstance(tool_input, dict) and "expression" in tool_input
                        )
                        if verbose:
                            logger.info("\nInput:")
                            logger.info("```")
                            for line in tool_input["expression"].split("\n"):
                                logger.info(f"{line}")
                            logger.info("```")
                        result = handler(tool_input["expression"])
                        if verbose:
                            logger.info("\nOutput:")
                            logger.info("```")
                            logger.info(result)
                            logger.info("```")
                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
                        # Generic handler call
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        # If we have tool uses, add them to the conversation
        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})

            messages.append({"role": "user", "content": tool_results})

            # If an answer was submitted, return it
            if submitted_answer is not None:
                if verbose:
                    logger.info(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            # No tool use, conversation might be complete
            if verbose:
                logger.info("\nNo tool use in response, ending loop.")
            break

    if verbose:
        logger.info(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None

async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    grading_func: Any,
    verbose: bool = False,
    best_run_data: dict = None, # NEW ARGUMENT
) -> tuple[int, bool, float, str]:
    if verbose:
        logger.info(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    code_string = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=10,
        verbose=verbose,
    )
    code_string = str(code_string) if code_string is not None else ""
    failure_reason = "" # FIX: Initialized here to avoid UnboundLocalError
    accuracy = 0.0
    success = False

    # FAILURE CATEGORY 1: No Submission
    if not code_string:
        failure_reason = "1. No Submission: Model failed to call the 'submit_answer' tool or ran out of steps."
        logger.info(f"Run {run_id} FAILURE: {failure_reason}")
    
    # Check for submission before proceeding
    if code_string:
        # 1. Get accuracy and execution status
        accuracy, exec_error = get_accuracy_from_code(code_string)
        
        # FAILURE CATEGORY 2 & 3: Code Execution Error or Missing Output Format
        if exec_error:
            # 2. Code Execution Error or 3. Missing Output Format
            if exec_error.startswith("Code Execution Error"):
                failure_reason = f"2. {exec_error}"
            else:
                failure_reason = f"3. {exec_error}"
            
            logger.info(f"Run {run_id} FAILURE: {failure_reason}") 
        else:
            # If no execution error, check performance
            success = grading_func(accuracy)

            if not success:
                # FAILURE CATEGORY 4: Low Accuracy
                failure_reason = f"4. Low Accuracy: Code submitted ({accuracy:.3f}), expected >= {EXPECTED_ANSWER:.2f}."
                logger.info(f"Run {run_id} FAILURE: {failure_reason}") 
            
            # Check for new best run and save file (only if code was submitted and executed successfully)
            is_new_best = False
            if best_run_data is not None and accuracy > best_run_data["accuracy"]:
                best_run_data["accuracy"] = accuracy
                best_run_data["code"] = code_string
                is_new_best = True
            
    # Output the result simply (PASSED or FAILED with reason)
    if success:
        logger.info(f"Run {run_id}: PASSED (Accuracy: {accuracy:.3f})")
        print(f"Run {run_id}: PASSED (Accuracy: {accuracy:.3f})")
        save_run_pipeline(code_string, run_id, accuracy, is_new_best=is_new_best) #It will save only when it
    else:
        logger.info(f"Run {run_id}: FAILED - {failure_reason}")
        print(f"Run {run_id}: FAILED - {failure_reason}")

    return run_id, success, accuracy, failure_reason

async def main(concurrent: bool = True):
    tools: list[ToolUnionParam] = [
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

    tool_handlers = {
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }

    # Run the test 10 times and track success rate
    num_runs = 10
    expected_answer = EXPECTED_ANSWER

    # Initialize global state tracking for the best run
    best_run_data = {"accuracy": -1.0, "code": ""}
    
    execution_mode = "concurrently" if concurrent else "sequentially"
    logger.info("=" * 60)
    logger.info(f"Running {num_runs} test iterations {execution_mode}...")
    logger.info("=" * 60)

    # 3. Create all test coroutines
    tasks: list[Awaitable[tuple[int, bool, float, str]]] = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=PROMPT,
            tools=tools,
            tool_handlers=tool_handlers,
            grading_func = grading_func,
            verbose=False, # Enforce overall quiet mode
            best_run_data=best_run_data,
        )
        for i in range(num_runs)
    ]

    # 4. Run concurrently or sequentially based on the flag
    if concurrent:
        results = []
        # Process results as they complete
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
    else:
        results = [await task for task in tasks]

    # 5. Count successes and display results
    successes = 0
    for _, success, _, _ in results:
        if success:
            successes += 1
    
    # Calculate and display pass rate
    pass_rate = (successes / num_runs) * 100
    logger.info(f"\n{'=' * 60}")
    logger.info("Test Results Summary:")
    logger.info(f"  Passed: {successes}/{num_runs}")
    logger.info(f"  Failed: {num_runs - successes}/{num_runs}")
    logger.info(f"  Pass Rate: {pass_rate:.1f}%")
    logger.info(f"{'=' * 60}")
        
    if best_run_data["accuracy"] > 0:
        logger.info(f"\nThe overall best pipeline achieved an accuracy of {best_run_data['accuracy']:.3f} and has been saved.")
        print(f"\nThe overall best pipeline achieved an accuracy of {best_run_data['accuracy']:.3f} and has been saved.")
    else:
        logger.info("\nNo successful pipeline runs were recorded.")
        print("\nNo successful pipeline runs were recorded.")

if __name__ == "__main__":
    # Set to True for concurrent execution, False for sequential execution
    asyncio.run(main(concurrent=False))
