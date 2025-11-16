from typing import Dict, List, Union
from prompt.planner import planner_messages
from prompt.tool_setup import parameter_prompt, script_prompt_fixed
from prompt.result_parse import result_parse_prompt
from prompt.result_judge import result_judge_prompt
from prompt.info_query import api_call_prompt

def get_prompt(prompt_type: str, **kwargs) -> List[Dict[str, str]]:
    """
    Select a prompt template by type, fill placeholders,
    and wrap it into a chat-style message list.

    Args:
        prompt_type (str): The type of prompt to use (e.g., "planner").
        **kwargs: Values to substitute into the template (tool, question, etc.).

    Returns:
        List[Dict[str, str]]: Chat-style messages ready for LLM.
    """
    if prompt_type == "planner":
        template = planner_messages
    elif prompt_type == "parameter":
        template = parameter_prompt
    elif prompt_type == "script":
        template = script_prompt_fixed
    elif prompt_type == "script_fixed":
        template = script_prompt_fixed
    elif prompt_type == "result_parse":
        template = result_parse_prompt
    elif prompt_type == "result_judge":
        template = result_judge_prompt
    elif prompt_type == "api_call":
        template = api_call_prompt
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    messages: List[Dict[str, str]] = []

    # --- inject header for previous_memory ---
    pm = kwargs.get("previous_memory", "")
    if pm is not None and str(pm) != "":
        kwargs["previous_memory"] = "\n ### Memory of previous subproblems\n" + str(pm) + "\n"
    # ----------------------------------------
    # --- inject header for initial_structures ---
    qi = kwargs.get("initial_structures", "")
    if qi is not None and str(qi) != "":
        kwargs["query_info"] = "\n ### ### Initial Structures. Use the following initial structures as the starting atomic configurations.\n" + str(qi) + "\n"
    else:
        kwargs["query_info"] = ""
    # ----------------------------------------
    # --- inject header for previous_run ---
    pr = kwargs.get("previous_run", "")
    if pr is not None and str(pr) != "":
        kwargs["previous_run"] = "\n ### Previous incorrect parameter configurations (for your context only, do not repeat them)\n" + str(pr) + "\n"
    else:
        kwargs["previous_run"] = ""
    # ----------------------------------------

    if isinstance(template, list):
        # system + user
        for msg in template:
            content = msg["content"].format(**kwargs)
            messages.append({"role": msg.get("role", "user"), "content": content})
    elif isinstance(template, dict):
        # single dict
        try:
            content = template["content"].format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing placeholder: {e.args[0]} in template.") from e
        messages.append({"role": template.get("role", "user"), "content": content})
    else:
        raise TypeError("Template must be dict or list of dicts.")

    return messages
