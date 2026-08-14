from typing import Dict, List, Union
from prompt.planner import planner_messages, planner_messages_no_force, plan_refine_messages
from prompt.tool_setup import parameter_prompt, script_prompt_fixed
from prompt.result_parse import result_parse_prompt
from prompt.result_judge import result_judge_prompt
from prompt.info_query import api_call_prompt
from prompt.slurm_execution import slurm_execution_prompt

def _tool_vocabulary() -> str:
    """Render the dispatchable tool set from FN_MAP, so the planner prompt cannot
    drift out of sync with what the code can actually run.

    Each tool carries its own executable, mode and description — including the
    constraints that used to be hand-written as planner rules (e.g. matdyn_post
    needs q2r_post first; pw_phonon_gamma is Gamma-only and gives no dispersion).
    Sourcing them here keeps the prompt free of duplicated domain heuristics: the
    vocabulary describes itself.
    """
    from tool.tool_map import FN_MAP, ALLOWED_FNS

    lines = []
    for name in sorted(ALLOWED_FNS):
        spec = FN_MAP.get(name)
        if spec is None:
            continue
        binary = spec.exec + (f" ({spec.mode})" if spec.mode else "")
        desc = " ".join((spec.description or "").split())
        lines.append(f"    - {name}  [{binary}]" + (f": {desc}" if desc else ""))
    return "\n".join(lines)


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
        # Default ON: structures come from MP as initial guesses, so always
        # relax first. Set force_vc_relax=False to let the planner decide.
        force_vc_relax = kwargs.get("force_vc_relax", True)
        template = planner_messages if force_vc_relax else planner_messages_no_force
    elif prompt_type == "plan_refine":
        template = plan_refine_messages
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
    elif prompt_type == "slurm":
        template = slurm_execution_prompt
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    messages: List[Dict[str, str]] = []

    # Planner-family prompts render the live tool set rather than a hand-copied list.
    if prompt_type in ("planner", "plan_refine"):
        kwargs.setdefault("tool_vocabulary", _tool_vocabulary())

    # --- inject header for previous_memory ---
    pm = kwargs.get("previous_memory", "")
    if pm is not None and str(pm) != "":
        kwargs["previous_memory"] = "\n ### Memory of previous subproblems\n" + str(pm) + "\n"
    # ----------------------------------------
    # --- inject header for initial_structures ---
    # qi = kwargs.get("initial_structures", "")
    # if qi is not None and str(qi) != "":
    #     kwargs["query_info"] = "\n ### ### Initial Structures. Use the following initial structures as the starting atomic configurations.\n" + str(qi) + "\n"
    # else:
    #     kwargs["query_info"] = ""
    # We are using conventional structure info for now
    # qi = kwargs.get("conventional_structure", "")
    # if qi is not None and str(qi) != "":
    #     kwargs["query_info"] = \
    #     """
    #     ### Initial Structures: Use the following CONVENTIONAL unit-cell structure as the starting atomic configuration.
    #     - The provided structure is a conventional unit-cell representation.
    #     - All lattice lengths are in angstrom (Å); lattice angles are in degrees.
    #     - Atomic positions are fractional (crystal) coordinates with respect to the lattice vectors.
    #     This structure should be used to construct Quantum ESPRESSO inputs.
    #     """ + str(qi) + "\n"
    # else:
    #     kwargs["query_info"] = ""
    # Now we are using primitive structure info
    qi = kwargs.get("primitive_structure", "")
    if qi is not None and str(qi) != "":
        kwargs["query_info"] = \
        """
        ### Initial Structures: Use the following PRIMITIVE unit-cell structure as the starting atomic configuration.
        - The provided structure is a primitive unit-cell representation.
        - All lattice lengths are in angstrom (Å); lattice angles are in degrees.
        - Atomic positions are fractional (crystal) coordinates with respect to the lattice vectors.
        This structure should be used to construct Quantum ESPRESSO inputs.
        """ + str(qi) + "\n"
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
