parameter_prompt = { 
    "role": "user",
    "content": ("""You are a computational materials scientist solving a DFT subproblem.
    Now you are asked to propose ONLY the minimal necessary input parameter guesses for this subproblem.

    ### Instructions
    1) Propose only the parameters that are directly required or explicitly important for this query.
    2) Make reasonable guesses based on existing data (e.g., Materials Project or known trends).
    Avoid performing convergence tests; instead, give an educated estimate.
    3) When guessing parameters, aim for a Pareto-efficient trade-off:
    choose values that are accurate and numerically stable (i.e., likely to converge),
    while keeping computational cost (time and memory) reasonably low.
    4) Do NOT output any {tool} input files.
    5) Keep scientific choices dynamic rather than relying on fixed material-name rules. When relevant to the current problem, explicitly decide bulk/slab dimensionality and phase, vdW treatment, magnetic order and initial site moments, collinear/noncollinear spin, SOC, DFT+U formulation/value, occupations, and requested orbital projections.
    6) Record a short reason and confidence for every relevant vdW, magnetism, SOC, or DFT+U decision. If evidence is insufficient, say that the choice is uncertain instead of silently omitting it.
    7) Preserve scientific settings supplied in previous workflow memory within their declared branch and scope. A dependent step must not independently change XC, pseudopotential identity/relativity, cutoffs, the complete vdW model (including vdw_corr, dftd3_version, damping and three-body controls), spin, SOC, Hubbard settings, magnetic species labels, or structure provenance. Do not propagate an electronic-only SOC refinement backward into relaxation or into the scalar-relativistic baseline branch.

    ### Output Schema
    {{
    "material": "<element/compound>",
    "structure": {{
        "prototype": "<fcc|bcc|rocksalt|diamond|...>"
    }},
    "parameter_guesses": {{
        "<only the parameters directly relevant to the query>": "<value>"
    }},
    "scientific_decisions": {{
        "<only relevant decisions such as vdw|magnetism|soc|dft_plus_u|dimensionality>": {{
            "choice": "<selected choice or explicit not-used decision>",
            "reason": "<short physical justification>",
            "confidence": "high|medium|low"
        }}
    }}
    }}

    ### The whole user query is:
    {query}.
    {pseudo_context}
    {previous_memory}
    ### Current Problem to Solve:
    {subproblem}.
    Please respond strictly in the required JSON schema, without any additional explanation.
    """)
}

script_prompt_fixed = {
    "role": "user",
    "content": """You are a computational materials scientist.

    Given the parameter JSON below, GENERATE ONLY {bin_tool} input files in {tool_mode} mode (Quantum ESPRESSO style unless otherwise specified).
    Wrap ALL generated inputs inside ONE <scripts> ... </scripts> block; each file inside a <script> ... </script> tag.

    {tool_requirements}

    {previous_memory}

    {previous_inputs}

    {available_files}

    {previous_run}

    Only apply targeted corrections to the script based on the errors above; do not change any other guessed parameters unnecessarily.
    
    ### The whole user query is:
    {query}.
    ### Input (parameter JSON)
    {params_json}

    {query_info}
    ### Current Problem to Solve:
    {subproblem}.

    ### Output Schema
    <scripts>\\n<script> ...one complete {bin_tool} input... </script>\\n<script> ...next... </script>\\n...</scripts>

    ### Pitfall
    Do NOT regenerate or reinterpret the entire query. Only generate new or corrected scripts corresponding to the *specific subproblem(s)* currently being handled.  
    For example, if the overall query involves multiple tasks on the same system (e.g., `vc-relax`, `scf`, and `bandgap`), generate scripts **only** for the current subproblem (e.g., `vc-relax`). Future subproblems like `scf` or `bandgap` will be generated later.

    Please provide your output, containing only the text wrapped in <scripts> and <script> tags, and do not generate any other text.
    You MUST output raw plain text. Do NOT format or render anything. Output exactly the text content only. Output must be copy-paste ready for a Unix shell or input file.
    Do NOT escape or encode any characters. In particular, NEVER output "&amp;".
    Use literal Quantum ESPRESSO namelists like "&control", "&system", "&electrons".
    """
}

    # 12) Symmetry breaking for Output Schema or non-centrosymmetric phases:
    # - If the target structure is non-centrosymmetric, you MUST ensure the calculation does not enforce inversion or higher symmetry.
    # - This can be achieved in one of two ways:
    # (a) Explicitly set in &system: nosym = .true., noinv = .true.
    # (b) Provide slightly perturbed atomic positions that break inversion symmetry (e.g. displacing cations or anions by ±0.005 along z).


# parameter_prompt = { 
#     "role": "user",
#     "content": ("""You are a computational materials scientist solving a DFT subproblem.
#     Now you are asked to propose ONLY the minimal necessary input parameter guesses for this subproblem.

#     ### Instructions
#     1) Propose only the parameters that are directly required or explicitly important for this query.
#     2) Make reasonable guesses based on existing data (e.g., Materials Project or known trends).
#     Avoid performing convergence tests; instead, give an educated estimate.
#     3) When guessing parameters, aim for a Pareto-efficient trade-off:
#     choose values that are accurate and numerically stable (i.e., likely to converge),
#     while keeping computational cost (time and memory) reasonably low.
#     4) Do NOT output any {tool} input files.

#     ### Output Schema
#     {{
#     "material": "<element/compound>",
#     "structure": {{
#         "prototype": "<fcc|bcc|rocksalt|diamond|...>"
#     }},
#     "parameter_guesses": {{
#         "<only the parameters directly relevant to the query>": "<value>"
#     }}
#     }}

#     ### The whole user query is:
#     {query}.
#     {previous_memory}
#     ### Current Problem to Solve:
#     {subproblem}.
#     Please respond strictly in the required JSON schema, without any additional explanation.
#     """)
# }
