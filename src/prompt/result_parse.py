result_parse_prompt = {
    "role": "user",
    "content": ("""Now you are asked to analyze the output of a finished DFT run, which corresponds to a parameter configuration.

    ### Clarification
    - param_json: the planned parameter configuration and sweep guesses for this subproblem.
    - input_file: the actual {fn} input text used in this run.
    - output_text: the raw output result produced by {fn}.

    ### Parse Requirement
    {parse_requirement}

    ### Instructions
    1) Parse and summarize the result in concise, machine-consumable JSON.
    2) Include:
    - success: true/false (whether run finished correctly)
    - param_used: copy param_json here, include the sweep value actually used in this run
    - key_findings: dictionary of extracted important quantities (e.g. total energy, Fermi level, band gap, lattice constant, forces, stress)
    - conclusion: 1-2 sentence human-readable summary
    - errors: list of error/warning messages if any

    ### Input
    Now the input is:
    param_json = {input_json}
    input_file = {input_file}
    output_text = {output_text}

    ### Output Schema
    {{
    "prefix": "system_<number>",
    "success": <bool>,
    "param_used": <dict with parameters used in this run>,
    "key_findings": "<findings required by parse_requirement, Please output the unit as well>",
    "conclusion": "<short summary>",
    "errors": ["<warning or error messages>"]
    }}

    ### Final Requirement
    Output MUST be a single valid JSON object matching the schema above.
    Do NOT include explanations, reasoning, markdown, or any extra text.
    If you cannot find a value, use null.
    Do NOT attempt to generate code to solve this problem; directly provide the answer in the JSON format below.
    """)
}

### log 0909: The model can output parsing result, but with analysis
# result_parse_prompt = {
#     "role": "user",
#     "content": ("""You are a computational materials scientist analyzing DFT results.

#     Now you are asked to analyze the output of a finished DFT run, which corresponds to a parameter configuration.

#     ### Clarification
#     - param_json: the planned parameter configuration and sweep guesses for this subproblem.
#     - input_file: the actual {fn} input text used in this run.
#     - output_text: the raw output result produced by {fn}.

#     ### Instructions
#     1) Parse and summarize the result in concise, machine-consumable JSON.
#     2) Include:
#     - success: true/false (whether run finished correctly)
#     - param_used: copy param_json here, include the sweep value actually used in this run
#     - key_findings: dictionary of extracted important quantities (e.g. total energy, Fermi level, band gap, lattice constant, forces, stress)
#     - conclusion: 1–2 sentence human-readable summary
#     - errors: list of error/warning messages if any

#     ### Input
#     Now the input is:
#     param_json = {input_json}
#     input_file = {input_file}
#     output_text = {output_text}

#     ### Output Schema
#     {{
#     "success": <bool>,
#     "param_used": <the value of the sweep parameter actually used in this run>,
#     "key_findings": {{
#     "total_energy": <float or null>,
#     "fermi_level": <float or null>,
#     "band_gap": <float or null>,
#     "lattice_constant": <float or null>,
#     "forces_max": <float or null>,
#     "stress_max": <float or null>
#     }},
#     "conclusion": "<short summary>",
#     "errors": [ "<warning or error messages>" ]
#     }}

#     Onlu output the expected json structure. Do NOT output any other text. The expected json output is:\n\n
#     """)
# }
