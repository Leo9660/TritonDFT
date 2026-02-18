result_parse_prompt = {
    "role": "user",
    "content": ("""Now you are asked to parse the output of a finished DFT run.

    ### Clarification
    - param_json: the planned parameter configuration for this subproblem.
    - input_file: the actual {fn} input text used in this run.
    - output_text: the raw output result produced by {fn}.

    ### Parse Requirement
    {parse_requirement}

    ### Instructions
    1) Determine whether the run and required post-processing steps finished correctly.
    2) Extract only the quantities explicitly required by parse_requirement.
    3) Keep the output minimal and machine-consumable.

    ### Completion Rules (IMPORTANT)
    - Mark the subproblem as completed if the run finished and the required output files
      or parsing steps were executed, even if some physical quantities are null.
    - Do NOT treat missing values (e.g., band_gap, VBM, CBM) as a failure by themselves.
    - If a quantity cannot be determined, set it to null and continue.

    ### Input
    param_json = {input_json}
    input_file = {input_file}
    output_text = {output_text}

    ### Output Schema
    {{
      "prefix": "system_<number>",
      "success": <bool>,
      "key_findings": {{
        "<required_quantity_name_1>": "<value with unit or null>",
        "<required_quantity_name_2>": "<value with unit or null>"
      }}
      "conclusion": Briefly state the run outcome and if fails, speficy the reason why. Limit to 1-2 concise sentences.
    }}

    ### Final Requirement
    - Output MUST be a single valid JSON object matching the schema above.
    - key_findings MUST NOT contain any keys other than those listed in parse_requirement.
    - Do NOT include explanations, reasoning, markdown, or any extra text.
    - If a value cannot be found, use null.
    - If the output_text contains an explicit completion indicator such as "JOB DONE" (case-insensitive),
      you MUST mark the subproblem as completed: set "success" = true, regardless of missing quantities
      (e.g., band structure data not found). Missing outputs only mean the parser could not locate them,
      NOT that the run failed; set those quantities to null and continue.
    """)
}

# result_parse_prompt_backup = {
#     "role": "user",
#     "content": ("""Now you are asked to parse the output of a finished DFT run.

#     ### Clarification
#     - param_json: the planned parameter configuration for this subproblem.
#     - input_file: the actual {fn} input text used in this run.
#     - output_text: the raw output result produced by {fn}.

#     ### Parse Requirement
#     {parse_requirement}

#     ### Instructions
#     1) Determine whether the run and required post-processing steps finished correctly.
#     2) Extract only the quantities explicitly required by parse_requirement.
#     3) Keep the output minimal and machine-consumable.

#     ### Completion Rules (IMPORTANT)
#     - Mark the subproblem as completed if the run finished and the required output files
#       or parsing steps were executed, even if some physical quantities are null.
#     - Do NOT treat missing values (e.g., band_gap, VBM, CBM) as a failure by themselves.
#     - If a quantity cannot be determined, set it to null and continue.

#     ### Input
#     param_json = {input_json}
#     input_file = {input_file}
#     output_text = {output_text}

#     ### Output Schema
#     {{
#       "prefix": "system_<number>",
#       "success": <bool>,
#       "key_findings": {{
#         "<required_quantity_name>": "<value with unit or null>"
#       }}
#       "conclusion": Briefly state the run outcome and if fails, speficy the reason why. Limit to 1-2 concise sentences.
#     }}

#     ### Final Requirement
#     Output MUST be a single valid JSON object matching the schema above.
#     Do NOT include explanations, reasoning, markdown, or any extra text.
#     If a value cannot be found, use null.
#     """)
# }

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
