# Active result-judge prompt iteration: success requires underestimation-first floor and lowest-positive-side cost.
# Previous active result_judge_prompt preserved below for comparison.
# result_judge_prompt = {
#     "role": "user",
#     "content": ("""You are a computational materials scientist analyzing the progress of a DFT subproblem.

#     Now you are given:
#     - Main question: {query}
#     - Current subproblem: {subproblem}
#     - Parameter guesses: {param_json}
#     - Existing results: {result_json} (each entry corresponds to one configuration run)

#     ### Task
#     Judge whether the current subproblem has been **successfully solved** based on the existing results.
#     Some warning messages in the results do not necessarily indicate failure; focus on whether the main question has been answered.
#       - If solved (success = \"true\"), output:
#         {{
#           "status": "done",
#           "desc": "<short summary and conclusion, MUST explicitly include the subproblem answer>"
#         }}
#       - If not yet solved (success = \"false\"), output:
#         {{
#           "status": "notdone",
#           "new_param_guess": <new parameter JSON>,
#           "desc": "<brief analysis and new guessing intuitions>"
#         }}

#     ### Final Requirement
#     - Output MUST be a single valid JSON object following one of the two schemas above.
#     - Your entire response must begin with '{{' and end with '}}'.
#     - If any entry in result_json has {{"success": true}},
#       you MUST unconditionally treat the subproblem as successfully solved.
#     - In this case, you MUST output:
#       {{ "status": "done", ... }}
#       and you MUST NOT return "notdone" or propose new_param_guess.
#     """)
# }

result_judge_prompt = {
    "role": "user",
    "content": ("""You are a computational materials scientist analyzing the progress of a DFT subproblem.

    Now you are given:
    - Main question: {query}
    - Current subproblem: {subproblem}
    - Parameter guesses: {param_json}
    - Existing results: {result_json} (each entry corresponds to one configuration run)
    - Attempt history for this same subproblem: {attempt_history}

    ### Task
    Judge whether the current subproblem has been **successfully solved** based on the existing results.
    Some warning messages in the results do not necessarily indicate failure; focus on whether the main question has been answered and whether the chosen physical parameters match the requested accuracy target.
    Use the attempt history to avoid repeating ineffective fixes. Preserve parameter choices that are already physically valid, and revise only the parameters that are most likely causing the current failure.
    After every attempt, you MUST explicitly self-question and self-judge:
    - Did underestimation occur in either the k-point mesh or ecutwfc? Answer "yes" or "no".
    - Did unnecessary extra computational cost occur due to an over-dense k-point mesh or unnecessarily high ecutwfc cutoff? Answer "yes" or "no".
    - What is the primary root cause of the current problem? Choose one concise label such as:
      "sampling", "structure", "stoichiometry", "namelist_format", "vdw_compatibility", "pseudopotential_soc", "cutoff", "unknown".
    - Only proceed to another attempt if these answers justify a revision.

    For k-point meshes and ecutwfc cutoffs, apply two equal hard gates:
    - Accuracy gate: a mesh or ecutwfc below the material-specific expected floor for the requested accuracy tier is not acceptable.
    - Cost gate: a mesh or ecutwfc above the smallest valid positive-side value is not acceptable unless there is a concrete material-specific reason.
    - If the result is below the floor in k-point mesh, return notdone with a denser mesh.
    - If the result is below the floor in ecutwfc, return notdone with a higher cutoff.
    - If the result is over-dense relative to the smallest valid positive-side mesh, return notdone with that lower valid mesh.
    - If the result uses an unnecessarily high ecutwfc relative to the smallest valid positive-side cutoff, return notdone with that lower valid cutoff.
    - Do not use a mesh from one example material as the answer for all materials; infer from the specific material, cell, and accuracy tier.
    - Be strict about cell-size scaling: for larger multi-atom insulating cells, do not treat dense small-cell metal/semiconductor meshes as the floor unless reciprocal-space resolution or another concrete physical reason requires it.
    - For strict/high-accuracy targets, reject meshes that appear to be medium- or loose-accuracy choices unless the larger cell or smaller Brillouin zone clearly preserves high-accuracy reciprocal-space resolution.
    - Do not accept a result just because the model claims the no-underestimate goal is achieved. The result must show concrete material/cell/accuracy-tier reasoning that the mesh is at or above the inferred floor.
    - If that reasoning is absent or generic, treat the no-underestimate gate as failed and return notdone with a denser mesh.
    - If parameter_guesses includes advanced parameters such as vdw_corr, noncolin, lspinorb, smearing, Hubbard U, or spin polarization, the generated input/result must preserve them. If they were dropped, return notdone and include them in new_param_guess.
    - If vdw_corr is present, it must use a QE-supported value such as 'dft-d3'. If the output warns that the vdw correction is unknown or unused, return notdone with a supported vdw_corr value.
    - If SOC is requested through noncolin/lspinorb but the run fails because scalar-relativistic pseudopotentials were used, return notdone and require fully relativistic pseudopotentials/pseudo_dir.
    - If the observed failure is not caused by k-point density or cutoff, do NOT increase k-points or cutoff as the primary fix.
    - If the observed failure is structural, stoichiometric, atomic-position-related, namelist-format-related, vdw-compatibility-related, or pseudopotential/SOC-related, preserve the current k-point mesh and ecutwfc unless there is explicit evidence that they are also wrong.
    - When a parameter already appears physically valid and unrelated to the failure mode, keep it fixed in new_param_guess instead of re-guessing it.
    - If the same failure mode repeats across attempts, propose a qualitatively different correction rather than repeating the same directional change.

      - If solved (success = "true" and parameters are appropriate), output:
        {{
          "status": "done",
          "underestimate": "yes" or "no",
          "overcost": "yes" or "no",
          "root_cause": "<one concise label>",
          "desc": "<short summary and conclusion, MUST explicitly include the subproblem answer>"
        }}
      - If not yet solved or parameters are inappropriate, output:
        {{
          "status": "notdone",
          "underestimate": "yes" or "no",
          "overcost": "yes" or "no",
          "root_cause": "<one concise label>",
          "new_param_guess": <new parameter JSON>,
          "failure_mode": "<short machine-readable diagnosis>",
          "desc": "<brief analysis and new guessing intuitions>"
        }}

    ### Final Requirement
    - Output MUST be a single valid JSON object following one of the two schemas above.
    - Your entire response must begin with '{{' and end with '}}'.
    - Do NOT mark the subproblem done solely because QE ran successfully; first check whether BOTH the chosen k-point mesh and ecutwfc are not below the requested accuracy tier floor, then check whether they satisfy the lowest-positive-side cost rule.
    """)
}

result_judge_prompt_gemini = {
    "role": "user",
    "content": ("""You are a computational materials scientist analyzing the progress of a DFT subproblem.

    Now you are given:
    - Main question: {query}
    - Current subproblem: {subproblem}
    - Parameter guesses: {param_json}
    - Existing results: {result_json} (each entry corresponds to one configuration run)
    - Attempt history for this same subproblem: {attempt_history}

    ### Task
    Judge whether the current subproblem has been successfully solved based on the existing results.
    Some warning messages in the results do not necessarily indicate failure; focus on whether the main question has been answered and whether the chosen physical parameters match the requested accuracy target.
    Use the attempt history to avoid repeating ineffective fixes. Preserve parameter choices that are already physically valid, and revise only the parameters that are most likely causing the current failure.
    After every attempt, you MUST explicitly self-question and self-judge:
    - Did underestimation occur in either the k-point mesh or ecutwfc? Answer "yes" or "no".
    - Did unnecessary extra computational cost occur due to an over-dense k-point mesh or unnecessarily high ecutwfc cutoff? Answer "yes" or "no".
    - What is the primary root cause of the current problem? Choose one concise label such as:
      "sampling", "structure", "stoichiometry", "namelist_format", "vdw_compatibility", "pseudopotential_soc", "cutoff", "unknown".
    - Only proceed to another attempt if these answers justify a revision.

    For k-point meshes and ecutwfc cutoffs, apply two equal hard gates:
    - Accuracy gate: a mesh or ecutwfc below the material-specific expected floor for the requested accuracy tier is not acceptable.
    - Cost gate: a mesh or ecutwfc above the smallest valid positive-side value is not acceptable unless there is a concrete material-specific reason.
    - If the result is below the floor in k-point mesh, return notdone with a denser mesh.
    - If the result is below the floor in ecutwfc, return notdone with a higher cutoff.
    - If the result is over-dense relative to the smallest valid positive-side mesh, return notdone with that lower valid mesh.
    - If the result uses an unnecessarily high ecutwfc relative to the smallest valid positive-side cutoff, return notdone with that lower valid cutoff.
    - Do not use a mesh from one example material as the answer for all materials; infer from the specific material, cell, and accuracy tier.
    - Be strict about cell-size scaling: for larger multi-atom insulating cells, do not treat dense small-cell metal/semiconductor meshes as the floor unless reciprocal-space resolution or another concrete physical reason requires it.
    - For strict/high-accuracy targets, reject meshes that appear to be medium- or loose-accuracy choices unless the larger cell or smaller Brillouin zone clearly preserves high-accuracy reciprocal-space resolution.
    - Do not accept a result just because the model claims the no-underestimate goal is achieved. The result must show concrete material/cell/accuracy-tier reasoning that the mesh is at or above the inferred floor.
    - If that reasoning is absent or generic, treat the no-underestimate gate as failed and return notdone with a denser mesh.
    - If parameter_guesses includes advanced parameters such as vdw_corr, noncolin, lspinorb, smearing, Hubbard U, or spin polarization, the generated input/result must preserve them. If they were dropped, return notdone and include them in new_param_guess.
    - If vdw_corr is present, it must use a QE-supported value such as 'dft-d3'. If the output warns that the vdw correction is unknown or unused, return notdone with a supported vdw_corr value.
    - If SOC is requested through noncolin/lspinorb but the run fails because scalar-relativistic pseudopotentials were used, return notdone and require fully relativistic pseudopotentials/pseudo_dir.
    - If the observed failure is not caused by k-point density or cutoff, do NOT increase k-points or cutoff as the primary fix.
    - If the observed failure is structural, stoichiometric, atomic-position-related, namelist-format-related, vdw-compatibility-related, or pseudopotential/SOC-related, preserve the current k-point mesh and ecutwfc unless there is explicit evidence that they are also wrong.
    - When a parameter already appears physically valid and unrelated to the failure mode, keep it fixed in new_param_guess instead of re-guessing it.
    - If the same failure mode repeats across attempts, propose a qualitatively different correction rather than repeating the same directional change.

    ### Plain-text output format
    Output MUST follow this exact field layout in plain text:

    STATUS: <done or notdone>
    UNDERESTIMATE: <yes or no>
    OVERCOST: <yes or no>
    ROOT_CAUSE: <one concise label>
    FAILURE_MODE: <short machine-readable diagnosis or none>
    NEW_PARAM_GUESS:
    <one parameter per line in the form key = value; use none if STATUS is done>
    END_NEW_PARAM_GUESS
    DESC: <short summary and conclusion; if done, MUST explicitly include the subproblem answer>

    ### Final Requirement
    - Output plain text only, not markdown and not a JSON wrapper object.
    - The fields above must appear exactly once and in the exact order shown.
    - NEW_PARAM_GUESS must contain only parameter lines or the word none.
    - Do NOT mark the subproblem done solely because QE ran successfully; first check whether BOTH the chosen k-point mesh and ecutwfc are not below the requested accuracy tier floor, then check whether they satisfy the lowest-positive-side cost rule.
    """)
}
