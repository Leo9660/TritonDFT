parameter_self_judge_prompt = {
    "role": "user",
    "content": ("""You are a computational materials scientist reviewing a proposed DFT parameter guess before any QE run is executed.

    Now you are given:
    - Main question: {query}
    - Current subproblem: {subproblem}
    - Current parameter guesses: {param_json}
    - Optional category reference sample: {category_reference_sample}
    - Attempt history for this same subproblem: {attempt_history}

    ### Task
    Critically review the current parameter guess before running QE.
    Your job is to catch likely mistakes early, reduce unnecessary computational cost, and preserve physically correct parameters when they already look sound.

    You MUST explicitly self-question and self-judge:
    - Did underestimation likely occur in either the k-point mesh or ecutwfc? Answer "yes" or "no".
    - Did unnecessary extra computational cost likely occur due to an over-dense k-point mesh or an unnecessarily high ecutwfc cutoff? Answer "yes" or "no".
    - What is the primary root cause of concern in the current guess? Choose one concise label such as:
      "sampling", "structure", "stoichiometry", "advanced_parameters", "vdw_compatibility", "pseudopotential_soc", "cutoff", "none", "unknown".

    ### Review rules
    - For k-point meshes and ecutwfc cutoffs, apply two equal hard gates:
      - Accuracy gate: do not allow a guess below the inferred floor required by the requested accuracy.
      - Cost gate: do not allow a denser k-point mesh or a higher ecutwfc than the smallest value that is still safely at or above the inferred floor, unless there is a concrete named physical reason.
    - Infer the floor from physical principles only: primitive vs conventional cell, reciprocal-cell size, metallicity or gap, symmetry, anisotropy, material class, dimensionality, and requested energy accuracy.
    - If a category reference sample is provided, use it only as a category-level calibration anchor; do NOT copy its mesh/cutoff blindly and do NOT let it override specific material physics.
    - For any requested tier, treat the same-tier category reference sample as a floor candidate for BOTH k-point mesh and ecutwfc.
    - You MUST NOT lower the mesh below the category-sample floor unless you can point to a concrete material-specific reason that the system truly needs less raw sampling while still preserving the same requested reciprocal-space resolution.
    - You MUST NOT lower ecutwfc below the category-sample floor unless you can point to a concrete material-specific reason that the system truly needs a lower basis cutoff while still preserving the same requested accuracy tier.
    - If that justification is missing or uncertain, mark underestimate as "yes" or keep/revise upward; do NOT accept the lower mesh/cutoff as ready.
    - For strict targets, uncertainty should bias toward possible underestimation, not toward accepting a cheaper mesh or lower cutoff.
    - Do not use a dense mesh from one example material as a generic answer for others.
    - If the material is layered or anisotropic, consider whether the out-of-plane sampling should be lower than the in-plane sampling while still preserving the requested accuracy, but do not treat anisotropy as permission to drop below the inferred tier floor.
    - If advanced parameters such as vdw_corr, noncolin, lspinorb, smearing, Hubbard U, or spin polarization are physically justified, preserve them.
    - If lspinorb/noncolin is present, require fully relativistic pseudopotentials.
    - If vdw_corr is present, it must be a QE-supported value such as 'dft-d3'.
    - If the current guess already looks physically consistent, do not invent changes.
    - If attempt history shows the same kind of ineffective change repeatedly, avoid repeating it.

    ### Output rules
    - If the current guess is ready to run, output:
      {{
        "status": "ready",
        "underestimate": "yes" or "no",
        "overcost": "yes" or "no",
        "root_cause": "<one concise label>",
        "keep_fixed": ["<parameter_name>", "..."],
        "new_param_guess": {{}},
        "desc": "<brief review summary>"
      }}
    - If the current guess should be revised before QE, output:
      {{
        "status": "revise",
        "underestimate": "yes" or "no",
        "overcost": "yes" or "no",
        "root_cause": "<one concise label>",
        "keep_fixed": ["<parameter_name>", "..."],
        "new_param_guess": <new parameter JSON>,
        "desc": "<brief review summary>"
      }}

    ### Final Requirement
    - Output MUST be a single valid JSON object.
    - Your entire response must begin with '{{' and end with '}}'.
    - If a parameter already appears physically valid and unrelated to the concern, include it in keep_fixed.
    - Do NOT output QE input text.
    """)
}

parameter_self_judge_prompt_gemini = {
    "role": "user",
    "content": ("""You are a computational materials scientist reviewing a proposed DFT parameter guess before any QE run is executed.

    Now you are given:
    - Main question: {query}
    - Current subproblem: {subproblem}
    - Current parameter guesses: {param_json}
    - Optional category reference sample: {category_reference_sample}
    - Attempt history for this same subproblem: {attempt_history}

    ### Task
    Critically review the current parameter guess before running QE.
    Your job is to catch likely mistakes early, reduce unnecessary computational cost, and preserve physically correct parameters when they already look sound.

    You MUST explicitly self-question and self-judge:
    - Did underestimation likely occur in either the k-point mesh or ecutwfc? Answer "yes" or "no".
    - Did unnecessary extra computational cost likely occur due to an over-dense k-point mesh or an unnecessarily high ecutwfc cutoff? Answer "yes" or "no".
    - What is the primary root cause of concern in the current guess? Choose one concise label such as:
      "sampling", "structure", "stoichiometry", "advanced_parameters", "vdw_compatibility", "pseudopotential_soc", "cutoff", "none", "unknown".

    ### Review rules
    - For k-point meshes and ecutwfc cutoffs, apply two equal hard gates:
      - Accuracy gate: do not allow a guess below the inferred floor required by the requested accuracy.
      - Cost gate: do not allow a denser k-point mesh or a higher ecutwfc than the smallest value that is still safely at or above the inferred floor, unless there is a concrete named physical reason.
    - Infer the floor from physical principles only: primitive vs conventional cell, reciprocal-cell size, metallicity or gap, symmetry, anisotropy, material class, dimensionality, and requested energy accuracy.
    - If a category reference sample is provided, use it only as a category-level calibration anchor; do NOT copy its mesh/cutoff blindly and do NOT let it override specific material physics.
    - For any requested tier, treat the same-tier category reference sample as a floor candidate for BOTH k-point mesh and ecutwfc.
    - You MUST NOT lower the mesh below the category-sample floor unless you can point to a concrete material-specific reason that the system truly needs less raw sampling while still preserving the same requested reciprocal-space resolution.
    - You MUST NOT lower ecutwfc below the category-sample floor unless you can point to a concrete material-specific reason that the system truly needs a lower basis cutoff while still preserving the same requested accuracy tier.
    - If that justification is missing or uncertain, mark underestimate as "yes" or keep/revise upward; do NOT accept the lower mesh/cutoff as ready.
    - For strict targets, uncertainty should bias toward possible underestimation, not toward accepting a cheaper mesh or lower cutoff.
    - Do not use a dense mesh from one example material as a generic answer for others.
    - If the material is layered or anisotropic, consider whether the out-of-plane sampling should be lower than the in-plane sampling while still preserving the requested accuracy, but do not treat anisotropy as permission to drop below the inferred tier floor.
    - If advanced parameters such as vdw_corr, noncolin, lspinorb, smearing, Hubbard U, or spin polarization are physically justified, preserve them.
    - If lspinorb/noncolin is present, require fully relativistic pseudopotentials.
    - If vdw_corr is present, it must be a QE-supported value such as 'dft-d3'.
    - If the current guess already looks physically consistent, do not invent changes.
    - If attempt history shows the same kind of ineffective change repeatedly, avoid repeating it.

    ### Plain-text output format
    Output MUST follow this exact field layout in plain text:

    STATUS: <ready or revise>
    UNDERESTIMATE: <yes or no>
    OVERCOST: <yes or no>
    ROOT_CAUSE: <one concise label>
    KEEP_FIXED: <comma-separated parameter names or none>
    NEW_PARAM_GUESS:
    <one parameter per line in the form key = value; use none if STATUS is ready>
    END_NEW_PARAM_GUESS
    DESC: <brief review summary>

    ### Final Requirement
    - Output plain text only, not markdown and not a JSON wrapper object.
    - The fields above must appear exactly once and in the exact order shown.
    - NEW_PARAM_GUESS must contain only parameter lines or the word none.
    - If a parameter already appears physically valid and unrelated to the concern, include it in KEEP_FIXED.
    - Do NOT output QE input text.
    """)
}
