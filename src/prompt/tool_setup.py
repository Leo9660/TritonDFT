# Active prompt iteration: stronger one-sided accuracy bias.
# Iteration result summary:
# - Test case: Si, primitive diamond, LDA, vc-relax, strict 1 meV/atom target
# - Successful QE run folder:
#   /workspace/TritonDFT/2026-04-14_si_highacc_iter1/2026-04-14/Si_vc-relax_163523_2ddff17b
# - Parameter stage guess: ecutwfc = 50 Ry, k-point mesh = 6x6x6
# - Final QE input used: ecutwfc = 80 Ry, K_POINTS automatic / 6 6 6 1 1 1
# - Conclusion: strengthening parameter_prompt alone did not move the Si 1 meV/atom
#   mesh above 6x6x6, so the next likely bottleneck is elsewhere in the prompt stack.
#
# Active parameter prompt iteration: underestimation-first target floor with mandatory lowest-positive-side calibration.
# Previous active parameter_prompt preserved below for comparison.
# parameter_prompt = {
#     "role": "user",
#     "content": ("""You are a computational materials scientist solving a DFT subproblem.
#     Now you are asked to propose ONLY the minimal necessary input parameter guesses for this subproblem.
#
#     ### Instructions
#     1) Propose only the parameters that are directly required or explicitly important for this query.
#     2) Infer parameters from the requested scientific accuracy target and known physical trends of the material.
#     Avoid performing convergence tests; instead, give a conservative expert-grade estimate.
#     3) For this task, accuracy is more important than computational cost.
#     If there is uncertainty, you MUST bias the estimate upward rather than downward.
#     In particular:
#        - Do NOT underestimate ecutwfc.
#        - Do NOT underestimate k-point density.
#        - If two k-point meshes are both plausible, choose the denser one.
#        - A slightly over-conservative parameter is preferred over a too-coarse parameter.
#     4) When the query specifies a strict target such as 1 meV/atom, treat underestimation as unacceptable.
#     Your guess should be at least as strict as a typical expert manual setup for that material class.
#     5) Do NOT output any {tool} input files.
#
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
#
#     ### The whole user query is:
#     {query}.
#     {previous_memory}
#     ### Current Problem to Solve:
#     {subproblem}.
#     Please respond strictly in the required JSON schema, without any additional explanation.
#     """)
# }

parameter_prompt = {
    "role": "user",
    "content": ("""You are a computational materials scientist solving a DFT subproblem.
    Now you are asked to propose ONLY the minimal necessary input parameter guesses for this subproblem.

    ### Instructions
    1) Propose only the parameters that are directly required or explicitly important for this query.
    2) Infer parameters from the requested scientific accuracy target and known physical trends of the specific material.
    Avoid performing convergence tests; instead, give an expert estimate.
    3) For k-point density and cutoffs, use two equal hard gates:
       - Accuracy gate: NEVER choose below the material-specific floor required by the requested accuracy.
       - Cost gate: NEVER choose above the smallest mesh/cutoff that is still safely at or above that floor, unless a concrete material-specific reason requires extra density.
       - First infer the floor, then select the closest valid value from the positive side.
       - Do NOT add an extra "safety" increment after reaching the inferred floor; strict accuracy changes the floor estimate, not a second upward rounding step.
       - Treat `ecutwfc` exactly like the k-point mesh: it MUST NOT be below the inferred cutoff floor for the requested tier, and it MUST NOT be kept above the smallest valid positive-side cutoff unless a concrete reason requires it.
       - Infer the lower bound from physical principles: primitive vs conventional cell, reciprocal-cell size, metallicity or gap, symmetry, anisotropy, material class, and requested energy accuracy.
       - Evaluate plausible k-point meshes and plausible ecutwfc values in increasing cost order and stop at the first value that is not below the inferred floor.
       - If two adjacent meshes or cutoffs are plausible, choose the lower one if it is not below the floor; choose the higher one only if the lower one may be below the floor.
       - Be strict about not inflating the floor: larger primitive cells, multi-atom insulating cells, and small Brillouin zones usually require less raw nk density than one- or two-atom primitive metals/semiconductors.
       - Do NOT transfer a dense mesh appropriate for a small primitive metal/semiconductor to a larger insulating cell unless the reciprocal-space scale and accuracy target require it.
       - Do not downgrade to a medium- or loose-accuracy mesh or cutoff for a strict/high-accuracy target; reduce raw density only when reciprocal-space resolution and basis quality clearly remain at the requested floor.
    4) Advanced physical parameters are not defaults, but they are required when the material physics strongly indicates them.
       Before finalizing parameter_guesses, actively decide whether each of the following is needed: vdw_corr, spin polarization, Hubbard U, noncolin, lspinorb, and smearing.
       - Van der Waals / dispersion: include vdw_corr when the material is likely layered, weakly bonded between structural units, molecular, van der Waals bonded, or contains stacked quintuple/septuple layers. This often includes layered chalcogenides/tellurides, topological insulators, graphite-like systems, and molecular crystals. Use a QE-supported value such as vdw_corr='dft-d3'; do NOT invent values such as 'dft-d3.abc'.
       - Spin / magnetism: include spin polarization for magnetic materials, transition-metal magnets, open-shell atoms/ions, or materials described as ferromagnetic, antiferromagnetic, or ferrimagnetic.
       - Hubbard U: include Hubbard U only for localized correlated d/f electron systems where DFT+U is physically expected.
       - Spin-orbit coupling: include noncolin=.true. and lspinorb=.true. for heavy-element topological materials, strong-SOC compounds, materials where band inversion/topology is central, or when SOC is requested. This requires fully relativistic pseudopotentials.
       - Smearing: include smearing for metals and semimetals; omit smearing for clear insulators or semiconductors unless needed for convergence.
       Do not include advanced parameters when the material class does not justify them.
    5) You may consult the optional category reference sample below as a category-level calibration anchor.
       - Do NOT copy its k-point mesh, ecutwfc, cutoff, or advanced-parameter choices blindly.
       - The actual answer must still be inferred from the specific material, cell size, dimensionality, anisotropy, metallicity/gap, SOC needs, and requested accuracy tier.
       - If the specific material clearly differs from the category exemplar, prefer the specific-material reasoning over the exemplar.
       - Treat the same-tier category sample as a floor candidate for both k-point mesh and ecutwfc, not just a vague suggestion.
       - Do NOT go below that category-sample floor unless you can state a concrete material-specific reason that the present system genuinely needs less raw k-point density or lower cutoff while still remaining at the same requested accuracy floor.
       - If that concrete justification is absent, keeping or moving to the floor candidate is safer than lowering below it.
    6) Do not hard-code a k-point mesh or advanced-parameter answer from one example material as a general answer for other materials.
    7) Do NOT output any {tool} input files.

    ### Output Schema
    {{
    "material": "<element/compound>",
    "structure": {{
        "prototype": "<fcc|bcc|rocksalt|diamond|...>"
    }},
    "parameter_guesses": {{
        "<only the parameters directly relevant to the query>": "<value>"
    }}
    }}

    ### The whole user query is:
    {query}.
    {previous_memory}
    {category_reference_sample}
    {attempt_history}
    ### Current Problem to Solve:
    {subproblem}.
    Please respond strictly in the required JSON schema, without any additional explanation.
    """)
}

parameter_prompt_gemini = {
    "role": "user",
    "content": ("""You are a computational materials scientist solving a DFT subproblem.
    Now you are asked to propose ONLY the minimal necessary input parameter guesses for this subproblem.

    ### Instructions
    1) Propose only the parameters that are directly required or explicitly important for this query.
    2) Infer parameters from the requested scientific accuracy target and known physical trends of the specific material.
    Avoid performing convergence tests; instead, give an expert estimate.
    3) For k-point density and cutoffs, use two equal hard gates:
       - Accuracy gate: NEVER choose below the material-specific floor required by the requested accuracy.
       - Cost gate: NEVER choose above the smallest mesh/cutoff that is still safely at or above that floor, unless a concrete material-specific reason requires extra density.
       - First infer the floor, then select the closest valid value from the positive side.
       - Do NOT add an extra "safety" increment after reaching the inferred floor; strict accuracy changes the floor estimate, not a second upward rounding step.
       - Treat `ecutwfc` exactly like the k-point mesh: it MUST NOT be below the inferred cutoff floor for the requested tier, and it MUST NOT be kept above the smallest valid positive-side cutoff unless a concrete reason requires it.
       - Infer the lower bound from physical principles: primitive vs conventional cell, reciprocal-cell size, metallicity or gap, symmetry, anisotropy, material class, and requested energy accuracy.
       - Evaluate plausible k-point meshes and plausible ecutwfc values in increasing cost order and stop at the first value that is not below the inferred floor.
       - If two adjacent meshes or cutoffs are plausible, choose the lower one if it is not below the floor; choose the higher one only if the lower one may be below the floor.
       - Be strict about not inflating the floor: larger primitive cells, multi-atom insulating cells, and small Brillouin zones usually require less raw nk density than one- or two-atom primitive metals/semiconductors.
       - Do NOT transfer a dense mesh appropriate for a small primitive metal/semiconductor to a larger insulating cell unless the reciprocal-space scale and accuracy target require it.
       - Do not downgrade to a medium- or loose-accuracy mesh or cutoff for a strict/high-accuracy target; reduce raw density only when reciprocal-space resolution and basis quality clearly remain at the requested floor.
    4) Advanced physical parameters are not defaults, but they are required when the material physics strongly indicates them.
       Before finalizing parameter_guesses, actively decide whether each of the following is needed: vdw_corr, spin polarization, Hubbard U, noncolin, lspinorb, and smearing.
       - Van der Waals / dispersion: include vdw_corr when the material is likely layered, weakly bonded between structural units, molecular, van der Waals bonded, or contains stacked quintuple/septuple layers. This often includes layered chalcogenides/tellurides, topological insulators, graphite-like systems, and molecular crystals. Use a QE-supported value such as vdw_corr='dft-d3'; do NOT invent values such as 'dft-d3.abc'.
       - Spin / magnetism: include spin polarization for magnetic materials, transition-metal magnets, open-shell atoms/ions, or materials described as ferromagnetic, antiferromagnetic, or ferrimagnetic.
       - Hubbard U: include Hubbard U only for localized correlated d/f electron systems where DFT+U is physically expected.
       - Spin-orbit coupling: include noncolin=.true. and lspinorb=.true. for heavy-element topological materials, strong-SOC compounds, materials where band inversion/topology is central, or when SOC is requested. This requires fully relativistic pseudopotentials.
       - Smearing: include smearing for metals and semimetals; omit smearing for clear insulators or semiconductors unless needed for convergence.
       Do not include advanced parameters when the material class does not justify them.
    5) You may consult the optional category reference sample below as a category-level calibration anchor.
       - Do NOT copy its k-point mesh, ecutwfc, cutoff, or advanced-parameter choices blindly.
       - The actual answer must still be inferred from the specific material, cell size, dimensionality, anisotropy, metallicity/gap, SOC needs, and requested accuracy tier.
       - If the specific material clearly differs from the category exemplar, prefer the specific-material reasoning over the exemplar.
       - Treat the same-tier category sample as a floor candidate for both k-point mesh and ecutwfc, not just a vague suggestion.
       - Do NOT go below that category-sample floor unless you can state a concrete material-specific reason that the present system genuinely needs less raw k-point density or lower cutoff while still remaining at the same requested accuracy floor.
       - If that concrete justification is absent, keeping or moving to the floor candidate is safer than lowering below it.
    6) Do not hard-code a k-point mesh or advanced-parameter answer from one example material as a general answer for other materials.
    7) Do NOT output any {tool} input files.

    ### Plain-text output format
    Output MUST follow this exact field layout in plain text:

    MATERIAL: <element/compound>
    STRUCTURE_PROTOTYPE: <fcc|bcc|rocksalt|diamond|...>
    PARAMETER_GUESSES:
    <one parameter per line in the form key = value>
    END_PARAMETER_GUESSES

    Notes for values:
    - For list-like values, use JSON-style brackets, for example: [4, 4, 4]
    - For booleans, use true or false
    - For null, use null
    - For strings, output the raw string value without extra commentary

    ### The whole user query is:
    {query}.
    {previous_memory}
    {category_reference_sample}
    {attempt_history}
    ### Current Problem to Solve:
    {subproblem}.
    Output plain text only, using the exact field order above, with no markdown and no extra explanation.
    """)
}

#
#
# # Original baseline prompt preserved for k-point prompt-tuning experiments.
# # Baseline run summary:
# # - Test case: Si, primitive diamond, LDA, vc-relax, strict 1 meV/atom target
# # - Successful QE run folder:
# #   /workspace/TritonDFT/2026-04-14_si_highacc_local_abs/2026-04-14/Si_vc-relax_040316_1f022cae
# # - Agent-chosen mesh in input_1_1.in: K_POINTS automatic / 6 6 6 1 1 1
# # - Baseline conclusion: the original prompt underestimates the desired Si 1 meV/atom
# #   k-point target, so later edits should aim to move the mesh upward first.
# # Weakness note:
# # - The prompt asks for "reasonable" and "educated" guesses and explicitly frames
# #   parameter choice as a Pareto trade-off that should keep computational cost low.
# # - That wording is appropriate for balanced efficiency, but not for a strict one-sided
# #   accuracy goal where underestimating k-point density is considered wrong.
# # - In practice, this likely biases the model toward conservative-cost meshes such as
# #   6x6x6 instead of deliberately rounding upward for the 1 meV/atom Si test.
# # parameter_prompt = { 
# #     "role": "user",
# #     "content": ("""You are a computational materials scientist solving a DFT subproblem.
# #     Now you are asked to propose ONLY the minimal necessary input parameter guesses for this subproblem.
# #
# #     ### Instructions
# #     1) Propose only the parameters that are directly required or explicitly important for this query.
# #     2) Make reasonable guesses based on existing data (e.g., Materials Project or known trends).
# #     Avoid performing convergence tests; instead, give an educated estimate.
# #     3) When guessing parameters, aim for a Pareto-efficient trade-off:
# #     choose values that are accurate and numerically stable (i.e., likely to converge),
# #     while keeping computational cost (time and memory) reasonably low.
# #     4) Do NOT output any {tool} input files.
# #
# #     ### Output Schema
# #     {{
# #     "material": "<element/compound>",
# #     "structure": {{
# #         "prototype": "<fcc|bcc|rocksalt|diamond|...>"
# #     }},
# #     "parameter_guesses": {{
# #         "<only the parameters directly relevant to the query>": "<value>"
# #     }}
# #     }}
# #
# #     ### The whole user query is:
# #     {query}.
# #     {previous_memory}
# #     ### Current Problem to Solve:
# #     {subproblem}.
# #     Please respond strictly in the required JSON schema, without any additional explanation.
# #     """)
# # }

# Active script-generation prompt iteration: enforce underestimation-first lowest-positive-side mesh.
# Previous active script_prompt_fixed preserved below for comparison.
# script_prompt_fixed = {
#     "role": "user",
#     "content": """You are a computational materials scientist.
#
#     Given the parameter JSON below, GENERATE ONLY {bin_tool} input files in {tool_mode} mode (Quantum ESPRESSO style unless otherwise specified).
#     Wrap ALL generated inputs inside ONE <scripts> ... </scripts> block; each file inside a <script> ... </script> tag.
#
#     {tool_requirements}
#
#     {previous_memory}
#     
#     {previous_run}
#
#     Only apply targeted corrections to the script based on the errors above; do not change any other guessed parameters unnecessarily.
#     
#     ### The whole user query is:
#     {query}.
#     ### Input (parameter JSON)
#     {params_json}
#
#     {query_info}
#     ### Current Problem to Solve:
#     {subproblem}.
#
#     ### Output Schema
#     <scripts>\\n<script> ...one complete {bin_tool} input... </script>\\n<script> ...next... </script>\\n...</scripts>
#
#     ### Pitfall
#     Do NOT regenerate or reinterpret the entire query. Only generate new or corrected scripts corresponding to the *specific subproblem(s)* currently being handled.  
#     For example, if the overall query involves multiple tasks on the same system (e.g., `vc-relax`, `scf`, and `bandgap`), generate scripts **only** for the current subproblem (e.g., `vc-relax`). Future subproblems like `scf` or `bandgap` will be generated later.
#
#     Please provide your output, containing only the text wrapped in <scripts> and <script> tags, and do not generate any other text.
#     You MUST output raw plain text. Do NOT format or render anything. Output exactly the text content only. Output must be copy-paste ready for a Unix shell or input file.
#     Do NOT escape or encode any characters. In particular, NEVER output "&amp;".
#     Use literal Quantum ESPRESSO namelists like "&control", "&system", "&electrons".
#     """
# }

script_prompt_fixed = {
    "role": "user",
    "content": """You are a computational materials scientist.

    Given the parameter JSON below, GENERATE ONLY {bin_tool} input files in {tool_mode} mode (Quantum ESPRESSO style unless otherwise specified).
    Wrap ALL generated inputs inside ONE <scripts> ... </scripts> block; each file inside a <script> ... </script> tag.

    {tool_requirements}

    {previous_memory}

    {previous_run}

    {attempt_history}

    When correcting a failed script, apply only targeted corrections based on the errors above.
    However, do not blindly preserve a guessed k-point mesh or ecutwfc if it conflicts with the requested accuracy tier.
    For BOTH K_POINTS and ecutwfc, enforce the same two equal hard gates as the requirements:
    reject any mesh or cutoff below the inferred floor, and reject any denser mesh or higher cutoff when a lower value is clearly still at or above that floor.
    Do NOT add a second upward safety step after the floor is reached.
    If a smaller mesh is still at or above the inferred floor, you MUST use the smaller mesh unless there is a concrete named physical reason not to.
    If a smaller mesh is below the inferred floor, you MUST NOT use it even if it is cheaper.
    If a lower ecutwfc is still at or above the inferred floor, you MUST use the lower cutoff unless there is a concrete named physical reason not to.
    If a lower ecutwfc is below the inferred floor, you MUST NOT use it even if it is cheaper.
    Be strict about cell-size scaling: for larger multi-atom insulating cells, do not inherit dense small-cell meshes when a lower mesh satisfies the same reciprocal-space resolution.
    If the generated script cannot clearly justify that the lower mesh or lower ecutwfc is still at the requested accuracy floor, use the next denser mesh or higher cutoff.
    Once a mesh or ecutwfc is clearly at or above the inferred floor, high computational cost is also invalid: do NOT choose or keep a denser mesh or higher cutoff unless the parameter JSON or the material physics gives a concrete reason.
    If a parameter JSON explicitly contains advanced parameters such as vdw_corr, noncolin, lspinorb, occupations, smearing, degauss, Hubbard U, or spin polarization, you MUST preserve them in the generated QE input unless QE syntax makes them impossible.
    If lspinorb=.true. or noncolin=.true. is used for spin-orbit coupling, you MUST use fully relativistic pseudopotentials and a fully relativistic pseudo_dir. Do not combine SOC with scalar-relativistic pseudopotentials.
    If vdw_corr is present in the parameter JSON, it MUST appear inside &system in the generated QE input.
    Use only QE-supported vdw_corr values. Prefer vdw_corr='dft-d3' for D3; do NOT output unsupported strings such as 'dft-d3.abc'.

    ### The whole user query is:
    {query}.
    ### Input (parameter JSON)
    {params_json}

    {query_info}
    ### Current Problem to Solve:
    {subproblem}.

    ### Output Schema
    <scripts>\n<script> ...one complete {bin_tool} input... </script>\n<script> ...next... </script>\n...</scripts>

    ### Pitfall
    Do NOT regenerate or reinterpret the entire query. Only generate new or corrected scripts corresponding to the *specific subproblem(s)* currently being handled.
    For example, if the overall query involves multiple tasks on the same system (e.g., `vc-relax`, `scf`, and `bandgap`), generate scripts **only** for the current subproblem (e.g., `vc-relax`). Future subproblems like `scf` or `bandgap` will be generated later.

    Please provide your output, containing only the text wrapped in <scripts> and <script> tags, and do not generate any other text.
    You MUST output raw plain text. Do NOT format or render anything. Output exactly the text content only. Output must be copy-paste ready for a Unix shell or input file.
    Do NOT escape or encode any characters. In particular, NEVER output "&amp;".
    Use literal Quantum ESPRESSO namelists like "&control", "&system", "&electrons".
    Do NOT put commas after namelist assignments in QE input files.
    Put pseudo_dir only in &control, never in &system.
    All QE numeric fields must contain evaluated numeric literals only. Do NOT write arithmetic expressions such as "10.26 / 0.529177" in namelists.
    """
}

script_prompt_fixed_gemini = {
    "role": "user",
    "content": """You are a computational materials scientist.

    Given the parameter JSON below, GENERATE ONLY one complete {bin_tool} input file in {tool_mode} mode (Quantum ESPRESSO style unless otherwise specified).

    {tool_requirements}

    {previous_memory}

    {previous_run}

    {attempt_history}

    When correcting a failed script, apply only targeted corrections based on the errors above.
    However, do not blindly preserve a guessed k-point mesh or ecutwfc if it conflicts with the requested accuracy tier.
    For BOTH K_POINTS and ecutwfc, enforce the same two equal hard gates as the requirements:
    reject any mesh or cutoff below the inferred floor, and reject any denser mesh or higher cutoff when a lower value is clearly still at or above that floor.
    Do NOT add a second upward safety step after the floor is reached.
    If a smaller mesh is still at or above the inferred floor, you MUST use the smaller mesh unless there is a concrete named physical reason not to.
    If a smaller mesh is below the inferred floor, you MUST NOT use it even if it is cheaper.
    If a lower ecutwfc is still at or above the inferred floor, you MUST use the lower cutoff unless there is a concrete named physical reason not to.
    If a lower ecutwfc is below the inferred floor, you MUST NOT use it even if it is cheaper.
    Be strict about cell-size scaling: for larger multi-atom insulating cells, do not inherit dense small-cell meshes when a lower mesh satisfies the same reciprocal-space resolution.
    If the generated script cannot clearly justify that the lower mesh or lower ecutwfc is still at the requested accuracy floor, use the next denser mesh or higher cutoff.
    Once a mesh or ecutwfc is clearly at or above the inferred floor, high computational cost is also invalid: do NOT choose or keep a denser mesh or higher cutoff unless the parameter JSON or the material physics gives a concrete reason.
    If a parameter JSON explicitly contains advanced parameters such as vdw_corr, noncolin, lspinorb, occupations, smearing, degauss, Hubbard U, or spin polarization, you MUST preserve them in the generated QE input unless QE syntax makes them impossible.
    If lspinorb=.true. or noncolin=.true. is used for spin-orbit coupling, you MUST use fully relativistic pseudopotentials and a fully relativistic pseudo_dir. Do not combine SOC with scalar-relativistic pseudopotentials.
    If vdw_corr is present in the parameter JSON, it MUST appear inside &system in the generated QE input.
    Use only QE-supported vdw_corr values. Prefer vdw_corr='dft-d3' for D3; do NOT output unsupported strings such as 'dft-d3.abc'.

    ### The whole user query is:
    {query}.
    ### Input (parameter JSON)
    {params_json}

    {query_info}
    ### Current Problem to Solve:
    {subproblem}.

    ### Output format
    Output exactly one complete raw QE input file as plain text.
    Do NOT wrap it in <scripts>, <script>, markdown fences, JSON, or explanations.
    Start directly from the QE input content itself.

    Use literal Quantum ESPRESSO namelists like "&control", "&system", "&electrons".
    Do NOT put commas after namelist assignments in QE input files.
    Put pseudo_dir only in &control, never in &system.
    All QE numeric fields must contain evaluated numeric literals only. Do NOT write arithmetic expressions such as "10.26 / 0.529177" in namelists.
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
