# NOTE: every template below is rendered with str.format(**kwargs) in
# prompt/utils.py — literal braces must be doubled. Keep the output format
# XML-ish (<subproblemN>) rather than JSON so no escaping is needed.

# Shared rule text, so the forced / non-forced planner variants can't drift apart.
_BAND_RULES = """
    Band / electronic-structure rule (IMPORTANT — read carefully):
    - `pw_nscf` uses a UNIFORM k-grid (K_POINTS automatic). `pw_bands` uses an
      explicit HIGH-SYMMETRY PATH (K_POINTS crystal_b). They are NOT interchangeable.
    - To PLOT a band structure / dispersion: pw_scf -> `pw_bands` -> `bands_post`.
      NEVER use `pw_nscf` as the k-path step: bands.x would post-process a uniform
      grid and produce a meaningless plot.
    - BAND GAP RULE (applies whenever the query asks for a gap AT ALL, in any
      wording — including "band gap along a high-symmetry path"): the plan MUST
      contain a `pw_nscf` step on a dense UNIFORM grid. A `pw_bands` run does not
      determine occupations or the Fermi level (its k-point weights are path
      points, not a BZ sampling), so pw.x prints no "highest occupied, lowest
      unoccupied level" and NO GAP CAN BE READ FROM IT. `pw_nscf` is the only
      step that yields the gap.
        * gap only, no plot  -> pw_scf -> pw_nscf                 (do NOT add pw_bands/bands_post)
        * gap AND a path/plot -> pw_scf -> pw_nscf -> pw_bands -> bands_post
      Mentioning a high-symmetry path NEVER removes the pw_nscf step; it only
      ADDS the pw_bands + bands_post steps after it.
    - `bands_post` (bands.x) is post-processing ONLY and MUST be preceded by a
      `pw_bands` step. Never place `bands_post` directly after `pw_nscf`.
    - `dos_post` requires a uniform-grid `pw_nscf`, never a `pw_bands` step.

    Phonon post-processing rule:
    - Use `matdyn_post` ONLY for full phonon dispersion / DOS along q-paths, and only AFTER `q2r_post` has produced real-space force constants (flfrc).
    - Use `dynmat_post` for a SINGLE-q (e.g. Gamma-only) ph.x dynamical matrix file (.dynG / .dyn). Do NOT pair `dynmat_post` with `q2r_post`.
    - For a Gamma-only stability check, ph.x already prints frequencies in cm-1 in its own output; an additional dynmat_post step is optional, not mandatory.
"""

_OUTPUT_RULES = """
    Output requirements:
    - Decompose the user query into 1..N subproblems.
    - Each subproblem must be wrapped as <subproblem1>...</subproblem1>, <subproblem2>...</subproblem2>, etc. (in order).
    - Each subproblem must contain exactly three fields:
    Problem: What to calculate
    Tool: Tool to use
    Required input: Required input parameters (Do not give any concrete parameter value here, just describe what is needed)
    These fields MUST appear on separate lines, each separated by a newline; otherwise, the output is considered incorrect.
    - Keep each subproblem short (2-3 lines).
    - Do not output anything outside <subproblem> blocks.

    Core rules:
    Allowed tools: pw_scf, pw_nscf, pw_relax, pw_vc_relax, pw_bands, bands_post, dos_post, projwfc_post, pp_post, q2r_post, matdyn_post, dynmat_post, pw_phonon_gamma, elastic_post.
"""

planner_messages = {
    "role": "user",
    "content": """
    <|system|>
    You are a strict planning assistant for Quantum ESPRESSO ({tool}).
""" + _OUTPUT_RULES + """
    Structure rule (IMPORTANT):
    - Every query starts from an INITIAL structure fetched from the Materials Project. This is a STARTING GUESS, NOT the equilibrium geometry — the cell and atomic positions are generally NOT relaxed.
    - Therefore the FIRST subproblem MUST ALWAYS be a `pw_vc_relax` that relaxes BOTH the cell and the atomic positions to obtain the equilibrium structure. Do this even when the query does not explicitly mention relaxation, and even if a lattice constant is mentioned.
    - All subsequent subproblems (pw_scf, pw_nscf, pw_bands, bands_post, dos_post, phonon steps, elastic_post, etc.) MUST take the RELAXED structure produced by the vc_relax step as their starting structure — never the raw initial structure.
    - Only exception: if the user EXPLICITLY asks to skip relaxation or to use a fixed/given geometry as-is, you may start directly from the provided structure.
""" + _BAND_RULES + """
    <|user|>
    You are a senior Quantum ESPRESSO planner.

    ### In-Context Example 1 (band STRUCTURE / dispersion plot)
    Query: Calculate the band structure of silicon in the diamond structure.

    <subproblem1>
    Problem: Relax the cell and atomic positions to obtain the equilibrium structure
    Tool: pw_vc_relax
    Required input: initial diamond Si structure
    </subproblem1>

    <subproblem2>
    Problem: Do an SCF calculation to converge charge density
    Tool: pw_scf
    Required input: relaxed structure from the vc_relax step
    </subproblem2>

    <subproblem3>
    Problem: Compute eigenvalues along the high-symmetry k-path
    Tool: pw_bands
    Required input: relaxed structure, SCF charge density, high-symmetry path
    </subproblem3>

    <subproblem4>
    Problem: Post-process the band eigenvalues into plottable band-structure data
    Tool: bands_post
    Required input: pw_bands results with the same prefix/outdir
    </subproblem4>

    ---

    ### In-Context Example 2 (band GAP number only — no plot)
    Query: What is the band gap of silicon?

    <subproblem1>
    Problem: Relax the cell and atomic positions to obtain the equilibrium structure
    Tool: pw_vc_relax
    Required input: initial diamond Si structure
    </subproblem1>

    <subproblem2>
    Problem: Do an SCF calculation to converge charge density
    Tool: pw_scf
    Required input: relaxed structure from the vc_relax step
    </subproblem2>

    <subproblem3>
    Problem: Run NSCF on a dense uniform k-grid to resolve the valence band maximum and conduction band minimum
    Tool: pw_nscf
    Required input: relaxed structure, SCF charge density
    </subproblem3>

    ---

    ### In-Context Example 3 (band GAP *and* a high-symmetry path — needs BOTH)
    Query: Calculate the band gap of silicon along the high-symmetry path.

    <subproblem1>
    Problem: Relax the cell and atomic positions to obtain the equilibrium structure
    Tool: pw_vc_relax
    Required input: initial diamond Si structure
    </subproblem1>

    <subproblem2>
    Problem: Do an SCF calculation to converge charge density
    Tool: pw_scf
    Required input: relaxed structure from the vc_relax step
    </subproblem2>

    <subproblem3>
    Problem: Run NSCF on a dense uniform k-grid to resolve the valence band maximum and conduction band minimum
    Tool: pw_nscf
    Required input: relaxed structure, SCF charge density
    </subproblem3>

    <subproblem4>
    Problem: Compute eigenvalues along the high-symmetry k-path
    Tool: pw_bands
    Required input: relaxed structure, SCF charge density, high-symmetry path
    </subproblem4>

    <subproblem5>
    Problem: Post-process the band eigenvalues into plottable band-structure data
    Tool: bands_post
    Required input: pw_bands results with the same prefix/outdir
    </subproblem5>

    ---

    ### In-Context Example 4 (equilibrium lattice constant unknown)
    Query: Calculate the equilibrium lattice constant of Na in the BCC structure.

    <subproblem1>
    Problem: Find equilibrium lattice constant by relaxing the cell volume and atomic positions
    Tool: pw_vc_relax
    Required input: initial BCC Na structure
    Output: equilibrium lattice constant (Å)
    </subproblem1>
    ---

    ### Now handle this query:
    Query: {question}

    - Do NOT include reasoning, explanations, or justification.
    - The output must ONLY be <subproblemN>...</subproblemN> blocks, nothing else.

    <|assistant|>
    """
}

# Variant used when force_vc_relax is OFF: the planner decides whether to relax
# based on the query (e.g. given lattice constant -> straight to SCF; unknown
# equilibrium -> relax first). Kept in sync with planner_messages above except
# for the structure rule and In-Context Example 1.
planner_messages_no_force = {
    "role": "user",
    "content": """
    <|system|>
    You are a strict planning assistant for Quantum ESPRESSO ({tool}).
""" + _OUTPUT_RULES + """
    Structure rule:
    - If key structural information (e.g., the equilibrium lattice constant) is already provided or known, you may go straight to the property calculation (e.g. pw_scf).
    - If the equilibrium geometry is unknown or uncertain, relax it first with pw_vc_relax and use the relaxed structure downstream.
""" + _BAND_RULES + """
    <|user|>
    You are a senior Quantum ESPRESSO planner.

    ### In-Context Example 1 (band STRUCTURE with given lattice constant)
    Query: Calculate the band structure of silicon in the diamond structure (a0 = 5.43 Å).

    <subproblem1>
    Problem: Do an SCF calculation to converge charge density
    Tool: pw_scf
    Required input: diamond Si structure
    </subproblem1>

    <subproblem2>
    Problem: Compute eigenvalues along the high-symmetry k-path
    Tool: pw_bands
    Required input: same structure, SCF charge density, high-symmetry path
    </subproblem2>

    <subproblem3>
    Problem: Post-process the band eigenvalues into plottable band-structure data
    Tool: bands_post
    Required input: pw_bands results with the same prefix/outdir
    </subproblem3>

    ---

    ### In-Context Example 2 (band GAP number only — no plot)
    Query: What is the band gap of silicon (a0 = 5.43 Å)?

    <subproblem1>
    Problem: Do an SCF calculation to converge charge density
    Tool: pw_scf
    Required input: diamond Si structure
    </subproblem1>

    <subproblem2>
    Problem: Run NSCF on a dense uniform k-grid to resolve the valence band maximum and conduction band minimum
    Tool: pw_nscf
    Required input: same structure, SCF charge density
    </subproblem2>

    ---

    ### In-Context Example 3 (equilibrium lattice constant unknown)
    Query: Calculate the equilibrium lattice constant of Na in the BCC structure.

    <subproblem1>
    Problem: Find equilibrium lattice constant by relaxing the cell volume and atomic positions
    Tool: pw_vc_relax
    Required input: initial BCC Na structure
    Output: equilibrium lattice constant (Å)
    </subproblem1>
    ---

    ### Now handle this query:
    Query: {question}

    - Do NOT include reasoning, explanations, or justification.
    - The output must ONLY be <subproblemN>...</subproblemN> blocks, nothing else.

    <|assistant|>
    """
}


# Assistant mode: the user reviewed the plan and asked for a change in natural
# language. Re-emit the WHOLE plan (the suggestion may add, drop, reorder or
# retarget steps — we deliberately don't try to localise it to one subproblem).
plan_refine_messages = {
    "role": "user",
    "content": """
    <|system|>
    You are a strict planning assistant for Quantum ESPRESSO ({tool}).
    You are REVISING an existing plan according to a user's instruction.
""" + _OUTPUT_RULES + _BAND_RULES + """
    Revision rules:
    - Apply the user's instruction faithfully. It may target one step or the whole plan.
    - Keep every part of the original plan that the instruction does not ask to change,
      including wording, unless it violates a rule above.
    - Renumber the subproblems consecutively from 1 after any insertion or removal.
    - If the user's instruction would produce a physically invalid workflow (e.g.
      `bands_post` without a preceding `pw_bands`), fix the workflow so it stays valid
      while still honouring the intent.

    <|user|>
    ### Original query
    {question}

    ### Current plan
    {current_plan}

    ### The user's requested change
    {suggestion}

    Re-emit the COMPLETE revised plan.
    - Do NOT include reasoning, explanations, or justification.
    - The output must ONLY be <subproblemN>...</subproblemN> blocks, nothing else.

    <|assistant|>
    """
}
