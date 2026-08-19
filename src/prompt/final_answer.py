# The synthesis step. Everything before it judges ONE subproblem at a time and
# has no view of the question that was actually asked, so the run used to end
# with a pile of per-step verdicts and no answer.
#
# This is judgment, not transcription, so it is a prompt rather than code: what
# counts as the answer depends entirely on what was asked. The one hard rule is
# that it may not invent numbers — everything it states has to come from the
# step results it is shown.
final_answer_prompt = {
    "role": "user",
    "content": ("""You are a computational materials scientist writing the final answer
    for a DFT study that has just finished running.

    ### The question that was asked
    {query}

    ### What was planned
    {plan}

    ### What each step concluded, in order
    {step_conclusions}

    ### What is actually on disk, parsed from the outputs
    {extracted}

    ### Task
    Answer the question. Write for someone who asked it and did not watch the run.

    Requirements:
    - LEAD with the answer. First sentence states the result and its number with
      units — e.g. "The PBE band gap of silicon is 0.61 eV (indirect, Gamma to
      near X)." Do not open with what was run.
    - Then, briefly, how it was obtained: the workflow and the settings that
      materially affect the number (functional, pseudopotential library, cutoff,
      k-mesh). Two or three sentences.
    - Then any caveat that a scientist would insist on. State known systematic
      errors plainly — for example, semilocal functionals underestimate band
      gaps by roughly half for many semiconductors; a Gamma-only phonon run
      gives no dispersion; a single unit cell cannot show an antiferromagnetic
      ground state.
    - If the question asked for something the run did NOT produce, say so
      explicitly instead of answering a nearby question you can answer.

    Hard constraints:
    - Use ONLY numbers that appear above. Never substitute a literature or
      remembered value. If a quantity was not computed, say it was not computed.
    - The parsed section is the authority on outcomes. If a step said it would
      write a file and the parsed section says none was written, the file was
      NOT written — report that, do not repeat the step's intention as fact.
      If a step says a quantity could not be determined but the parsed section
      has it, use the parsed value.
    - If the steps disagree, say which one you trust and why.
    - Plain prose. No JSON, no markdown headings, no bullet lists. Under 200 words.
    - Do not describe your own reasoning process.
    """)
}
