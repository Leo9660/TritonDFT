# The follow-up agent's loop.
#
# A follow-up used to be answered from the conversation text alone, which holds
# the streamed log and nothing else — so "what cutoff did step 3 use?" was
# unanswerable even though 03_nscf.in was sitting on disk. Rather than widen the
# context until everything fits, the model is shown a manifest of what exists
# and asks for the files it wants.
followup_prompt = {
    "role": "user",
    "content": ("""You are answering a question about calculations this service has
    already run for this user, in this conversation.

    ### The question
    {query}

    ### The conversation so far
    {context}

    ### Calculations in this conversation
    {manifest}

    ### Files you have already read
    {documents}

    ### Task
    Either answer, or ask for the files you need first.

    Reply with ONE JSON object and nothing else.

    To read files (you may do this at most twice, so ask for everything you need
    at once, and no more than 4 files):
      {{"action": "read",
        "files": [{{"job": "<job id from the manifest>", "name": "<exact filename>"}}]}}

    To answer:
      {{"action": "answer", "answer": "<the reply to the user>"}}

    Rules:
    - Read a file when the answer depends on what was actually in an input or an
      output — cutoffs, k-meshes, occupations, convergence, a specific energy.
      Input files (.in) are small and are usually what you want. Output files
      (.out) can be very large; ask for one only when the question is about what
      the calculation produced or how it failed.
    - Do not ask for a file when the conversation above already answers the
      question. An extra round costs the user time for nothing.
    - Answer with numbers that appear in the material you were given. If what is
      being asked was never computed, say so and say what would have to be run.
      Never substitute a remembered or literature value for a computed one.
    - Plain prose, no markdown headings, under 250 words.
    """)
}
