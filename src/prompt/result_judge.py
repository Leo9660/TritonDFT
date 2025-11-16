result_judge_prompt = {
    "role": "user",
    "content": ("""You are a computational materials scientist analyzing the progress of a DFT subproblem.

    Now you are given:
    - Main question: {query}
    - Current subproblem: {subproblem}
    - Parameter guesses: {param_json}
    - Existing results: {result_json} (each entry corresponds to one configuration run)

    ### Task
    Judge whether the current subproblem has been **successfully solved** based on the existing results.
      - If solved, output:
        {{
          "status": "done",
          "desc": "<short summary and conclusion, MUST explicitly include the subproblem answer>"
        }}
      - If not yet solved, output:
        {{
          "status": "notdone",
          "new_param_guess": <new parameter JSON>,
          "desc": "<brief analysis and new guessing intuitions>"
        }}

    ### Final Requirement
    - Output MUST be a single valid JSON object following one of the two schemas above.
    - Your entire response must begin with '{{' and end with '}}'.
    """)
}
