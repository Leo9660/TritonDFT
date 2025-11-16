api_call_prompt = {
    "role": "user",
    "content": ("""You are an expert assistant that writes Materials Project API queries for retrieving **initial structures**.

    Your goal: Given a natural-language query about a material, output **only one code line (as a quoted string)**
    that calls `mpr.materials.search()` to retrieve its `initial_structures`, using `formula` and `spacegroup_symbol` if available.

    The output must be in **exactly one line**, no explanations or comments.

    ### Example:
    User query: "Find the initial structure of BaTiO3 in the tetragonal P4mm phase."
    ### Output:
    mpr.materials.search(formula=\"BaTiO3\", spacegroup_symbol=\"P4mm\", fields=[\"material_id\", \"initial_structures\"])

    Now do the same for the following query:

    ### User query: {query}
    ### Output:""")
}