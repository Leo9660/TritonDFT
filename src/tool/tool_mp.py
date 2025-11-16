import re, ast
from typing import Dict, Any, List, Optional
from mp_api.client import MPRester


# ---------- Helper extraction functions ----------

def _extract_str(snippet: str, key: str) -> Optional[str]:
    """Extract a string argument like key="value"."""
    m = re.search(rf"{key}\s*=\s*([\"'])(.*?)\1", snippet)
    return m.group(2) if m else None


def _extract_int(snippet: str, key: str) -> Optional[int]:
    """Extract an integer argument like key=123."""
    m = re.search(rf"{key}\s*=\s*(\d+)", snippet)
    return int(m.group(1)) if m else None


def _extract_list(snippet: str, key: str) -> Optional[List[Any]]:
    """Extract a Python-style list argument like key=["A","B"]."""
    m = re.search(rf"{key}\s*=\s*(\[[^\]]*\])", snippet, re.DOTALL)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except Exception:
        return None


# ---------- Core function ----------

def fetch_initial_structures_from_api_snippet(snippet: str, limit: int = 25, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Parse an MP API query snippet (e.g. 'mpr.materials.summary.search(formula="BaTiO3", spacegroup_symbol="P4mm")'),
    execute the query to obtain material IDs, then fetch their corresponding initial structures.

    Returns:
        {
          "mp-12345": {
              "initial_structures": [Structure, ...],   # May be empty if not stored
              "relaxed_structure": Structure or None    # Provided as fallback
          },
          ...
        }
    """

    # Step 1. Extract material_ids if explicitly given.
    material_ids = _extract_list(snippet, "material_ids")

    # Step 2. Otherwise parse formula/elements/spacegroup fields.
    formula = _extract_str(snippet, "formula")
    elements = _extract_list(snippet, "elements")
    sg_symbol = _extract_str(snippet, "spacegroup_symbol")
    sg_number = _extract_int(snippet, "spacegroup_number")
    chemsys = _extract_str(snippet, "chemsys")

    # Step 3. Query summary endpoint to get material_ids (avoid `limit=`; truncate locally).
    if not material_ids:
        with MPRester() as mpr:
            docs_iter = mpr.materials.summary.search(
                formula=formula,
                elements=elements,
                spacegroup_symbol=sg_symbol,
                spacegroup_number=sg_number,
                chemsys=chemsys,
                fields=["material_id", "structure"],
            )
            docs = list(docs_iter)[:limit]
        material_ids = [d.material_id for d in docs]
        relaxed_lookup = {d.material_id: d.structure for d in docs}
    else:
        with MPRester() as mpr:
            docs_iter = mpr.materials.summary.search(
                material_ids=material_ids,
                fields=["material_id", "structure"],
            )
            docs = list(docs_iter)
        relaxed_lookup = {d.material_id: d.structure for d in docs}

    if not material_ids:
        return {}

    # Step 4. Query materials endpoint for initial_structures (attribute access, not subscripting).
    with MPRester() as mpr:
        mats_iter = mpr.materials.search(
            material_ids=material_ids,
            fields=["material_id", "initial_structures"],
        )
        mats = list(mats_iter)

    # Step 5. Merge results into final dict (attribute access).
    result = {
        "material_ids": [],
        "initial_structures": [],
        "relaxed_structures": []
    }

    mats_by_id = {m.material_id: m for m in mats}

    ehull_min = float("inf")
    min_id = None
    min_subid = None
    init_list = {}
    for mid in material_ids:
        mdoc = mats_by_id.get(mid)

        # extract initial structures
        if mdoc and getattr(mdoc, "initial_structures", None):
            init_list[mid] = [e.to(fmt="cif") for e in mdoc.initial_structures]

        docs = mpr.summary.search(
            material_ids=mid,
            fields=[
                "energy_above_hull",
            ],
        )

        for i, doc in enumerate(docs):
            if doc.energy_above_hull < ehull_min:
                ehull_min = doc.energy_above_hull
                min_id = mid
                min_subid = i
                print(f"New min ehull: {ehull_min} for {min_id} (subid {min_subid})")

    result["material_ids"].append(min_id)
    result["initial_structures"].append(init_list[min_id][min_subid])
    result["relaxed_structures"].append(relaxed_lookup.get(mid)[min_subid])

    return result

# ---------- Example usage inside main() ----------

def main():
    snippet = 'mpr.materials.summary.search(formula="BaTiO3", spacegroup_symbol="P4mm")'
    out = fetch_initial_structures_from_api_snippet(snippet, limit=5)

    for mid, payload in out.items():
        print(f"\n================= {mid} =================")
        init_structs = payload["initial_structures"]

        if not init_structs:
            print("No initial structures found.")
        else:
            for i, s in enumerate(init_structs):
                print(f"\n# --- Initial structure {i} ---")
                print(s.to(fmt="cif"))  # print as CIF text directly

        if payload["relaxed_structure"]:
            print(f"\n# --- Relaxed structure ---")
            print(payload["relaxed_structure"].to(fmt="cif"))


if __name__ == "__main__":
    main()
