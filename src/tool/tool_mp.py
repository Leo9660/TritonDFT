import re, ast
from typing import Dict, Any, List, Optional


def _as_structure(x):
    """Coerce an MP API payload into a pymatgen Structure.

    With use_document_model=False the API returns raw dicts, and whether a
    nested structure is monty-decoded to a Structure or left as a plain dict
    varies across mp-api/emmet-core versions (e.g. 0.45.x decodes, 0.46.x does
    not). Normalize both so downstream code is version-independent.
    """
    from pymatgen.core import Structure

    if x is None or isinstance(x, Structure):
        return x
    if isinstance(x, dict):
        return Structure.from_dict(x)
    return x


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


def _spacegroup_number_from_symbol(symbol: str):
    """International number for a Hermann-Mauguin symbol, or None.

    Accepts the loose forms an LLM writes ("P63/mmc") as well as the canonical
    ones ("P6_3/mmc"), since the two differ only by underscores before screw-axis
    digits.
    """
    try:
        from pymatgen.symmetry.groups import SpaceGroup
    except Exception:
        return None
    candidates = [symbol, symbol.replace("_", "")]
    # "P63/mmc" -> "P6_3/mmc": reinsert the underscore before a screw digit.
    candidates.append(re.sub(r"([A-Za-z])(\d)", r"\1_\2", symbol.replace("_", ""), count=1))
    for cand in dict.fromkeys(candidates):
        try:
            return int(SpaceGroup(cand).int_number)
        except Exception:
            continue
    return None

def fetch_material_info_from_api_snippet(snippet: str, limit: int = 25, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
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
    try:
        from mp_api.client import MPRester
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError as exc:
        raise ImportError(
            "Materials Project lookup requires the optional packages `mp_api` "
            "and `pymatgen`. Install them, or run with CLUSTER_AGENT_NO_QUERY_INFO=true "
            "to generate inputs without querying Materials Project."
        ) from exc

    # Step 1. Extract material_ids if explicitly given.
    material_ids = _extract_list(snippet, "material_ids")

    # Step 2. Otherwise parse formula/elements/spacegroup fields.
    formula = _extract_str(snippet, "formula")
    elements = _extract_list(snippet, "elements")
    sg_symbol = _extract_str(snippet, "spacegroup_symbol")
    sg_number = _extract_int(snippet, "spacegroup_number")
    chemsys = _extract_str(snippet, "chemsys")

    # The Materials Project API returns HTTP 500 for any spacegroup_symbol
    # containing a slash — the URL-encoded "%2F" breaks it server-side. That hits
    # a large fraction of real space groups (P6_3/mmc, P4/mmm, I4/mcm, C2/m...),
    # so translate the symbol to its international number, which works fine.
    if sg_symbol and "/" in sg_symbol and sg_number is None:
        sg_number = _spacegroup_number_from_symbol(sg_symbol)
        if sg_number is not None:
            if verbose:
                print(f"[MP] spacegroup_symbol={sg_symbol!r} would 500; querying by number {sg_number}")
            sg_symbol = None
        else:
            # Unknown symbol: drop the constraint rather than fail the whole run.
            if verbose:
                print(f"[MP] could not map spacegroup {sg_symbol!r} to a number; querying without it")
            sg_symbol = None

    # Step 3. Query summary endpoint to get material_ids (avoid `limit=`; truncate locally).
    if not material_ids:
        with MPRester(use_document_model=False) as mpr:
            docs_iter = mpr.materials.summary.search(
                formula=formula,
                elements=elements,
                spacegroup_symbol=sg_symbol,
                spacegroup_number=sg_number,
                chemsys=chemsys,
                fields=["material_id", "structure"],
            )
            docs = list(docs_iter)[:limit]
        material_ids = [d["material_id"] for d in docs]
        relaxed_lookup = {d["material_id"]: _as_structure(d["structure"]) for d in docs}
    else:
        with MPRester(use_document_model=False) as mpr:
            docs_iter = mpr.materials.summary.search(
                material_ids=material_ids,
                fields=["material_id", "structure"],
            )
            docs = list(docs_iter)
        relaxed_lookup = {d["material_id"]: _as_structure(d["structure"]) for d in docs}

    if not material_ids:
        return {}

    # Step 4. Query materials endpoint for initial_structures.
    # use_document_model=False -> raw dicts, bypassing emmet-core's MPID
    # validation. The live MP API now returns alias ids like "mp-aaaaaaft"
    # that fail the strict MPID regex in every emmet-core version, raising
    # "Invalid MPID Format" and killing the whole query. Raw dicts avoid it.
    with MPRester(use_document_model=False) as mpr:
        mats_iter = mpr.materials.search(
            material_ids=material_ids,
            fields=["material_id", "initial_structures"],
        )
        mats = list(mats_iter)

    # Step 5. Merge results into final dict (attribute access).
    result = {
        "material_ids": [],
        "initial_structures": [],
        "relaxed_structures": [],
        "conventional_structure": [],
        "primitive_structure": [],
    }

    mats_by_id = {m["material_id"]: m for m in mats}

    ehull_min = float("inf")
    min_id = None
    min_subid = None
    init_list = {}
    for mid in material_ids:
        mdoc = mats_by_id.get(mid)

        # extract initial structures
        if mdoc and mdoc.get("initial_structures"):
            init_list[mid] = [_as_structure(e).to(fmt="cif") for e in mdoc["initial_structures"]]

        with MPRester(use_document_model=False) as mpr:
            docs_iter = mpr.materials.summary.search(
                material_ids=[mid],
                fields=[
                    "energy_above_hull",
                ],
            )
            docs = list(docs_iter)

            for i, doc in enumerate(docs):
                if doc["energy_above_hull"] < ehull_min:
                    ehull_min = doc["energy_above_hull"]
                    min_id = mid
                    min_subid = i
                    # print(f"New min ehull: {ehull_min} for {min_id} (subid {min_subid})")

    retrieved_structure = relaxed_lookup.get(min_id)
    # 使用 SpacegroupAnalyzer 进行标准化处理
    sga = SpacegroupAnalyzer(retrieved_structure)

    # 1. 获取真正的原始胞 (LiNbO3 应该是 10 原子的那个)
    primitive = sga.get_primitive_standard_structure() 

    # 2. 获取常规胞 (LiNbO3 应该是 30 原子的那个)
    conventional = sga.get_conventional_standard_structure()

    # 存入结果
    result["primitive_structure"].append(primitive)
    result["conventional_structure"].append(conventional)

    result["material_ids"].append(min_id)
    result["initial_structures"].append(init_list[min_id][min_subid])
    result["relaxed_structures"].append(relaxed_lookup.get(mid)[min_subid])
    # result["conventional_structure"].append(conventional)
    # result["primitive_structure"].append(primitive)

    gt = {}
    try:
        s = relaxed_lookup.get(min_id)
        if s is not None:
            lat = s.lattice
            gt["a"] = lat.a
            gt["b"] = lat.b
            gt["c"] = lat.c
            gt["alpha"] = lat.alpha
            gt["beta"] = lat.beta
            gt["gamma"] = lat.gamma
            try:
                sga = SpacegroupAnalyzer(s, symprec=1e-3, angle_tolerance=5)
                gt["space_group"] = sga.get_space_group_symbol()
                gt["space_group_number"] = sga.get_space_group_number()
                gt["point_group"] = sga.get_point_group_symbol()
                gt["crystal_system"] = sga.get_crystal_system()
            except Exception:
                pass
    except Exception:
        pass
    result["ground_truth"] = gt

    # Compact, structured summary for the streamed log. The full structures used
    # to be printed via Structure.__repr__ — dozens of lines of site coordinates
    # that drowned out everything else. They're written to CIF files by the
    # caller instead, and surfaced behind a "details" toggle in the UI.
    summary = {"material_id": min_id}
    try:
        summary["formula"] = primitive.composition.reduced_formula
        summary["n_sites_primitive"] = len(primitive)
        summary["n_sites_conventional"] = len(conventional)
    except Exception:
        pass
    if ehull_min != float("inf"):
        summary["energy_above_hull"] = round(ehull_min, 4)
    for k in ("space_group", "space_group_number", "crystal_system", "point_group"):
        if k in gt:
            summary[k] = gt[k]
    for k in ("a", "b", "c", "alpha", "beta", "gamma"):
        if k in gt:
            summary[k] = round(gt[k], 4)
    result["summary"] = summary

    if verbose:
        sg = summary.get("space_group", "?")
        sgn = summary.get("space_group_number")
        sg_txt = f"{sg} (#{sgn})" if sgn else sg
        bits = [
            str(min_id),
            summary.get("formula", "?"),
            sg_txt,
            summary.get("crystal_system", "?"),
        ]
        print(f"[MP] {' · '.join(bits)}")
        if "a" in summary:
            print(f"[MP] a={summary['a']} b={summary['b']} c={summary['c']} Å · "
                  f"α={summary['alpha']}° β={summary['beta']}° γ={summary['gamma']}° · "
                  f"{summary.get('n_sites_primitive', '?')} sites (primitive)")

    # print("[MP result]", result)

    return result

# ---------- Example usage inside main() ----------

def main():
    snippet = 'mpr.materials.summary.search(formula="BaTiO3", spacegroup_symbol="P4mm")'
    out = fetch_material_info_from_api_snippet(snippet, limit=5)

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
