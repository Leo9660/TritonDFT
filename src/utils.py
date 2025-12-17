import json
import re
import os
from typing import Optional, Any, Dict, List

def get_qe_prefix(self):
    return getattr(self, "qe_bin_prefix", None) or os.environ.get("QE_BIN_PREFIX", "")

def patch_qe_input_file(
    in_path: str,
    *,
    new_pseudo_dir: Optional[str] = None,
    new_outdir: Optional[str] = None,
    new_prefix: Optional[str] = None,
    # Pseudopotential replacement policy (choose one or both; pp_map has higher priority):
    pp_dir: Optional[str] = None,          # replace the 3rd column .UPF file with pp_dir/original-filename
    pp_map: Optional[Dict[str, str]] = None,  # species name -> pseudopotential file path (absolute or relative)
    pp_dir_clean: Optional[bool] = False
) -> None:
    """
    In-place modification of QE input file:
    1) Replace pseudo_dir/outdir/prefix values if provided.
    2) In ATOMIC_SPECIES block, replace the 3rd column (.UPF file):
       - with pp_dir/filename if pp_dir is given, or
       - with pp_map[species] if provided (pp_map has higher priority).
    """
    with open(in_path, "r", encoding="utf-8") as f:
        text = f.read()

    def _replace_key(text: str, key: str, value: str) -> str:
        # Match lines like  key = '...'  or  key='...'  or  key = ... (possibly ending with a comma)
        # Try to preserve trailing comments
        pattern = re.compile(
            rf"(^\s*{re.escape(key)}\s*=\s*)(?:['\"]?)([^,'\"\n]+)(?:['\"]?)(\s*,?\s*)(?=$)",
            re.IGNORECASE | re.MULTILINE
        )
        # If no match, try a more relaxed version (allowing trailing comments)
        if not pattern.search(text):
            pattern = re.compile(
                rf"(^\s*{re.escape(key)}\s*=\s*)(?:['\"]?)([^,'\"\n]+)(?:['\"]?)(\s*,?\s*)(?=.*$)",
                re.IGNORECASE | re.MULTILINE
            )
        return pattern.sub(rf"\1'{value}'\3", text)

    # if new_pseudo_dir:
    #     text = _replace_key(text, "pseudo_dir", new_pseudo_dir)
    if new_outdir:
        text = _replace_key(text, "outdir", new_outdir)
    # if new_prefix:
    #     text = _replace_key(text, "prefix", new_prefix)

    # Process ATOMIC_SPECIES block
    lines = text.splitlines()
    new_lines: List[str] = []
    in_species = False

    # Keywords indicating the end of ATOMIC_SPECIES block
    species_end_keys = re.compile(
        r"^\s*(ATOMIC_POSITIONS|K_POINTS|CELL_PARAMETERS|ATOMIC_FORCES|OCCUPATIONS|CONSTRAINTS|&\w+)\b",
        re.IGNORECASE
    )

    for raw in lines:
        line = raw
        if not in_species:
            if re.match(r"^\s*ATOMIC_SPECIES\b", line, re.IGNORECASE):
                in_species = True
                new_lines.append(line)
                continue
            else:
                new_lines.append(line)
                continue

        # Inside ATOMIC_SPECIES
        if not line.strip() or species_end_keys.match(line):
            in_species = False
            new_lines.append(line)
            continue

        # Preserve inline comments (after '#')
        parts = line.split("#", 1)
        body = parts[0].rstrip()
        comment = (" #"+parts[1]) if len(parts) == 2 else ""

        # Expected format: species mass pseudopotential
        tokens = body.split()
        if len(tokens) >= 3:
            species, mass, ppfile = tokens[:3]
            # Only modify .upf files
            if ppfile.lower().endswith(".upf"):
                new_pp = ppfile
                # Priority: pp_dir_clean > pp_map > pp_dir
                if pp_dir_clean:
                    new_pp = f"{species}.upf".lower()
                elif pp_map and species in pp_map:
                    new_pp = pp_map[species]
                elif pp_dir:
                    new_pp = os.path.join(pp_dir, os.path.basename(ppfile))
                # Rebuild line (preserve extra tokens if present)
                rest = " ".join(tokens[3:]) if len(tokens) > 3 else ""
                rebuilt = f"{species} {mass} {new_pp}"
                if rest:
                    rebuilt += " " + rest
                line = rebuilt + comment
            else:
                line = body + comment
        else:
            line = raw

        new_lines.append(line)

    new_text = "\n".join(new_lines)
    if new_text != text:
        with open(in_path, "w", encoding="utf-8") as f:
            f.write(new_text)

def parse_scripts_block(generated: str) -> list[str]:
    # </scripts> or <\scripts>
    scripts_block = re.search(
        r"<scripts>(.*?)</scripts>|<scripts>(.*?)<\\scripts>",
        generated,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not scripts_block:
        raise ValueError("No <scripts>...</scripts> block found in model output.")

    full_block = scripts_block.group(1) if scripts_block.group(1) is not None else scripts_block.group(2)

    # parsing multiple <script>...</script>
    scripts = re.findall(
        r"<script>(.*?)</script>|<script>(.*?)<\\script>",
        full_block,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # normalize
    normalized = [(s[0] if s[0] else s[1]).strip() for s in scripts]

    if not normalized:
        if full_block.strip():
            normalized = [full_block.strip()]
        else:
            raise ValueError("Empty <scripts> block.")

    normalized = [s.replace("\\n", "\n") for s in normalized]

    return normalized


def write_inputs(work_dir: str, scripts: list, prefix: str = "input", suffix: str = ".in"):
    """
    write multiple input scripts to files in work_dir.
    """
    os.makedirs(work_dir, exist_ok=True)
    paths = []
    for idx, content in enumerate(scripts, start=1):
        path = os.path.join(work_dir, f"{prefix}_{idx}{suffix}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.rstrip() + "\n")
        paths.append(path)
    return paths

def get_qe_result(work_dir: str, input_paths: list, verbose: bool = False) -> list:
    input_text = []
    output_text = []
    for idx, in_path in enumerate(input_paths, start=1):
        out_path = os.path.join(work_dir, f"output_{idx}.out")

        # Read the input file
        try:
            with open(in_path, "r", encoding="utf-8") as f:
                input_text.append(f.read())
        except FileNotFoundError:   
            if verbose:
                print(f"[parser] Input file not found: {in_path}")
            input_text.append("")
        
        # Read the output file
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                output_text.append(f.read())
        except FileNotFoundError:   
            if verbose:
                print(f"[parser] Output file not found: {out_path}")
            output_text.append("")
    
    return input_text, output_text

def parse_plan_string(s: str) -> List[Dict[str, Any]]:
    """
    Parse a plan string containing <subproblemN>...</subproblemN> blocks.
    Each block must explicitly include lines starting with:
    Problem:, Tool:, Required input:, Sweep:
    
    Returns a list of dicts with keys: id, problem, tool, input, sweep.
    """
    subproblems = []
    pattern = re.compile(r"<subproblem(\d+)>(.*?)</subproblem\1>", re.DOTALL | re.IGNORECASE)

    for match in pattern.finditer(s):
        pid = int(match.group(1))
        body = match.group(2).strip()
        lines = [line.strip() for line in body.splitlines() if line.strip()]

        # Defaults
        problem, tool, req_input, sweep = None, None, None, None

        for line in lines:
            if line.lower().startswith("problem:"):
                problem = line.split(":", 1)[1].strip()
            elif line.lower().startswith("tool:"):
                tool = line.split(":", 1)[1].strip()
            elif line.lower().startswith("required input:"):
                req_input = line.split(":", 1)[1].strip()
            elif line.lower().startswith("sweep:"):
                sweep = line.split(":", 1)[1].strip()

        subproblems.append({
            "id": pid,
            "problem": problem,
            "tool": tool,
            "input": req_input,
            "sweep": sweep
        })

    if not subproblems:
        raise ValueError("No <subproblem> blocks found in input string")

    return subproblems

def extract_json_brutal(s: str):
    """
    get json structure
    """
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError(f"No json structure in {s}")
    return json.loads(s[start:end+1])

def output_to_log_file(work_dir: str, file_name: str, output: str, new: bool = False):
    os.makedirs(work_dir, exist_ok=True)
    log_path = os.path.join(work_dir, file_name)
    mode = "w" if new else "a"
    with open(log_path, mode, encoding="utf-8") as f:
        f.write(output)
        if not output.endswith("\n"):
            f.write("\n")

def parse_test():
    test_str = '''### Answer:
    ```json
    {
    "question": "Calculate the band structure of silicon (fcc + 2 atoms basis, lattice constant = 5.43) along high-symmetry paths.",
    "work_dir": "jobs/si-fcc-bands/",
    "subproblems": [
        {
        "subquestion": "SCF to converge charge density (structure provided in question)",
        "call": {
            "tool": "quantum espresso",
            "fn": "pw_scf",
            "args": {
            "structure": { "lattice": "fcc", "a0": 5.43, "ibrav": 0, "species": ["Si"], "basis": ["Si"] },
            "ecutwfc": 40,
            "ecutrho": 320,
            "kpoints": [8,8,8],
            "occupations": "fixed",
            "conv_thr": 1e-6
            }
        }
        },
        {
        "subquestion": "NSCF along high-symmetry path",
        "call": {
            "tool": "quantum espresso",
            "fn": "pw_nscf",
            "args": {
            "structure_from": "pw_scf",
            "kpath": "seekpath:auto",
            "kpath_points": 200,
            "occupations": "fixed"
            }
        }
        },
        {
        "subquestion": "Postprocess band structure",
        "call": {
            "tool": "quantum espresso",
            "fn": "bands_post",
            "args": { "input_from": "pw_nscf", "plot": true }
        }
        }
    ],
    "notes": "Structure came from the question; SCF → NSCF (k-path) → bands.x post."
    }
    ```'''
    print(parse_plan_string(test_str))

    test_str = """    <scripts>
    <script>
    &control
    calculation = 'bands'
    restart_mode = 'from_scratch'
    prefix = 'X_fcc_a4p00_k12p00p12_e100'
    outdir = './out/'
    pseudo_dir = './pseudo/'
    /
    &system
    ibrav = 2
    celldm(1) = 4.00
    nat = 1
    ntyp = 1
    ecutwfc = 100
    /
    &electrons
    diagonalization = 'david'
    mixing_beta = 0.7
    /
    ATOMIC_SPECIES
    X  1.0  X.pbe.UPF
    ATOMIC_POSITIONS
    X  0.0  0.0  0.0
    K_POINTS
    automatic
    12  12  12  0  0  0
    </script>
    <script>
    &control
    calculation = 'bands'
    restart_mode = 'from_scratch'
    prefix = 'X_fcc_a4p00_k8p00p8p8_e100'
    outdir = './out/'
    pseudo_dir = './pseudo/'
    /
    &system
    ibrav = 2
    celldm(1) = 4.00
    nat = 1
    ntyp = 1
    ecutwfc = 100
    /
    &electrons
    diagonalization = 'david'
    mixing_beta = 0.7
    /
    ATOMIC_SPECIES
    X  1.0  X.pbe.UPF
    ATOMIC_POSITIONS
    X  0.0  0.0  0.0
    K_POINTS
    automatic
    8  8  8  0  0  0
    </script>
    <script>
    &control
    calculation = 'bands'
    restart_mode = 'from_scratch'
    prefix = 'X_fcc_a4p00_k10p00p10p10_e100'
    outdir = './out/'
    pseudo_dir = './pseudo/'
    /
    &system
    ibrav = 2
    celldm(1) = 4.00
    nat = 1
    ntyp = 1
    ecutwfc = 100
    /
    &electrons
    diagonalization = 'david'
    mixing_beta = 0.7
    /
    ATOMIC_SPECIES
    X  1.0  X.pbe.UPF
    ATOMIC_POSITIONS
    X  0.0  0.0  0.0
    K_POINTS
    automatic
    10  10  10  0  0  0
    </script>
    </scripts>"""

    ret = parse_scripts_block(generated = test_str)

    patch_qe_input_file("/workspace/DFTagent/test/job_pw.x_20250908_043840/input_1.in",
    new_pseudo_dir="/workspace/DFTagent/SSSP/",
    new_outdir="./tmp",
    new_prefix="scf_1",
    pp_dir="Si.pbe-n-rrkjus_psl.1.0.0.UPF"
    )

if __name__ == "__main__":
    parse_test()
