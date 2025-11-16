# -*- coding: utf-8 -*-
"""
qe_metrics.py
-------------
Rebuild a clean QE input using:
  - &CONTROL block copied verbatim from original .in (case-insensitive)
  - ATOMIC_SPECIES copied from original .in
  - ATOMIC_POSITIONS and CELL_PARAMETERS from the final-geometry window of .out
  - K_POINTS automatic (and its grid line) taken as the LAST one from original .in
Then print the WHOLE rebuilt input and compute lattice/symmetry via pymatgen.

Usage:
    python qe_metrics.py <scf.in> <relax.out>
"""

import re
import sys
from typing import Optional, Tuple, Dict

from pymatgen.io.pwscf import PWInput
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element

BOHR_TO_ANG = 0.52917721092

def get_lattice_param_str(qe_input_str: str):
    """Return (a,b,c, alpha,beta,gamma) from a QE input string."""
    normed = _normalize_qe_for_pymatgen(qe_input_str)

    # 尝试走 pymatgen 自带的 parser
    try:
        pw = PWInput.from_str(normed)
        s = pw.structure
    except Exception:
        # 兜底：自己解析
        s = _minimal_qe_to_structure(normed)

    a, b, c = s.lattice.abc
    alpha, beta, gamma = s.lattice.angles
    print(f"a={a:.6f} Å, b={b:.6f} Å, c={c:.6f} Å")
    print(f"α={alpha:.6f}°, β={beta:.6f}°, γ={gamma:.6f}°")
    return a, b, c, alpha, beta, gamma


def get_symm_str(qe_input_str: str):
    """Return (SG symbol, SG number, point group, crystal system) from QE input string."""
    normed = _normalize_qe_for_pymatgen(qe_input_str)

    try:
        pw = PWInput.from_str(normed)
        s = pw.structure
    except Exception:
        s = _minimal_qe_to_structure(normed)

    sga = SpacegroupAnalyzer(s, symprec=1e-3, angle_tolerance=5)
    SG = sga.get_space_group_symbol()
    SG_num = sga.get_space_group_number()
    PG = sga.get_point_group_symbol()
    CS = sga.get_crystal_system()
    print("SG symbol:", SG)
    print("SG number:", SG_num)
    print("Point group:", PG)
    print("Crystal system:", CS)
    return SG, SG_num, PG, CS


# ================== 下面是这两个函数依赖的轻量 helper ================== #

def _normalize_qe_for_pymatgen(src: str) -> str:
    """
    如果没有 CELL_PARAMETERS，就尝试把常见 QE 的 ibrav/celldm 形式展开成显式晶格。
    当前策略：
      - 已有 CELL_PARAMETERS：直接返回
      - ibrav in (1, 2, 3): 展开成立方/面心/体心
      - ibrav == 4 或者出现 celldm(3): 当成六方，展开
      - 其他情况：保持原样
    """
    import re
    from math import sqrt

    # 1) 已经有 CELL_PARAMETERS 就不用动
    if re.search(r"(?mi)^\s*CELL_PARAMETERS", src):
        return src

    # 2) 找 &system 块
    m_sys = re.search(r"(?mis)^\s*&\s*system\b.*?^\s*/\s*$", src)
    if not m_sys:
        return src
    system_blk = m_sys.group(0)

    # 3) 取 ibrav
    m_ibrav = re.search(r"ibrav\s*=\s*([0-9\-]+)", system_blk, flags=re.I)
    if not m_ibrav:
        return src
    ibrav_val = int(m_ibrav.group(1))

    # 4) 取 celldm(1)
    m_celldm1 = re.search(r"celldm\(1\)\s*=\s*([0-9\.EeDd+\-]+)", system_blk, flags=re.I)
    if not m_celldm1:
        return src
    celldm1_bohr = float(m_celldm1.group(1).replace("D", "E").replace("d", "e"))
    a_ang = celldm1_bohr * BOHR_TO_ANG

    cell_lines = None

    # ====== cubic: 1/2/3 ======
    if ibrav_val in (1, 2, 3):
        half = 0.5 * a_ang
        if ibrav_val == 1:
            cell_lines = [
                (a_ang, 0.0,   0.0),
                (0.0,   a_ang, 0.0),
                (0.0,   0.0,   a_ang),
            ]
        elif ibrav_val == 2:
            cell_lines = [
                (0.0,   half,  half),
                (half,  0.0,   half),
                (half,  half,  0.0),
            ]
        else:  # ibrav == 3
            cell_lines = [
                (-half,  half,  half),
                ( half, -half,  half),
                ( half,  half, -half),
            ]

    else:
        # ====== 非立方的 check 在这儿加上 ======
        # 情况A: ibrav = 4 → 六方
        # 情况B: 不管 ibrav 是多少，只要有 celldm(3) 我们也当成六方来展开
        m_celldm3 = re.search(r"celldm\(3\)\s*=\s*([0-9\.EeDd+\-]+)", system_blk, flags=re.I)
        if ibrav_val == 4 or m_celldm3:
            if not m_celldm3:
                # 六方必须有 c/a，没有就没法展开
                return src
            celldm3 = float(m_celldm3.group(1).replace("D", "E").replace("d", "e"))
            c_ang = a_ang * celldm3
            cell_lines = [
                (a_ang,            0.0,                 0.0),
                (-0.5 * a_ang,  0.5 * sqrt(3) * a_ang,  0.0),
                (0.0,             0.0,                 c_ang),
            ]
        else:
            # 其他 ibrav 先不猜，保持原样
            return src

    # 走到这里说明我们已经构造出了 cell_lines
    cell_blk = "CELL_PARAMETERS (angstrom)\n"
    for x, y, z in cell_lines:
        cell_blk += f"{x:.16f}   {y:.16f}   {z:.16f}\n"

    # 把 ibrav 改成 0
    patched_system = re.sub(r"ibrav\s*=\s*{}".format(ibrav_val), "ibrav = 0", system_blk, flags=re.I)

    # 写回原文本
    start, end = m_sys.span()
    rebuilt = src[:start] + patched_system + src[end:]
    if not rebuilt.endswith("\n"):
        rebuilt += "\n"
    rebuilt += cell_blk
    return rebuilt


def _minimal_qe_to_structure(qe_str: str) -> Structure:
    """
    更健壮的兜底版：
      1. 找 CELL_PARAMETERS，往后扫，捞出 3 行≥3个数的向量；
      2. 找 ATOMIC_POSITIONS，往后扫，捞出 element + 3 个数；
      3. 单位只处理 angstrom / crystal 这两种常见的。
    """
    # ========== 1) CELL_PARAMETERS ==========
    m_cell = re.search(r"(?mi)^\s*CELL_PARAMETERS(?:\s*\(([^)]+)\))?\s*$", qe_str, re.M)
    if not m_cell:
        raise ValueError("Cannot find CELL_PARAMETERS in QE input (minimal parser).")

    cell_unit = (m_cell.group(1) or "").strip().lower()
    cell_lines = []
    idx = m_cell.end()
    lines = qe_str[idx:].splitlines()

    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        # 抓这一行里的所有数字（支持E/D指数）
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?", stripped)
        if len(nums) >= 3:
            # 只要前三个
            x, y, z = (float(n.replace("D", "E").replace("d", "e")) for n in nums[:3])
            cell_lines.append((x, y, z))
        # 收够 3 行就可以停了
        if len(cell_lines) == 3:
            break

    if len(cell_lines) != 3:
        raise ValueError("CELL_PARAMETERS block does not contain 3 valid vector lines.")

    # 单位转换
    if "bohr" in cell_unit:
        cell_lines = [[c * BOHR_TO_ANG for c in vec] for vec in cell_lines]
    # "(alat=...)" 这种我们前面 normalize 的时候已经转过了，这里就当成 angstrom

    lattice = Lattice(cell_lines)

    # ========== 2) ATOMIC_POSITIONS ==========
    m_pos = re.search(r"(?mi)^\s*ATOMIC_POSITIONS(?:\s*\(([^)]+)\))?\s*$", qe_str, re.M)
    if not m_pos:
        raise ValueError("Cannot find ATOMIC_POSITIONS in QE input (minimal parser).")

    pos_unit = (m_pos.group(1) or "").strip().lower()
    idx = m_pos.end()
    lines = qe_str[idx:].splitlines()

    species = []
    coords = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            # 遇到空行就认为 positions 结束了
            if species:
                break
            else:
                continue
        parts = stripped.split()
        if len(parts) < 4:
            # 不是元素 + 3坐标 了，认为结束
            if species:
                break
            else:
                continue
        sp = parts[0]
        # 只取后三个数字
        x = float(parts[1].replace("D", "E").replace("d", "e"))
        y = float(parts[2].replace("D", "E").replace("d", "e"))
        z = float(parts[3].replace("D", "E").replace("d", "e"))
        species.append(sp)
        coords.append([x, y, z])

    if not species:
        raise ValueError("No atomic positions parsed from QE input (minimal parser).")

    # ========== 3) 构造结构 ==========
    if "crystal" in pos_unit or "alat" in pos_unit:
        struct = Structure(lattice, species, coords)
    else:
        struct = Structure(lattice, species, coords, coords_are_cartesian=True)

    return struct

# ----------------------- Final-geometry block extractors -----------------------

ALLCAPS_HDR = re.compile(r"(?m)^\s*[A-Z_][A-Z0-9_ ]*(?:\s*\([^\)]*\))?\s*$")

def _slice_final_window(text: str) -> Optional[str]:
    """Text between last 'Begin final coordinates' and the subsequent 'End final coordinates'."""
    b = list(re.finditer(r"Begin final coordinates", text))
    e = list(re.finditer(r"End final coordinates", text))
    if not b or not e:
        return None
    b_end = b[-1].end()
    e_after = [m.start() for m in e if m.start() > b_end]
    if not e_after:
        return None
    return text[b_end:e_after[0]]

def _is_float(tok: str) -> bool:
    """Float check that accepts Fortran D/d exponents."""
    try:
        float(tok.replace("D","E").replace("d","e"))
        return True
    except Exception:
        return False

def _extract_cell(win: str) -> str:
    """CELL_PARAMETERS header + exactly 3 vector lines (≥3 float-like tokens per line)."""
    m = re.search(r"(?mi)^[ \t\r]*CELL_PARAMETERS(?:\s*\([^\)]*\))?\s*$", win, re.M)
    if not m:
        raise ValueError("CELL_PARAMETERS not found in final window.")
    s = win.rfind("\n", 0, m.end()) + 1
    e = win.find("\n", m.end());  e = len(win) if e == -1 else e
    header = win[s:e].rstrip()
    i = e + 1
    vecs = []
    while i <= len(win) and len(vecs) < 3:
        j = win.find("\n", i); j = len(win) if j == -1 else j
        ln = win[i:j]; i = j + 1
        if not ln.strip(): 
            continue
        if sum(_is_float(t) for t in ln.split()) >= 3:
            vecs.append(ln.rstrip())
    if len(vecs) != 3:
        raise ValueError("Incomplete CELL_PARAMETERS vectors (need 3).")
    return header + "\n" + "\n".join(vecs) + "\n"

def _extract_pos(win: str) -> str:
    """ATOMIC_POSITIONS header + contiguous element lines (Element + 3 floats)."""
    m = re.search(r"(?mi)^[ \t\r]*ATOMIC_POSITIONS(?:\s*\([^\)]*\))?\s*$", win, re.M)
    if not m:
        raise ValueError("ATOMIC_POSITIONS not found in final window.")
    s = win.rfind("\n", 0, m.end()) + 1
    e = win.find("\n", m.end());  e = len(win) if e == -1 else e
    header = win[s:e].rstrip()
    i = e + 1
    lines = [header]
    while i <= len(win):
        j = win.find("\n", i); j = len(win) if j == -1 else j
        ln = win[i:j]; i = j + 1
        parts = ln.strip().split()
        if len(parts) >= 4 and parts[0][:1].isalpha() and _is_float(parts[1]) and _is_float(parts[2]) and _is_float(parts[3]):
            lines.append(ln.rstrip())
        elif not ln.strip():
            continue
        else:
            break
    if len(lines) == 1:
        raise ValueError("ATOMIC_POSITIONS header found but no atom lines.")
    return "\n".join(lines) + "\n"

BOHR_TO_ANG = 0.52917721092

def _normalize_cell_to_angstrom(cell_blk: str) -> str:
    """
    If cell_blk header is '(alat=...)' or '(bohr)', convert the 3x vectors to angstrom
    and return a block with header 'CELL_PARAMETERS (angstrom)'. If already angstrom,
    return as-is.

    Accepts Fortran D/d exponents in numbers.
    """
    lines = [ln for ln in cell_blk.splitlines() if ln.strip() != ""]
    if not lines:
        raise ValueError("Empty CELL_PARAMETERS block.")

    header = lines[0].strip()
    body   = lines[1:]

    # QE sometimes prints exactly 3 lines; be robust if more appear, we still take first 3 vector lines
    if len(body) < 3:
        raise ValueError("Incomplete CELL_PARAMETERS vectors (need 3).")

    hlow = header.lower().replace(" ", "")
    # Already angstrom
    if "(angstrom)" in hlow:
        # normalize header wording minimally
        return "CELL_PARAMETERS (angstrom)\n" + "\n".join(body[:3]) + "\n"

    factor = None

    # bohr -> angstrom
    if "(bohr)" in hlow:
        factor = BOHR_TO_ANG

    # alat=? (alat is in Bohr)
    if "(alat=" in hlow:
        m = re.search(r"alat\s*=\s*([0-9+\-EeDd\.]+)", header, flags=re.I)
        if not m:
            raise ValueError("Found '(alat=...)' but could not parse the numeric value.")
        alat_str = m.group(1).replace("D", "E").replace("d", "e")
        try:
            alat_bohr = float(alat_str)
        except Exception:
            raise ValueError(f"Failed to parse alat value: {alat_str!r}")
        factor = alat_bohr * BOHR_TO_ANG

    # plain '(alat)' without value
    if factor is None and "(alat)" in hlow:
        raise ValueError("CELL_PARAMETERS uses '(alat)' without a numeric value; cannot convert to angstrom without celldm(1).")

    if factor is None:
        # Fall back: return as-is
        return cell_blk if cell_blk.endswith("\n") else cell_blk + "\n"

    out_vecs = []
    for i in range(3):
        raw = body[i]
        tokens = raw.strip().split()
        if len(tokens) < 3:
            raise ValueError(f"CELL_PARAMETERS line {i+1} has fewer than 3 numbers: {raw!r}")

        def _to_float(x: str) -> float:
            return float(x.replace("D","E").replace("d","e"))

        x, y, z = (_to_float(tokens[0]), _to_float(tokens[1]), _to_float(tokens[2]))
        X, Y, Z = (x * factor, y * factor, z * factor)

        out_vecs.append(f"{X:.16f}   {Y:.16f}   {Z:.16f}")

    return "CELL_PARAMETERS (angstrom)\n" + "\n".join(out_vecs) + "\n"

# ----------------------- Pull pieces from original .in -----------------------

def _extract_control_block(inp: str) -> str:
    """Copy &CONTROL ... / verbatim (case-insensitive)."""
    m = re.search(r"(?mis)^\s*&\s*control\b.*?^\s*/\s*$", inp)
    if not m:
        raise ValueError("&CONTROL block not found in input.")
    block = m.group(0)
    # ensure trailing newline
    if not block.endswith("\n"): block += "\n"
    return block

def _extract_species_block(inp: str) -> str:
    """ATOMIC_SPECIES block from its header up to the next ALLCAPS header (or EOF)."""
    m = re.search(r"(?mi)^\s*ATOMIC_SPECIES\s*$", inp, re.M)
    if not m:
        raise ValueError("ATOMIC_SPECIES not found in input.")
    start = inp.rfind("\n", 0, m.end()) + 1
    nxt = ALLCAPS_HDR.search(inp, m.end())
    end = nxt.start() if nxt else len(inp)
    block = inp[start:end]
    if not block.endswith("\n"): block += "\n"
    return block

def _extract_last_kpoints_block(inp: str) -> Optional[str]:
    """
    Return the LAST 'K_POINTS automatic' header with its immediate grid line.
    If not found, return None.
    """
    hits = list(re.finditer(r"(?mi)^\s*K_POINTS\s+automatic\s*$", inp, re.M))
    if not hits:
        return None
    m = hits[-1]
    # take the very next non-empty line as grid
    i = inp.find("\n", m.end())
    if i == -1:
        return "K_POINTS automatic\n"
    i += 1
    while i < len(inp):
        j = inp.find("\n", i); j = len(inp) if j == -1 else j
        ln = inp[i:j]
        i = j + 1
        if ln.strip():
            return "K_POINTS automatic\n" + ln.strip() + "\n"
    return "K_POINTS automatic\n"

# ----------------------- Rebuild clean input (only keep &CONTROL) -----------------------

def rebuild_clean_input_keep_control(original_in: str, pos_blk: str, cell_blk: str) -> str:
    """
    Build a minimal, clean QE input containing:
      - &CONTROL (copied verbatim from original .in)
      - &SYSTEM (minimal): ibrav = 0, nat = (# atoms from positions), ntyp = (# species lines)
      - ATOMIC_SPECIES (copied from original .in)
      - ATOMIC_POSITIONS (from final window)
      - CELL_PARAMETERS (from final window)
      - K_POINTS automatic (LAST one from original .in; default to 1 1 1 if missing)
    """
    control = _extract_control_block(original_in)
    species = _extract_species_block(original_in)
    kpts    = _extract_last_kpoints_block(original_in) or "K_POINTS automatic\n1 1 1 0 0 0\n"

    # Count nat from positions; ntyp from species lines (exclude header)
    nat = len([ln for ln in pos_blk.splitlines()[1:] if ln.strip()])
    ntyp = len([ln for ln in species.splitlines()[1:] if ln.strip()])

    system = (
        "&system\n"
        "    ibrav = 0,\n"
        f"    nat = {nat},\n"
        f"    ntyp = {ntyp},\n"
        "/\n"
    )

    parts = [control.rstrip() + "\n", system, species, pos_blk, cell_blk, kpts]
    return "\n".join(part if part.endswith("\n") else part + "\n" for part in parts)

# ----------------------- Public entry & CLI -----------------------

def run_relax_metrics(scf_in_path: str, relax_out_path: str) -> Dict:
    original_in = open(scf_in_path,  "r", encoding="utf-8", errors="ignore").read()
    relax_out   = open(relax_out_path,"r", encoding="utf-8", errors="ignore").read()

    win = _slice_final_window(relax_out) or relax_out
    cell_blk = _extract_cell(win)
    cell_blk = _normalize_cell_to_angstrom(cell_blk)
    pos_blk  = _extract_pos(win)

    cleaned = rebuild_clean_input_keep_control(original_in, pos_blk, cell_blk)

    # Print the whole rebuilt input
    print("===== PATCHED (CLEAN) QE INPUT BEGIN =====")
    print(cleaned, end="" if cleaned.endswith("\n") else "\n")
    print("===== PATCHED (CLEAN) QE INPUT END =====")

    # Metrics on the cleaned string
    a, b, c, alpha, beta, gamma = get_lattice_param_str(cleaned)
    SG, SG_num, PG, CS = get_symm_str(cleaned)

    return {
        "a": a, "b": b, "c": c,
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "space_group": SG,
        "space_group_number": SG_num,
        "point_group": PG,
        "crystal_system": CS,
    }

def run_relax_metrics_input(scf_in_path: str) -> Dict:
    original_in = open(scf_in_path,  "r", encoding="utf-8", errors="ignore").read()

    # Metrics on the original input string
    a, b, c, alpha, beta, gamma = get_lattice_param_str(original_in)
    SG, SG_num, PG, CS = get_symm_str(original_in)

    return {
        "a": a, "b": b, "c": c,
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "space_group": SG,
        "space_group_number": SG_num,
        "point_group": PG,
        "crystal_system": CS,
    }

def _cli():
    if len(sys.argv) != 3:
        print("Usage: python qe_metrics.py <scf.in> <relax.out>")
        sys.exit(1)
    _ = run_relax_metrics(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    _cli()
