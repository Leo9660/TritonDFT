from __future__ import annotations

import json
import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path


TEXT_SUFFIXES = {".in", ".out", ".err", ".log", ".txt", ".json", ".xml", ".dat", ".dos", ".gnu", ".band", ".cif", ".yaml", ".yml"}
SKIP_PARTS = {".git", "__pycache__", "pseudos"}
STOP_WORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "give", "how", "i", "in", "is", "it", "me", "of", "on", "or", "show", "the", "this", "to", "was", "what", "where", "which", "with"}
ALIASES = {
    "gap": ("band gap", "highest occupied", "lowest unoccupied", "homo", "lumo"),
    "fermi": ("fermi energy", "the fermi energy is"),
    "lattice": ("cell_parameters", "lattice parameter", "celldm"),
    "moment": ("magnetic moment", "magnetization", "magn="),
    "raman": ("raman", "raman tensor", "raman cross section"),
    "phonon": ("freq", "frequency", "cm-1", "thz"),
    "energy": ("total energy", "!    total energy", "fermi energy"),
    "distance": ("cell_parameters", "atomic_positions", "lattice parameter"),
    "axis": ("cell_parameters", "atomic_positions"),
    "alpha": ("cell_parameters",),
    "beta": ("cell_parameters",),
    "gamma": ("cell_parameters",),
    "converged": ("convergence has been achieved", "end of self-consistent"),
}


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    path: Path
    relative_path: str
    line_number: int
    line: str
    context: tuple[str, ...]
    score: float

    def prompt_text(self) -> str:
        return f"[{self.evidence_id}] FILE: {self.relative_path}\nLINE: {self.line_number}\nEXACT LINE: {self.line}\nCONTEXT:\n" + "\n".join(self.context)


def _query_terms(question: str) -> tuple[set[str], set[str]]:
    words = {word for word in re.findall(r"[a-z0-9_+.-]+", question.lower()) if len(word) > 1 and word not in STOP_WORDS}
    phrases = {phrase for word in words for phrase in ALIASES.get(word, ())}
    if "band" in words and "gap" in words:
        phrases.add("band gap")
    return words, phrases


def _candidate_files(run_dir: Path):
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(run_dir)
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & SKIP_PARTS or any(part.endswith(".save") for part in lowered_parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "CRASH":
            continue
        try:
            if path.stat().st_size > 12_000_000:
                continue
        except OSError:
            continue
        yield path


def search_workflow_evidence(run_dir: Path, question: str, limit: int = 18) -> list[Evidence]:
    """Return ranked, line-addressable evidence from files under run_dir only."""
    run_dir = run_dir.resolve()
    words, phrases = _query_terms(question)
    if not words and not phrases:
        return []
    matches = []
    for path in _candidate_files(run_dir):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        relative_lower = str(path.relative_to(run_dir)).lower()
        structural_question = bool(words & {"distance", "axis", "alpha", "beta", "gamma", "angle", "lattice", "structure"})
        structural_indices: set[int] = set()
        if structural_question:
            for header_index, header in enumerate(lines):
                header_lower = header.lower()
                if "cell_parameters" in header_lower:
                    structural_indices.update(range(header_index, min(len(lines), header_index + 4)))
                if "atomic_positions" in header_lower:
                    structural_indices.update(range(header_index, min(len(lines), header_index + 80)))
        for index, line in enumerate(lines):
            lowered = line.lower()
            word_hits = sum(word in lowered for word in words)
            phrase_hits = sum(phrase in lowered for phrase in phrases)
            structural_hit = index in structural_indices
            # Within ATOMIC_POSITIONS, retain only the header and species that
            # occur in the question; otherwise large structures swamp ranking.
            if structural_hit and not ("atomic_positions" in lowered or "cell_parameters" in lowered):
                first = lowered.split(maxsplit=1)[0] if lowered.split() else ""
                if index not in structural_indices or (first and first not in words and not re.match(r"^[+-]?\d", first)):
                    structural_hit = index > 0 and "cell_parameters" in lines[index - 1].lower()
            if not word_hits and not phrase_hits and not structural_hit:
                continue
            path_hits = sum(word in relative_lower for word in words)
            source_priority = 0.0
            if path.name == "relaxed_structure.in":
                source_priority += 6.0
            elif path.suffix.lower() == ".out":
                source_priority += 2.0
            if "approved_inputs" in {part.lower() for part in path.relative_to(run_dir).parts}:
                source_priority -= 3.0
            score = word_hits + 2.5 * phrase_hits + 0.3 * path_hits + (2.0 if structural_hit else 0.0) + source_priority
            start, end = max(0, index - 3), min(len(lines), index + 4)
            context = tuple(f"{number + 1}: {lines[number]}" for number in range(start, end))
            matches.append((score, path, index + 1, line, context))
    matches.sort(key=lambda item: (-item[0], str(item[1]), item[2]))
    return [Evidence(f"E{number}", path, str(path.relative_to(run_dir)), line_number, line, context, score)
            for number, (score, path, line_number, line, context) in enumerate(matches[:limit], 1)]


def evidence_prompt(question: str, evidence: list[Evidence]) -> str:
    excerpts = "\n\n".join(item.prompt_text() for item in evidence)
    return f"""Answer a question about one DFT workflow using ONLY the supplied evidence. Do not use outside knowledge. Do not infer a numerical result that is absent. Distinguish directly reported results from interpretation. If evidence is insufficient, say so.
When the requested value is not printed directly but can be obtained by straightforward mathematics from the evidence, you MUST derive it. Supply one arithmetic expression using only numeric constants and these permitted functions: sqrt, acos, asin, atan, cos, sin, tan, abs, min, max, degrees, radians. Use radians internally for trigonometry. For lattice angles use the vector dot-product definition. For periodic fractional-coordinate separations use the minimum-image difference when appropriate.
Return only valid JSON: {{"answer":"concise answer, mentioning that it is calculated when derived", "evidence_ids":["E1"], "confidence":"high|medium|low", "derivation":"formula/explanation", "calculation":{{"expression":"numeric expression or empty", "unit":"eV|angstrom|degree|...", "description":"name of calculated quantity"}}}}
Every evidence ID must directly support the answer. Never invent paths or quotations; the application displays verified source lines.

QUESTION:\n{question}\n\nEVIDENCE:\n{excerpts}\n"""


def parse_evidence_answer(raw: str, available_ids: set[str]) -> dict:
    raw = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    candidate = fenced.group(1) if fenced else raw
    try:
        payload = json.loads(candidate)
    except (TypeError, ValueError):
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise ValueError("The model did not return the required JSON response.")
        payload = json.loads(match.group(0))
    answer = str(payload.get("answer", "")).strip()
    if not answer:
        raise ValueError("The model returned an empty answer.")
    calculation = payload.get("calculation", {})
    if not isinstance(calculation, dict):
        calculation = {}
    return {"answer": answer,
            "evidence_ids": [str(item) for item in payload.get("evidence_ids", []) if str(item) in available_ids],
            "confidence": str(payload.get("confidence", "low")),
            "derivation": str(payload.get("derivation", "")).strip(),
            "calculation": {
                "expression": str(calculation.get("expression", "")).strip(),
                "unit": str(calculation.get("unit", "")).strip(),
                "description": str(calculation.get("description", "calculated value")).strip(),
            }}


_CALC_FUNCTIONS = {
    "sqrt": math.sqrt, "acos": math.acos, "asin": math.asin, "atan": math.atan,
    "cos": math.cos, "sin": math.sin, "tan": math.tan, "abs": abs,
    "min": min, "max": max, "degrees": math.degrees, "radians": math.radians,
}
_CALC_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
                ast.Pow: lambda a, b: a ** b}


def evaluate_calculation(expression: str) -> float:
    """Evaluate a small numeric expression without Python eval or arbitrary names."""
    if not expression or len(expression) > 600:
        raise ValueError("No valid calculation expression was supplied.")
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _CALC_BINOPS:
            return _CALC_BINOPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _CALC_FUNCTIONS and not node.keywords:
            return _CALC_FUNCTIONS[node.func.id](*(visit(argument) for argument in node.args))
        raise ValueError("The proposed calculation contains a disallowed operation.")

    result = float(visit(tree))
    if not math.isfinite(result):
        raise ValueError("The proposed calculation did not produce a finite result.")
    return result


def verify_evidence(item: Evidence) -> bool:
    try:
        lines = item.path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    return 0 < item.line_number <= len(lines) and lines[item.line_number - 1] == item.line
