from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk


def run_monitor(run_dir: Path) -> None:
    root = tk.Tk()
    root.title(f"TritonDFT workflow — {run_dir.name}")
    root.geometry("1000x620")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    status_frame = ttk.Frame(notebook)
    tree = ttk.Treeview(
        status_frame,
        columns=("id", "task", "branch", "state", "attempt", "jobs"),
        show="headings",
    )
    for column, width in (("id", 45), ("task", 280), ("branch", 150), ("state", 110), ("attempt", 80), ("jobs", 180)):
        tree.heading(column, text=column.title())
        tree.column(column, width=width, anchor="w")
    tree.pack(fill="both", expand=True)
    status_label = ttk.Label(status_frame, text="Waiting for workflow state…")
    status_label.pack(fill="x", pady=6)
    error_label = ttk.Label(status_frame, text="", wraplength=940, justify="left")
    error_label.pack(fill="x", pady=(0, 6))
    notebook.add(status_frame, text="Execution status")

    validation = tk.Text(notebook, wrap="word", font=("Menlo", 11))
    validation.configure(state="disabled")
    notebook.add(validation, text="Validation")

    def refresh() -> None:
        try:
            state = json.loads((run_dir / "workflow_state.json").read_text(encoding="utf-8"))
            for item in tree.get_children():
                tree.delete(item)
            for step in state.get("steps", []):
                tree.insert("", "end", values=(
                    step.get("id"), step.get("problem"), step.get("branch"),
                    step.get("status"), step.get("attempts", 0),
                    ", ".join(step.get("job_ids", [])),
                ))
            status_label.configure(text=f"Workflow: {state.get('status', 'unknown')}   Updated: {state.get('updated_at', '')}")
            waiting = next(
                (step for step in state.get("steps", []) if step.get("status") == "awaiting_user"),
                None,
            )
            if waiting:
                error_label.configure(
                    text=(
                        f"Terminal awaiting recovery instruction for step {waiting.get('id')}.\n"
                        f"Latest error: {waiting.get('last_error', '')}"
                    )
                )
            else:
                error_label.configure(text="")
        except (OSError, ValueError):
            pass
        report_path = run_dir / "validation_report.txt"
        report = report_path.read_text(encoding="utf-8", errors="replace") if report_path.is_file() else "No validation report yet."
        validation.configure(state="normal")
        validation.delete("1.0", "end")
        validation.insert("1.0", report)
        validation.configure(state="disabled")
        root.after(2000, refresh)

    refresh()
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep a live TritonDFT status and validation window open.")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_monitor(Path(args.run_dir).expanduser().resolve())


if __name__ == "__main__":
    main()
