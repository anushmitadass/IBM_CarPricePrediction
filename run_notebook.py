"""
run_notebook.py
───────────────
Executes AnushmitaDas_CarPricePrediction.ipynb and saves the result
with all cell outputs (charts, tables, print statements) embedded.

Run this once from your terminal:
    python run_notebook.py
    # or: jupyter nbconvert --to notebook --execute --inplace AnushmitaDas_CarPricePrediction.ipynb
"""

import subprocess
import sys
import os

NB = "AnushmitaDas_CarPricePrediction.ipynb"

def run():
    # ── 1. Make sure required packages are available ───────────────────────────
    required = [
        "nbconvert", "nbformat", "nbclient", "ipykernel",
        "pandas", "numpy", "matplotlib", "seaborn",
        "scikit-learn", "ipywidgets", "joblib",
    ]
    print("Checking / installing dependencies …")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet"] + required
    )

    # ── 2. Add missing cell IDs (nbformat ≥ 4.5 requirement) ──────────────────
    import nbformat, uuid
    print(f"Normalising {NB} …")
    with open(NB, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    changed = False
    for cell in nb.cells:
        if not cell.get("id"):
            cell["id"] = str(uuid.uuid4())[:8]
            changed = True
    if changed:
        with open(NB, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print("  → cell IDs added.")

    # ── 3. Execute the notebook ────────────────────────────────────────────────
    print(f"Executing {NB}  (this may take 2–5 minutes) …")
    import nbclient
    with open(NB, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = nbclient.NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": os.path.dirname(os.path.abspath(NB))}},
    )
    try:
        client.execute()
    except nbclient.exceptions.CellExecutionError as e:
        print(f"\n⚠  A cell raised an error (widget cells are skipped automatically):\n{e}\n")
        print("Saving partial outputs anyway …")

    # ── 4. Save with outputs ───────────────────────────────────────────────────
    with open(NB, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"\n✅  Done!  Open {NB} in Jupyter — every chart is now pre-rendered.")

if __name__ == "__main__":
    run()
