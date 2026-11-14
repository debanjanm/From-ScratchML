"""Re-run the notebooks and report which ones pass.

    python run_notebooks.py                      # run all 42 (slow: some take a minute or more), change nothing
    python run_notebooks.py 15_decision_tree 34_pca   # only some (folder names or prefixes like 15)
    python run_notebooks.py --save               # also write the fresh outputs back into the notebooks

A notebook fails if any cell raises, and the notebooks assert correctness (tests and the comparison
with scikit-learn), so a failure means an algorithm broke. Exit code is 1 if any notebook failed."""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).parent / "algorithms"


def main(args):
    save = "--save" in args
    wanted = [a for a in args if not a.startswith("--")]
    folders = sorted(p for p in ROOT.iterdir() if p.is_dir() and (not wanted or any(p.name.startswith(w) for w in wanted)))
    if not folders:
        sys.exit(f"no folder matches {wanted}")
    failed = []
    for folder in folders:
        path = folder / "notebook.ipynb"
        start = time.time()
        nb = nbformat.read(path, as_version=4)
        try:
            NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(folder)}}).execute()
            status = "PASS"
            if save:
                nbformat.write(nb, path)
        except Exception as e:  # a cell raised
            status = "FAIL"
            failed.append(folder.name)
            print(f"   {type(e).__name__}: {str(e).strip().splitlines()[-1] if str(e).strip() else ''}")
        print(f"{status}  {folder.name:34s} {time.time() - start:5.1f}s", flush=True)
    print(f"\n{len(folders) - len(failed)}/{len(folders)} notebooks passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
