# Path: tests/integration/test_notebooks.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

import json
from pathlib import Path


def test_notebooks_are_valid_and_do_not_import_legacy_src_package() -> None:
    notebooks = list(Path("notebooks").glob("*.ipynb"))
    assert notebooks
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook.get("cells")
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        assert "from src." not in source
        assert "import src." not in source
