"""
setup_modules.py
----------------
Run once to create all module stub files.
In Colab: !python setup_modules.py
"""

import os

STUBS = {
    "modules/__init__.py":  None,
    "modules/cleaning.py":  "load_and_clean",
    "modules/metrics.py":   "compute_all_metrics",
    "modules/analyses.py":  "run_all_analyses",
    "modules/gpt_scoring.py": "run_gpt_scoring",
    "modules/word_freq.py": "run_word_freq",
    "modules/export.py":    "write_excel",
}

STUB_TEMPLATE = '''"""
{name}
-----------
TODO: implement this module.
"""
import pandas as pd


def {func}(*args, **kwargs) -> pd.DataFrame:
    """Placeholder — returns empty DataFrame until implemented."""
    return pd.DataFrame()
'''

os.makedirs("modules", exist_ok=True)

for filepath, func in STUBS.items():
    if func is None:
        content = "# modules package\n"
    else:
        name = os.path.basename(filepath)
        content = STUB_TEMPLATE.format(name=name, func=func)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"created: {filepath}")

print("\nAll stubs created.")
