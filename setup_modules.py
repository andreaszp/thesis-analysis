"""
setup_modules.py
----------------
Run this once to create all module stub files.
Run from the root of the repository:
    python setup_modules.py
"""

import os

STUBS = {
    "modules/__init__.py": ("# modules package", None),
    "modules/cleaning.py": ("cleaning.py", "load_and_clean"),
    "modules/formatting.py": ("formatting.py", "write_excel"),
    "modules/sheet_a_correlations.py": ("sheet_a_correlations.py", "run_sheet_a"),
    "modules/sheet_b_tone_effects.py": ("sheet_b_tone_effects.py", "run_sheet_b"),
    "modules/sheet_c_ai_perceptions.py": ("sheet_c_ai_perceptions.py", "run_sheet_c"),
    "modules/sheet_d_feedback_quality.py": ("sheet_d_feedback_quality.py", "run_sheet_d"),
    "modules/sheet_e_chatbot_eval.py": ("sheet_e_chatbot_eval.py", "run_sheet_e"),
    "modules/sheet_f_mediation.py": ("sheet_f_mediation.py", "run_sheet_f"),
    "modules/sheet_g_gpt_scoring.py": ("sheet_g_gpt_scoring.py", "run_sheet_g"),
    "modules/sheet_h_word_freq.py": ("sheet_h_word_freq.py", "run_sheet_h"),
    "modules/sheet_i_demographics.py": ("sheet_i_demographics.py", "run_sheet_i"),
    "modules/sheet_j_toc.py": ("sheet_j_toc.py", "run_sheet_j"),
    "modules/sheet_l_variable_definitions.py": ("sheet_l_variable_definitions.py", "run_sheet_l"),
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

for filepath, (name, func) in STUBS.items():
    if func is None:
        content = name  # special case for __init__.py
    else:
        content = STUB_TEMPLATE.format(name=name, func=func)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"created: {filepath}")

print("\nAll stubs created.")
