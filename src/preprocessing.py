import re

def clean_code(code: str) -> str:
    """Strip comments and normalize whitespace.
    We remove comments because a developer may write things like
    '// fix overflow bug' AFTER finding a bug -- leaving that in would
    let the model cheat off the comment instead of the code."""
    if not isinstance(code, str):
        return ""
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"[ \t]+", " ", code)
    code = re.sub(r"\n\s*\n+", "\n", code)
    return code.strip()

def basic_stats(code: str) -> dict:
    return {
        "num_chars": len(code),
        "num_lines": code.count("\n") + 1,
        "num_braces": code.count("{") + code.count("}"),
        "num_semicolons": code.count(";"),
    }
