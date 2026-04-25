"""AST-checked Python sandbox.

Used as one of the MCP tools a task might expose. Validates that submitted
code contains no obviously dangerous constructs before execution, then
executes with restricted builtins.
"""

import ast
from typing import Any, Dict

BLOCKED_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil",
    "open", "__import__", "eval", "exec", "compile",
    "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "__builtins__",
}

BLOCKED_NODES = (ast.Import, ast.ImportFrom)


def static_check(src: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {"ok": False, "reason": f"syntax_error: {e}"}
    for node in ast.walk(tree):
        if isinstance(node, BLOCKED_NODES):
            return {"ok": False, "reason": "imports_blocked"}
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            return {"ok": False, "reason": f"name_blocked: {node.id}"}
        if isinstance(node, ast.Attribute) and (
            isinstance(node.value, ast.Name) and node.value.id in BLOCKED_NAMES
        ):
            return {"ok": False, "reason": f"attr_blocked: {node.value.id}"}
    return {"ok": True}


SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    "range": range, "round": round, "sorted": sorted, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "float": float, "int": int,
    "str": str, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "any": any, "all": all, "print": lambda *a, **kw: None,
}


def sandboxed_exec(src: str) -> Dict[str, Any]:
    check = static_check(src)
    if not check["ok"]:
        return {"ok": False, "error": check["reason"]}
    globs: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    locs: Dict[str, Any] = {}
    try:
        exec(compile(src, "<sandbox>", "exec"), globs, locs)
    except Exception as e:
        return {"ok": False, "error": f"runtime_error: {e}"}
    result = locs.get("result")
    return {"ok": True, "result": result}
