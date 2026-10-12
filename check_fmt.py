"""Catch format strings split across consecutive print calls.

The standard check tests each % BinOp alone. It cannot see a print whose
literal carries a %d while the % operator lives on the NEXT print. That is
the exact bug this file had, and the handover records it as a known failure
shape, so it gets its own checker.
"""
import ast, re, sys

SPEC = re.compile(r'%(?!%)[-+#0 ]*[0-9.*]*[hlL]?[diouxXeEfFgGcrsa]')

def check(path):
    tree = ast.parse(open(path).read())
    bad = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for stmt in body:
            if not (isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and getattr(stmt.value.func, "id", None) == "print"):
                continue
            for arg in stmt.value.args:
                # a bare string literal that carries a conversion spec and
                # is NOT the left side of a % operator
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if SPEC.search(arg.value):
                        bad.append((stmt.lineno, arg.value[:60]))
    return bad

ok = True
for p in sys.argv[1:]:
    bad = check(p)
    print(f"{p}: {'CLEAN' if not bad else 'PROBLEMS'}")
    for ln, txt in bad:
        ok = False
        print(f"   line {ln}: unformatted spec in a bare print -> {txt!r}")
sys.exit(0 if ok else 1)
