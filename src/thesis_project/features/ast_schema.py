"""Fixed, compact AST syntax schema shared by extraction and the encoder."""

import inspect
import javalang.tree as tree
from javalang.ast import Node

NODE_TYPES = sorted(name for name, cls in vars(tree).items()
                    if inspect.isclass(cls) and issubclass(cls, Node) and cls is not Node)
ROLES = ["ROOT", "UNKNOWN"] + sorted({attr for name in NODE_TYPES
                                       for attr in getattr(tree, name).attrs})
OPERATORS = ["NONE", "UNKNOWN", "=", "+", "-", "*", "/", "%", "==", "!=",
             "<", "<=", ">", ">=", "&&", "||", "&", "|", "^", "<<", ">>", ">>>",
             "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", ">>>=",
             "instanceof"]
LITERALS = ["NONE", "NULL", "BOOLEAN", "ZERO", "ONE", "INTEGER", "FLOAT",
            "STRING", "CHAR", "UNKNOWN"]
UNARY = ["+", "-", "!", "~", "++", "--"]
# depth, degree, identifier flag, three categorical IDs, two order values,
# then prefix and postfix counts for the six unary operators.
FEATURE_DIM = 8 + 2 * len(UNARY)
FEATURE_NAMES = ["depth", "out_degree", "has_identifier", "operator_id", "literal_id",
                 "child_role_id", "log_sibling_index", "relative_sibling_index"] + [
    f"{side}_{op}_log_count" for side in ("prefix", "postfix") for op in UNARY
]
CATEGORY_SIZES = (len(OPERATORS), len(LITERALS), len(ROLES))


def literal_category(value: str) -> str:
    if value == "null":
        return "NULL"
    if value in {"true", "false"}:
        return "BOOLEAN"
    if value.startswith('"'):
        return "STRING"
    if value.startswith("'"):
        return "CHAR"
    text = value.replace("_", "").lower()
    # Hex digits e/f/d are not decimal exponent/suffix markers.
    if "." in text or ("p" in text if text.startswith("0x") else any(c in text for c in "efd")):
        return "FLOAT"
    try:
        text = text.removesuffix("l")
        base = 16 if text.startswith("0x") else 2 if text.startswith("0b") else 8 if len(text) > 1 and text.startswith("0") else 10
        number = int(text, base)
        return "ZERO" if number == 0 else "ONE" if number == 1 else "INTEGER"
    except ValueError:
        return "UNKNOWN"
