"""
app/tools/calculator.py
Safe arithmetic tool for LLM tool-calling.
Evaluates math expressions without using eval() on arbitrary code.
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Union

_SAFE_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS = {
    "round": round,
    "abs":   abs,
    "sqrt":  math.sqrt,
    "log":   math.log,
    "ceil":  math.ceil,
    "floor": math.floor,
    "min":   min,
    "max":   max,
    "sum":   sum,
}


class UnsafeExpression(ValueError):
    pass


def _eval_node(node: ast.AST) -> Union[int, float]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise UnsafeExpression(f"Non-numeric constant: {node.value}")

    if isinstance(node, ast.BinOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise UnsafeExpression(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise UnsafeExpression(f"Unsupported unary op: {type(node.op).__name__}")
        return op_func(_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpression("Only simple function calls allowed")
        func = _SAFE_FUNCS.get(node.func.id)
        if func is None:
            raise UnsafeExpression(f"Function not allowed: {node.func.id}")
        args = [_eval_node(a) for a in node.args]
        return func(*args)

    if isinstance(node, ast.List):
        return [_eval_node(e) for e in node.elts]

    raise UnsafeExpression(f"Unsupported AST node: {type(node).__name__}")


def calculate(expression: str) -> dict:
    """
    Safely evaluate a math expression string.

    Returns:
        {"result": <number>, "expression": <input>, "error": None}
        or
        {"result": None, "expression": <input>, "error": <message>}

    Example:
        calculate("40 * 180000 * 1.15")  → {"result": 8280000.0, ...}
        calculate("(2.1 - 1) * 250000 * 40")  → {"result": 11000000.0, ...}
    """
    expression = expression.strip()
    try:
        tree   = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return {"result": result, "expression": expression, "error": None}
    except ZeroDivisionError:
        return {"result": None, "expression": expression, "error": "Division by zero"}
    except UnsafeExpression as exc:
        return {"result": None, "expression": expression, "error": f"Unsafe: {exc}"}
    except SyntaxError as exc:
        return {"result": None, "expression": expression, "error": f"Syntax error: {exc}"}
    except Exception as exc:
        return {"result": None, "expression": expression, "error": str(exc)}
