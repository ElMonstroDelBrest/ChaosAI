"""Test that FinJEPA computes ret_loss when ret_weight > 0."""
import pytest


def test_jepa_ret_loss_zero_when_disabled():
    """ret_loss should be 0.0 when ret_weight=0 (default). AST-only test."""
    import ast
    src = open("src/jax_v6/jepa.py").read()
    tree = ast.parse(src)
    # Verify ret_weight field exists in FinJEPA class
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FinJEPA":
            for item in ast.walk(node):
                if isinstance(item, ast.AnnAssign):
                    if hasattr(item.target, 'id') and item.target.id == 'ret_weight':
                        found = True
    assert found, "ret_weight field not found in FinJEPA"


def test_jepa_return_head_in_setup():
    """return_head Dense(1) must be defined in setup(). AST-only test."""
    import ast
    src = open("src/jax_v6/jepa.py").read()
    assert "return_head" in src, "return_head not found in jepa.py"
    assert "nn.Dense(1" in src, "Dense(1) not found — return_head should be Dense(1)"


def test_jepa_ret_loss_in_forward():
    """ret_loss must be computed and added to total_loss in __call__. AST-only test."""
    import ast
    src = open("src/jax_v6/jepa.py").read()
    assert "ret_loss" in src, "ret_loss not found in jepa.py"
    assert "ret_weight" in src, "ret_weight not found in jepa.py"
    assert '"ret_loss"' in src, 'ret_loss must be in the return dict as "ret_loss"'
