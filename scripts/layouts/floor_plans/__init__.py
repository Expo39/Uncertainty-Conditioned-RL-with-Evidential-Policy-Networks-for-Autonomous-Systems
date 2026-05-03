"""
@file __init__.py
@brief Floor plan subpackage. Each module defines one parking lot geometry.
"""

from scripts.layouts.floor_plans.irregular_a import generate as generate_irregular_a
from scripts.layouts.floor_plans.rectangle import generate as generate_rectangle
from scripts.layouts.floor_plans.trapezoid import generate as generate_trapezoid

__all__ = ["generate_rectangle", "generate_trapezoid", "generate_irregular_a"]
