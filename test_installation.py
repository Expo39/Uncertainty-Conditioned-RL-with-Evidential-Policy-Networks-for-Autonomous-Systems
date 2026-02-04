#!/usr/bin/env python3
"""Test installation and verify all components are working.

This script performs basic sanity checks to ensure the package is
correctly installed and all dependencies are available.
"""
import sys
import importlib
from typing import List, Tuple


def check_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Check if a module can be imported.
    
    Args:
        module_name: Name of module to import.
        package_name: Optional friendly package name.
        
    Returns:
        Tuple of (success, message).
    """
    if package_name is None:
        package_name = module_name
        
    try:
        importlib.import_module(module_name)
        return True, f"✓ {package_name}"
    except ImportError as e:
        return False, f"✗ {package_name}: {str(e)}"


def main():
    """Run installation tests."""
    print("=" * 60)
    print("Testing Uncertainty-Conditioned RL Installation")
    print("=" * 60)
    
    # Core dependencies
    print("\n1. Checking core dependencies...")
    core_deps = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("gymnasium", "Gymnasium"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("yaml", "PyYAML"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
    ]
    
    core_results = []
    for module, name in core_deps:
        success, msg = check_import(module, name)
        core_results.append(success)
        print(f"  {msg}")
    
    # Optional dependencies
    print("\n2. Checking optional dependencies...")
    optional_deps = [
        ("carla", "CARLA"),
        ("rclpy", "ROS 2 Python"),
        ("sensor_msgs", "ROS 2 Sensor Messages"),
    ]
    
    optional_results = []
    for module, name in optional_deps:
        success, msg = check_import(module, name)
        optional_results.append(success)
        if success:
            print(f"  {msg}")
        else:
            print(f"  {msg} (optional)")
    
    # Package modules
    print("\n3. Checking package modules...")
    sys.path.insert(0, 'src')
    
    package_modules = [
        ("uncertainty_rl.networks.evidential_policy", "Evidential Policy Network"),
        ("uncertainty_rl.envs.carla_parking", "CARLA Parking Environment"),
        ("uncertainty_rl.utils.logging", "Logging Utilities"),
        ("uncertainty_rl.utils.visualisation", "Visualisation Utilities"),
    ]
    
    package_results = []
    for module, name in package_modules:
        success, msg = check_import(module, name)
        package_results.append(success)
        print(f"  {msg}")
    
    # Configuration files
    print("\n4. Checking configuration files...")
    import os
    config_files = [
        "configs/train_config.yaml",
        "configs/eval_config.yaml",
        "configs/ros2_config.yaml",
    ]
    
    config_results = []
    for config_file in config_files:
        if os.path.exists(config_file):
            config_results.append(True)
            print(f"  ✓ {config_file}")
        else:
            config_results.append(False)
            print(f"  ✗ {config_file} not found")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    core_passed = sum(core_results)
    core_total = len(core_results)
    print(f"Core dependencies: {core_passed}/{core_total} passed")
    
    optional_passed = sum(optional_results)
    optional_total = len(optional_results)
    print(f"Optional dependencies: {optional_passed}/{optional_total} available")
    
    package_passed = sum(package_results)
    package_total = len(package_results)
    print(f"Package modules: {package_passed}/{package_total} passed")
    
    config_passed = sum(config_results)
    config_total = len(config_results)
    print(f"Configuration files: {config_passed}/{config_total} found")
    
    # Overall status
    print("\n" + "=" * 60)
    if core_passed == core_total and package_passed == package_total and config_passed == config_total:
        print("✓ Installation test PASSED!")
        print("\nReady to use! Try running:")
        print("  python examples/evidential_network_demo.py")
        return 0
    else:
        print("✗ Installation test FAILED")
        print("\nSome components are missing. Please check the installation guide.")
        if core_passed < core_total:
            print("\nInstall missing dependencies with:")
            print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
