"""
@file setup.py
@brief ROS 2 ament_python package setup for uncertainty_rl_ros2.
"""

import os
from glob import glob

from setuptools import setup

package_name = "uncertainty_rl_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name, f"{package_name}.sensor_relay"],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Antonio Galdes",
    maintainer_email="antoniogaldes2@outlook.com",
    description="EKF covariance extraction for uncertainty-conditioned RL.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "covariance_extractor = uncertainty_rl_ros2.covariance_extractor:main",
            "covariance_monitor = uncertainty_rl_ros2.covariance_extractor:main_monitor",
            "sensor_relay = uncertainty_rl_ros2.sensor_relay.sensor_relay:main",
        ],
    },
)
