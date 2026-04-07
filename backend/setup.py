"""Install ``heartbeat_ai`` into the environment (needed for ``python -m heartbeat_ai.run`` on PaaS)."""

from setuptools import find_packages, setup

setup(
    name="heartbeat_ai",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
)
