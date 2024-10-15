"""Setup script for module

To install, use

    python -m pip install .

or, for an editable install,

    python -m pip install --editable .

"""

from setuptools import setup

long_description = """
Hybridisable DG discretisation of the incompressible Euler equations.

Implement several timestepping methods based on the DG discretisations described
in http://arxiv.org/abs/2410.09790.

For further details and instructions on how to use this code see README.md.
"""

# Extract requirements from requirements.txt file
with open("requirements.txt", "r", encoding="utf8") as f:
    requirements = [line.strip() for line in f.readlines()]

# Run setup
setup(
    name="incompressible_euler",
    author="Eike Mueller",
    author_email="e.mueller@bath.ac.uk",
    description="Hybridisable DG discretisation of the incompressible Euler equations",
    long_description=long_description,
    version="1.0.0",
    install_requires=[
        'importlib-metadata; python_version == "3.8"',
    ]
    + requirements,
    url="https://github.com/eikehmueller/IncompressibleEulerHDG",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
