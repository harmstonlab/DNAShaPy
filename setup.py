from setuptools import setup, find_packages
import re
def get_version():
    with open("DNAShaPy/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return re.match(r'__version__\s*=\s*"(.+)"', line).group(1)
    raise RuntimeError("Version not found")

setup(
    name="DNAShaPy",
    version=get_version(),
    description="A tool for predicting DNA shape features from sequence using pre-computed lookup tables from DeepDNAShape.",
    author="Kieran Howard",
    author_email="howardkj1@cardiff.ac.uk",
    url="",
    packages=find_packages(),
    install_requires=[
        "biopython>=1.85",
        "fastparquet>=2024.11.0",
        "numba>=0.61.2",
        "numpy>=2.2.6",
        "pandas>=2.3.2",
        "pyarrow>=21.0.0",
        "tqdm>=4.67.1",
        "pooch>=1.8.0",
    ],
    entry_points={
        "console_scripts": ["DNAShaPy=DNAShaPy.cli:main"],
    },
)
