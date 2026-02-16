from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "pandas",
    "pyarrow",
    "tqdm",
    "pyyaml",
    "tabulate",
    "requests",
    "tqdm"
]


setup(
    name="nematode",
    version="0.1.0",
    packages=find_packages(exclude=("tests", "tests.*", "prototypes", "prototypes.*")),
    py_modules=[],
    python_requires=">=3.11",
    install_requires=install_requires,
)
