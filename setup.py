from setuptools import find_packages, setup


DEV_REQUIRES = [
    "black",
    "build",
    "cookiecutter-project-upgrader",
    "coverage",
    "ipykernel",
    "ipython",
    "isort",
    "mypy",
    "pytest",
    "ruff",
]


API_REQUIRES = [
    "fastapi",
    "pydantic",
    "uvicorn",
]


NN_REQUIRES = [
    "torch",
]


ALL_REQUIRES = [
    *DEV_REQUIRES,
    *API_REQUIRES,
    *NN_REQUIRES,
]


setup(
    name="alpha-cc",
    install_requires=[
        "maturin",
        "numpy",
        "tqdm-loggable",
    ],
    packages=find_packages(),
    extras_require={
        "api": API_REQUIRES,
        "dev": DEV_REQUIRES,
        "nn": NN_REQUIRES,
        "all": ALL_REQUIRES,
    },
)
