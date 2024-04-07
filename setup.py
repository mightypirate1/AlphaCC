from setuptools import find_packages, setup


setup(
    name="alpha-cc",
    install_requires=[
        "maturin",
        "numpy",
        "pydantic",
        "tqdm-loggable",
    ],
    packages=find_packages(),
    extras_require={
        "dev": [
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
    },
)
