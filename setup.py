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
    "nbformat>4.2.0",
    "pandas",
    "plotly-express",
    "pytest",
    "ruff",
]


API_REQUIRES = [
    "fastapi",
    "pydantic",
    "uvicorn",
]


ALL_REQUIRES = [
    *DEV_REQUIRES,
    *API_REQUIRES,
]


setup(
    name="alpha-cc",
    install_requires=[
        "apscheduler",
        "click",
        "dill",
        "einops",
        "maturin",
        "numpy",
        "python-dotenv",
        "redis",
        "scipy",
        "tensorboard",
        "torch",
        "tqdm-loggable",
    ],
    packages=find_packages(),
    extras_require={
        "api": API_REQUIRES,
        "dev": DEV_REQUIRES,
        "all": ALL_REQUIRES,
    },
    entry_points = {
        "console_scripts": [
            "alphacc-trainer = alpha_cc.entrypoints.trainer_thread:main",
            "alphacc-worker = alpha_cc.entrypoints.worker_thread:main",
            "alphacc-nn-service = alpha_cc.entrypoints.nn_service_thread:main",
            "alphacc-singlethread-training = alpha_cc.entrypoints.singlethread_training:main",
            "alphacc-eval-weights = alpha_cc.entrypoints.eval_weights:main",
        ],
    }
)
