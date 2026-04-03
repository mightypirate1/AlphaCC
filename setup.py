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
    "plotext",
    "plotly-express",
    "pytest",
    "ruff",
    "standard-imghdr",
    #"standard-imwdb",  # local tensorboard needed this after py313 upgrade
    "grpcio-tools",
]


API_REQUIRES = [
    "fastapi",
    "pydantic",
    "uvicorn",
]


ONNX_REQUIRES = [
    "onnxruntime-gpu",
    "onnxscript",
]

ALL_REQUIRES = [
    *DEV_REQUIRES,
    *API_REQUIRES,
    *ONNX_REQUIRES,
]


setup(
    name="alpha-cc",
    install_requires=[
        "apscheduler",
        "click",
        "dill",
        "einops",
        "grpcio",
        "maturin[patchelf]",
        "numpy",
        "protobuf",
        "python-dotenv",
        "redis",
        "rich",
        "scipy",
        "tensorboard",
        "torch",
        "tqdm-loggable",
    ],
    packages=find_packages(),
    extras_require={
        "api": API_REQUIRES,
        "dev": DEV_REQUIRES,
        "onnx": ONNX_REQUIRES,
        "all": ALL_REQUIRES,
    },
    entry_points = {
        "console_scripts": [
            "alphacc-trainer = alpha_cc.entrypoints.trainer_thread:main",
            "alphacc-worker = alpha_cc.entrypoints.worker_thread:main",
            "alphacc-nn-service = alpha_cc.entrypoints.nn_service_thread:main",
            "alphacc-singlethread-training = alpha_cc.entrypoints.singlethread_training:main",
            "alphacc-eval-weights = alpha_cc.entrypoints.eval_weights:main",
            "alphacc-param-schedule-eval = alpha_cc.entrypoints.param_schedule_eval:main",
            "alphacc-db-game-inspection = alpha_cc.entrypoints.db_game_inspection:main",
            "alphacc-tournament = alpha_cc.entrypoints.tournament:main",
        ],
    }
)
