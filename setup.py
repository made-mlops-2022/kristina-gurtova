from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="mlops_project",
    packages=find_packages(),
    version="0.1.0",
    description="Heart cleveland dataset research",
    author="Kristina Gurtova",
    entry_points={
        "console_scripts": [
            "ml_train = ml_project.model_usage.train:train"
        ]
    },
    install_requires=required,
)
