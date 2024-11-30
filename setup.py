from setuptools import setup, find_packages

setup(
    name="iris_ml_package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "joblib",
    ],
)
