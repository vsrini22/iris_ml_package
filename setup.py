from setuptools import setup, find_packages

setup(
    name='iris_ml_package',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn'
    ],
    description='A machine learning package for the Iris dataset.',
    author='Your Name',
    author_email='your_email@example.com',
)
