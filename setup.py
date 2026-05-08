from setuptools import setup, find_packages

setup(
    name="waymo-unsupervised",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas",
        "pyarrow",
        "fastparquet",
        "numpy",
        "opencv-python",
        "pillow",
        "torch",
        "torchvision",
        "filterpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "scikit-learn",
        "hdbscan",
        "pyyaml",
    ],
)
