from setuptools import setup, find_packages

setup(
    name="cellpaintmono",
    version="0.1.0",
    author="Diya Srivastava",
    description="A library for analyzing and comparing Cell Painting data with MONO predictions",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=1.5.1",
        "numpy>=1.23.4",
        "matplotlib>=3.6.2",
        "seaborn>=0.12.1",
        "scipy>=1.9.3",
        "scikit-learn>=1.1.3",
        "umap-learn>=0.5.3",
        "tqdm>=4.64.1",
        "plotly>=5.11.0",
        "pycytominer",
    ],
)
