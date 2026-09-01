from setuptools import setup, find_packages

setup(
    name="plumegym-marl",
    version="0.1.0",
    author="Shaurjesh Basu",
    author_email="sbasu@research.edu",
    description="Physics-Informed Multi-Agent RL for Wildfire Perimeter Tracking",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/DaMaker1291/my-researrch-paper",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "torch>=1.10",
    ],
    extras_require={
        "dev": ["matplotlib", "scipy"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    keywords="marl wildfire drone safety reinforcement-learning gaussian-process",
)
