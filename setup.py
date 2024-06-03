#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="chattts-fork",
    description="fork from https://github.com/2noise/ChatTTS to PYPI",
    version="0.0.6",
    license="# Attribution-NonCommercial-NoDerivatives 4.0 International",
    author="lich99 CH.Li, yihong0618",
    author_email="zouzou0208@gmail.com",
    url="https://github.com/yihong0618/ChatTTS",
    packages=find_packages(),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "omegaconf~=2.3.0",
        "torch~=2.0",
        "tqdm",
        "einops",
        "vector_quantize_pytorch",
        "transformers~=4.41.1",
        "vocos",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": ["chattts = ChatTTS.cli:main"],
    },
)
