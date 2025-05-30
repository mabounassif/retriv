import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="retriv",
    version="0.2.3",
    author="Elias Bassani",
    author_email="elias.bssn@gmail.com",
    description="retriv: A Python Search Engine for Humans.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AmenRa/retriv",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
        "multipipe",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: General",
    ],
    keywords=[
        "information retrieval",
        "search engine",
        "bm25",
        "numba",
        "sparse retrieval",
        "dense retrieval",
        "hybrid retrieval",
        "neural information retrieval",
    ],
    python_requires=">=3.8",
    extras_require={
        "sparse": [
            "numpy",
            "numba>=0.54.1",
            "tqdm",
            "orjson",
            "krovetzstemmer",
            "pystemmer==2.0.1",
            "unidecode",
            "scikit-learn",
            "ranx",
            "indxr",
            "oneliner_utils",
            "optuna",
        ],
        "dense": [
            "numpy",
            "torch",
            "torchvision",
            "torchaudio",
            "transformers[torch]",
            "faiss-cpu",
            "autofaiss",
            "multipipe",
        ],
        "hybrid": ["retriv[sparse]", "retriv[dense]"],
        "all": ["retriv[sparse]", "retriv[dense]"],
    },
)
