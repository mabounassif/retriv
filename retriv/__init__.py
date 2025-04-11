__all__ = [
    "ANN_Searcher",
    "DenseRetriever",
    "Encoder",
    "SearchEngine",
    "SparseRetriever",
    "HybridRetriever",
    "Merger",
]

import os
from pathlib import Path

import pkg_resources

from .merger.merger import Merger

# Set environment variables ----------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if (
    "RETRIV_BASE_PATH" not in os.environ
):  # allow user to set a different path in .bash_profile
    os.environ["RETRIV_BASE_PATH"] = str(Path.home() / ".retriv")


def set_base_path(path: str):
    os.environ["RETRIV_BASE_PATH"] = path


# Check which extras are installed
installed_packages = {pkg.key for pkg in pkg_resources.working_set}
has_sparse = (
    "retriv[sparse]" in installed_packages or "retriv[all]" in installed_packages
)
has_dense = "retriv[dense]" in installed_packages or "retriv[all]" in installed_packages
has_hybrid = (
    "retriv[hybrid]" in installed_packages or "retriv[all]" in installed_packages
)

# Import base classes that are always available
from .base_retriever import BaseRetriever

# Conditionally import flavor-specific classes
if has_sparse:
    from .experimental import AdvancedRetriever
    from .sparse_retriever import SparseRetriever

    # Alias for backward compatibility
    SearchEngine = SparseRetriever

if has_dense:
    from .dense_retriever.ann_searcher import ANN_Searcher
    from .dense_retriever.dense_retriever import DenseRetriever
    from .dense_retriever.encoder import Encoder

if has_hybrid:
    from .hybrid_retriever import HybridRetriever
