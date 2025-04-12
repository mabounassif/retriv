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


# Import base classes that are always available
from .base_retriever import BaseRetriever

try:
    from .experimental import AdvancedRetriever
    from .sparse_retriever.sparse_retriever import SparseRetriever

    # Alias for backward compatibility
    SearchEngine = SparseRetriever
except ImportError:
    pass
try:
    from .dense_retriever.ann_searcher import ANN_Searcher
    from .dense_retriever.dense_retriever import DenseRetriever
    from .dense_retriever.encoder import Encoder

    # Alias for backward compatibility
    SearchEngine = DenseRetriever
except ImportError:
    pass

try:
    from .hybrid_retriever import HybridRetriever
except ImportError:
    pass
