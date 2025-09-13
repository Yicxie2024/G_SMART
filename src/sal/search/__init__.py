# src/sal/search/__init__.py

# ---- best-of-n ----
from .best_of_n import best_of_n
from .best_of_n_conf import best_of_n_conf
from .best_of_n_smart import smart_best_of_n

speculative_best_of_n = smart_best_of_n  # 兼容旧名

# ---- beam-search ----
from .beam_search import beam_search
from .beam_search_conf import beam_search_conf
from .beam_search_smart import smart_beam_search
from .beam_search_smart_conf import smart_beam_search_conf

speculative_beam_search = smart_beam_search  # 兼容旧名
speculative_beam_search_conf = smart_beam_search_conf  # 兼容旧名

# 可选：DVTS（存在才导出）
try:
    from .diverse_verifier_tree_search import dvts
except Exception:
    dvts = None

__all__ = [
    "best_of_n",
    "best_of_n_conf",
    "smart_best_of_n",
    "speculative_best_of_n",
    "beam_search",
    "beam_search_conf",
    "smart_beam_search",
    "smart_beam_search_conf",
    "speculative_beam_search",
    "speculative_beam_search_conf",
    "dvts",
]
