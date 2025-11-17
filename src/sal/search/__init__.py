# src/sal/search/__init__.py

# --- best-of-n 系列 ---
try:
    from .best_of_n import best_of_n as speculative_best_of_n  # 兼容旧名
except Exception:
    pass
try:
    from .best_of_n import best_of_n
except Exception:
    pass
try:
    from .best_of_n_conf import best_of_n_conf
except Exception:
    pass
try:
    from .best_of_n_smart import smart_best_of_n as best_of_n_smart
except Exception:
    pass

# --- beam-search 系列 ---
try:
    from .beam_search import beam_search
except Exception:
    pass
try:
    from .beam_search_conf import beam_search_conf
except Exception:
    pass
try:
    from .beam_search_smart import smart_beam_search as beam_search_smart
except Exception:
    pass
try:
    from .beam_search_smart_conf import smart_beam_search_conf as beam_search_smart_conf
except Exception:
    pass
try:
    from .beam_search_smart_random_score import smart_beam_search_random_score as beam_search_smart_random_score
except Exception:
    pass
try:
    from .beam_search_smart_random_score import split_dataset_by_thresholds
except Exception:
    pass
try:
    from .beam_search_smart_conf_multi_threshold import smart_beam_search_conf_multi_threshold as beam_search_smart_conf_multi_threshold
except Exception:
    pass
try:
    from .beam_search_smart_conf_multi_threshold import split_dataset_by_thresholds as split_dataset_by_uq_thresholds
except Exception:
    pass
try:
    from .beam_search_smart_cocoa import smart_beam_search_cocoa as beam_search_smart_cocoa
except Exception:
    pass
try:
    from .beam_search_smart_cocoa_default import smart_beam_search_cocoa_default as beam_search_smart_cocoa_default
except Exception:
    pass
try:
    from .beam_search_smart_cocoa_multi_threshold import smart_beam_search_cocoa_multi_threshold as beam_search_smart_cocoa_multi_threshold
except Exception:
    pass
try:
    from .beam_search_smart_uhead import smart_beam_search_uhead as beam_search_smart_uhead
except Exception:
    pass
try:
    from .beam_search_slm_only import smart_beam_search_slm_only as beam_search_slm_only
except Exception:
    pass
try:
    from .beam_search_llm_only import smart_beam_search_llm_only as beam_search_llm_only
except Exception:
    pass
try:
    from .beam_search_smart_prm_only import smart_beam_search as beam_search_smart_prm_only
except Exception:
    pass
# --- utils ---
try:
    from .utils import *
except Exception:
    pass

__all__ = [name for name in globals().keys() if not name.startswith("_")]
