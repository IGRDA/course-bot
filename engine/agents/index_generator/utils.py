import math


def get_module_count(total_pages: int) -> int:
    """Determine number of modules using smooth logarithmic scaling."""
    if total_pages <= 0:
        raise ValueError("total_pages must be a positive integer")

    log_pages = math.log10(total_pages)
    n_modules = 2 + (log_pages * 2)
    return max(2, min(12, round(n_modules)))


def compute_layout(total_pages: int) -> tuple[int, int, int]:
    """
    Compute course layout: (n_modules, n_submodules, n_sections)
    Uses logarithmic scaling so density grows smoothly across course sizes.
    """
    if total_pages <= 0:
        raise ValueError("total_pages must be a positive integer")

    n_modules = get_module_count(total_pages)

    pages_per_module = total_pages / n_modules
    submodule_scale = math.log10(max(pages_per_module, 1))
    n_submodules = 2 + (submodule_scale * 1.8)
    n_submodules = max(2, min(8, round(n_submodules)))

    pages_per_submodule = pages_per_module / n_submodules
    n_sections = max(2, round(pages_per_submodule / 2))

    return n_modules, n_submodules, n_sections