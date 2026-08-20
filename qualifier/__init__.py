"""Company intent-qualification engine: hard-filter gates + NAICS/keyword
scoring + local embedding similarity, with no per-company LLM calls."""

from .pipeline import QualificationPipeline
from .models import Company

__all__ = ["QualificationPipeline", "Company"]
