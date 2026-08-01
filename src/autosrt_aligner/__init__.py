"""Forced-alignment subtitle generator package."""

from .pipeline_v2 import run_alignment_job_v2 as run_alignment_job

__all__ = ["run_alignment_job"]
