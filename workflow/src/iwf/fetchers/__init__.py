"""Data-extension fetchers (plan §4): OpenAlex, UEFISCDI journal seeds, S2 gap-fill.

Every fetcher yields the same normalized paper dict (see fetchers.base.NORMALIZED_KEYS)
and inserts into the workflow-owned extended DB — the professor's store/ is never written.
"""
