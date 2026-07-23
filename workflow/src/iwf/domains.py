"""Canonical research-domain vocabulary shared by the OpenAlex fetcher's field-balancing
and the LLM domain auto-fill (main_extend_db's llm_fill pass): OpenAlex's standard ~26-field
taxonomy (the same partition ``fetch_domains: primary_topic.field.id:fields/N`` filters
against — see https://api.openalex.org/fields). The LLM auto-fill must pick from this list
verbatim so a filled ``domains`` value is never a domain the rest of the pipeline has never
seen.
"""

from __future__ import annotations

ALLOWED_DOMAINS: tuple[str, ...] = (
    "Agricultural and Biological Sciences",
    "Arts and Humanities",
    "Biochemistry, Genetics and Molecular Biology",
    "Business, Management and Accounting",
    "Chemical Engineering",
    "Chemistry",
    "Computer Science",
    "Decision Sciences",
    "Dentistry",
    "Earth and Planetary Sciences",
    "Economics, Econometrics and Finance",
    "Energy",
    "Engineering",
    "Environmental Science",
    "Health Professions",
    "Immunology and Microbiology",
    "Materials Science",
    "Mathematics",
    "Medicine",
    "Neuroscience",
    "Nursing",
    "Pharmacology, Toxicology and Pharmaceutics",
    "Physics and Astronomy",
    "Psychology",
    "Social Sciences",
    "Veterinary",
)

# OpenAlex's 26 field ids (11..36), one per top-level scientific field — see
# https://api.openalex.org/fields. Used to spread a crawl across every discipline.
OPENALEX_FIELD_IDS: tuple[int, ...] = tuple(range(11, 37))


def all_field_filters() -> list[str]:
    """OpenAlex filter strings, one per scientific field (wide-domain crawl)."""
    return [f"primary_topic.field.id:fields/{i}" for i in OPENALEX_FIELD_IDS]
