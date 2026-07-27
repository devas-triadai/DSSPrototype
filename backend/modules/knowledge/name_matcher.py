"""Shared name-matching utilities for knowledge module evidence builders.

Provides bidirectional matching between detection names (e.g. "K9_Vajra")
and knowledge base names (e.g. "K9 Vajra-T") by normalizing underscores
and hyphens, then comparing tokens.
"""

import re


def normalize_name(name: str) -> str:
    """Lowercase and replace underscores/hyphens with spaces."""
    return re.sub(r"[_\-]+", " ", name.lower()).strip()


def extract_tokens(name: str) -> set[str]:
    """Extract significant tokens (>= 2 chars) from a name."""
    normalized = normalize_name(name)
    return {t for t in re.findall(r"[a-z0-9]+", normalized) if len(t) >= 2}


def names_match(obj_name: str, kb_name: str) -> bool:
    """Check if a detection name matches a knowledge base name.

    Bidirectional token matching: tokens from either string must appear
    in the other. Handles mismatches like ``K9_Vajra`` vs ``K9 Vajra-T``
    or ``Baktar_Shikan_ATGM`` vs ``Baktar Shikan ATGM``.
    """
    obj_tokens = extract_tokens(obj_name)
    kb_tokens = extract_tokens(kb_name)

    if not obj_tokens or not kb_tokens:
        return False

    # Require at least one meaningful token overlap
    overlap = obj_tokens & kb_tokens
    # Filter out trivially short tokens from overlap consideration
    meaningful_overlap = {t for t in overlap if len(t) >= 3}

    # If we have meaningful overlap, it's a match
    if meaningful_overlap:
        return True

    # Fallback: check if all tokens of the shorter name appear in the longer
    shorter, longer = (
        (obj_tokens, kb_tokens)
        if len(obj_tokens) <= len(kb_tokens)
        else (kb_tokens, obj_tokens)
    )
    if shorter and shorter.issubset(longer):
        return True

    return False


def is_substring_match(obj_name: str, kb_value: str) -> bool:
    """Check if *kb_value* is a substring of *obj_name* OR vice versa.

    The original code only checked ``kb_value in obj_name`` which fails
    when the KB value is longer than the detection name.  This function
    checks both directions after normalizing underscores/hyphens.
    """
    obj_norm = normalize_name(obj_name)
    kb_norm = normalize_name(kb_value)
    return kb_norm in obj_norm or obj_norm in kb_norm
