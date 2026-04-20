"""Backwards-compatibility shim. ``vcztools.filter`` was renamed to
``vcztools.bcftools_filter`` when the generic
:class:`~vcztools.variant_filter.VariantFilter` protocol was introduced.

New code should import from :mod:`vcztools.bcftools_filter` directly.
This shim will be removed in a future release.
"""

from vcztools.bcftools_filter import *  # noqa: F401, F403
from vcztools.bcftools_filter import (
    BcftoolsFilter,
    ParseError,
)

FilterExpression = BcftoolsFilter

__all__ = ["BcftoolsFilter", "FilterExpression", "ParseError"]
