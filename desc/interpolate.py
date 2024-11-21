"""Dummy module with deprecation warning."""

import warnings

import interpax

warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)
warnings.warn(
    "desc.interpolate has been deprecated, please use the interpax package instead",
    DeprecationWarning,
)


def _getattr_deprecated(name):
    warnings.warn(
        "desc.interpolate has been deprecated, please use the interpax package instead",
        DeprecationWarning,
    )
    return getattr(interpax, name)


__getattr__ = _getattr_deprecated
