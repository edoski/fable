"""Context-free typed loading for named config groups."""

from __future__ import annotations

from typing import cast

from .group_catalog import ConfigGroup, GroupSpec, group_spec
from .models import ChainSpec

CHAIN = cast(GroupSpec[ChainSpec], group_spec(ConfigGroup.CHAIN))
