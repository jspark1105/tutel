# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
from torch import Tensor

from .jit_compiler import tutel_custom_kernel


def ms_all2all(data, group=None):
    tutel_custom_kernel.ms_all2all(
        data if not isinstance(data, Tensor) else [data],
        group if group is not None else dist.group.WORLD,
    )
    return data
