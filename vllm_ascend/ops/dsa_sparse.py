# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""DSA Sparse operator integration boundary.

The framework currently depends only on ``DSASparseLookupOperator``. The new
fused SIMT implementation will be connected here in the operator milestone.
No existing ``torch.ops`` lookup ABI is used by the framework adaptation.
"""
