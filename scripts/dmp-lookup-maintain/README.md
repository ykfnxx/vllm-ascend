# DMP Lookup/Maintain standalone operators

This directory contains `AsuHbmIndexLookup`, `AsuHbmIndexMaintainAicpu`, and
the scheme-4-only `DmpLookupKvGather`. Lookup emits original token positions
for hits and 10K staging slots for misses. Gather writes the fixed misses into
arbitrary staging slots with one invocation.

Revision 9 follows the upstream fixed-workload operator: the final 300 TopK
entries are misses and Maintain performs 300 eviction accesses per request.
The local 144K index and direct `int32` hit/miss sparse outputs are retained.
This is a profiling workload, not a cache-policy accuracy model.

Hit SFA reads the existing full vLLM KV cache. Scheme-4 KVGather copies only
the current 300 misses into the 2K staging region, allowing hit SFA to overlap
the Gather on a separate stream.

The OPP installs below this directory and the Torch extension uses the
`dmp_lookup_maintain` namespace. It does not replace the operator package
bundled with vllm-ascend.

The token index capacity is 144K so a 128K prompt plus decode tokens remains
inside the kernel's fixed index range.
