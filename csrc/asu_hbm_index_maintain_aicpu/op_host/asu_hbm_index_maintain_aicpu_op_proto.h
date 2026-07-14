#ifndef ASU_HBM_INDEX_MAINTAIN_AICPU_OP_PROTO_H
#define ASU_HBM_INDEX_MAINTAIN_AICPU_OP_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
REG_OP(AsuHbmIndexMaintainAicpu)
    .INPUT(index, TensorType({DT_INT32}))
    .INPUT(slot_to_index, TensorType({DT_INT32}))
    .INPUT(free_slots, TensorType({DT_INT32}))
    .INPUT(free_head, TensorType({DT_INT32}))
    .INPUT(req_pool_entries, TensorType({DT_INT32}))
    .INPUT(last_query_slots, TensorType({DT_INT32}))
    .OUTPUT(index_out, TensorType({DT_INT32}))
    .OUTPUT(slot_to_index_out, TensorType({DT_INT32}))
    .OUTPUT(free_slots_out, TensorType({DT_INT32}))
    .OUTPUT(free_head_out, TensorType({DT_INT32}))
    .REQUIRED_ATTR(req_num, Int)
    .REQUIRED_ATTR(seed, Int)
    .OP_END_FACTORY_REG(AsuHbmIndexMaintainAicpu)
}  // namespace ge

#endif  // ASU_HBM_INDEX_MAINTAIN_AICPU_OP_PROTO_H
