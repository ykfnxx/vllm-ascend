/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "register/op_def_registry.h"

namespace ops {
namespace {

void AddInt32Input(OpDef& op, const char* name)
{
    op.Input(name)
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
}

void AddBoolInput(OpDef& op, const char* name)
{
    op.Input(name)
        .ParamType(REQUIRED)
        .DataType({ge::DT_BOOL})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
}

}  // namespace

class DsaSparseLookupUpdate : public OpDef {
public:
    explicit DsaSparseLookupUpdate(const char* name) : OpDef(name)
    {
        AddInt32Input(*this, "tokenToHot");
        AddInt32Input(*this, "hotToToken");
        AddInt32Input(*this, "lruSlots");
        AddInt32Input(*this, "stateSeatEpoch");
        AddInt32Input(*this, "rowToCacheSeat");
        AddInt32Input(*this, "rowSeatEpoch");
        AddInt32Input(*this, "queryPositions");
        AddInt32Input(*this, "queryToRow");
        AddInt32Input(*this, "queryToLane");
        AddBoolInput(*this, "queryValidMask");
        AddInt32Input(*this, "validTopkCounts");
        AddInt32Input(*this, "seqLens");
        AddInt32Input(*this, "topkPositions");
        AddInt32Input(*this, "resolvedHotIndices");
        AddBoolInput(*this, "missMask");
        AddInt32Input(*this, "workspace");

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(DsaSparseLookupUpdate);

}  // namespace ops
