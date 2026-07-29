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

void AddInt32Output(OpDef& op, const char* name)
{
    op.Output(name)
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
}

}  // namespace

class DsaSparseLookupUpdate : public OpDef {
public:
    explicit DsaSparseLookupUpdate(const char* name) : OpDef(name)
    {
        AddInt32Input(*this, "index");
        AddInt32Input(*this, "slotToIndex");
        AddInt32Input(*this, "freeSlots");
        AddInt32Input(*this, "freeHead");
        AddInt32Input(*this, "reqPoolEntries");
        AddInt32Input(*this, "queryIndex");
        AddInt32Input(*this, "lookupMask");
        AddInt32Output(*this, "slotOut");
        AddInt32Output(*this, "missOut");
        this->Attr("req_num").AttrType(REQUIRED).Int();

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(DsaSparseLookupUpdate);

}  // namespace ops
