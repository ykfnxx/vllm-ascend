#include "register/register.h"

using namespace ge;

namespace domi {
REGISTER_CUSTOM_OP("MockKVSelect")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MockKVSelect")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
