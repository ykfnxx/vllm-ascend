target_sources(opapi PRIVATE
        aclnn_asu_hbm_index_maintain_aicpu.cpp
        l0_asu_hbm_index_maintain_aicpu.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
            aclnn_asu_hbm_index_maintain_aicpu.cpp
            l0_asu_hbm_index_maintain_aicpu.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
            aclnn_asu_hbm_index_maintain_aicpu.cpp
            l0_asu_hbm_index_maintain_aicpu.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        asu_hbm_index_maintain_aicpu_op_proto.cpp
        asu_hbm_index_maintain_aicpu_proto.cpp
)

install(FILES aclnn_asu_hbm_index_maintain_aicpu.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
