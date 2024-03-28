# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_openvino_net)

if ("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)

    mmdeploy_add_net(${PROJECT_NAME} openvino_net.cpp)
    target_link_libraries(${PROJECT_NAME} PRIVATE openvino::runtime)
    add_library(mmdeploy::openvino_net ALIAS ${PROJECT_NAME})
else ()
    message(ERROR "'openvino_net' is NOT supported in target devices: ${MMDEPLOY_TARGET_DEVICES}")
endif ()
