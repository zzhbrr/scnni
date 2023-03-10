/*
 * @Author: xzj
 * @Date: 2023-03-06 12:11:49
 * @LastEditTime: 2023-03-09 12:44:24
 * @Description: 
 * @FilePath: /SCNNI/src/layers/input.cpp
 */


#include "scnni/tensor.hpp"
#include "scnni/layer_factory.hpp"
#include "scnni/layers/input.hpp"
#include "scnni/macros.h"
#include <memory>
#include <vector>

namespace scnni {
auto InputLayer::Forward(const std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &output_blobs) const -> int {
    SCNNI_ASSERT(input_blobs.empty(), "InputLayer has input blobs");
    return 0;
} 
auto GetInputLayer(const std::shared_ptr<Operator> &op) -> Layer* {
    return new InputLayer();
}

LayerRegistelrWrapper input_layer_registe("pnnx.Input", LayerRegister::layer_creator_function(GetInputLayer));
}  // namespace scnni
