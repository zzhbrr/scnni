/*
 * @Author: zzh
 * @Date: 2023-03-09
 * @LastEditTime: 2023-03-10 06:46:16
 * @Description: 
 * @FilePath: /SCNNI/src/layers/relu.cpp
 */
#include "scnni/layer_factory.hpp"
#include "scnni/tensor.hpp"
#include "scnni/layers/relu.hpp"
#include "scnni/macros.h"
#include "scnni/logger.hpp"
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

namespace scnni {
auto ReluLayer::Forward(const std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &output_blobs) const -> int {
    SCNNI_ASSERT(!input_blobs.empty(), "ReluLayer's input blobs empty");
    SCNNI_ASSERT(input_blobs.size() == 1, "ReluLayer has multiple inputs");
    SCNNI_ASSERT(!output_blobs.empty(), "ReluLayer's output blobs empty");
    for (size_t batch = 0; batch < input_blobs[0].size(); batch++) {
        auto input_tensor_shptr = input_blobs[0][batch];
        std::shared_ptr<Tensor<float>> feat = output_blobs[0].at(0);
        // LOG_DEBUG("input_tensor_shptr's row: %d", input_tensor_shptr->Rows());
        for (size_t i = 0; i < input_tensor_shptr->Rows(); i ++) {
            for (size_t j = 0; j < input_tensor_shptr->Cols(); j ++) {
                for (size_t k = 0; k < input_tensor_shptr->Channels(); k ++) {
                    feat->At(k, i, j) = input_tensor_shptr->At(k, i, j) > 0 ? input_tensor_shptr->At(k, i, j) : 0; 
                }
            }
        }
    }
    return 0;
} 
auto GetReluLayer(const std::shared_ptr<Operator> &op) -> Layer* {
    return new ReluLayer();
} 

LayerRegistelrWrapper relu_layer_registe("nn.ReLU", LayerRegister::layer_creator_function(GetReluLayer));

}  // namespace scnni