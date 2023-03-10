/*
 * @Author: zzh
 * @Date: 2023-03-06
 * @LastEditTime: 2023-03-09 12:39:47
 * @Description: 
 * @FilePath: /SCNNI/src/layers/output.cpp
 */
#include "scnni/layer.hpp"
#include "scnni/tensor.hpp"
#include "scnni/layer_factory.hpp"
#include "scnni/layers/output.hpp"
#include "scnni/macros.h"
#include <memory>
#include <vector>

namespace scnni {
auto OutputLayer::Forward(const std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &output_blobs) const -> int {
    SCNNI_ASSERT(output_blobs.empty(), "OututLayer has output blobs");
    return 0;
}
auto GetOutputLayer(const std::shared_ptr<Operator> &op) -> Layer* {
    return new OutputLayer();
} 
LayerRegistelrWrapper output_layer_registe("pnnx.Output", LayerRegister::layer_creator_function(GetOutputLayer));
}  // namespace scnni