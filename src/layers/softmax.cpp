/*
 * @Author: xzj
 * @Date: 2023-03-13 11:18:42
 * @LastEditTime: 2023-03-13 14:05:43
 * @Description: 
 * @FilePath: /scnni/src/layers/softmax.cpp
 */
#include "scnni/layer_factory.hpp"
#include "scnni/tensor.hpp"
#include "scnni/layers/softmax.hpp"
#include "scnni/operator.hpp"
#include "scnni/macros.h"
#include "scnni/logger.hpp"
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>

namespace scnni {
auto SoftmaxLayer::Forward(
    const std::vector<std::vector<std::shared_ptr<Tensor<float>>>> &input_blobs,
    std::vector<std::vector<std::shared_ptr<Tensor<float>>>> &output_blobs) const -> int {
    SCNNI_ASSERT(!input_blobs.empty(), "SoftmaxLayer's input blobs empty");
    SCNNI_ASSERT(input_blobs.size() == 1, "SoftmaxLayer has multiple inputs");
    SCNNI_ASSERT(input_blobs.size() == output_blobs.size(), "SoftmaxLayer: input_blobs are not equal to output_blobs");
    SCNNI_ASSERT(!output_blobs.empty(), "SoftmaxLayer's output blobs empty");
    int dim = dim_;
    // SCNNI_ASSERT(dim == -1, "SoftmaxLayer: dim not implemented");
    for (size_t batch = 0; batch < input_blobs[0].size(); batch++) {
        const auto input_tensor_shptr = input_blobs[0][batch];
        const std::shared_ptr<Tensor<float>> feat = output_blobs[0].at(batch);

        float sum = 0;
        LOG_DEBUG("Softmax: input shape [%d, %d, %d]", input_tensor_shptr->Rows(), input_tensor_shptr->Cols(), input_tensor_shptr->Channels());
        for (size_t i = 0; i < input_tensor_shptr->Rows(); i ++) {
            sum += exp(input_tensor_shptr->At(0, i, 0));
            LOG_DEBUG("%f", input_tensor_shptr->At(0, i, 0));
        }
        std::cout << sum << std::endl;
        for (size_t i = 0; i < input_tensor_shptr->Rows(); i ++) {
            feat->At(0, i, 0) = exp(input_tensor_shptr->At(0, i, 0)) / sum;
        }
    }
    return 0;
}

void SoftmaxLayer::SetDim(int dim) {
    dim_ = dim;
}

auto GetSoftmaxLayer(const std::shared_ptr<Operator> &op) -> Layer* {
    auto* layer = new SoftmaxLayer();
    Parameter p_dim = op->GetParam("dim");
    layer->SetDim(p_dim.GetValueInt());
    return layer;
}

LayerRegistelrWrapper softmax_layer_registe("F.softmax", LayerRegister::layer_creator_function(GetSoftmaxLayer));

}  // namespace scnni