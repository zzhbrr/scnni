/*
 * @Author: zzh
 * @Date: 2023-03-13 12:28:42
 * @LastEditTime: 2023-03-13 12:32:55
 * @Description: 
 * @FilePath: /SCNNI/src/layers/linear.cpp
 */
#include "scnni/layers/linear.hpp"
#include "scnni/layer_factory.hpp"
#include "scnni/operator.hpp"
#include "scnni/macros.h"
#include "scnni/logger.hpp"
#include <Eigen/src/Core/util/Constants.h>
#include <bits/stdint-uintn.h>
#include <cfloat>
#include <iostream>

namespace scnni {
auto LinearLayer::Forward(const std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &output_blobs) const -> int {
    SCNNI_ASSERT(!input_blobs.empty(), "Maxpool2dLayer's input blobs empty");
    SCNNI_ASSERT(input_blobs.size() == 1, "Maxpool2dLayer has multiple inputs");
    SCNNI_ASSERT(!output_blobs.empty(), "Maxpool2dLayer's output blobs empty");
    // LOG_DEBUG("FlattenLayer forward: start_dim: %d, end_dim: %d", start_dim_, end_dim_);
    for (size_t batch = 0; batch < input_blobs[0].size(); batch++) {
        const auto input_tensor_shptr = input_blobs[0][batch];
        const std::shared_ptr<Tensor<float>> feat = output_blobs[0].at(0);
        
        auto in_shape = input_tensor_shptr->Shapes();
    }
    
    return 0;
}
void LinearLayer::SetBias(bool bias) {
    bias_ = bias;
}
void LinearLayer::SetInFetures(int in_features) {
    in_features_ = in_features;
}
void LinearLayer::SetOutFeatures(int out_features) {
    out_features_ = out_features;
}
auto GetLinearLayer(const std::shared_ptr<Operator> &op) -> Layer* {
    auto* layer = new LinearLayer();
    Parameter p = op->GetParam("bias");
    layer->SetBias(p.GetValueBool());
    p = op->GetParam("in_features");
    layer->SetInFetures(p.GetValueInt());
    p = op->GetParam("out_features");
    layer->SetOutFeatures(p.GetValueInt());
    return layer;
} 

LayerRegistelrWrapper linear_layer_registe("nn.Linear", LayerRegister::layer_creator_function(GetLinearLayer));

}  // namespace scnni