/*
 * @Author: zzh
 * @Date: 2023-03-13 12:28:42
 * @LastEditTime: 2023-03-15 07:39:53
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
#include <memory>
#include <vector>

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

        if (in_shape[0] == 1 && in_shape[2] == 1) { // channel = 0, cols = 0, only has one dimension
            SCNNI_ASSERT(feat->RawShapes().size() == 1, "LinearLayer: input tensor has one dimension, but output tensor has more than one dimensions");
            for (uint32_t k = 0; k < feat->Rows(); k ++) {
                feat->At(0, k, 0) = 0;
                for (uint32_t i = 0; i < in_shape[1]; i ++) {
                  feat->At(0, k, 0) +=
                      weights_->At(0, k, i) * input_tensor_shptr->At(0, i, 0);
                }
            }
            if (bias_) {
              for (uint32_t k = 0; k < feat->Rows(); k++) {
                feat->At(0, k, 0) += bias_v_->At(0, k, 0);
              }
            }
        } else {
            UNREACHABLE("LinearLayer input has more than one dimensions, but not implemented");
        }
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
void LinearLayer::SetWeights(const Attribute &att) {
    SCNNI_ASSERT(att.shape_.size() == 2, "Linear: weight att shape != 2");
    std::vector<float> weights = att.Get();
    this->weights_ = std::make_shared<Tensor<float>>(1, att.shape_[0], att.shape_[1]);
    // std::cout << "Linear weight:" << std::endl;
    for (int i = 0; i < att.shape_[0]; i ++) {
        for (int j = 0; j < att.shape_[1]; j ++) {
            this->weights_->At(0, i, j) = weights.at(j + i * att.shape_[1]);
            // std::cout << this->weights_->At(0, i, j) << " ";
        }
        // std::cout << std::endl;
    }
}
void LinearLayer::SetBiasValue(const Attribute &att) {
    SCNNI_ASSERT(att.shape_.size() == 1, "Linear: bias att shape != 1");
    std::vector<float> bias_value = att.Get();
    this->bias_v_ = std::make_shared<Tensor<float>>(1, att.shape_[0], 1);
    // std::cout << "Linear bias:" << std::endl;
    for (int i = 0; i < att.shape_[0]; i ++) {
      this->bias_v_->At(0, i, 0) = bias_value.at(i);
    //   std::cout << this->bias_v_->At(0, i, 0) << " ";
    }
    // std::cout << std::endl;
}
auto GetLinearLayer(const std::shared_ptr<Operator> &op) -> Layer* {
    auto* layer = new LinearLayer();
    Parameter p = op->GetParam("bias");
    bool use_bias = p.GetValueBool();
    layer->SetBias(p.GetValueBool());
    p = op->GetParam("in_features");
    layer->SetInFetures(p.GetValueInt());
    p = op->GetParam("out_features");
    layer->SetOutFeatures(p.GetValueInt());
    Attribute att = op->GetAttr("weight");
    layer->SetWeights(att);
    if (use_bias) {
      att = op->GetAttr("bias");
      layer->SetBiasValue(att);
    }
    return layer;
} 

LayerRegistelrWrapper linear_layer_registe("nn.Linear", LayerRegister::layer_creator_function(GetLinearLayer));

}  // namespace scnni