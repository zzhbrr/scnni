/*
 * @Author: zzh
 * @Date: 2023-03-12 
 * @LastEditTime: 2023-03-13 12:33:27
 * @Description: 
 * @FilePath: /scnni/src/layers/maxpool2d.cpp
 */

#include "scnni/layers/maxpool2d.hpp"
#include "scnni/layer_factory.hpp"
#include "scnni/operator.hpp"
#include "scnni/macros.h"
#include "scnni/logger.hpp"
#include <Eigen/src/Core/util/Constants.h>
#include <bits/stdint-uintn.h>
#include <cfloat>
#include <iostream>

namespace scnni {
auto Maxpool2dLayer::Forward(const std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &output_blobs) const -> int {
    SCNNI_ASSERT(!input_blobs.empty(), "Maxpool2dLayer's input blobs empty");
    SCNNI_ASSERT(input_blobs.size() == 1, "Maxpool2dLayer has multiple inputs");
    SCNNI_ASSERT(!output_blobs.empty(), "Maxpool2dLayer's output blobs empty");
    // LOG_DEBUG("FlattenLayer forward: start_dim: %d, end_dim: %d", start_dim_, end_dim_);
    if (return_indices_) {
        LOG_ERROR("Maxpool2dLayer: return_indices not implemented");
    }
    if (ceil_mode_) {
        LOG_ERROR("Maxpool2dLayer: ceil_mode not implemented");
    }
    if (!(dilation_[0] == 1 && dilation_[1] == 1)) {
        LOG_ERROR("Maxpool2dLayer: dilation not implemented");
    }
    for (size_t batch = 0; batch < input_blobs[0].size(); batch++) {
        auto input_tensor_shptr = input_blobs[0][batch];
        std::shared_ptr<Tensor<float>> feat = output_blobs[0].at(0);
        
        auto in_shape = input_tensor_shptr->Shapes();
        uint32_t h_out = 0;
        uint32_t w_out = 0;
        if (ceil_mode_) {
            h_out = ceil((in_shape[1] + 2 * padding_[0] - dilation_[0] * (kernel_size_[0] - 1) - 1) / stride_[0] + 1);
            w_out = ceil((in_shape[2] + 2 * padding_[1] - dilation_[1] * (kernel_size_[1] - 1) - 1) / stride_[1] + 1);
        } else {
            h_out = (in_shape[1] + 2 * padding_[0] - dilation_[0] * (kernel_size_[0] - 1) - 1) / stride_[0] + 1;
            w_out = (in_shape[2] + 2 * padding_[1] - dilation_[1] * (kernel_size_[1] - 1) - 1) / stride_[1] + 1;
        }
        auto out_shape = feat->Shapes();
        SCNNI_ASSERT(h_out == out_shape[1] && w_out == out_shape[2], "Maxpool2dLayer: output shape uncorrect");

        Tensor<float> after_padding = *input_tensor_shptr;

        after_padding.Padding({static_cast<unsigned int>(padding_[0]),
                               static_cast<unsigned int>(padding_[0]),
                               static_cast<unsigned int>(padding_[1]),
                               static_cast<unsigned int>(padding_[1])},
                              -FLT_MAX);
        LOG_DEBUG("Get Padding right");

        for (uint32_t c = 0; c < in_shape[0]; c ++) {
            for (uint32_t i = 0; i < h_out; i ++) {
                for (uint32_t j = 0; j < w_out; j ++) {
                    float res = -FLT_MAX;
                    for (int p = 0; p < kernel_size_[0]; p ++) {
                        for (int q = 0; q < kernel_size_[1]; q ++) {
                            if (i * stride_[0] + p > in_shape[1] || j * stride_[1] + q > in_shape[2]) { // ceil mode
                                continue;
                            }
                            res = std::max(res, after_padding.At(c, i * stride_[0] + p, j * stride_[1] + q));
                        }
                    }
                    LOG_DEBUG("maxpool output [%u, %u, %u] = %.2f", c, i, j, res);
                    feat->At(c, i, j) = res;
                }
            }
        }
    }
    return 0;
}
void Maxpool2dLayer::SetCeilMode(bool ceil_mode) {
    ceil_mode_ = ceil_mode;
}
void Maxpool2dLayer::SetDilation(const std::vector<int> &dilation) {
    dilation_ = dilation;
}
void Maxpool2dLayer::SetKernelSize(const std::vector<int> &kernel_size) {
    kernel_size_ = kernel_size;
}
void Maxpool2dLayer::SetPadding(const std::vector<int> &padding) {
    padding_ = padding;
}
void Maxpool2dLayer::SetReturnIndices(bool return_indices) {
    return_indices_ = return_indices;
}
void Maxpool2dLayer::SetStride(const std::vector<int> &stride) {
    stride_ = stride;
}
auto GetMaxpool2dLayer(const std::shared_ptr<Operator> &op) -> Layer* {
    auto* layer = new Maxpool2dLayer();
    Parameter p = op->GetParam("ceil_mode");
    layer->SetCeilMode(p.GetValueBool());
    p = op->GetParam("dilation");
    layer->SetDilation(p.GetValueIntArray());
    p = op->GetParam("kernel_size");
    layer->SetKernelSize(p.GetValueIntArray());
    p = op->GetParam("padding");
    layer->SetPadding(p.GetValueIntArray());
    p = op->GetParam("return_indices");
    layer->SetReturnIndices(p.GetValueBool());
    p = op->GetParam("stride");
    layer->SetStride(p.GetValueIntArray());
    return layer;
} 

LayerRegistelrWrapper maxpool2d_layer_registe("F.max_pool2d", LayerRegister::layer_creator_function(GetMaxpool2dLayer));

}  // namespace scnni