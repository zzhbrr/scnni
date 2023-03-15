/*
 * @Author: zzh
 * @Description: 
 * @FilePath: /scnni/src/layers/flatten.cpp
 */

#include "scnni/layers/flatten.hpp"
#include "scnni/layer_factory.hpp"
#include "scnni/operator.hpp"
#include "scnni/macros.h"
#include "scnni/logger.hpp"
#include <bits/stdint-uintn.h>

namespace scnni {
auto FlattenLayer::Forward(const std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &output_blobs) const -> int {
    SCNNI_ASSERT(!input_blobs.empty(), "FlattenLayer's input blobs empty");
    SCNNI_ASSERT(input_blobs.size() == 1, "FlattenLayer has multiple inputs");
    SCNNI_ASSERT(!output_blobs.empty(), "FlattenLayer's output blobs empty");
    // LOG_DEBUG("FlattenLayer forward: start_dim: %d, end_dim: %d", start_dim_, end_dim_);
    int start_dim = start_dim_;
    int end_dim = end_dim_;
    if (start_dim < 0) {
        start_dim = 4 + start_dim;
    }
    if (end_dim < 0) {
        end_dim = 4 + end_dim;
    }
    start_dim -= 1; // 去除Batchsize维度
    end_dim -= 1;   // 去除Batchsize维度
    SCNNI_ASSERT(start_dim <= end_dim, "End dim must larger than start dim");
    //遍历batch_size
    for (size_t batch = 0; batch < input_blobs[0].size(); batch++) {
        const auto input_tensor_shptr = input_blobs[0][batch];
        const std::shared_ptr<Tensor<float>> feat = output_blobs[0].at(0);
        
        const auto in_shape = input_tensor_shptr->Shapes();
        uint32_t element_size = 1;
        for (int i = start_dim; i <= end_dim; i ++) {
            element_size *= in_shape[i];
        }

        auto out_shape = feat->Shapes();

        Tensor<float> tmp(feat->Shapes()[0], feat->Shapes()[1], feat->Shapes()[2]);
        tmp = *input_tensor_shptr;
        tmp.ReShape(feat->RawShapes());
        auto rawsp = feat->RawShapes();
        // LOG_DEBUG("Flatten layer: feat shape:[%d, %d, %d]", feat->Shapes()[0], feat->Shapes()[1], feat->Shapes()[2]);
        // LOG_DEBUG("Flatten layer: feat rawshape:[%d, %d, %d]", rawsp[0], rawsp[1], rawsp[2]);
        *feat = tmp;
    }
    return 0;
} 
void FlattenLayer::SetEndDim(int enddim) {
    end_dim_ = enddim;
}
void FlattenLayer::SetStartDim(int startdim) {
    start_dim_ = startdim;
}

auto GetFlattenLayer(const std::shared_ptr<Operator> &op) -> Layer* {
    auto* layer = new FlattenLayer();
    Parameter p_enddim = op->GetParam("end_dim");
    layer->SetEndDim(p_enddim.GetValueInt());
    Parameter p_startdim = op->GetParam("start_dim");
    layer->SetEndDim(p_startdim.GetValueInt());
    return layer;
} 

// 注册算子: name="torch.flatten" and type = GetFlattenLayer
LayerRegistelrWrapper flatten_layer_registe("torch.flatten", LayerRegister::layer_creator_function(GetFlattenLayer));

}  // namespace scnni