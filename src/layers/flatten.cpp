/*
 * @Author: zzh
 * @Date: 2023-03-10 
 * @LastEditTime: 2023-03-11 14:13:26
 * @Description: 
 * @FilePath: /SCNNI/src/layers/flatten.cpp
 */

#include "scnni/layers/flatten.hpp"
#include "scnni/layer_factory.hpp"
#include "scnni/operator.hpp"
#include "scnni/macros.h"
#include "scnni/logger.hpp"

namespace scnni {
auto FlattenLayer::Forward(const std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &output_blobs) const -> int {
    SCNNI_ASSERT(!input_blobs.empty(), "FlattenLayer's input blobs empty");
    SCNNI_ASSERT(input_blobs.size() == 1, "FlattenLayer has multiple inputs");
    SCNNI_ASSERT(!output_blobs.empty(), "FlattenLayer's output blobs empty");
    LOG_DEBUG("FlattenLayer forward: start_dim: %d, end_dim: %d", start_dim_, end_dim_);
    for (size_t batch = 0; batch < input_blobs[0].size(); batch++) {
        auto input_tensor_shptr = input_blobs[0][batch];
        std::shared_ptr<Tensor<float>> feat = output_blobs[0].at(0);
        
        auto in_shape = input_tensor_shptr->Shapes();
        auto out_shape = feat->Shapes();

        Tensor<float> tmp(feat->Shapes()[0], feat->Shapes()[1], feat->Shapes()[2]);
        tmp = *input_tensor_shptr;
        tmp.ReShape(feat->Shapes());
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

LayerRegistelrWrapper flatten_layer_registe("torch.flatten", LayerRegister::layer_creator_function(GetFlattenLayer));


}