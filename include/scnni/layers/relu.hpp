/*
 * @Author: zzh
 * @Description: 
 * @FilePath: /scnni/include/scnni/layers/relu.hpp
 */
#ifndef SCNNI_LAYERS_RELU_HPP_
#define SCNNI_LAYERS_RELU_HPP_

#include "scnni/layer.hpp"
#include "scnni/tensor.hpp"
#include <vector>

namespace scnni {
class ReluLayer: public Layer {
    public:
        auto Forward(
            const std::vector<std::vector<std::shared_ptr<Tensor<float>>>> &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>> &output_blobs) const -> int override;
        // auto GetLayer(const std::shared_ptr<Operator>& op) -> std::unique_ptr<Layer> override; 
        ~ReluLayer() override = default;
};
}  // namespace scnni

#endif