/*
 * @Author: xzj
 * @Date: 2023-03-06 12:11:58
 * @LastEditTime: 2023-03-13 11:25:18
 * @Description: 
 * @FilePath: /scnni/include/scnni/layers/input.hpp
 */

#ifndef SCNNI_LAYERS_INPUT_HPP_
#define SCNNI_LAYERS_INPUT_HPP_

#include "scnni/layer.hpp"
#include "scnni/tensor.hpp"
#include <vector>

namespace scnni {
class InputLayer: public Layer {
    public:
        auto Forward(
            const std::vector<std::vector<std::shared_ptr<Tensor<float>>>> &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>> &output_blobs) const -> int override;
        // auto GetLayer(const std::shared_ptr<Operator>& op) -> std::unique_ptr<Layer> override; 
        ~InputLayer() override = default;
};
}  // namespace scnni

#endif