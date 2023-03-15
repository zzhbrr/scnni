/*
 * @Author: xzj
 * @Description: 
 * @FilePath: /scnni/include/scnni/layers/softmax.hpp
 */

#ifndef SCNNI_LAYERS_SOFTMAX_HPP_
#define SCNNI_LAYERS_SOFTMAX_HPP_

#include "scnni/layer.hpp"
#include "scnni/tensor.hpp"
#include <vector>

namespace scnni {
class SoftmaxLayer: public Layer {
    public:
        auto Forward(
            const std::vector<std::vector<std::shared_ptr<Tensor<float>>>> &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>> &output_blobs) const -> int override;
        // auto GetLayer(const std::shared_ptr<Operator>& op) -> std::unique_ptr<Layer> override; 
        ~SoftmaxLayer() override = default;
        void SetDim(int dim);
    private:
        int dim_;
};
}  // namespace scnni

#endif