/*
 * @Author: zzh
 * @Description: Declare variable, forward_function and set_function of linear_layer
 * @FilePath: /scnni/include/scnni/layers/linear.hpp
 */

#ifndef SCNNI_LINEAR_HPP_
#define SCNNI_LINEAR_HPP_

#include "scnni/layer.hpp"
#include "scnni/operator.hpp"

namespace scnni {
class LinearLayer: public Layer {
    public:
        auto Forward(
            const std::vector<std::vector<std::shared_ptr<Tensor<float>>>> &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>> &output_blobs) const -> int override;
        ~LinearLayer() override = default;

        void SetBias(bool bias);
        void SetInFetures(int in_features);
        void SetOutFeatures(int out_features);
        void SetWeights(const Attribute& att);
        void SetBiasValue(const Attribute& att);

    private:
        bool bias_;  // If set to False, the layer will not learn an additive bias. Default: True
        int in_features_;  // size of each input sample
        int out_features_;  // size of each output sample
        std::shared_ptr<Tensor<float>> weights_;  // the learnable weights of the module of shape (out_features, in_features)
        std::shared_ptr<Tensor<float>> bias_v_;  // the learnable bias of the module of shape (out_features)
};
} // namespace scnni


#endif
