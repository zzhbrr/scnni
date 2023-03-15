/*
 * @Author: zzh
 * @Description
 * @FilePath: /scnni/include/scnni/layers/linear.hpp
 */

#ifndef SCNNI_LINEAR_HPP_
#define SCNNI_LINEAR_HPP_

#include "scnni/layer.hpp"
#include "scnni/operator.hpp"

namespace scnni {
class LinearLayer: public Layer {
  public:
     auto Forward(const std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &output_blobs) const -> int override;
    ~LinearLayer() override = default;

    void SetBias(bool bias);
    void SetInFetures(int in_features);
    void SetOutFeatures(int out_features);
    void SetWeights(const Attribute& att);
    void SetBiasValue(const Attribute& att);

  private:
    bool bias_;
    int in_features_;
    int out_features_;
    std::shared_ptr<Tensor<float>> weights_;
    std::shared_ptr<Tensor<float>> bias_v_;
};
} // namespace scnni


#endif
