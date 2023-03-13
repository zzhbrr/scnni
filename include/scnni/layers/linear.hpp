/*
 * @Author: zzh
 * @Date: 2023-03-13
 * @LastEditTime: 2023-03-13 12:30:37
 * @Description
 * @FilePath: /SCNNI/include/scnni/layers/linear.hpp
 */

#ifndef SCNNI_LINEAR_HPP_
#define SCNNI_LINEAR_HPP_

#include "scnni/layer.hpp"

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

  private:
    bool bias_;
    int in_features_;
    int out_features_;
      std::vector<std::shared_ptr<Tensor<float>>> weights_;
      std::vector<std::shared_ptr<Tensor<float>>> bias_v_;
};
} // namespace scnni


#endif
