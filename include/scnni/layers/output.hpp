/*
 * @Author: zzh
 * @Date: 2023-03-06
 * @LastEditTime: 2023-03-09 12:45:24
 * @Description: 
 * @FilePath: /SCNNI/include/scnni/layers/output.hpp
 */
#ifndef SCNNI_LAYERS_OUTPUT_HPP_
#define SCNNI_LAYERS_OUTPUT_HPP_

#include "scnni/layer.hpp"
#include "scnni/tensor.hpp"
#include <vector>

namespace scnni {
class OutputLayer: public Layer {
  public:
      auto Forward(const std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &output_blobs) const -> int override;
    //   auto GetLayer(const std::shared_ptr<Operator>& op) -> std::unique_ptr<layer> override; 
    ~OutputLayer() override = default;
};
}  // namespace scnni

#endif