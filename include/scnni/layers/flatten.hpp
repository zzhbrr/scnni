/*
 * @Author: zzh
 * @Description: 
 * @FilePath: /scnni/include/scnni/layers/flatten.hpp
 */

#ifndef SCNNI_FLATTEN_HPP_
#define SCNNI_FLATTEN_HPP_

#include "scnni/layer.hpp"

namespace scnni {
class FlattenLayer: public Layer {
  public:
     auto Forward(const std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &output_blobs) const -> int override;
    ~FlattenLayer() override = default;

    void SetEndDim(int enddim);
    void SetStartDim(int startdim);
  private:
    int end_dim_;
    int start_dim_;
};
} // namespace scnni


#endif