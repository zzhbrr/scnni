/*
 * @Author: zzh
 * @Date: 2023-03-06
 * @LastEditTime: 2023-03-06 17:45:34
 * @Description: 
 * @FilePath: /SCNNI/include/scnni/layers/output.hpp
 */
#include "scnni/layer.hpp"
#include "scnni/tensor.hpp"
#include <vector>

namespace scnni {
class OutputLayer: public Layer {
  private:
    int c_, h_, w_;
    std::vector<Tensor<float>> weight_;
};
}