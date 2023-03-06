/*
 * @Author: xzj
 * @Date: 2023-03-06 12:11:58
 * @LastEditTime: 2023-03-06 12:26:51
 * @Description: 
 * @FilePath: /scnni/include/scnni/layers/input.hpp
 */
#include "scnni/layer.hpp"
#include "scnni/tensor.hpp"
#include <vector>

namespace scnni {
class InputLayer: public Layer {
  private:
    int c_, h_, w_;
    std::vector<Tensor<float>> weight_;
};
}