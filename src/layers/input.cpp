/*
 * @Author: xzj
 * @Date: 2023-03-06 12:11:49
 * @LastEditTime: 2023-03-06 12:31:27
 * @Description: 
 * @FilePath: /scnni/src/layers/input.cpp
 */

#include "scnni/tensor.hpp"
#include "scnni/layers/input.hpp"
#include "scnni/macros.h"
#include <memory>
#include <vector>

namespace scnni {
auto InputLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& input_blobs, 
                        std::shared_ptr<std::vector<std::shared_ptr<Tensor<float>>>> output_blobs) const -> int {
    SCNNI_ASSERT(input_blobs.empty(), "InputLayer has input blobs");
    for (size_t i = 0; i < output_blobs->size(); i ++) {
        output_blobs->at(i) = weight_.at(i);
    }
    return 0;
} 
auto InputLayer::Forward(const std::shared_ptr<Tensor<float>>& input_blobs, std::shared_ptr<Tensor<float>> output_blobs) const ->int;
    SCNNI_ASSERT(input_blobs == nullptr, "InputLayer has input blobs");
    output_blob = weight;
    return 0;
}