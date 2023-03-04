/*
 * @Author: zzh
 * @Date: 2023-03-04
 * @LastEditTime: 2023-03-04 09:37:15
 * @Description: 
 * @FilePath: /SCNNI/include/layer.hpp
 */

#ifndef SCNNI_LAYER_HPP_
#define SCNNI_LAYER_HPP_ 
#include "tensor.hpp"
#include <memory>
#include <string>
#include <vector>

namespace scnni {

class Layer {
  public:
    explicit Layer(std::string layer_name): layer_name_(std::move(layer_name)) {};
    virtual ~Layer() = default;

    /**
     * @description: Layer的前向计算
     * @param {vector<std::shared_ptr<Tensor<float>>>& input_blobs, std::shared_ptr<std::vector<std::shared_ptr<Tensor<float>>>>} output_blobs
     * @return {*}
     */
    virtual auto Forward(const std::vector<std::shared_ptr<Tensor<float>>>& input_blobs, std::shared_ptr<std::vector<std::shared_ptr<Tensor<float>>>> output_blobs) const -> int; 
    virtual auto Forward(const std::shared_ptr<Tensor<float>>& input_blobs, std::shared_ptr<Tensor<float>> output_blobs) const ->int;

    auto LayerName() const -> const std::string & { return this->layer_name_; }

  protected:
    std::string layer_name_;
    std::string layer_type_;
};

} // namespace scnni
#endif