/*
 * @Author: zzh
 * @Date: 2023-03-04
 * @LastEditTime: 2023-03-08 10:10:23
 * @Description: 
 * @FilePath: /SCNNI/include/scnni/layer.hpp
 */

#ifndef SCNNI_LAYER_HPP_
#define SCNNI_LAYER_HPP_ 
#include "tensor.hpp"
#include <memory>
#include <string>
#include <vector>

namespace scnni {
class Blob;
class Operator;
class Layer {
  public:
    Layer() = default;
    explicit Layer(std::string layer_name): layer_name_(std::move(layer_name)) {};
    virtual ~Layer() = default;

    /**
     * @description: Layer的前向计算
     * @param std::vector<std::shared_ptr<Tensor<float>>>& input_blobs
     * @param std::shared_ptr<std::vector<std::shared_ptr<Tensor<float>>>> output_blobs
     * @return {*}
     */
    virtual auto Forward(const std::vector<std::vector<std::shared_ptr<Tensor<float>>>>& input_blobs,  std::vector<std::vector<std::shared_ptr<Tensor<float>>>>&output_blobs) const -> int; 
    // virtual auto Forward(const Tensor<float>& input_blobs, Tensor<float> &output_blobs) const ->int;
    // virtual auto ForwardInplace(std::vector<Tensor<float>> &blobs) const -> int; 
    // virtual auto ForwardInplace(Tensor<float> &blobs) const -> int; 
    auto LayerName() const -> const std::string & { return this->layer_name_; }

    virtual auto GetLayer(const std::shared_ptr<Operator>& op) -> Layer;

  protected:
    std::string layer_name_;
    std::string layer_type_;
};

} // namespace scnni
#endif