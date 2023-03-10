/*
 * @Author: zzh
 * @Date: 2023-03-04
 * @LastEditTime: 2023-03-09 12:34:47
 * @Description: 
 * @FilePath: /SCNNI/include/scnni/layer.hpp
 */
// TODO: 提前分配计算所需要的内存

#ifndef SCNNI_LAYER_HPP_
#define SCNNI_LAYER_HPP_ 
#include "scnni/tensor.hpp"
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
     * @param const std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
     *          &input_blobs, 第一个vecotor表示多个输入, 第二个vector表示多个batch, shared_pter指向blob对应的Tensor
     * @param std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &output_blobs, 第一个vector表示多个输出, 第二个vector表示多个batch 
     * @return {*}
     */
    virtual auto
    Forward(const std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &output_blobs) const -> int = 0;

    // virtual auto ForwardInplace(std::vector<Tensor<float>> &blobs) const -> int; 
    auto LayerName() const -> const std::string & { return this->layer_name_; }

    // virtual auto GetLayer(const std::shared_ptr<Operator>& op) -> std::unique_ptr<Layer>;

  protected:
    std::string layer_name_;
    std::string layer_type_;
};

} // namespace scnni
#endif