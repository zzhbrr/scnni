/*
 * @Author: zzh
 * @Date: 2023-03-04 
 * @LastEditTime: 2023-03-04 08:34:36
 * @Description: 
 * @FilePath: /SCNNI/include/tensor.hpp
 */

#ifndef SCNNI_TENSOR_HPP_
#define SCNNI_TENSOR_HPP_
#include "macros.h"
#include <bits/stdint-uintn.h>
#include <vector>


namespace scnni {
template <typename T>
class Tensor {};


template <>
class Tensor<float> {
  public:
    explicit Tensor() = default;

    /**
     * @description: 构造函数
     * @param {uint32_t} channels
     * @param {uint32_t} rows
     * @param {uint32_t} cols
     * @return {*}
     */
    explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

    explicit Tensor(const std::vector<uint32_t>& shapes);

    /**
     * @description: 拷贝构造函数
     * @param {Tensor&} tensor
     * @return {*}
     */
    Tensor(const Tensor& tensor);

    auto operator=(const Tensor& tensor) -> Tensor<float>&;

    /**
     * @description: 移动构造函数
     * @param {Tensor&&} tensor
     * @return {*}
     */
    Tensor(Tensor&& tensor) noexcept ;

    auto operator=(Tensor&& tenson) noexcept -> Tensor<float>&;

    // auto Rows() const -> uint32_t { SCNNI_ASSERT(!this->data_.empty(), ""); return this->data_->n_rows; }
  private:
    void Review(const std::vector<uint32_t>& shapes);
    std::vector<uint32_t> raw_shapes_;
    // data_;
};

} // namespace scnni


#endif // SCNNI_TENSOR_HPP_