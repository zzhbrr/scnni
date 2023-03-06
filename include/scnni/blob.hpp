/*
 * @Author: zzh
 * @Date: 2023-03-04 12:37:46
 * @LastEditTime: 2023-03-05 15:49:11
 * @Description: 
 * @FilePath: /SCNNI/include/scnni/blob.hpp
 */
#ifndef SCNNI_BLOB_HPP_
#define SCNNI_BLOB_HPP_
#include "scnni/tensor.hpp"
#include "scnni/operator.hpp"
#include <memory>
#include <string>
#include <vector>
namespace scnni {
class Layer;
class Blob {
  public:
    Blob();

    int id_{-1};
    std::shared_ptr<Operator> producer_; // 一个产生者：产生 blob 的layer
    std::vector<std::shared_ptr<Operator>> consumers_; // 多个消费者：以此 blob 为输入的 layer

    std::vector<int> shape_;
    std::vector<Tensor<float>> data_;  //操作数：可包含多个bz的Tensor
};
} // namespace scnni
#endif