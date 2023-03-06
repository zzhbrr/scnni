/*
 * @Author: zzh
 * @Date: 2023-03-04 12:37:46
 * @LastEditTime: 2023-03-06 11:20:11
 * @Description: 
 * @FilePath: /scnni/include/scnni/blob.hpp
 */
#ifndef SCNNI_BLOB_HPP_
#define SCNNI_BLOB_HPP_
#include "scnni/tensor.hpp"
#include "scnni/operator.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>
namespace scnni {
class Layer;
class Blob {
  public:
    enum class BlobType {Unknow, Float32, Float16, Int32, Int8}; // Only float32 supported now
    Blob();
    explicit Blob(std::string name): name_(std::move(name)) {};

    int id_{-1};
    BlobType type_{BlobType::Unknow};
    std::string name_;
    std::shared_ptr<Operator> producer_; // 产生 blob 的layer
    std::vector<std::shared_ptr<Operator>> consumers_; // 以此 blob 为输入的 layer

    std::vector<int> shape_;
    std::vector<Tensor<float>> data_;  //操作数：可包含多个bz的Tensor
};
} // namespace scnni
#endif