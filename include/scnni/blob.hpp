/*
 * @Author: zzh
 * @Date: 2023-03-04 12:37:46
 * @LastEditTime: 2023-03-05 12:30:32
 * @Description: 
 * @FilePath: /SCNNI/include/scnni/blob.hpp
 */
#ifndef SCNNI_BLOB_HPP_
#define SCNNI_BLOB_HPP_
#include "scnni/tensor.hpp"
#include <memory>
#include <string>
#include <vector>
namespace scnni {
class Layer;
class Blob {
  public:
    Blob();

    int id_{-1};
    std::shared_ptr<Layer> producer_; // 产生 blob 的layer
    std::vector<std::shared_ptr<Layer>> consumers_; // 以此 blob 为输入的 layer

    std::vector<int> shape_;
    std::vector<Tensor<float>> data_;
};
} // namespace scnni
#endif