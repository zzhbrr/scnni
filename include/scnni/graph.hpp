/*
 * @Author: zzh
 * @Date: 2023-03-04 
 * @LastEditTime: 2023-03-06 06:32:29
 * @Description: 
 * @FilePath: /SCNNI/include/scnni/graph.hpp
 */
#ifndef SCNNI_GRAPH_HPP_
#define SCNNI_GRAPH_HPP_
#include "scnni/layer.hpp"
#include "scnni/blob.hpp"
#include <cstddef>
#include <memory>
#include <vector>
namespace scnni {
class Excecutor;
class Graph {
  public:
    Graph() = default;
    ~Graph() = default;

    auto LoadParam(const std::string &path) -> int;
    auto LoadWeight() -> int;
    auto GetBlobByName(const std::string &name) -> std::shared_ptr<Blob>;

    auto CreateExcecutor() const -> Excecutor;

    std::vector<std::shared_ptr<Layer>> layers_;
    std::vector<std::shared_ptr<Blob>> blobs_;

  protected:
    friend class Excecutor;
    auto ForwardLayer(int layer_id, std::shared_ptr<std::vector<Tensor<float>>> blobs) const;
};

class Excecutor {
  public:
    Excecutor() = delete;
    ~Excecutor();

    auto Input(const std::string &blob_name, const Tensor<float>& data_in) -> int;
    auto Forward() -> int;

  protected:
    friend auto Graph::CreateExcecutor() const -> Excecutor;
    Excecutor(std::shared_ptr<const Graph> graph, size_t blob_count);

  private:
   std::shared_ptr<const Graph> graph_;
   std::vector<Tensor<float>> blob_tensors_; 
};
}  // namespace scnni
#endif