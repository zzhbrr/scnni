/*
 * @Author: zzh
 * @Date: 2023-03-04 
 * @LastEditTime: 2023-03-08 09:57:05
 * @Description: 
 * @FilePath: /SCNNI/include/scnni/graph.hpp
 */
#ifndef SCNNI_GRAPH_HPP_
#define SCNNI_GRAPH_HPP_
#include "scnni/operator.hpp"
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

    auto LoadModel(const std::string &parampath, const std::string &binpath) -> int;

    auto GetBlobByName(const std::string &name) const -> std::shared_ptr<Blob>;

    std::vector<std::shared_ptr<Operator>> operators_;
    std::vector<std::shared_ptr<Blob>> blobs_;
    std::vector<std::string> input_names_;

};

class Excecutor {
  public:
    Excecutor() = delete;
    explicit Excecutor(std::unique_ptr<Graph>graph);
    ~Excecutor() = default;

    auto Input(const std::string &blob_name, const std::vector<Tensor<float>>& data_in) -> int;
    auto Output() -> std::vector<Tensor<float>>; // 目前只支持一个输出节点
    auto Forward() -> int;
    auto ForwardOp(const std::shared_ptr<Operator>&op) -> int;

  protected:
    std::vector<std::shared_ptr<Operator>> inputs_ops_;
    std::vector<std::shared_ptr<Operator>> outputs_ops_; // 目前只有一个输出算子
    std::unique_ptr<const Graph> graph_;
};
}  // namespace scnni
#endif