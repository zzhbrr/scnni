/*
 * @Author: zzh
 * @Date: 2023-03-05 
 * @LastEditTime: 2023-03-06 11:21:02
 * @Description: 
 * @FilePath: /scnni/include/scnni/operator.hpp
 */
#ifndef SCNNI_OPERATOR_HPP_
#define SCNNI_OPERATOR_HPP_

#include "scnni/layer.hpp"
#include <memory>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace scnni {
class Blob;
class Parameter {
  public:
    enum class ParamType {Unknow, Int, Float, String, Bool, IntArray, StringArray, FloatArray};
    Parameter() = default;
    explicit Parameter(int i): type_(ParamType::Int), i_(i) {};
    explicit Parameter(float f): type_(ParamType::Float), f_(f) {};

    explicit Parameter(std::string s): type_(ParamType::String), s_(std::move(s)) {};
    explicit Parameter(bool b): type_(ParamType::Bool), b_(b) {};
    explicit Parameter(std::vector<int> ia): type_(ParamType::IntArray), ia_(std::move(ia)) {};
    explicit Parameter(std::vector<std::string> sa): type_(ParamType::StringArray), sa_(std::move(sa)) {};
    explicit Parameter(std::vector<float> fa): type_(ParamType::FloatArray), fa_(std::move(fa)) {};

    static auto GetFromString(const std::string &s) -> Parameter;
  private:
    ParamType type_{ParamType::Unknow};
    
    // value
    int i_;
    float f_;
    std::string s_;
    bool b_;
    std::vector<int> ia_;
    std::vector<std::string> sa_;
    std::vector<float> fa_;

};
class Attribute {
  public:
    enum class AttrType {Unknow, Null, Float32, Float64, Float16, Int32, Int64, Int16, Int8, Uint8, Bool};
    AttrType type_{AttrType::Unknow};
    std::vector<char> weight_; 
    std::vector<int> shape_;  

  /**
   * 从节点中加载权重参数
   * @tparam T 权重类型
   * @return 权重参数数组
   */
  template<class T> 
  auto Get() -> std::vector<T>;
};


class Operator {
  public:
    Operator(std::string type, std::string name): type_(std::move(type)), name_(std::move(name)) {};
    std::vector<std::shared_ptr<Blob>> inputs_;
    std::vector<std::shared_ptr<Blob>> outputs_;

    std::string type_;
    std::string name_;

    std::vector<std::string> inputnames_; // input blob names
    std::map<std::string, Parameter> params_;
    std::map<std::string, Attribute> attrs_;
};
} // namespace scnni

#endif