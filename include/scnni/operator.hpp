/*
 * @Author: zzh
 * @Date: 2023-03-05 
 * @LastEditTime: 2023-03-13 17:11:50
 * @Description: 
 * @FilePath: /SCNNI/include/scnni/operator.hpp
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

    auto GetValueInt() -> int; 
    auto GetValueFloat() -> float; 
    auto GetValueString() -> std::string; 
    auto GetValueBool() -> bool; 
    auto GetValueIntArray() -> std::vector<int>; 
    auto GetValueStringArray() -> std::vector<std::string>; 
    auto GetValueFloatArray() -> std::vector<float>; 
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

  // /**
  //  * 从节点中加载权重参数
  //  * @tparam T 权重类型
  //  * @return 权重参数数组
  //  */
  // template<typename T> 
  // auto Get() const -> std::vector<T>;
    auto Get() const -> std::vector<float>;
};


class Operator {
  public:
    enum class OpState {NoInit, Inited, Executed};
    Operator(std::string type, std::string name): type_(std::move(type)), name_(std::move(name)) {};
    ~Operator();
    std::vector<std::shared_ptr<Blob>> inputs_; // 这里定义的input一定要与Layer实现相符合
    std::vector<std::shared_ptr<Blob>> outputs_; // 这里定义的output一定要与layer实现相符合
    int refcnt_;
    OpState state_{OpState::NoInit};

    std::string type_;
    std::string name_;

    std::vector<std::string> inputnames_; // input blob names
    std::map<std::string, Parameter> params_;
    std::map<std::string, Attribute> attrs_;

    bool can_forward_inplace_{false};
    // std::unique_ptr<Layer> layer_;
    Layer* layer_;

  public:
    auto GetParam(const std::string &param_name) -> Parameter;
    auto GetAttr(const std::string &attr_name) -> Attribute;
};
} // namespace scnni

#endif