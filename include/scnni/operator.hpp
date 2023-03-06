/*
 * @Author: zzh
 * @Date: 2023-03-05 
 * @LastEditTime: 2023-03-05 16:16:53
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
    enum class ParamType {Unkown, Int, Float, String, Bool, IntArray, StringArray, FloatArray};
    Parameter() = default;;
    explicit Parameter(int i): type_(ParamType::Int), i_(i) {};
    explicit Parameter(float f): type_(ParamType::Float), f_(f) {};

    explicit Parameter(std::string s): type_(ParamType::String), s_(std::move(s)) {};
    explicit Parameter(bool b): type_(ParamType::Bool), b_(b) {};
    explicit Parameter(std::vector<int> ia): type_(ParamType::IntArray), ia_(std::move(ia)) {};
    explicit Parameter(std::vector<std::string> sa): type_(ParamType::StringArray), sa_(std::move(sa)) {};
    explicit Parameter(std::vector<float> fa): type_(ParamType::FloatArray), fa_(std::move(fa)) {};

    static auto InitFromString(const std::string &s) -> Parameter;
  private:
    ParamType type_{ParamType::Unkown};
    
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

};

class Operator {
    std::vector<std::shared_ptr<Blob>> inputs_;
    std::vector<std::shared_ptr<Blob>> outputs_;

    std::string type_;
    std::string name_;

    std::vector<std::string> inputnames_;
    std::map<std::string, Parameter> params_;  // 参数形式
    std::map<std::string, Attribute> attrs_;  // 参数值

    std::shared_ptr<Layer> layer;
};
} // namespace scnni

#endif