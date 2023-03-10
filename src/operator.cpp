/*
 * @Author: zzh
 * @Date: 2023-03-05
 * @LastEditTime: 2023-03-09 12:36:58
 * @Description: 
 * @FilePath: /SCNNI/src/operator.cpp
 */

#include "scnni/operator.hpp"
#include "scnni/logger.hpp"
#include "scnni/macros.h"
#include <sstream>
namespace scnni {
auto Parameter::GetFromString(const std::string &s) -> Parameter {
    Parameter p;
    p.type_ = ParamType::Unknow;

    if (s == "None" || s == "()" || s == "[]")
    {
        return p;
    }

    if (s == "True" || s == "False")
    {
        // bool
        p.type_ = ParamType::Bool;
        p.b_ = s == "True";
        return p;
    }

    if (s[0] == '(' || s[0] == '[')
    {
        // list
        std::string lc = s.substr(1, s.size() - 2);
        std::istringstream lcss(lc);

        while (!lcss.eof())
        {
            std::string elem;
            std::getline(lcss, elem, ',');

            if ((elem[0] != '-' && (elem[0] < '0' || elem[0] > '9')) || (elem[0] == '-' && (elem[1] < '0' || elem[1] > '9')))
            {
                // string
                p.type_ = ParamType::StringArray;
                p.sa_.push_back(elem);
            }
            else if (elem.find('.') != std::string::npos || elem.find('e') != std::string::npos)
            {
                // float
                p.type_ = ParamType::FloatArray;
                p.fa_.push_back(std::stof(elem));
            }
            else
            {
                // integer
                p.type_ = ParamType::IntArray;
                p.ia_.push_back(std::stoi(elem));
            }
        }
        return p;
    }

    if ((s[0] != '-' && (s[0] < '0' || s[0] > '9')) || (s[0] == '-' && (s[1] < '0' || s[1] > '9')))
    {
        // string
        p.type_ = ParamType::String;
        p.s_ = s;
        return p;
    }

    if (s.find('.') != std::string::npos || s.find('e') != std::string::npos)
    {
        // float
        p.type_ = ParamType::Float;
        p.f_ = std::stof(s);
        return p;
    }

    // integer
    p.type_ = ParamType::Int;
    p.i_ = std::stoi(s);
    return p;
}

template<class T>
auto Attribute::Get() -> std::vector<T> {
  /// 检查节点属性中的权重类型
  SCNNI_ASSERT(!weight_.empty(), "weight empty");
  SCNNI_ASSERT(type_ != AttrType::Unknow, "attrtype unknow");
  std::vector<T> weights;
  switch (type_) {
    case AttrType::Float32: {
      const bool is_float = std::is_same<T, float>::value;
      SCNNI_ASSERT(is_float == true, "");
      const uint32_t float_size = sizeof(float);
      SCNNI_ASSERT(weight_.size() % float_size == 0, "");
      for (uint32_t i = 0; i < weight_.size() / float_size; ++i) {
        float weight = *(reinterpret_cast<float *>(weight_.data()) + i);
        weights.push_back(weight);
      }
      break;
    }
    default: {
      LOG_ERROR("AttrType Unsupport");
    }
  }
  return weights;
}

auto Parameter::GetValueInt() -> int {
  SCNNI_ASSERT(type_ == ParamType::Int, "Param Type Error");
  return i_;
}
auto Parameter::GetValueFloat() -> float {
  SCNNI_ASSERT(type_ == ParamType::Float, "Param Type Error");
  return f_;
}
auto Parameter::GetValueString() -> std::string {
    SCNNI_ASSERT(type_ == ParamType::String, "Param Type Error");
    return s_;
}
auto Parameter::GetValueBool() -> bool {
    SCNNI_ASSERT(type_ == ParamType::Bool, "Param Type Error");
    return b_;
}
auto Parameter::GetValueIntArray() -> std::vector<int> {
    SCNNI_ASSERT(type_ == ParamType::IntArray, "Param Type Error");
    return ia_;
}
auto Parameter::GetValueStringArray() -> std::vector<std::string> {
    SCNNI_ASSERT(type_ == ParamType::StringArray, "Param Type Error");
    return sa_;
}

Operator::~Operator() {
  delete[] layer_;
}

} // namespace scnni