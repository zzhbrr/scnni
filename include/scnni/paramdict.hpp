/*
 * @Author: zzh
 * @Date: 2023-03-05 
 * @LastEditTime: 2023-03-05 08:20:08
 * @Description: 
 * @FilePath: /SCNNI/include/paramdict.hpp
 */
#ifndef SCNNI_PARAMDICT_HPP_
#define SCNNI_PARAMDICT_HPP_
#include "scnni/graph.hpp"
#include <sstream>
namespace scnni {
class ParamDict {
  public:
    explicit ParamDict(std::stringstream &line_stream);
    enum class ParamType {Unkown, Int, Float, String, Bool, IntArray, StringArray, FloatArray};
    template<typename T> 
    auto Get(int id, const T &ret_default) -> T;

  private:
    friend class Graph;
    void Init(std::stringstream &line_stream);
    void Clear();
    
};
} // namespace scnni
#endif