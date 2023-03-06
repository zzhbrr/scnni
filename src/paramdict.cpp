/*
 * @Author: zzh
 * @Date: 2023-03-05 
 * @LastEditTime: 2023-03-06 08:15:22
 * @Description: 
 * @FilePath: /SCNNI/src/paramdict.cpp
 */

#include "scnni/paramdict.hpp"
#include <sstream>

namespace scnni {
ParamDict::ParamDict(std::stringstream &line_stream) {
    Clear();
    Init(line_stream);
}
// void ParamDict::Init(std::stringstream &line_stream) {

// }

void ParamDict::Clear() {
    
}

} // namespace scnni