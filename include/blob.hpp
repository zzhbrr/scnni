/*
 * @Author: zzh
 * @Date: 2023-03-04 12:37:46
 * @LastEditTime: 2023-03-04 13:05:14
 * @Description: 
 * @FilePath: /SCNNI/include/blob.hpp
 */

#include <string>
#include <vector>
namespace scnni {
class Blob {
  public:
    Blob();

    std::string name_;
    int producer_{-1};
    std::vector<int> consumers_;
};
}