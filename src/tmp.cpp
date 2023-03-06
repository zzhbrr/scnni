/*
 * @Author: xzj
 * @Date: 2023-03-06 12:32:47
 * @LastEditTime: 2023-03-06 14:21:27
 * @Description: 
 * @FilePath: /scnni/src/tmp.cpp
 */

#include "scnni/tensor.hpp"
#include <glog/logging.h>
#include <memory>
#include <iostream>

Eigen::Tensor<float, 3> data(2, 2, 2);
auto main() -> int{
    data.setConstant(3.2);
    std::cout << data(1);
}