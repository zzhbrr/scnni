/*
 * @Author: xzj
 * @Date: 2023-03-06 12:32:47
 * @LastEditTime: 2023-03-07 11:04:44
 * @Description: 
 * @FilePath: /SCNNI/src/tmp.cpp
 */

#include "scnni/tensor.hpp"
#include <memory>
#include <iostream>

Eigen::Tensor<float, 3> data(2, 2, 2);
auto main() -> int{
    data.setConstant(3.2);
    std::cout << data(1);
}