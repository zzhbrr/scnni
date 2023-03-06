/*
 * @Author: zzh
 * @Date: 2023-03-05 09:56:09
 * @LastEditTime: 2023-03-06 08:09:06
 * @Description: 
 * @FilePath: /SCNNI/test/test_graph.cpp
 */

#include "scnni/graph.hpp"
#include <string>
#include <gtest/gtest.h>
#include <iostream>

TEST(graph_test, load_params) {
    scnni::Graph g = scnni::Graph();
    g.LoadParam("../demo_net/demo_net.pnnx.param");

}