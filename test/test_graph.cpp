/*
 * @Author: zzh
 * @Date: 2023-03-05 09:56:09
 * @LastEditTime: 2023-03-08 11:00:44
 * @Description: 
 * @FilePath: /SCNNI/test/test_graph.cpp
 */

#include "scnni/graph.hpp"
// #include "pnnx/ir.h"
#include <string>
#include <gtest/gtest.h>
#include <iostream>

TEST(GGGGGGG, load_model) {
    // std::unique_ptr<pnnx::Graph> g = std::make_unique<pnnx::Graph>();
    // g->load("/ws/CourseProject/SCNNI/demo_net/demo_net.pnnx.param","/ws/CourseProject/SCNNI/demo_net/demo_net.pnnx.bin");
    std::cout << "In graph_test load params" << std::endl;
    scnni::Graph g = scnni::Graph();
    g.LoadModel("/ws/CourseProject/SCNNI/demo_net/demo_net.pnnx.param","/ws/CourseProject/SCNNI/demo_net/demo_net.pnnx.bin");
    EXPECT_EQ(g.blobs_.size(), 12);
    EXPECT_EQ(g.operators_.size(), 13);

    EXPECT_EQ(g.operators_[12]->outputs_.size(), 0);
    EXPECT_EQ(g.operators_[12]->inputs_.size(), 1);

    EXPECT_EQ(g.operators_[11]->params_.find("dim")->second.GetValueInt(), -1);
    EXPECT_FALSE(g.operators_[3]->params_.find("ceil_mode")->second.GetValueBool());
}