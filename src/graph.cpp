/*
 * @Author: zzh
 * @Date: 2023-03-04
 * @LastEditTime: 2023-03-05 14:38:37
 * @Description: 
 * @FilePath: /SCNNI/src/graph.cpp
 */

#include "scnni/graph.hpp"
#include "scnni/layer_factory.hpp"
#include "scnni/logger.hpp"
#include "scnni/macros.h"
#include <cstddef>
#include <exception>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

namespace scnni {

auto Graph::LoadParam(const std::string &path) -> int {
    if (path.empty()) {
        LOG_ERROR("LoadModel: file path is empty");
        return 0;
    }
    std::ifstream infile(path);
    std::string line_str;
    std::stringstream line_stream;

    int magic = -1;
    int layer_count = -1;
    int blob_count = -1;

    int loaded_layer_count = 0;

    while(infile.good()) {
        std::getline(infile, line_str);
        if (line_str.empty()) {
            break;
        }
        line_stream.clear();
        line_stream.str(line_str);
        std::string token;

        if (magic == -1) { // load magic
            line_stream >> token;
            try {
                magic = std::stoi(token);
            } catch (std::exception &e) {
                LOG_ERROR("LoadModel: load failed when read magic");
            }
            LOG_DEBUG("magic: %d\n", magic);
            SCNNI_ASSERT(magic == 767517, "LoadModel: Unsupportable Version");
        } else if (layer_count == -1 && blob_count == -1) { // load layer_count and blob_count
            while (line_stream.good()) {
                line_stream >> layer_count;
                line_stream >> blob_count;
                LOG_DEBUG("layer_count: %d, blob_count: %d\n", layer_count,
                          blob_count);
            }
            layers_.resize(layer_count);
            blobs_.resize(blob_count);
        } else { // load layer
            if (loaded_layer_count == layer_count) {
                    break;
            } 
            std::string layer_name;
            std::string layer_type;
            int input_blobs_count = 0;
            int output_blobs_count = 0;
            
            line_stream >> layer_type;
            line_stream >> layer_name;
            line_stream >> input_blobs_count;
            line_stream >> output_blobs_count;

            std::shared_ptr<Layer> layer = LayerRegister::CreateLayer(layer_type);

            layer->inputs_.resize(input_blobs_count);
            layer->outputs_.resize(output_blobs_count);

            // read input blobs
            for (int i = 0; i < input_blobs_count; i ++) {
                int blob_id;
                line_stream >> blob_id;
                SCNNI_ASSERT(blob_id >= 0 && blob_id < blob_count, "LoadModel: Wrong file: blob_id error");
                std::shared_ptr<Blob> blob = blobs_[blob_id];
                blob->consumers_.push_back(layer);
                layer->inputs_[i] = blob;
            }

            // read output blobs
            for (int i = 0; i < output_blobs_count; i ++) {
                int blob_id;
                line_stream >> blob_id;
                SCNNI_ASSERT(blob_id >= 0 && blob_id < blob_count, "LoadModel: Wrong file: blob_id error");
                std::shared_ptr<Blob> blob = blobs_[blob_id];
                blob->producer_ = layer;
                layer->outputs_[i] = blob;
            }

            // read layer attributes
            while(!line_stream.eof()) {
                std::string key;
                std::string value;
                std::getline(line_stream, key, '=');
                std::getline(line_stream, value);
                // line_stream >> value;

                if (key[0] == '@') {

                } else if (key[0] == '$') {
                    
                } else if (key[0] == '#'){

                } else {

                }
            }
            
            ++ loaded_layer_count;
        }
    }
    
    return 1;
}

auto Graph::LoadWeight() -> int {
    return 1;
}


}  // namespace scnni