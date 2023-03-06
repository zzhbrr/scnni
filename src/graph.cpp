/*
 * @Author: zzh
 * @Date: 2023-03-04
 * @LastEditTime: 2023-03-06 10:14:45
 * @Description: 
 * @FilePath: /SCNNI/src/graph.cpp
 */

#include "scnni/graph.hpp"
#include "scnni/blob.hpp"
#include "scnni/layer_factory.hpp"
#include "scnni/logger.hpp"
#include "scnni/macros.h"
#include "scnni/operator.hpp"
#include <cstddef>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <iostream>

namespace scnni {

void LoadParameter(const std::shared_ptr<Operator>& op, const std::string& key, const std::string& value) {
    op->params_[key] = Parameter::GetFromString(value);
}
void LoadAttributeSize(const std::shared_ptr<Operator>& op, const std::string& key, const std::string& value) {
    Attribute& att = op->attrs_[key];
    if (value.substr(value.find_last_of(')') + 1) != "f32") {
        UNREACHABLE("Attribute Type Unsupported");
    }
    att.type_ = Attribute::AttrType::Float32;
    std::string shape_str = value.substr(1, value.find_last_of(')') - 1);
    std::istringstream shape_stream(shape_str);
    
    while(!shape_stream.eof()) {
        std::string token;
        std::getline(shape_stream, token, ',');
        att.shape_.push_back(std::stoi(token));
    }
}
void LoadShape(const std::shared_ptr<Operator>& op, const std::string& key, const std::string& value) {
    std::shared_ptr<Blob> blob;
    for (const auto& b: op->inputs_) {
        if (b->name_ == key) {
            blob = b;
            break;
        }
    }
    if (blob == nullptr) {
      for (const auto &b : op->outputs_) {
        if (b->name_ == key) {
          blob = b;
          break;
        }
      }
    }
    SCNNI_ASSERT(blob != nullptr, "Operator has no corresponding blob");
    if (value.substr(value.find_last_of(')') + 1) != "f32") {
        UNREACHABLE("Blob Type Unsupported");
    }
    blob->type_ = Blob::BlobType::Float32;
    std::string shape_str = value.substr(1, value.find_last_of(')') - 1);
    std::istringstream shape_stream(shape_str);
    
    while(!shape_stream.eof()) {
        std::string token;
        std::getline(shape_stream, token, ',');
        if (token == "?") {
          blob->shape_.push_back(-1);
        } else {
          blob->shape_.push_back(std::stoi(token));
        }
    }

}

auto Graph::GetBlobByName(const std::string &name) -> std::shared_ptr<Blob> {
    for (const std::shared_ptr<Blob>& b : blobs_) {
        if (b == nullptr) {
            continue;
        }
        if (b->name_ == name) {
            return b;
        }
    }
    UNREACHABLE("Blob Not Found");
}

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

    LOG_DEBUG("Loading Param");

    SCNNI_ASSERT(infile.good(), "file not good");
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
            LOG_DEBUG("magic: %d", magic);
            SCNNI_ASSERT(magic == 7767517, "LoadModel: Unsupportable Version");
        } else if (layer_count == -1 && blob_count == -1) { // load layer_count and blob_count
            while (line_stream.good()) {
                line_stream >> layer_count;
                line_stream >> blob_count;
                LOG_DEBUG("layer_count: %d, blob_count: %d\n", layer_count,
                          blob_count);
            }
            operators_.resize(layer_count);
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

            LOG_DEBUG("layer_type: %s, layer_name: %s, inputs_num: %d, outputs_num: %d", layer_type.c_str(), layer_name.c_str(), input_blobs_count, output_blobs_count);


            std::shared_ptr<Operator> op = std::make_shared<Operator>(layer_type, layer_name);

            op->inputs_.resize(input_blobs_count);
            op->outputs_.resize(output_blobs_count);
            op->inputnames_.resize(input_blobs_count);

            // read input blobs
            for (int i = 0; i < input_blobs_count; i ++) {
                std::string blob_name;
                line_stream >> blob_name;
                printf("input blob name : %s\n", blob_name.c_str());
                std::shared_ptr<Blob> blob = GetBlobByName(blob_name); 
                blob->consumers_.push_back(op);
                op->inputs_[i] = blob;
                op->inputnames_.push_back(blob_name);
            }

            // read output blobs
            for (int i = 0; i < output_blobs_count; i ++) {
                std::string blob_name;
                line_stream >> blob_name;
                printf("output blob name : %s\n", blob_name.c_str());
                std::shared_ptr<Blob> blob = std::make_shared<Blob>(blob_name);
                blob->producer_ = op;
                op->outputs_[i] = blob;
                blobs_.push_back(blob);
            }

            // read layer attributes
            while(!line_stream.eof()) {
                std::string key;
                std::string value;
                std::getline(line_stream, key, '=');
                // std::getline(line_stream, value);
                line_stream >> value;
                printf("value is %s\n", value.c_str());
                key.erase(key.begin());

                if (key[0] == '@') {
                    LoadAttributeSize(op, key.substr(1), value);
                } else if (key[0] == '$') {
                    ;
                } else if (key[0] == '#') {
                    LoadShape(op, key.substr(1), value);
                } else {
                    LoadParameter(op, key, value);
                }
            }
            ++ loaded_layer_count;
        }
    }
    
    return 1;
}

auto Graph::LoadWeight(const std::string& path) -> int {

    for (size_t i = 0; i < operators_.size(); i ++) {
        std::shared_ptr<Operator> op = operators_[i];

    }
}


}  // namespace scnni