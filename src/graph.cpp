/*
 * @Author: zzh
 * @Date: 2023-03-04
 * @LastEditTime: 2023-03-08 10:09:27
 * @Description: 
 * @FilePath: /SCNNI/src/graph.cpp
 */

#include "scnni/graph.hpp"
#include "scnni/blob.hpp"
#include "scnni/layer_factory.hpp"
#include "scnni/logger.hpp"
#include "scnni/macros.h"
#include "scnni/operator.hpp"
#include "scnni/store_zip.hpp"
#include <cstddef>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <iostream>
#include <queue>

namespace scnni {

void LoadParameter(const std::shared_ptr<Operator>& op, const std::string& key, const std::string& value) {
    LOG_DEBUG("Loading Params: %s", key.c_str());
    op->params_[key] = Parameter::GetFromString(value);
}
void LoadAttribute(const std::shared_ptr<Operator>& op, const std::string& key, const std::string& value, pnnx::StoreZipReader& szr) {
    LOG_DEBUG("Loading Attributes: %s", key.c_str());
    Attribute& att = op->attrs_[key];
    if (value.substr(value.find_last_of(')') + 1) != "f32") {
        UNREACHABLE("Attribute Type Unsupported");
    }
    att.type_ = Attribute::AttrType::Float32;
    std::string shape_str = value.substr(1, value.find_last_of(')') - 1);
    std::istringstream shape_stream(shape_str);
    
    att.shape_.clear();
    while(!shape_stream.eof()) {
        std::string token;
        std::getline(shape_stream, token, ',');
        att.shape_.push_back(std::stoi(token));
    }
    if (att.shape_.empty()) {
        return ;
    }

    size_t size = 1;
    for (int i : att.shape_)
    {
        size *= i;
    }

    size_t bytesize = size * 4; // float size

    std::string filename = op->name_ + "." + key;

    size_t filesize = szr.get_file_size(filename);

    if (filesize == 0)
    {
        // no such file
        return;
    }

    if (filesize != bytesize)
    {
        fprintf(stderr, "file size not match expect %lu but got %lu\n", bytesize, filesize);
    }

    att.weight_.resize(bytesize);
    szr.read_file(filename, static_cast<char*>(att.weight_.data()));

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

auto Graph::GetBlobByName(const std::string &name) const -> std::shared_ptr<Blob> {
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

auto Graph::LoadModel(const std::string &parampath, const std::string &binpath) -> int {
    if (parampath.empty() || binpath.empty()) {
        LOG_ERROR("LoadModel: file path is empty");
        return 0;
    }
    std::ifstream infile(parampath);
    pnnx::StoreZipReader szr;
    if (szr.open(binpath) != 0) {
        fprintf(stderr, "open failed\n");
        return -1;
    }

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

        if (magic == -1) { // load magic
            std::string token;
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
            // operators_.resize(layer_count);
            // blobs_.resize(blob_count);
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
            operators_.push_back(op);

            op->inputs_.resize(input_blobs_count);
            op->outputs_.resize(output_blobs_count);
            op->inputnames_.resize(input_blobs_count);
            op->refcnt_ = input_blobs_count;

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
                printf("key is %s, ", key.c_str());
                printf("value is %s\n", value.c_str());
                key.erase(key.begin());

                if (key[0] == '@') {
                    LoadAttribute(op, key.substr(1), value, szr);
                } else if (key[0] == '$') {
                    ;
                } else if (key[0] == '#') {
                    LoadShape(op, key.substr(1), value);
                } else {
                    LoadParameter(op, key, value);
                }
            }
            ++ loaded_layer_count;
            op->layer_ = LayerRegister::CreateLayer(op->type_, op);
            op->state_ = Operator::OpState::Inited;
        }
    }

    return 1;
}

Excecutor::Excecutor(std::unique_ptr<Graph> graph) : graph_(std::move(graph)){
    for (const auto& op : graph_->operators_) {
        if (op->type_ == "pnnx.Input") {
            inputs_ops_.push_back(op);
        }
        if (op->type_ == "pnnx.Output") {
            outputs_ops_.push_back(op);
        }
    }
}

auto Excecutor::Input(const std::string &blob_name, const std::vector<Tensor<float>> &data_in) -> int {
    std::shared_ptr<Blob> blob = graph_->GetBlobByName(blob_name);
    size_t batchsize = data_in.size();
    for (size_t i = 0; i < batchsize; i ++) {
        // blob[i] = data_in[i]; // 为输入blob赋值
    }
    return 1;
}

auto Excecutor::Output() -> std::vector<Tensor<float>> {
    std::vector<Tensor<float>> ret;
    for (const auto& op: outputs_ops_) { // 因为只有一个输出算子，所以只进行一次循环
        if (op->state_ != Operator::OpState::Executed) {
            LOG_ERROR("Operator haven't benn executed");
            return ret;
        }
        for (const auto& input_blob_ptr: op->inputs_) {
            for (const auto& one_batch_tensor: input_blob_ptr->data_) {
              Tensor<float> new_tensor = *one_batch_tensor;
              ret.push_back(new_tensor);
            }
        }
    }
    return ret;
}

auto Excecutor::ForwardOp(const std::shared_ptr<Operator> &op) -> int {
  if (op->state_ != Operator::OpState::Inited) {
    LOG_ERROR("Operator haven't init");
    return -1;
  }
  std::vector<std::vector<std::shared_ptr<Tensor<float>>>> inputblobs;
  std::vector<std::vector<std::shared_ptr<Tensor<float>>>> outputblobs;
  for (const auto &blob : op->inputs_) {
    if (blob->data_.empty()) {
      LOG_ERROR("Op's input data empty");
      return -1;
    }
    std::vector<std::shared_ptr<Tensor<float>>> tensors;
    for (const auto &tensor : blob->data_) {
      tensors.push_back(tensor);
    }
    inputblobs.push_back(tensors);
  }
  for (const auto &blob : op->outputs_) {
    if (blob->data_.empty()) {
      LOG_ERROR("Op's output data empty");
      return -1;
    }
    std::vector<std::shared_ptr<Tensor<float>>> tensors;
    for (const auto &tensor : blob->data_) {
      tensors.push_back(tensor);
    }
    outputblobs.push_back(tensors);
  }
  int ret = op->layer_->Forward(inputblobs, outputblobs);
  op->state_ = Operator::OpState::Executed;
  return ret;
}

auto Excecutor::Forward() -> int {
    std::queue<std::shared_ptr<Operator>> exe_q;
    if (inputs_ops_.empty()) {
        LOG_ERROR("No input operators");
        return -1;
    }
    for (const auto& op: inputs_ops_) {
        exe_q.push(op);
    }
    while(!exe_q.empty()) {
        const std::shared_ptr<Operator>& op = exe_q.front();
        exe_q.pop();
        int ret = ForwardOp(op);
        if (ret != 0) {
            LOG_ERROR("Layer forward failed");
            return ret;
        }
        for (const auto& blob: op->outputs_) {
            for (const auto& nxt_op: blob->consumers_) {
                if (--nxt_op->refcnt_ == 0) {
                    exe_q.push(nxt_op);
                }
            }
        }
    }
    return 0;
}
} // namespace scnni