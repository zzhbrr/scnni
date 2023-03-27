//
// Created by xzj on 22-11-12.
//

#include "scnni/tensor.hpp"
#include "scnni/macros.h"
#include "scnni/logger.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#include <Eigen/src/Core/util/Constants.h>
#include <bits/stdint-uintn.h>
#include <iterator>
#include <memory>
#include <cmath>
#include <iostream>
#include <unsupported/Eigen/CXX11/src/util/EmulateArray.h>
#include <fstream>
#include <cstdint>


namespace scnni {

//!
Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
    data_ = Eigen::Tensor<float, 3>(rows, cols, channels);
    if (channels == 1 && cols == 1) {
        this->raw_shapes_ = std::vector<uint32_t>{rows};
    } else if (channels == 1) {
        this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
    } else {
        this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
    }
}

/**
 * @description: 构造
 */
Tensor<float>::Tensor(const std::vector<uint32_t>& shapes) {
    SCNNI_ASSERT(shapes.size() == 3, "Find shapes.size() != 3");
    uint32_t channels = shapes.at(0);
    uint32_t rows = shapes.at(1);
    uint32_t cols = shapes.at(2);

    data_ = Eigen::Tensor<float, 3>(rows, cols, channels);
    if (channels == 1 && cols == 1) {
        this->raw_shapes_ = std::vector<uint32_t>{rows};
    } else if (channels == 1) {
        this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
    } else {
        this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
    }
}

/**
 * @description: 复制拷贝
 * @param {Tensor&} tensor
 * @return {*}
 */
Tensor<float>::Tensor(const Tensor& tensor) {
    if (this != &tensor) {
        this->data_ = tensor.data_;
        this->raw_shapes_ = tensor.raw_shapes_;
    }
}

Tensor<float>::Tensor(Tensor<float>&& tensor) noexcept {
    if (this != &tensor) {
        this->data_ = tensor.GetData();
        this->raw_shapes_ = tensor.raw_shapes_;
    }
}

// auto Tensor<float>::operator=(Tensor<float>&& tensor) noexcept -> Tensor<float>& {
//   if (this != &tensor) {
//     this->data_ = tensor.GetData();
//     this->raw_shapes_ = tensor.raw_shapes_;
//   }
//   return *this;
// }

auto Tensor<float>::operator=(const Tensor& tensor) -> Tensor<float>& {
    if (this != &tensor) {
        this->data_ = tensor.data_;
        this->raw_shapes_ = tensor.raw_shapes_;
    }
    return *this;
}

//!
auto Tensor<float>::Rows() const -> uint32_t {
	SCNNI_ASSERT(this->data_.size(), "data_ is empty"); 
    return this->data_.dimensions()[0];
}

//!
auto Tensor<float>::Cols() const -> uint32_t {
    SCNNI_ASSERT(this->data_.size(), "data_ is empty"); 
    return this->data_.dimensions()[1];
}

//!
auto Tensor<float>::Channels() const -> uint32_t {
    SCNNI_ASSERT(this->data_.size(), "data_ is empty");
    return this->data_.dimensions()[2];
}

//!
auto Tensor<float>::Size() const -> uint32_t {
    SCNNI_ASSERT(this->data_.size(), "data_ is empty");
    return this->data_.size();
}

//!
void Tensor<float>::SetData(const Eigen::Tensor<float, 3>& data) {
    SCNNI_ASSERT(data.dimensions()[0] == this->data_.dimensions()[0], "Data shape not meet");
    SCNNI_ASSERT(data.dimensions()[1] == this->data_.dimensions()[1], "Data shape not meet");
    SCNNI_ASSERT(data.dimensions()[2] == this->data_.dimensions()[2], "Data shape not meet");
    this->data_ = data;
}

//!
auto Tensor<float>::Empty() const -> bool {
	// return this->data_.size();
	return false;
}

//!
auto Tensor<float>::Index(uint32_t offset) const -> float {
    // CHECK(offset < this->data_.size()) << "Tensor capacity is not enough!";
    return this->data_(offset);
}

//!
auto Tensor<float>::Index(uint32_t offset) -> float & {
    SCNNI_ASSERT(offset < this->data_.size(), "Tensor capacity is not enough!");
    return this->data_(offset);
}

//!
auto Tensor<float>::Shapes() const -> std::vector<uint32_t> {
    SCNNI_ASSERT(this->data_.size(), "data_ is empty");
    return {this->Channels(), this->Rows(), this->Cols()};
}

//!
auto Tensor<float>::RawShapes() const -> const std::vector<uint32_t>& {
    SCNNI_ASSERT(!this->raw_shapes_.empty(),  "raw_shapes_ is empty");
    return this->raw_shapes_;
}

//!
auto Tensor<float>::GetData() -> Eigen::Tensor<float, 3> & {
	return this->data_;
}

//!
auto Tensor<float>::GetData() const -> const Eigen::Tensor<float, 3>& {
	return this->data_;
}

auto Tensor<float>::Slice(uint32_t channel) const -> Eigen::Tensor<float, 3> {
	Eigen::array<Eigen::DenseIndex, 3> offsets = { 0, 0, channel };  //起点
	Eigen::array<Eigen::DenseIndex, 3> extends = { this->Rows(), this->Cols(), 1 };  //扩充
	Eigen::Tensor<float, 2> tmp = this->data_.slice(offsets, extends);
  return tmp;
}
// //!
// /**
//  * @description: 返回张量第channel通道中的数据
//  * @param {uint32_t} channel
//  * @return {*}
//  */
// Eigen::Tensor<float, 2>& Tensor<float>::Slice(uint32_t channel) {
//   CHECK_LT(channel, this->Channels());
// 	Eigen::array<Eigen::DenseIndex, 3> offsets = { channel, 0, 0 };  //起点
// 	Eigen::array<Eigen::DenseIndex, 3> extends = { 1, this->Rows(), this->Cols() };  //扩充
// 	Eigen::Tensor<float, 2> &tmp = this->data_.slice(offsets, extends);
// 	return tmp;
// }

// //!
// const Eigen::Tensor<float, 2>& Tensor<float>::Slice(uint32_t channel) const {
//   CHECK_LT(channel, this->Channels());
//   Eigen::array<Eigen::DenseIndex, 3> offsets = { channel, 0, 0 };
// 	Eigen::array<Eigen::DenseIndex, 3> extends = { 1, this->Rows(), this->Cols() };
// 	return this->data_.slice(offsets, extends);
// }

//!
auto Tensor<float>::At(uint32_t channel, uint32_t row, uint32_t col) const -> float {
  // CHECK_LT(row, this->Rows());
  // CHECK_LT(col, this->Cols());
  // CHECK_LT(channel, this->Channels());
  return this->data_(row, col, channel);
}

//!
auto Tensor<float>::At(uint32_t channel, uint32_t row, uint32_t col) -> float& {
  // CHECK_LT(row, this->Rows());
  // CHECK_LT(col, this->Cols());
  // CHECK_LT(channel, this->Channels());
  return this->data_(row, col, channel);
}

//!
void Tensor<float>::Padding(const std::vector<uint32_t>& pads, float padding_value) {
    SCNNI_ASSERT(this->data_.size(), "data_ is empty");
    SCNNI_ASSERT(pads.size() == 4, "Padding: pads size != 4");
    uint32_t pad_rows1 = pads.at(0);  // up
    uint32_t pad_rows2 = pads.at(1);  // bottom
    uint32_t pad_cols1 = pads.at(2);  // left
    uint32_t pad_cols2 = pads.at(3);  // right
    // 设定shape
    Eigen::Tensor<float, 3> new_data(this->data_.dimensions()[0] + pad_rows1 + pad_rows2,
                                    this->data_.dimensions()[1] + pad_cols1 + pad_cols2,
                                    this->data_.dimensions()[2]);
    raw_shapes_[0] = new_data.dimensions()[2];
    raw_shapes_[1] = new_data.dimensions()[0];
    raw_shapes_[2] = new_data.dimensions()[1];
    // 全部填充padding_value
    new_data.setConstant(padding_value);
    // 内部再还原原来的data_
    for (int c = 0; c < this->data_.dimensions()[2]; c ++) {
        for (int i = pad_rows1; i < pad_rows1 + this->data_.dimensions()[0]; i++) {
            for (int j = pad_cols1; j < pad_cols1 + this->data_.dimensions()[1]; j ++) {
                new_data(i, j, c) = this->data_(i-pad_rows1, j-pad_cols1, c);
            }
        }
    }
    this->data_ = new_data;
}

//!
void Tensor<float>::Fill(float value) {
  SCNNI_ASSERT(this->data_.size(), "data_ is empty");
  this->data_.setConstant(value);
}


//!
void Tensor<float>::Ones() {
  SCNNI_ASSERT(this->data_.size(), "data_ is empty");
  this->Fill(1.0);
}

//!
void Tensor<float>::Rand() {
  SCNNI_ASSERT(this->data_.size(), "data_ is empty");
  this->data_.setRandom();
}

// //!
void Tensor<float>::Show() {
  for (uint32_t c = 0; c < this->Channels(); c ++) {
    for (uint32_t i = 0; i < this->Rows(); i ++) {
      for (uint32_t j = 0; j < this->Cols(); j ++) {
        std::cout << this->data_(i, j, c) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

//!
void Tensor<float>::ReRawshape(const std::vector<uint32_t>& shapes) {
  SCNNI_ASSERT(this->data_.size(), "data_ is empty");
  SCNNI_ASSERT(!shapes.empty(), "shapes is empty");
  
	const uint32_t origin_size = this->Size();  // 原始大小
  uint32_t current_size = 1;  //reshape后大小
  for (uint32_t s : shapes) {
    current_size *= s;
  }
  SCNNI_ASSERT(shapes.size() <= 3, "shapes.size() > 3");
  SCNNI_ASSERT(current_size == origin_size, "Find current_size != origin_size");

  if (shapes.size() == 3) {
		this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
		Eigen::array<Eigen::DenseIndex, 3> dim = {{shapes.at(1), shapes.at(2), shapes.at(0)}};
		this->data_ = this->data_.reshape(dim);
  } else if (shapes.size() == 2) {  //为保证data_为三维且列优先, 在0维度设1
    this->raw_shapes_ = {1, shapes.at(0), shapes.at(1)};
		Eigen::array<Eigen::DenseIndex, 3> dim = {{1, shapes.at(0), shapes.at(1)}};
		this->data_ = this->data_.reshape(dim);
  } else {  //为保证data_为三维且列优先, 在0维度和1维度设1
    this->raw_shapes_ = {1, 1, shapes.at(0)};
		Eigen::array<Eigen::DenseIndex, 3> dim = {{1, 1, shapes.at(0)}};
		this->data_ = this->data_.reshape(dim);
  }
}

void Tensor<float>::ReView(const std::vector<uint32_t>& shapes) {
  SCNNI_ASSERT(this->data_.size(), "data_ is empty");
  SCNNI_ASSERT(!shapes.empty(), "shapes is empty");
  
	const uint32_t origin_size = this->Size();
  uint32_t current_size = 1;
  for (uint32_t s : shapes) {
    current_size *= s;
  }
  SCNNI_ASSERT(shapes.size() <= 3, "shapes.size() > 3");
  SCNNI_ASSERT(current_size == origin_size, "Find current_size != origin_size");
	
  std::vector<uint32_t> target_shapes;  // channel row col
  if (shapes.size() == 3) {
    target_shapes = {shapes.at(0), shapes.at(1), shapes.at(2)};
    this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
  } else if (shapes.size() == 2) {
    target_shapes = {1, shapes.at(0), shapes.at(1)};
    this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
  } else {
    target_shapes = {1, shapes.at(0), 1};
    this->raw_shapes_ = {shapes.at(0)};
  }
  this->ReView(target_shapes);
}

void Tensor<float>::ReShape(const std::vector<uint32_t> &shapes) {
    SCNNI_ASSERT(this->data_.size(), "data_ is empty");
    SCNNI_ASSERT(!shapes.empty(), "shapes is empty");

	const uint32_t origin_size = this->Size();  // 原始大小
    uint32_t current_size = 1;  //reshape后大小
    for (uint32_t s : shapes) {
        current_size *= s;
    }
    SCNNI_ASSERT(shapes.size() <= 3, "shapes.size() > 3");
    SCNNI_ASSERT(current_size == origin_size, "Find current_size != origin_size");
    Eigen::array<Eigen::DenseIndex, 3> dim;

    if (shapes.size() == 3) {
        this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
        dim = {{shapes.at(1), shapes.at(2), shapes.at(0)}};
        // Eigen::array<Eigen::DenseIndex, 3> dim({shapes.at(1), shapes.at(2), shapes.at(0)});
        // this->data_ = this->data_.reshape(dim);
        // this->data_ = this->data_.reshape(dim).shuffle(Eigen::array<int, 3>{1, 0, 2});
        // this->data_ = this->data_.swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0}).reshape(dim).swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0});
        // Eigen::Tensor<float, 3> new_data = this->data_.swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0}).reshape(dim).swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0});
        // auto new_data = this->data_.swap_layout().shuffle(Eigen::array<int, 3>{1, 2, 0}).reshape(dim).swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0});
        // this->data_ = new_data;
    } else if (shapes.size() == 2) {  //为保证data_为三维且列优先, 在0维度设1
        this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
        dim = {{shapes.at(0), shapes.at(1), 1}};
        // Eigen::array<Eigen::DenseIndex, 3> dim({shapes.at(0), shapes.at(1), 1});
        // this->data_ = this->data_.reshape(dim);
        // this->data_ = this->data_.reshape(dim).shuffle(Eigen::array<int, 3>{1, 0, 2});
        // this->data_ = this->data_.swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0}).reshape(dim).swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0});
        // Eigen::Tensor<float, 3> new_data = this->data_.swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0}).reshape(dim).swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0});
        // auto new_data = this->data_.swap_layout().shuffle(Eigen::array<int, 3>{1, 2, 0}).reshape(dim).swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0});
        // this->data_ = new_data;
    } else {  //为保证data_为三维且列优先, 在0维度和1维度设1
        this->raw_shapes_ = {shapes.at(0)};
        dim = {{shapes.at(0), 1, 1}};
        // Eigen::array<Eigen::DenseIndex, 3> dim({shapes.at(0), 1, 1});
        // this->data_ = this->data_.reshape(dim);
        // this->data_ = this->data_.reshape(dim).shuffle(Eigen::array<int, 3>{1, 0, 2});
        // this->data_ = this->data_.shuffle(Eigen::array<int, 3>{2, 0, 1}).swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0}).reshape(dim).swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0}).shuffle(Eigen::array<int, 3>{1, 2, 0});
        // Eigen::Tensor<float, 3> new_data = (this->data_).swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0}).reshape(dim).swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0});
        // auto new_data = this->data_.swap_layout().shuffle(Eigen::array<int, 3>{1, 2, 0}).reshape(dim).swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0});
        // this->data_ = new_data;
    }
  auto *row_major_data = new float[this->data_.size()];
    for (int c = 0; c < this->data_.dimension(2); c++) {
        for (int h = 0; h < this->data_.dimension(0); h++) {
            for (int w = 0; w < this->data_.dimension(1); w++) {
                row_major_data[w + h * this->data_.dimension(1) + c * this->data_.dimension(0) * this->data_.dimension(1)] = this->data_(h, w, c);
            }
        }
    }
    this->data_ = Eigen::Tensor<float, 3>(dim[0], dim[1], dim[2]);
    for (int c = 0; c < this->data_.dimension(2); c++) {
        for (int h = 0; h < this->data_.dimension(0); h++) {
            for (int w = 0; w < this->data_.dimension(1); w++) {
                this->data_(h, w, c) = row_major_data[w + h * this->data_.dimension(1) + c * this->data_.dimension(0) * this->data_.dimension(1)];
            }
        }
    }
}
//!
void Tensor<float>::Flatten() {
    SCNNI_ASSERT(this->data_.size(), "data_ is empty");
    const uint32_t size = this->data_.size();
    this->ReRawshape({size});
}

auto Tensor<float>::Clone() -> std::shared_ptr<Tensor<float>> {
    return std::make_shared<Tensor<float>>(*this);
}
namespace img_util_func {
auto ReadJpg(const std::string& fileName, std::vector<unsigned char>& image, unsigned int& width, unsigned int& height, unsigned int& channels) -> bool {
    std::ifstream file(fileName, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // 按字节读取文件
    file.unsetf(std::ios::skipws);
    std::streampos file_size;
    file.seekg(0, std::ios::end);
    file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> file_data;
    file_data.reserve(file_size);
    file_data.insert(file_data.begin(), std::istream_iterator<unsigned char>(file), std::istream_iterator<unsigned char>());
    file.close();

    // 解析JPEG文件
    unsigned char* image_data = stbi_load_from_memory(file_data.data(), file_data.size(), reinterpret_cast<int*>(&width), reinterpret_cast<int*>(&height), reinterpret_cast<int*>(&channels), STBI_rgb);
    if (image_data == nullptr) {
        return false;
    }

    image.resize(width * height * channels);
    std::memcpy(image.data(), image_data, width * height * channels);
    stbi_image_free(image_data);

    return true;
}

auto WriteFloat(const std::string& fileName, const float* data, unsigned int size) -> bool {
    std::ofstream file(fileName, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.write(reinterpret_cast<const char*>(data), size * sizeof(float));
    file.close();

    return true;
}
}  // namespace img_util_func

//!
auto Tensor<float>::FromImage(const std::string &path, bool scaling) -> int {
    std::vector<unsigned char> image;
    unsigned int width;
    unsigned int height;
    unsigned int channels;
    bool success = img_util_func::ReadJpg(path, image, width, height, channels);
    if (!success) {
        std::cerr << "Failed to read image." << std::endl;
        return 1;
    }
    // std::cout <<"width: " << width << "  height: " << height << "  channels: " << channels << std::endl;
    // 将unsigned char数组转换为float数组
    auto* float_data = new float[width * height * channels];
    for (unsigned int i = 0; i < width * height * channels; i++) {
        float_data[i] = static_cast<float>(image[i]);
        if (scaling) {
            float_data[i] /= 255.0F;
        }
    }

    Eigen::Tensor<float, 3> img_tensor(height, width, channels);

    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            for (uint32_t c = 0; c < channels; c++) {
                img_tensor(i, j, c) = float_data[c+j*3+i*3*128];
                // if (c == 0 && i == 0) {
                //   std::cout << img_tensor(i, j, c) << " ";
                // }
            }
        }
    }
    // std::cout<<std::endl;

    this->data_ = img_tensor;
    this->raw_shapes_ = {channels, height, width};

    delete[] float_data;
    return 0;
}

auto TensorCreate(uint32_t channels, uint32_t rows, uint32_t cols) -> std::shared_ptr<Tensor<float>> {
  return std::make_shared<Tensor<float>>(channels, rows, cols);
}

auto TensorCreate(const std::vector<uint32_t>& shapes) -> std::shared_ptr<Tensor<float>> {
  SCNNI_ASSERT(shapes.size() == 3, "Find shapes.size() != 3");
  return TensorCreate(shapes.at(0), shapes.at(1), shapes.at(2));
}

auto TensorIsSame(const std::shared_ptr<Tensor<float>>& a,
                  const std::shared_ptr<Tensor<float>>& b) -> bool {
  SCNNI_ASSERT(a != nullptr, "a_ptr is nullptr");
  SCNNI_ASSERT(b != nullptr, "b_ptr is nullptr");
  if(a->Shapes() != b->Shapes()) {
    return false;
  }
	bool is_same = true;
	for(uint32_t i = 0; i < a->Channels(); i++){
		for(uint32_t j = 0; j < a->Rows(); j++){
			for(uint32_t k = 0; k < a->Cols(); k++){
				if(fabs(a->GetData()(i, j, k) - b->GetData()(i, j, k)) < 1e-5){
					is_same = false;
					break;
				}
			}
			if(!is_same) {
				break;
			}
		}
		if(!is_same){
			break;
		}
	}
  return is_same;
}


//!
void TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
                      const std::shared_ptr<Tensor<float>>& tensor2,
                      const std::shared_ptr<Tensor<float>>& output_tensor) {
  SCNNI_ASSERT(tensor1 != nullptr, "When tensorElementAdd, tensor1 is nullptr");
	SCNNI_ASSERT(tensor2 != nullptr, "When tensorElementAdd, tensor2 is nullptr");
	SCNNI_ASSERT(output_tensor != nullptr, "When tensorElementAdd, output_tensor is nullptr");
  if(tensor1->Shapes() == tensor2->Shapes()){
    SCNNI_ASSERT(tensor1->Shapes() == output_tensor->Shapes(), "");
    output_tensor->SetData(tensor1->GetData() + tensor2->GetData());
  } else{
		// broadcast
    SCNNI_ASSERT(tensor1->Channels() == tensor2->Channels(), "When tensorElementAdd, tensors shape are not adapting");
    const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
    SCNNI_ASSERT(output_tensor->Shapes() == input_tensor1->Shapes() &&
									output_tensor->Shapes() == input_tensor2->Shapes(), "");
    output_tensor->SetData(input_tensor1->GetData() + input_tensor2->GetData());
  }
}

//！
auto TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
																								const std::shared_ptr<Tensor<float>>& tensor2) -> std::shared_ptr<Tensor<float>> {
  SCNNI_ASSERT(tensor1 != nullptr, "When tensorElementAdd, tensor1 is nullptr");
	SCNNI_ASSERT(tensor2 != nullptr, "When tensorElementAdd, tensor2 is nullptr");
  if (tensor1->Shapes() == tensor2->Shapes()) {
    std::shared_ptr<Tensor<float>> output_tensor = TensorCreate(tensor1->Shapes());
		output_tensor->SetData(tensor1->GetData() + tensor2->GetData());
    return output_tensor;
  }
	// broadcast
	SCNNI_ASSERT(tensor1->Channels() == tensor2->Channels(), "When tensorElementAdd, tensors shape are not adapting");
	const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
	SCNNI_ASSERT(input_tensor1->Shapes() == input_tensor2->Shapes(), "");
	std::shared_ptr<Tensor<float>> output_tensor = TensorCreate(input_tensor1->Shapes());
	output_tensor->SetData(input_tensor1->GetData() + input_tensor2->GetData());
	return output_tensor;
}

//!
/**
 * @description: 矩阵点乘
 */
void TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2,
    const std::shared_ptr<Tensor<float>>& output_tensor) {
	SCNNI_ASSERT(tensor1 != nullptr, "When tensorElementMultiply, tensor1 is nullptr");
	SCNNI_ASSERT(tensor2 != nullptr, "When tensorElementMultiply, tensor2 is nullptr");
	SCNNI_ASSERT(output_tensor != nullptr, "When tensorElementMultiply, output_tensor is nullptr");
  if(tensor1->Shapes() == tensor2->Shapes()){
    SCNNI_ASSERT(tensor1->Shapes() == output_tensor->Shapes(), "");
		output_tensor->SetData(tensor1->GetData() * tensor2->GetData());
  } else{
		// broadcast
		SCNNI_ASSERT(tensor1->Channels() == tensor2->Channels(), "When tensorElementMultiply, tensors shape are not adapting");
		const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
		SCNNI_ASSERT(output_tensor->Shapes() == input_tensor1->Shapes() && output_tensor->Shapes() == input_tensor2->Shapes(), "");
		output_tensor->SetData(input_tensor1->GetData() + input_tensor2->GetData());
	}
}

//!
auto TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2) -> std::shared_ptr<Tensor<float>> {
  SCNNI_ASSERT(tensor1 != nullptr, "When tensorElementMultiply, tensor1 is nullptr");
	SCNNI_ASSERT(tensor2 != nullptr, "When tensorElementMultiply, tensor2 is nullptr");
  if (tensor1->Shapes() == tensor2->Shapes()) {
    std::shared_ptr<Tensor<float>> output_tensor = TensorCreate(tensor1->Shapes());
    output_tensor->SetData(tensor1->GetData() * tensor2->GetData());
    return output_tensor;
  }
	// broadcast
	SCNNI_ASSERT(tensor1->Channels() == tensor2->Channels(), "When tensorElementMultiply, tensors shape are not adapting");
	const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
	SCNNI_ASSERT(input_tensor1->Shapes() == input_tensor2->Shapes(), "");
	std::shared_ptr<Tensor<float>> output_tensor = TensorCreate(input_tensor1->Shapes());
	output_tensor->SetData(input_tensor1->GetData() + input_tensor2->GetData());
	return output_tensor;
}

//！没用到
auto TensorBroadcast(
    const std::shared_ptr<Tensor<float>>& s1,
	const std::shared_ptr<Tensor<float>>& s2) -> std::tuple<std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<float>>> {
	SCNNI_ASSERT(s1 != nullptr, "tensor1 is nullptr");
	SCNNI_ASSERT(s2 != nullptr, "tensor2 is nullptr");
    if(s1->Shapes() == s2->Shapes()){
        return {s1, s2};
    }
	SCNNI_ASSERT(s1->Channels() == s2->Channels(), "");
	if (s2->Rows() == 1 && s2->Cols() == 1) {  //s2为Flatten
		std::shared_ptr<Tensor<float>> s2add = TensorCreate(s2->Channels(), s1->Rows(), s1->Cols());
		SCNNI_ASSERT(s2->Size() == s2->Channels(), "");
		for (uint32_t c = 0; c < s2->Channels(); c++) {
			// s2add->Slice(c).Fill(s2->Index(c));
		}
		return {s1, s2add};
	}
	if (s1->Rows() == 1 && s1->Cols() == 1) {  //s1为Flatten
		std::shared_ptr<Tensor<float>> s1add = TensorCreate(s1->Channels(), s2->Rows(), s2->Cols());
		SCNNI_ASSERT(s1->Size() == s1->Channels(), "");
		for (uint32_t c = 0; c < s1->Channels(); c++) {
			// s1add->Slice(c).Fill(s1->Index(c));
		}
		return {s1add, s2};
	}
	// LOG(FATAL) << "Broadcast shape is not adapting!";
	return {s1, s2};
}

}  // namespace scnni