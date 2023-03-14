/*
 * @Author: zzh
 * @Date: 2023-03-04 08:53:12
 * @LastEditTime: 2023-03-13 15:15:41
 * @Description: 
 * @FilePath: /SCNNI/include/scnni/layer_factory.hpp
 */
#ifndef SCNNI_LAYER_FACTORY_HPP_
#define SCNNI_LAYER_FACTORY_HPP_

#include "scnni/layer.hpp"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
namespace scnni {
class Layer;
class Operator;

class LayerRegister {
	public:
        // 指定别名
        using layer_creator_function = std::function<Layer*(const std::shared_ptr<Operator> &)>;

        // Registry: Layer_type到layer_creator_func的映射
        using Registry = std::map<std::string, layer_creator_function>;

        /**
        * @description: 获取 Registry
        * @return {*}
        */
        static auto GetRegistry() -> Registry &;

        /**
        * @description: 创建Layer
        * @return {*}
        */
        static auto CreateLayer(const std::string &layer_type, const std::shared_ptr<Operator> &op) -> Layer*;

        /**
        * @description: 注册Layer到Registry中
        * @return {*}
        */
        static void RegistLayer(
                const std::string &layer_type,
                const layer_creator_function& creator);
};

class LayerRegistelrWrapper{
	public:
        LayerRegistelrWrapper(
            const std::string &layer_type,
            const LayerRegister::layer_creator_function& creator) {
            LayerRegister::RegistLayer(layer_type, creator);
        }
};

} // namespace scnni

#endif