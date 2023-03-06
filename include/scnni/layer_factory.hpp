/*
 * @Author: zzh
 * @Date: 2023-03-04 08:53:12
 * @LastEditTime: 2023-03-06 11:20:28
 * @Description: 
 * @FilePath: /scnni/include/scnni/layer_factory.hpp
 */
#ifndef SCNNI_LAYER_FACTORY_HPP_
#define SCNNI_LAYER_FACTORY_HPP_

#include "scnni/layer.hpp"
#include "scnni/operator.hpp"
#include <map>
#include <memory>
#include <string>
namespace scnni {
class LayerRegister {
  public:
    // using layer_creator_func = std::shared_ptr<Layer> (*)();
    using layer_creator_func = std::shared_ptr<Layer> (*)(const std::shared_ptr<Operator> &op);

    // Registry: Layer_type到layer_creator_func的映射
    using Registry = std::map<std::string, layer_creator_func>;

    /**
     * @description: 获取 Registry
     * @return {*}
     */
    static auto GetRegistry() -> Registry &;

    /**
     * @description: 创建Layer
     * @return {*}
     */
    static auto CreateLayer(const std::string &layer_type, const std::shared_ptr<Operator> &op) -> std::shared_ptr<Layer>;

    /**
     * @description: 注册Layer到Registry中
     * @return {*}
     */
    static void RegistLayer(const std::string &layer_type, const layer_creator_func &creator); 
};

class LayerRegistelrWrapper{
  public:
    LayerRegistelrWrapper(const std::string &layer_type, const LayerRegister::layer_creator_func &creator) {
      LayerRegister::RegistLayer(layer_type, creator);
    }
};

} // namespace scnni

#endif