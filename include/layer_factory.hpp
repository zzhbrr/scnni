/*
 * @Author: zzh
 * @Date: 2023-03-04 08:53:12
 * @LastEditTime: 2023-03-04 09:37:25
 * @Description: 
 * @FilePath: /SCNNI/include/layer_factory.hpp
 */
#ifndef SCNNI_LAYER_FACTORY_HPP_
#define SCNNI_LAYER_FACTORY_HPP_

#include "layer.hpp"
#include <map>
#include <memory>
#include <string>
namespace scnni {
class LayerRegister {
  public:
    using layer_creator_func = std::shared_ptr<Layer> (*)();

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
    static auto CreateLayer() -> std::shared_ptr<Layer*>;
}
} // namespace scnni

#endif