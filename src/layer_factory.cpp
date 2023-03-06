/*
 * @Author: zzh
 * @Date: 2023-03-04
 * @LastEditTime: 2023-03-06 05:27:50
 * @Description: 
 * @FilePath: /SCNNI/src/layer_factory.cpp
 */
#include "scnni/layer_factory.hpp"
#include "scnni/macros.h"
#include <memory>
#include <string>

namespace scnni {

auto LayerRegister::GetRegistry() -> LayerRegister::Registry & {
    static auto *k_registry = new Registry();
    SCNNI_ASSERT(k_registry != nullptr, "Registry init fail");
    return *k_registry;
}

void LayerRegister::RegistLayer(const std::string &layer_type, const layer_creator_func &creator) {
    SCNNI_ASSERT(creator != nullptr, "");
    Registry &registry = GetRegistry();
    SCNNI_ASSERT(registry.count(layer_type), "Layer has been registered");
    registry.insert({layer_type, creator});
}

auto LayerRegister::CreateLayer(const std::string &layer_type, const std::shared_ptr<Operator> &op) -> std::shared_ptr<Layer> {
    Registry &registry = GetRegistry();
    SCNNI_ASSERT(registry.count(layer_type) != 0, "Layer has now been registered");
    const layer_creator_func &creator = registry.find(layer_type)->second;
    SCNNI_ASSERT(creator != nullptr, "Layer Creator is null");
    std::shared_ptr<Layer> layer = creator(op);
    return layer;
}

} // namespace scnni