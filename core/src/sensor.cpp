// sensor.cpp — реализация SensorEncoder.
#include "fly/sensor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <stdexcept>

namespace fly {

void SensorEncoder::init(const SNN& snn) {
    flow_neurons_.clear();
    fear_neurons_.clear();
    motor_neurons_.clear();
    cursor_neurons_.clear();

    // 1. Ищем LC-нейроны (визуальные) для потока
    for (const auto& prefix : {"LC10", "LC12", "LC17", "LC9", "LC18",
                                "LC11", "LC13", "LC21"}) {
        auto ids = snn.find_all_by_prefix(prefix);
        for (auto id : ids) flow_neurons_.push_back(id);
    }

    // 2. Looming-нейроны (LC4, LC16) — «страх курсора»
    for (const auto& prefix : {"LC4", "LC16"}) {
        auto ids = snn.find_all_by_prefix(prefix);
        for (auto id : ids) fear_neurons_.push_back(id);
    }

    // 3. DN-нейроны (моторный выход)
    auto dn_ids = snn.find_all_by_prefix("DN");
    for (auto id : dn_ids) motor_neurons_.push_back(id);

    // 4. Курсорные нейроны — берём первые 4 DN для dx/dy уворота
    cursor_neurons_ = motor_neurons_;

    std::printf("SensorEncoder::init:\n");
    std::printf("  flow_neurons (LC): %zu\n", flow_neurons_.size());
    std::printf("  fear_neurons (LC4, LC16): %zu\n", fear_neurons_.size());
    std::printf("  motor_neurons (DN): %zu\n", motor_neurons_.size());

    if (flow_neurons_.empty()) {
        throw std::runtime_error("Не найдены LC-нейроны для потока");
    }
    if (fear_neurons_.empty()) {
        std::printf("  WARN: LC4/LC16 не найдены — страх не подключён\n");
    }
    if (motor_neurons_.empty()) {
        throw std::runtime_error("Не найдены DN-нейроны для моторов");
    }
}

void SensorEncoder::encode(const SensorInput& input, SNN& snn, float /*dt_ms*/) {
    // === 1. Поток (8×8) → LC-нейроны ===
    const std::size_t n_flow = flow_neurons_.size();
    if (n_flow > 0) {
        std::vector<float> flow_current(n_flow, 0.0f);
        for (std::size_t i = 0; i < 64; ++i) {
            std::size_t target = i % n_flow;
            flow_current[target] += input.flow.flow[i];
        }
        const float flow_gain = 10.0f;
        for (std::size_t i = 0; i < n_flow; ++i) {
            if (flow_current[i] > 0.0f) {
                snn.add_input(flow_neurons_[i], flow_current[i] * flow_gain);
            }
        }
    }

    // === 2. Курсор → LC4/LC16 («страх») ===
    if (!fear_neurons_.empty()) {
        float d = std::clamp(input.cursor_dist, 0.0f, 1.0f);
        float fear_signal = (1.0f - d) * (1.0f - d);
        const float fear_gain = 30.0f;
        for (auto id : fear_neurons_) {
            snn.add_input(id, fear_signal * fear_gain);
        }
    }

    // === 3. Курсор dx/dy → DN (рефлекс уворота) ===
    if (!motor_neurons_.empty() && input.cursor_dist < 0.5f) {
        float dx = -input.cursor_dx;
        float dy = -input.cursor_dy;
        float urgency = (1.0f - input.cursor_dist) * 15.0f;

        if (motor_neurons_.size() >= 4) {
            snn.add_input(motor_neurons_[0], std::max(0.0f, dx) * urgency);
            snn.add_input(motor_neurons_[1], std::max(0.0f, -dx) * urgency);
            snn.add_input(motor_neurons_[2], std::max(0.0f, dy) * urgency);
            snn.add_input(motor_neurons_[3], std::max(0.0f, -dy) * urgency);
        }
    }
}

} // namespace fly
