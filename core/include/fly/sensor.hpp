// sensor.hpp — интерфейс сенсоров: 8×8 поток + курсор → входные токи SNN.
#pragma once

#include <array>
#include <cstddef>      // size_t
#include <cstdint>      // uint32_t
#include <vector>

#include "fly/snn.hpp"

namespace fly {

// Оптический поток 8×8 (пиксели вокруг мухи)
struct OpticalFlow {
    // flow[y * 8 + x] = 0.0 (пусто) .. 1.0 (стена)
    std::array<float, 64> flow{};
};

// Внешний вход для мухи
struct SensorInput {
    OpticalFlow flow;          // 8×8 стены
    float cursor_dx = 0.0f;    // -1..1, относительный x курсора
    float cursor_dy = 0.0f;    // -1..1, относительный y курсора
    float cursor_dist = 1.0f;  // 0 (близко) .. 1 (далеко), нормированное
};

// Кодировщик сенсоров: раскладывает SensorInput на входные токи SNN.
class SensorEncoder {
public:
    // Находит нужные нейроны SNN по префиксам типов.
    void init(const SNN& snn);

    // Подаёт токи в SNN на основе SensorInput.
    void encode(const SensorInput& input, SNN& snn, float dt_ms);

    // Для отладки: сколько сенсорных нейронов найдено.
    std::size_t n_flow_neurons() const { return flow_neurons_.size(); }
    std::size_t n_cursor_neurons() const { return cursor_neurons_.size(); }

private:
    // Индексы нейронов SNN, куда идёт оптический поток.
    std::vector<std::size_t> flow_neurons_;      // 64 LC-нейрона
    // Индексы нейронов «страха» (LC4, LC16)
    std::vector<std::size_t> fear_neurons_;
    // Индексы моторных нейронов (DN)
    std::vector<std::size_t> motor_neurons_;
    // Индексы «курсорных» нейронов (для dx/dy)
    std::vector<std::size_t> cursor_neurons_;
};

} // namespace fly
