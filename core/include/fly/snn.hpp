// snn.hpp — спайковая нейросеть (LIF-нейроны + CSR-синапсы).
// Загружает coarse_graph.json (Hemibrain coarse-grained).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fly {

// LIF-нейрон (Leaky Integrate-and-Fire)
struct LIFNeuron {
    float v = 0.0f;             // мембранный потенциал
    float v_rest = 0.0f;        // потенциал покоя
    float v_thresh = 0.5f;      // порог спайка
    float v_reset = 0.0f;       // после спайка
    float tau_m = 20.0f;        // время релаксации, мс
    float tau_ref = 2.0f;       // рефрактерный период, мс
    float ref_counter = 0.0f;   // оставшийся рефрактер
    bool fired = false;
    float last_spike = -1000.0f;  // время последнего спайка (мс), для STDP
};

// Спайковая нейросеть
class SNN {
public:
    // --- Нейроны ---
    std::vector<LIFNeuron> neurons;

    // --- CSR для входящих синапсов ---
    // row_ptr[i]..row_ptr[i+1] — диапазон в col_idx/weights для нейрона i
    std::vector<uint32_t> row_ptr;   // размер N+1
    std::vector<uint32_t> col_idx;   // pre-нейроны
    std::vector<float>    weights;   // веса синапсов

    // --- Имена типов (для отладки) ---
    std::vector<std::string> type_names;

    // --- Временные буферы ---
    std::vector<float> input_current;   // внешний ток на текущий шаг
    std::vector<bool>  spike_buffer;    // спайки текущего шага
    std::vector<bool>  prev_spikes;     // спайки предыдущего шага

    // --- Время симуляции ---
    float sim_time = 0.0f;   // мс

    // --- Загрузка ---
    void load_from_json(const std::string& path);

    // --- Шаг симуляции ---
    void step(float dt_ms);

    // --- Доступ ---
    const std::vector<bool>& spikes() const { return spike_buffer; }
    const std::vector<std::string>& types() const { return type_names; }
    size_t size() const { return neurons.size(); }

    // --- Внешний вход ---
    void add_input(size_t i, float current);
    void set_input(size_t i, float current);
    void clear_inputs();

    // --- Поиск нейрона по имени типа ---
    // Возвращает SIZE_MAX, если не найден.
    size_t find_by_type(const std::string& type) const;

    // --- Поиск всех нейронов, чьё имя начинается с prefix ---
    std::vector<size_t> find_all_by_prefix(const std::string& prefix) const;
};

} // namespace fly
