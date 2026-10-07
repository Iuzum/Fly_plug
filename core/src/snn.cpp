// snn.cpp — реализация SNN. Загрузка coarse_graph.json, шаг симуляции.
// Нормализует raw weights из Hemibrain в разумный диапазон.
#include "fly/snn.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace fly {

void SNN::load_from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Не могу открыть: " + path);
    json j = json::parse(f);

    // --- Типы ---
    type_names = j["types"].get<std::vector<std::string>>();
    size_t N = type_names.size();
    if (N == 0) throw std::runtime_error("Пустой граф (0 типов)");

    // --- Рёбра ---
    auto& edges = j["edges"];

    // --- Подсчёт входящих синапсов ---
    std::vector<uint32_t> in_degree(N, 0);
    for (auto& e : edges) {
        uint32_t to = e["to_idx"].get<uint32_t>();
        if (to >= N) throw std::runtime_error("to_idx вне диапазона");
        in_degree[to]++;
    }

    // --- CSR: префиксная сумма ---
    row_ptr.resize(N + 1);
    row_ptr[0] = 0;
    for (size_t i = 0; i < N; ++i) {
        row_ptr[i + 1] = row_ptr[i] + in_degree[i];
    }

    size_t S = row_ptr[N];
    col_idx.resize(S);
    weights.resize(S);

    // --- Заполнение CSR ---
    std::vector<uint32_t> cursor(N, 0);
    for (auto& e : edges) {
        uint32_t from = e["from_idx"].get<uint32_t>();
        uint32_t to   = e["to_idx"].get<uint32_t>();
        float    w    = e["weight"].get<float>();

        if (from >= N) throw std::runtime_error("from_idx вне диапазона");

        uint32_t pos = row_ptr[to] + cursor[to];
        col_idx[pos] = from;
        weights[pos] = w;
        cursor[to]++;
    }

    // === НОРМАЛИЗАЦИЯ ВЕСОВ ===
    // Нормализуем ПО MAX: максимальный вес -> WEIGHT_MAX.
    // Это не сжимает слабые связи (в отличие от нормировки по mean).
    if (S > 0) {
        float w_max = 0.0f;
        for (auto w : weights) w_max = std::max(w_max, std::abs(w));
        if (w_max < 0.01f) w_max = 1.0f;

        // Max синаптический вес -> 0.5 (разумно для LIF с порогом 0.5)
        // Один сильный синапс -> половина порога. Много -> спайк.
        constexpr float WEIGHT_MAX = 0.5f;
        float scale = WEIGHT_MAX / w_max;

        for (auto& w : weights) w *= scale;

        float w_min_after = *std::min_element(weights.begin(), weights.end());
        float w_max_after = *std::max_element(weights.begin(), weights.end());
        std::printf("SNN weight normalization: max_before=%.0f, scale=%.8f, "
                    "range_after=[%.6f, %.4f]\n",
                    w_max, scale, w_min_after, w_max_after);
    }

    // --- Инициализация нейронов ---
    neurons.resize(N);
    for (auto& n : neurons) {
        n.v = n.v_rest;
        n.tau_m = 20.0f;
        n.tau_ref = 2.0f;
        n.v_thresh = 0.5f;
        n.v_reset = 0.0f;
    }

    // --- Буферы ---
    input_current.assign(N, 0.0f);
    spike_buffer.assign(N, false);
    prev_spikes.assign(N, false);
    sim_time = 0.0f;
}

void SNN::step(float dt_ms) {
    prev_spikes = spike_buffer;
    std::fill(spike_buffer.begin(), spike_buffer.end(), false);

    const size_t N = neurons.size();

    // 1) Синаптические токи от предыдущих спайков
    for (size_t i = 0; i < N; ++i) {
        float syn = 0.0f;
        for (uint32_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            if (prev_spikes[col_idx[k]]) {
                syn += weights[k];
            }
        }
        input_current[i] += syn;
    }

    // 2) Обновление нейронов
    for (size_t i = 0; i < N; ++i) {
        LIFNeuron& n = neurons[i];

        if (n.ref_counter > 0.0f) {
            n.ref_counter -= dt_ms;
            n.v = n.v_reset;
            n.fired = false;
            continue;
        }

        float dv = (-(n.v - n.v_rest) + input_current[i]) / n.tau_m;
        n.v += dv * dt_ms;

        if (n.v >= n.v_thresh) {
            n.v = n.v_reset;
            n.ref_counter = n.tau_ref;
            n.fired = true;
            n.last_spike = sim_time;
            spike_buffer[i] = true;
        } else {
            n.fired = false;
        }
    }

    // 3) Сброс токов
    std::fill(input_current.begin(), input_current.end(), 0.0f);

    // 4) Время
    sim_time += dt_ms;
}

void SNN::add_input(size_t i, float current) {
    if (i < input_current.size()) input_current[i] += current;
}

void SNN::set_input(size_t i, float current) {
    if (i < input_current.size()) input_current[i] = current;
}

void SNN::clear_inputs() {
    std::fill(input_current.begin(), input_current.end(), 0.0f);
}

size_t SNN::find_by_type(const std::string& type) const {
    for (size_t i = 0; i < type_names.size(); ++i) {
        if (type_names[i] == type) return i;
    }
    return SIZE_MAX;
}

std::vector<size_t> SNN::find_all_by_prefix(const std::string& prefix) const {
    std::vector<size_t> result;
    for (size_t i = 0; i < type_names.size(); ++i) {
        if (type_names[i].rfind(prefix, 0) == 0) {
            result.push_back(i);
        }
    }
    return result;
}

} // namespace fly
