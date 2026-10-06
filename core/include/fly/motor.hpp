// motor.hpp — маппинг DN (16) + выходных нейронов → 8 направлений полёта.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace fly {

// 8 направлений: N, NE, E, SE, S, SW, W, NW
// dx, dy ∈ {-1, 0, 1}
struct Action {
    int dx = 0;
    int dy = 0;
};

// Декодер: собирает активность моторных нейронов (DN) и выбирает направление.
class MotorDecoder {
public:
    // ids — индексы DN-нейронов в SNN (обычно 16 штук).
    // Веса: 16 × 8, где w[i][d] — вклад DN i в направление d.
    // Если пусто — инициализируется детерминированно (равномерно по кругу).
    void init(const std::vector<size_t>& dn_ids);

    // По спайкам SNN выбирает действие.
    Action decode(const std::vector<bool>& spikes) const;

    // Для отладки: сколько DN активно в данный момент.
    size_t active_count(const std::vector<bool>& spikes) const;

private:
    std::vector<size_t> dn_ids_;
    // w_[i * 8 + d] — вклад DN i в направление d
    std::vector<float> w_;
};

} // namespace fly
