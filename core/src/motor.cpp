// motor.cpp — реализация MotorDecoder. v3: правильный маппинг DN -> направлений.
#include "fly/motor.hpp"
#include "fly/snn.hpp"

#include <array>
#include <cmath>
#include <cstdio>

namespace fly {

void MotorDecoder::init(const std::vector<size_t>& dn_ids) {
    dn_ids_ = dn_ids;
    size_t N = dn_ids.size();

    // 8 направлений: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
    // Порядок DN: DN1pA, DN1pB, DN?, DNES1, DNa02, ...
    // Первые 4 DN — «прямые моторы» (sensor.cpp подаёт в них ток):
    //   DN[0] DN1pA  -> E (вправо)
    //   DN[1] DN1pB  -> W (влево)
    //   DN[2] DN?    -> S (вниз)
    //   DN[3] DNES1  -> N (вверх)
    // Остальные — диагонали.
    static const size_t DIRS[] = {
        2,   // DN[0] DN1pA  -> E
        6,   // DN[1] DN1pB  -> W
        4,   // DN[2] DN?    -> S
        0,   // DN[3] DNES1  -> N
        1,   // DN[4] DNa02  -> NE
        7,   // DN[5] DNa03  -> NW
        3,   // DN[6] DNb01  -> SE
        5,   // DN[7] DNb05  -> SW
    };
    constexpr size_t DIRS_SIZE = sizeof(DIRS) / sizeof(DIRS[0]);

    w_.assign(N * 8, 0.0f);
    for (size_t i = 0; i < N; ++i) {
        size_t d = (i < DIRS_SIZE) ? DIRS[i] : (i % 8);
        w_[i * 8 + d] = 1.0f;
    }

    std::printf("MotorDecoder::init: %zu DN -> 8 directions\n", N);
    std::printf("  mapping: DN[0]->E, DN[1]->W, DN[2]->S, DN[3]->N, ...\n");
}

size_t MotorDecoder::active_count(const std::vector<bool>& spikes) const {
    size_t count = 0;
    for (auto id : dn_ids_) {
        if (id < spikes.size() && spikes[id]) count++;
    }
    return count;
}

Action MotorDecoder::decode(const std::vector<bool>& spikes) const {
    std::array<float, 8> dir_acc{};
    dir_acc.fill(0.0f);

    for (size_t i = 0; i < dn_ids_.size(); ++i) {
        size_t id = dn_ids_[i];
        if (id >= spikes.size()) continue;
        if (!spikes[id]) continue;
        for (size_t d = 0; d < 8; ++d) {
            dir_acc[d] += w_[i * 8 + d];
        }
    }

    size_t best = 0;
    float best_v = dir_acc[0];
    for (size_t d = 1; d < 8; ++d) {
        if (dir_acc[d] > best_v) {
            best_v = dir_acc[d];
            best = d;
        }
    }

    if (best_v <= 0.0f) return {0, 0};

    static const int dirs[8][2] = {
        {0, -1},  // N
        {1, -1},  // NE
        {1, 0},   // E
        {1, 1},   // SE
        {0, 1},   // S
        {-1, 1},  // SW
        {-1, 0},  // W
        {-1, -1}  // NW
    };
    return {dirs[best][0], dirs[best][1]};
}

} // namespace fly
