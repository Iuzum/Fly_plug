// main.cpp — расширенные тесты SNN: подробная печать динамики.
#include "fly/snn.hpp"
#include "fly/motor.hpp"
#include "fly/sensor.hpp"

#include <cstdio>
#include <string>

namespace {

void run_test(const char* name, const fly::SensorInput& input,
              fly::SensorEncoder& encoder, fly::SNN& snn,
              fly::MotorDecoder& motor, int steps) {
    std::printf("\n=== %s ===\n", name);
    for (int t = 0; t < steps; ++t) {
        encoder.encode(input, snn, 1.0f);
        snn.step(1.0f);

        size_t n_spikes = 0;
        for (bool s : snn.spikes()) if (s) n_spikes++;

        auto action = motor.decode(snn.spikes());
        size_t active = motor.active_count(snn.spikes());

        std::printf("  шаг %3d: спайков %4zu, action=(%2d,%2d), DN активных %zu",
                    t, n_spikes, action.dx, action.dy, active);
        if (active > 0) {
            std::printf(" | DN:");
            for (auto id : snn.find_all_by_prefix("DN")) {
                if (id < snn.size() && snn.spikes()[id]) {
                    std::printf(" %s", snn.types()[id].c_str());
                }
            }
        }
        std::printf("\n");
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string graph_path = (argc > 1) ? argv[1] : "../data/coarse_graph.json";

    std::printf("fly-server v0.3.0 (full tests)\n");
    std::printf("Загружаю: %s\n", graph_path.c_str());

    fly::SNN snn;
    try {
        snn.load_from_json(graph_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ОШИБКА: %s\n", e.what());
        return 1;
    }

    std::printf("SNN: %zu нейронов, %zu синапсов\n",
                snn.size(), snn.col_idx.size());

    fly::SensorEncoder encoder;
    fly::MotorDecoder motor;

    try {
        encoder.init(snn);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SensorEncoder::init ошибка: %s\n", e.what());
        return 1;
    }

    auto dn_ids = snn.find_all_by_prefix("DN");
    motor.init(dn_ids);

    // === Тест 1: без входа, 30 шагов ===
    fly::SensorInput empty_input;
    run_test("Тест 1: без входа (30 шагов)", empty_input,
             encoder, snn, motor, 30);

    // === Тест 2: курсор близко справа ===
    fly::SensorInput cursor_input;
    cursor_input.cursor_dist = 0.1f;
    cursor_input.cursor_dx = 1.0f;   // курсор справа
    cursor_input.cursor_dy = 0.0f;
    run_test("Тест 2: курсор близко справа (30 шагов)", cursor_input,
             encoder, snn, motor, 30);

    // === Тест 3: курсор близко слева ===
    fly::SensorInput cursor_left;
    cursor_left.cursor_dist = 0.1f;
    cursor_left.cursor_dx = -1.0f;   // курсор слева
    cursor_left.cursor_dy = 0.0f;
    run_test("Тест 3: курсор близко слева (30 шагов)", cursor_left,
             encoder, snn, motor, 30);

    // === Тест 4: курсор сверху ===
    fly::SensorInput cursor_up;
    cursor_up.cursor_dist = 0.1f;
    cursor_up.cursor_dx = 0.0f;
    cursor_up.cursor_dy = -1.0f;     // курсор сверху
    run_test("Тест 4: курсор сверху (30 шагов)", cursor_up,
             encoder, snn, motor, 30);

    // === Тест 5: стены вокруг ===
    fly::SensorInput wall_input;
    wall_input.flow.flow.fill(1.0f);
    wall_input.cursor_dist = 1.0f;
    run_test("Тест 5: стены вокруг (30 шагов)", wall_input,
             encoder, snn, motor, 30);

    // === Тест 6: стена только справа ===
    fly::SensorInput wall_right;
    for (int i = 0; i < 64; ++i) {
        int x = i % 8;
        wall_right.flow.flow[i] = (x >= 6) ? 1.0f : 0.0f;
    }
    wall_right.cursor_dist = 1.0f;
    run_test("Тест 6: стена справа (30 шагов)", wall_right,
             encoder, snn, motor, 30);

    std::printf("\n=== Готово ===\n");
    return 0;
}
