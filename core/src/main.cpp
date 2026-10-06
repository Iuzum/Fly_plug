// main.cpp — тест загрузки SNN и первого шага симуляции.
#include "fly/snn.hpp"
#include "fly/motor.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string graph_path = (argc > 1) ? argv[1] : "../data/coarse_graph.json";

    std::cout << "fly-server v0.1.0 (SNN test)\n";
    std::cout << "Загружаю: " << graph_path << "\n";

    fly::SNN snn;
    try {
        snn.load_from_json(graph_path);
    } catch (const std::exception& e) {
        std::cerr << "ОШИБКА: " << e.what() << "\n";
        return 1;
    }

    std::cout << "SNN загружена:\n";
    std::cout << "  нейронов: " << snn.size() << "\n";
    std::cout << "  синапсов: " << snn.col_idx.size() << "\n";

    // Ищем DN-нейроны
    auto dn_ids = snn.find_all_by_prefix("DN");
    std::cout << "  DN-нейронов: " << dn_ids.size() << "\n";
    for (auto id : dn_ids) {
        std::cout << "    [" << id << "] " << snn.types()[id] << "\n";
    }

    // Прогоняем 20 шагов симуляции без входа
    std::cout << "\nСимуляция 20 шагов (dt=1ms), без входа:\n";
    for (int t = 0; t < 20; ++t) {
        snn.step(1.0f);

        size_t n_spikes = 0;
        for (bool s : snn.spikes()) if (s) n_spikes++;

        if (n_spikes > 0 || t < 5) {
            std::cout << "  шаг " << t << ": спайков " << n_spikes << "\n";
        }
    }

    std::cout << "\nВсё ок. SNN работает.\n";
    return 0;
}
