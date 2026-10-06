#include <iostream>
#include <vector>
#include <ranges>
#include <string>
#include <format>

int main(int argc, char** argv) {
    std::cout << "fly-server v0.1.0\n";
    std::cout << "C++ standard: " << __cplusplus << "\n";

    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto even = v | std::views::filter([](int n){ return n % 2 == 0; })
                  | std::views::transform([](int n){ return n * n; });

    std::cout << "C++20 ranges: ";
    for (int n : even) std::cout << n << " ";
    std::cout << "\n";

    std::string msg = std::format("fly-server ready, args={}", argc);
    std::cout << msg << "\n";

    return 0;
}
