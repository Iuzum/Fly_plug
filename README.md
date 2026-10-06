# Fly_plug

Биомиметическая модель мухи дрозофилы для Neovim/Vim.

## Что это

Плагин для Neovim/Vim: по окну летает муха, облетает препятствия
(непустые символы буфера), уворачивается от курсора и учится на ходу.

Мозг мухи — спайковая нейросеть (SNN) на C++, вдохновлённая
подсхемами дрозофилы (Giant Fiber, Lobula Plate, Central Complex,
Mushroom Body) и coarse-grained коннектомом Hemibrain.

## Уровни

- L1 — рефлекторный агент на Lua (готов, работает).
- L2 — SNN на C++ через UNIX-сокет (в разработке).
- L3 — coarse-grained Hemibrain в SNN (в разработке).

## Требования

- macOS (Apple Silicon)
- Neovim 0.10+ или Vim 9.0+ с +lua
- CMake 3.20+, Ninja
- Apple Clang 16+ или Homebrew LLVM 18+
- Python 3.11 (для экспорта Hemibrain)

## Сборка

    cd core
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ./build/fly-server

## Данные

Hemibrain (186 MB) — в data/hemibrain.pkl, не коммитится.

## Лицензия

MIT
