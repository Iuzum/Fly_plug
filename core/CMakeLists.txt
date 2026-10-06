cmake_minimum_required(VERSION 3.20)
project(fly_core CXX)

# Homebrew LLVM 22 — свежий clang, C++20/23.
# Требует CLT 26.6+ (SDK 26.5). Если нет — закомментируй A, раскомментируй B.
if(APPLE)
    set(CMAKE_C_COMPILER   "/opt/homebrew/opt/llvm/bin/clang")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")

    # Вариант B: Apple Clang (fallback)
    # set(CMAKE_C_COMPILER   "/usr/bin/clang")
    # set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
endif()

add_library(fly_core STATIC
    src/snn.cpp
    src/mushroom.cpp
    src/core.cpp
)
target_include_directories(fly_core PUBLIC include)

add_executable(fly-server src/main.cpp)
target_link_libraries(fly-server PRIVATE fly_core)
