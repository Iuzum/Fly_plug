cmake_minimum_required(VERSION 3.20)

# ВАЖНО: компилятор задаётся ДО project(),
# иначе cmake проигнорирует set() и зациклится на "cache delete".
if(APPLE)
    set(CMAKE_C_COMPILER   "/opt/homebrew/opt/llvm/bin/clang"   CACHE FILEPATH "")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++" CACHE FILEPATH "")
endif()

project(fly_core CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
endif()

# --- nlohmann/json через FetchContent ---
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# --- Ядро ---
add_library(fly_core STATIC
    src/snn.cpp
    src/mushroom.cpp
    src/core.cpp
)
target_include_directories(fly_core PUBLIC include)
target_link_libraries(fly_core PUBLIC nlohmann_json::nlohmann_json)

# --- Сервер ---
add_executable(fly-server src/main.cpp)
target_link_libraries(fly-server PRIVATE fly_core)
