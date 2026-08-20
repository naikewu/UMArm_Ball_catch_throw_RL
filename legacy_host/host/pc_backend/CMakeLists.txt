cmake_minimum_required(VERSION 3.16)

project(vnema_pc_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(vnema_backend
    src/main.cpp
    src/umarm_fk.cpp
    src/viz_shared_memory.cpp
)

if(WIN32)
    target_compile_definitions(vnema_backend PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_link_libraries(vnema_backend PRIVATE winmm)
endif()

target_compile_options(vnema_backend PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

target_link_options(vnema_backend PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-static -static-libgcc -static-libstdc++>
)
