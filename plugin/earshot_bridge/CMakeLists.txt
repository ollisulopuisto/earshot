cmake_minimum_required(VERSION 3.22)
project(earshot_bridge VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
FetchContent_Declare(
    clap_wrapper
    GIT_REPOSITORY https://github.com/free-audio/clap-wrapper.git
    GIT_TAG 806bba0a5f792e3966b99c1561ab9952d3316d7c
)
set(CLAP_WRAPPER_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "" FORCE)
set(CLAP_WRAPPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CLAP_WRAPPER_BUILD_STANDALONE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(clap_wrapper)

add_library(earshot_bridge_clap MODULE src/earshot_bridge_entry.cpp src/earshot_bridge_plugin.cpp)
target_include_directories(earshot_bridge_clap PRIVATE src)
target_link_libraries(earshot_bridge_clap PRIVATE clap)

# 1. VST3 target: works on macOS, Linux, Windows (Reaper, Ableton Live, Bitwig, Cubase, etc.)
target_add_vst3_wrapper(
    TARGET earshot_bridge_clap
    OUTPUT_NAME "EarshotBridge"
)

# 2. Audio Unit (AUv2) target: macOS (Logic Pro, Ableton Live, Reaper)
if(APPLE)
    add_library(earshot_bridge_au MODULE src/earshot_bridge_entry.cpp src/earshot_bridge_plugin.cpp)
    target_include_directories(earshot_bridge_au PRIVATE src)
    target_link_libraries(earshot_bridge_au PRIVATE clap)
    target_add_auv2_wrapper(
        TARGET earshot_bridge_au
        OUTPUT_NAME "EarshotBridge"
        BUNDLE_IDENTIFIER "fi.iki.dst.earshot.bridge.auv2"
        MANUFACTURER_NAME "earshot"
        MANUFACTURER_CODE "Ersh"
        SUBTYPE_CODE "ErBr"
        INSTRUMENT_TYPE "aufx"
    )
endif()

# 3. Pro Tools AAX target: when building with AAX SDK support (macOS/Windows)
if(CLAP_WRAPPER_CAN_BUILD_AAX AND CLAP_WRAPPER_BUILD_AAX)
    add_library(earshot_bridge_aax MODULE src/earshot_bridge_entry.cpp src/earshot_bridge_plugin.cpp)
    target_include_directories(earshot_bridge_aax PRIVATE src)
    target_link_libraries(earshot_bridge_aax PRIVATE clap)
    target_add_aax_wrapper(
        TARGET earshot_bridge_aax
        OUTPUT_NAME "EarshotBridge"
        BUNDLE_IDENTIFIER "fi.iki.dst.earshot.bridge.aax"
    )
endif()
