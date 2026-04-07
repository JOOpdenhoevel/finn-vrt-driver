#add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(config) 

add_executable(finnhpc FINNDriver.cpp)
target_include_directories(finnhpc PRIVATE ${FINN_SRC_DIR})
target_link_libraries(finnhpc PRIVATE finnc_core finnc_options Threads::Threads vrt uuid finn_config nlohmann_json::nlohmann_json JsonCpp::JsonCpp OpenMP::OpenMP_CXX)
set_target_properties(finnhpc
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)




