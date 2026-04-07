file(GLOB_RECURSE CORE_SRC *.cpp)

add_library(finnc_core STATIC ${CORE_SRC})
target_include_directories(finnc_core SYSTEM PUBLIC ${FINN_SRC_DIR})
target_link_libraries(finnc_core PUBLIC finnc_options finn_config Threads::Threads vrt JsonCpp::JsonCpp LibXml2::LibXml2 rt uuid)

add_library(finnc_core_test STATIC ${CORE_SRC})
target_include_directories(finnc_core_test SYSTEM PUBLIC ${FINN_SRC_DIR})
target_link_libraries(finnc_core_test PUBLIC finnc_options finn_config vrt JsonCpp::JsonCpp LibXml2::LibXml2)
target_compile_definitions(finnc_core_test PRIVATE UNITTEST=1)

# target_link_libraries(finnc_core_test PUBLIC ${Boost_LIBRARIES} finnc_options finnc_utils Threads::Threads OpenCL vrt JsonCpp::JsonCpp uuid)

set_target_properties(finnc_core
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(finnc_core_test
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
