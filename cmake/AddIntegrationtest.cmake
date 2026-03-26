function(add_integrationtest integrationtest_name)
  get_filename_component(integrationtest ${integrationtest_name} NAME_WE)

  list(APPEND CMAKE_MESSAGE_INDENT "  ") #indent +1
  message(STATUS "Set-up integrationtest: ${integrationtest}")
  list(POP_BACK CMAKE_MESSAGE_INDENT)    #indent -1
  
  add_executable(${integrationtest} EXCLUDE_FROM_ALL
    ${integrationtest_name}
  )
  add_dependencies(Integrationtests ${integrationtest})

  #target_compile_definitions(${integrationtest} PRIVATE UNITTEST=1)

  target_include_directories(${integrationtest} SYSTEM PRIVATE ${FINN_SRC_DIR})
  target_link_libraries(${integrationtest} PRIVATE gtest finnc_core finnc_options Threads::Threads vrt jsoncpp uuid nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)
endfunction()
