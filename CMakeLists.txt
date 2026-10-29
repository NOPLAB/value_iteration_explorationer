cmake_minimum_required(VERSION 3.8)
project(value_iteration_explorationer)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(Threads REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ExplorationMap.msg" DEPENDENCIES std_msgs nav_msgs)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)

add_library(planner STATIC
  src/planner.cpp src/actions.cpp src/global_planner.cpp src/local_planner.cpp
  src/map_layers.cpp src/exploration.cpp)
target_include_directories(planner PUBLIC include)
target_link_libraries(planner PUBLIC Threads::Threads)
target_compile_features(planner PUBLIC cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(planner PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(vi_node src/vi_node.cpp)
target_link_libraries(vi_node planner)
ament_target_dependencies(vi_node
  rclcpp ament_index_cpp nav_msgs geometry_msgs sensor_msgs tf2 tf2_ros tf2_geometry_msgs)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(vi_node PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(planner_test src/test_planner.cpp)
target_link_libraries(planner_test planner)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(planner_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

foreach(node exploration_map_node exploration_node)
  add_executable(${node} src/${node}.cpp)
  target_link_libraries(${node} planner "${cpp_typesupport_target}")
  ament_target_dependencies(${node}
    rclcpp ament_index_cpp std_msgs nav_msgs geometry_msgs sensor_msgs
    tf2 tf2_ros tf2_geometry_msgs)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${node} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endforeach()

foreach(check map_layers exploration)
  add_executable(${check}_test src/test_${check}.cpp)
  target_link_libraries(${check}_test planner)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${check}_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endforeach()

install(TARGETS vi_node exploration_map_node exploration_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(FILES example/cartographer_exploration.launch.py DESTINATION share/${PROJECT_NAME}/launch)
install(FILES example/exploration.rviz DESTINATION share/${PROJECT_NAME}/example)
install(PROGRAMS example/discrete_action_executor.py
  DESTINATION lib/${PROJECT_NAME} RENAME example_action_executor)

if(BUILD_TESTING)
  find_package(ament_cmake_test REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_cmake_cppcheck REQUIRED)
  find_package(ament_cmake_lint_cmake REQUIRED)
  find_package(ament_cmake_flake8 REQUIRED)
  find_package(ament_cmake_xmllint REQUIRED)
  ament_add_test(planner_test COMMAND "$<TARGET_FILE:planner_test>"
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO TIMEOUT 180)
  ament_add_pytest_test(map_service_test test/test_map_service.py TIMEOUT 90)
  foreach(check map_layers exploration)
    ament_add_test(${check}_test COMMAND "$<TARGET_FILE:${check}_test>"
      GENERATE_RESULT_FOR_RETURN_CODE_ZERO TIMEOUT 180)
  endforeach()
  ament_add_pytest_test(exploration_ros_test test/test_exploration_ros.py TIMEOUT 300)
  ament_add_pytest_test(scan_tf_queue_ros_test test/test_scan_tf_queue_ros.py TIMEOUT 90)
  ament_add_pytest_test(action_executor_test test/test_action_executor.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/example" TIMEOUT 30)
  # Ubuntu 22.04 ships cppcheck 2.7; run its analysis rather than skip it.
  ament_cppcheck()
  set_tests_properties(cppcheck PROPERTIES
    ENVIRONMENT "AMENT_CPPCHECK_ALLOW_SLOW_VERSIONS=1")
  ament_lint_cmake()
  ament_flake8()
  ament_xmllint()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
