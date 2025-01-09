cmake_minimum_required(VERSION 2.18...3.16)
project(fieldscale_publisher)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")
set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE MATCHES Release)
    message("Release Mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
elif (CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug Mode")
    add_definitions(-DDEBUG -DDEBUGMSF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
endif ()

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  cv_bridge
  image_transport
)

find_package(Eigen3 REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs cv_bridge image_transport
)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

set(SOURCE src/fieldscale.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCE})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(fieldscale_node src/fieldscale.cpp)
target_link_libraries(fieldscale_node ${PROJECT_NAME} ${catkin_LIBRARIES})
