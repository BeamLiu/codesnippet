cmake_minimum_required(VERSION 3.16)

project(agent_harmess_qt_sample VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC OFF)
set(CMAKE_AUTORCC ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(PROJECT_SOURCES
    main.cpp
    mainwindow.cpp
    mainwindow.h
)

if(QT_VERSION_MAJOR GREATER_EQUAL 6)
    qt_add_executable(agent_harmess
        ${PROJECT_SOURCES}
    )
else()
    add_executable(agent_harmess
        ${PROJECT_SOURCES}
    )
endif()

target_link_libraries(agent_harmess PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
