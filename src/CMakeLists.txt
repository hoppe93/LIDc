
set(lid_core
	"${PROJECT_SOURCE_DIR}/src/dreamoutput.cpp"
	"${PROJECT_SOURCE_DIR}/src/integrate.cpp"
)

set(lid_py
	"${PROJECT_SOURCE_DIR}/src/numpy.cpp"
	"${PROJECT_SOURCE_DIR}/src/py.cpp"
)

set(lid_exe
	"${PROJECT_SOURCE_DIR}/src/Main.cpp"
)

# Python library
add_library(lid SHARED ${lid_core} ${lid_py})
target_include_directories(lid PUBLIC "${PROJECT_BINARY_DIR}/include" "${PROJECT_SOURCE_DIR}/include")

# Executable
add_executable(lidx ${lid_core} ${lid_exe})
target_include_directories(lidx PUBLIC "${PROJECT_BINARY_DIR}/include" "${PROJECT_SOURCE_DIR}/include")

set_target_properties(lid PROPERTIES
	# Require C++17
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED YES
)
set_target_properties(lidx PROPERTIES
	# Require C++17
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED YES
)

# Math library
find_library(MATH_LIBRARY m)
if (MATH_LIBRARY)
	target_link_libraries(lid PUBLIC ${MATH_LIBRARY})
	target_link_libraries(lidx PUBLIC ${MATH_LIBRARY})
endif (MATH_LIBRARY)

# Python API
find_package(Python3 REQUIRED COMPONENTS Development NumPy)
if (Python3_FOUND)
	target_include_directories(lid PUBLIC "${Python3_INCLUDE_DIRS}" "${Python3_NumPy_INCLUDE_DIRS}")
	target_include_directories(lidx PUBLIC "${Python3_INCLUDE_DIRS}" "${Python3_NumPy_INCLUDE_DIRS}")
	target_link_libraries(lid PUBLIC "${Python3_LIBRARIES}")
	target_link_libraries(lidx PUBLIC "${Python3_LIBRARIES}")
endif (Python3_FOUND)

# SOFTLib
target_link_libraries(lid PUBLIC softlib)
target_link_libraries(lidx PUBLIC softlib)

