
set(lid_core
	"${PROJECT_SOURCE_DIR}/src/dreamoutput.cpp"
	"${PROJECT_SOURCE_DIR}/src/Main.cpp"
	"${PROJECT_SOURCE_DIR}/src/integrate.cpp"
)

add_executable(lid ${lid_core})
target_include_directories(lid PUBLIC "${PROJECT_BINARY_DIR}/include" "${PROJECT_SOURCE_DIR}/include")

set_target_properties(lid PROPERTIES
	# Require C++17
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED YES
)

# Math library
find_library(MATH_LIBRARY m)
if (MATH_LIBRARY)
	target_link_libraries(lid PUBLIC ${MATH_LIBRARY})
endif ()

# SOFTLib
target_link_libraries(lid PUBLIC softlib)

