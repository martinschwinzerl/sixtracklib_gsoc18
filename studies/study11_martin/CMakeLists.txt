cmake_minimum_required( VERSION 3.8 FATAL_ERROR )

project( sixtracklib_gsoc2018_study11 )

enable_language( C )
enable_language( CXX )
enable_language( CUDA )

# We know where to find the SixTracklib installation ->
# set the SixTracklib_DIR variable to help CMake finding it:

# set( SixTracklib_DIR "${CMAKE_SOURCE_DIR}/external/lib/cmake" )
# find_package( SixTracklib REQUIRED )

# -----------------------------------------------------------------------------
# convert sample_fodo.c into a shared library:

# add_library( sample_fodo SHARED sample_fodo.c )

# target_link_libraries( sample_fodo SixTracklib::sixtrack )

# set_target_properties(
#         sample_fodo PROPERTIES
#                     LINKER_LANGUAGE CXX
#                     POSITION_INDEPENDENT_CODE ON
#                     CXX_STANDARD 11
#                     CXX_STANDARD_REQUIRED ON
#                     DEBUG_POSTFIX d )
#
# set_property( TARGET sample_fodo
#               APPEND PROPERTY COMPILE_OPTIONS -Wall -Werror -pedantic )

# -----------------------------------------------------------------------------
# use sample_fodo in an executable:

add_executable( run_sample_fodo run_sample_fodo.cu )

set_target_properties( run_sample_fodo PROPERTIES LANGUAGE CUDA )

#Since sample_fodo explicitly depends on SixTracklib::sixtrack, we use do not
# have to add SixTracklib::sixtrack library explicitly as a dependency of the
# executable
# target_link_libraries( run_sample_fodo sample_fodo )
