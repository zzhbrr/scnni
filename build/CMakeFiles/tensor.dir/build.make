# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /ws/CourseProject/SCNNI

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /ws/CourseProject/SCNNI/build

# Include any dependencies generated for this target.
include CMakeFiles/tensor.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tensor.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tensor.dir/flags.make

CMakeFiles/tensor.dir/src/tensor.cpp.o: CMakeFiles/tensor.dir/flags.make
CMakeFiles/tensor.dir/src/tensor.cpp.o: ../src/tensor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/ws/CourseProject/SCNNI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tensor.dir/src/tensor.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tensor.dir/src/tensor.cpp.o -c /ws/CourseProject/SCNNI/src/tensor.cpp

CMakeFiles/tensor.dir/src/tensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tensor.dir/src/tensor.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ws/CourseProject/SCNNI/src/tensor.cpp > CMakeFiles/tensor.dir/src/tensor.cpp.i

CMakeFiles/tensor.dir/src/tensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tensor.dir/src/tensor.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ws/CourseProject/SCNNI/src/tensor.cpp -o CMakeFiles/tensor.dir/src/tensor.cpp.s

# Object files for target tensor
tensor_OBJECTS = \
"CMakeFiles/tensor.dir/src/tensor.cpp.o"

# External object files for target tensor
tensor_EXTERNAL_OBJECTS =

libtensor.so: CMakeFiles/tensor.dir/src/tensor.cpp.o
libtensor.so: CMakeFiles/tensor.dir/build.make
libtensor.so: CMakeFiles/tensor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/ws/CourseProject/SCNNI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libtensor.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tensor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tensor.dir/build: libtensor.so

.PHONY : CMakeFiles/tensor.dir/build

CMakeFiles/tensor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tensor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tensor.dir/clean

CMakeFiles/tensor.dir/depend:
	cd /ws/CourseProject/SCNNI/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /ws/CourseProject/SCNNI /ws/CourseProject/SCNNI /ws/CourseProject/SCNNI/build /ws/CourseProject/SCNNI/build /ws/CourseProject/SCNNI/build/CMakeFiles/tensor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tensor.dir/depend
