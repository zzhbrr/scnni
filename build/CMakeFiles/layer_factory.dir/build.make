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
include CMakeFiles/layer_factory.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/layer_factory.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/layer_factory.dir/flags.make

CMakeFiles/layer_factory.dir/src/layer_factory.cpp.o: CMakeFiles/layer_factory.dir/flags.make
CMakeFiles/layer_factory.dir/src/layer_factory.cpp.o: ../src/layer_factory.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/ws/CourseProject/SCNNI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/layer_factory.dir/src/layer_factory.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/layer_factory.dir/src/layer_factory.cpp.o -c /ws/CourseProject/SCNNI/src/layer_factory.cpp

CMakeFiles/layer_factory.dir/src/layer_factory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/layer_factory.dir/src/layer_factory.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ws/CourseProject/SCNNI/src/layer_factory.cpp > CMakeFiles/layer_factory.dir/src/layer_factory.cpp.i

CMakeFiles/layer_factory.dir/src/layer_factory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/layer_factory.dir/src/layer_factory.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ws/CourseProject/SCNNI/src/layer_factory.cpp -o CMakeFiles/layer_factory.dir/src/layer_factory.cpp.s

# Object files for target layer_factory
layer_factory_OBJECTS = \
"CMakeFiles/layer_factory.dir/src/layer_factory.cpp.o"

# External object files for target layer_factory
layer_factory_EXTERNAL_OBJECTS =

liblayer_factory.so: CMakeFiles/layer_factory.dir/src/layer_factory.cpp.o
liblayer_factory.so: CMakeFiles/layer_factory.dir/build.make
liblayer_factory.so: CMakeFiles/layer_factory.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/ws/CourseProject/SCNNI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library liblayer_factory.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/layer_factory.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/layer_factory.dir/build: liblayer_factory.so

.PHONY : CMakeFiles/layer_factory.dir/build

CMakeFiles/layer_factory.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/layer_factory.dir/cmake_clean.cmake
.PHONY : CMakeFiles/layer_factory.dir/clean

CMakeFiles/layer_factory.dir/depend:
	cd /ws/CourseProject/SCNNI/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /ws/CourseProject/SCNNI /ws/CourseProject/SCNNI /ws/CourseProject/SCNNI/build /ws/CourseProject/SCNNI/build /ws/CourseProject/SCNNI/build/CMakeFiles/layer_factory.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/layer_factory.dir/depend
