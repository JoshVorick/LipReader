# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:
.PHONY : .NOTPARALLEL

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/josh/git/LipReader

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/josh/git/LipReader

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/josh/git/LipReader/CMakeFiles /home/josh/git/LipReader/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/josh/git/LipReader/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named facerec_video

# Build rule for target.
facerec_video: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 facerec_video
.PHONY : facerec_video

# fast build rule for target.
facerec_video/fast:
	$(MAKE) -f CMakeFiles/facerec_video.dir/build.make CMakeFiles/facerec_video.dir/build
.PHONY : facerec_video/fast

facerec_video.o: facerec_video.cpp.o
.PHONY : facerec_video.o

# target to build an object file
facerec_video.cpp.o:
	$(MAKE) -f CMakeFiles/facerec_video.dir/build.make CMakeFiles/facerec_video.dir/facerec_video.cpp.o
.PHONY : facerec_video.cpp.o

facerec_video.i: facerec_video.cpp.i
.PHONY : facerec_video.i

# target to preprocess a source file
facerec_video.cpp.i:
	$(MAKE) -f CMakeFiles/facerec_video.dir/build.make CMakeFiles/facerec_video.dir/facerec_video.cpp.i
.PHONY : facerec_video.cpp.i

facerec_video.s: facerec_video.cpp.s
.PHONY : facerec_video.s

# target to generate assembly for a file
facerec_video.cpp.s:
	$(MAKE) -f CMakeFiles/facerec_video.dir/build.make CMakeFiles/facerec_video.dir/facerec_video.cpp.s
.PHONY : facerec_video.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... facerec_video"
	@echo "... rebuild_cache"
	@echo "... facerec_video.o"
	@echo "... facerec_video.i"
	@echo "... facerec_video.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
