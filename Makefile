CC      ?= cc
CFLAGS  := -std=c11 -Wall -Wextra
LDLIBS  := -lm

# each .c is compiled on its own into a .o, then the .o files are linked
# into one binary. editing one .c only recompiles that file
SRCS    := main.c arena.c prng.c

RELEASE_FLAGS := -O2
# sanitizers catch out-of-bounds reads/writes and undefined behavior,
# much slower, so it's a separate binary
DEBUG_FLAGS   := -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined

# release and debug objects are built with different flags, so they live in
# separate dirs or they'd overwrite each other
RELEASE_OBJS  := $(SRCS:%.c=build/release/%.o)
DEBUG_OBJS    := $(SRCS:%.c=build/debug/%.o)

.PHONY: all release debug run clean

all: release

release: main
debug: main_debug

# link step: $^ is every object. sanitizer flags have to be here too, the
# linker pulls in the sanitizer runtime
main: $(RELEASE_OBJS)
	$(CC) $(CFLAGS) $(RELEASE_FLAGS) $^ -o $@ $(LDLIBS)

main_debug: $(DEBUG_OBJS)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $^ -o $@ $(LDLIBS)

# compile step: -c stops before linking. -MMD also writes a .d file listing
# the headers this .c included, so editing a header rebuilds only the .c files
# that use it. -MP keeps the build working if a header gets deleted
build/release/%.o: %.c | build/release
	$(CC) $(CFLAGS) $(RELEASE_FLAGS) -MMD -MP -c $< -o $@

build/debug/%.o: %.c | build/debug
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) -MMD -MP -c $< -o $@

build/release build/debug:
	mkdir -p $@

run: main
	./main

clean:
	rm -rf build main main_debug main_debug.dSYM

# pull in the generated header deps. the - means don't error on the first
# build, before any .d files exist
-include $(RELEASE_OBJS:.o=.d) $(DEBUG_OBJS:.o=.d)
