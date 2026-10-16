CC       ?= cc
CFLAGS   := -std=c11 -Wall -Wextra
# -Isrc lets the examples #include "graph.h" without a path
# -MMD -MP write a .d file next to each .o listing the headers it included,
# so editing a header rebuilds only the .c files that use it
CPPFLAGS := -Isrc -MMD -MP
LDLIBS   := -lm

RELEASE_FLAGS := -O2
# sanitizers catch out-of-bounds reads/writes and undefined behavior,
# much slower, so it's a separate binary
DEBUG_FLAGS   := -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined

# the library is every .c in src/, each example is every .c in its dir
LIB_SRCS   := $(wildcard src/*.c)
MNIST_SRCS := $(wildcard examples/mnist/*.c)
ALL_SRCS   := $(LIB_SRCS) $(MNIST_SRCS)

# release and debug are built with different flags, so each gets its own
# tree under build/ that mirrors the source tree, or they'd overwrite each
# other's .o files
R := build/release
D := build/debug

.PHONY: all release debug run run-debug clean

all: release

release: $(R)/mnist
debug: $(D)/mnist

# make run SEED=42 reproduces a run, leave it off for a clock seed
run: $(R)/mnist
	./$< $(SEED)

run-debug: $(D)/mnist
	./$< $(SEED)

# compile step: one .c -> one .o, -c stops before linking
$(R)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(RELEASE_FLAGS) $(CPPFLAGS) -c $< -o $@

$(D)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(CPPFLAGS) -c $< -o $@

# the library: every src/ object bundled into one static archive (ar r adds
# the files, c creates the archive, s writes an index the linker uses)
# removed first so a deleted src file doesn't leave a stale .o inside it
$(R)/libml.a: $(LIB_SRCS:%.c=$(R)/%.o)
	rm -f $@
	$(AR) rcs $@ $^

$(D)/libml.a: $(LIB_SRCS:%.c=$(D)/%.o)
	rm -f $@
	$(AR) rcs $@ $^

# link step: the example's objects, then the library, then -lm
# order matters, the linker reads left to right and only pulls from an
# archive what's still missing at that point, so a library has to come after
# whatever uses it. sanitizer flags go here too, they pull in their runtime
$(R)/mnist: $(MNIST_SRCS:%.c=$(R)/%.o) $(R)/libml.a
	$(CC) $(CFLAGS) $(RELEASE_FLAGS) $^ -o $@ $(LDLIBS)

$(D)/mnist: $(MNIST_SRCS:%.c=$(D)/%.o) $(D)/libml.a
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $^ -o $@ $(LDLIBS)

clean:
	rm -rf build

# pull in the generated header deps, the - means don't error on the first
# build, before any .d files exist
-include $(ALL_SRCS:%.c=$(R)/%.d) $(ALL_SRCS:%.c=$(D)/%.d)
