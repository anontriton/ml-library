CC      ?= cc
CFLAGS  := -std=c11 -Wall -Wextra
LDLIBS  := -lm

SRC     := main.c
# unity build: main.c #includes arena.c and prng.c, so they're deps, not
# separate compile units
DEPS    := $(SRC) arena.c arena.h prng.c prng.h base.h

.PHONY: all release debug run clean

all: release

release: main
main: $(DEPS)
	$(CC) $(CFLAGS) -O2 $(SRC) -o $@ $(LDLIBS)

# sanitizers catch out-of-bounds reads/writes and undefined behavior,
# much slower, so it's a separate binary
debug: main_debug
main_debug: $(DEPS)
	$(CC) $(CFLAGS) -O1 -g -fno-omit-frame-pointer \
		-fsanitize=address,undefined $(SRC) -o $@ $(LDLIBS)

run: main
	./main

clean:
	rm -f main main_debug
	rm -rf main_debug.dSYM
