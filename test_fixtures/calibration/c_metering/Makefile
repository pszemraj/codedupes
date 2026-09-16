CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -pedantic -Iinclude
SOURCES := src/aggregate.c src/payouts.c src/rows.c src/adapters.c

.PHONY: test run clean

test: build/test_metering
	./build/test_metering

run: build/metering_demo
	./build/metering_demo

build/test_metering: $(SOURCES) tests/test_metering.c include/metering.h
	mkdir -p build
	$(CC) $(CFLAGS) $(SOURCES) tests/test_metering.c -o $@

build/metering_demo: $(SOURCES) src/main.c include/metering.h
	mkdir -p build
	$(CC) $(CFLAGS) $(SOURCES) src/main.c -o $@

clean:
	rm -rf build
