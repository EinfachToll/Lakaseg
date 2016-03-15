RESULT_LIB_NAME=liblakaseg.so
RESULT_BINARY_NAME=lakaseg
CC=g++
CFLAGS=-Wall -Wextra -Wformat=2 -Wpointer-arith -Wcast-qual -fopenmp
LDFLAGS=-lpthread -lX11 -lgomp
OPTIMIZATION=-O3 -DNDEBUG


$(RESULT_LIB_NAME): lakaseg.cpp
	LANG=en_EN $(CC) -c -fPIC -o lakaseg.o lakaseg.cpp $(CFLAGS) $(OPTIMIZATION) -isystem 3rd_party/
	LANG=en_EN $(CC) -shared -Wl,-soname,$(RESULT_LIB_NAME) -o $(RESULT_LIB_NAME) lakaseg.o $(LDFLAGS)


$(RESULT_BINARY_NAME): lakaseg.cpp
	LANG=en_EN $(CC) -o $(RESULT_BINARY_NAME) lakaseg.cpp $(CFLAGS) $(OPTIMIZATION) -isystem 3rd_party/ $(LDFLAGS)

bin: $(RESULT_BINARY_NAME)


debug: OPTIMIZATION=-g
debug: $(RESULT_LIB_NAME)

bin_debug: OPTIMIZATION=-g
bin_debug: $(RESULT_BINARY_NAME)

clean:
	rm -f $(RESULT_BINARY_NAME) $(RESULT_LIB_NAME) lakaseg.o

release: $(RESULT_LIB_NAME) bin

.PHONY: bin clean
