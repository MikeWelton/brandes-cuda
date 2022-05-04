PROGRAM = brandes
NVCC = nvcc
NVCCFLAGS = -g -lineinfo -arch=sm_61

all: $(PROGRAM)

FILES = brandes.cu

brandes: $(FILES)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

.PHONY: all clean

clean:
	rm -rf $(PROGRAM) *.o
