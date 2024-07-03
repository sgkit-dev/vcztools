
all: ext

ext: _vcztoolsmodule.c
	CFLAGS="-std=c99 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-cast-function-type" \
	       python3 setup.py build_ext --inplace

clean:
	rm -f *.so *.o tags
	rm -fR build
