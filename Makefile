
all: ext

# -O3 must be explicit: setting CFLAGS here clobbers CPython's defaults
# (which would otherwise add -O3 -fwrapv -DNDEBUG). Without -O3 the .so
# is built unoptimised and the BGEN kernel runs ~4x slower.
ext: vcztools/_vcztoolsmodule.c
	CFLAGS="-std=c99 -O3 -UNDEBUG -Wall -Wextra -Werror -Wno-unused-parameter -Wno-cast-function-type" \
	       uv run python setup.py build_ext --inplace

clean:
	rm -f vcztools/*.so
	rm -fR build
