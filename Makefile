
all: ext

ext: vcztools/_vcztoolsmodule.c
	CFLAGS="-std=c99 -UNDEBUG -Wall -Wextra -Werror -Wno-unused-parameter -Wno-cast-function-type" \
	       uv run python setup.py build_ext --inplace

clean:
	rm -f vcztools/*.so
	rm -fR build
