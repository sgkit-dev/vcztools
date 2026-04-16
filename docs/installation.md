(sec-installation)=
# Installation

`vcztools` is available on PyPI:

```bash
python3 -m pip install vcztools
```

This will install the `vcztools` program into your local Python path.

Alternatively, calling:
```bash
python3 -m vcztools <args>
```
is equivalent to:
```bash
vcztools <args>
```
and will always work.

## Shell completion

To enable shell completion for a particular session in Bash do:

```bash
eval "$(_VCZTOOLS_COMPLETE=bash_source vcztools)"
```

If you add this to your `.bashrc` shell completion should be available
in all new shell sessions.

See the [Click documentation](https://click.palletsprojects.com/en/8.1.x/shell-completion/#enabling-completion)
for instructions on how to enable completion in other shells.
