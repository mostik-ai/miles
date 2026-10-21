"""Which argument destinations the command line itself names."""

import sys


def dests_on_command_line(argv: list[str] | None = None) -> frozenset[str]:
    """The argparse destinations named by the ``--flag`` tokens of ``argv`` (``sys.argv`` by default).

    A ``--no-foo`` token is reported as both ``no_foo`` and ``foo``, so a caller asking whether the
    command line decided ``foo`` sees the negative spelling too.
    """
    named = set()
    for token in sys.argv[1:] if argv is None else argv:
        if token.startswith("--"):
            dest = token.removeprefix("--").partition("=")[0].replace("-", "_")
            named |= {dest, dest.removeprefix("no_")}
    return frozenset(named)
