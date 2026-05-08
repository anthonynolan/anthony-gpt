from pathlib import Path
import re
import sys


START_MARKER_RE = re.compile(
    r"^\s*\*{3}\s*START OF (?:THE|THIS)?\s*PROJECT GUTENBERG EBOOK\b.*\*{3}\s*$",
    re.IGNORECASE,
)
END_MARKER_RE = re.compile(
    r"^\s*\*{3}\s*END OF (?:THE|THIS)?\s*PROJECT GUTENBERG EBOOK\b.*\*{3}\s*$",
    re.IGNORECASE,
)


def strip_project_gutenberg_text(text: str) -> str:
    """Remove the Project Gutenberg header and footer from a plain-text book."""
    lines = text.splitlines(keepends=True)

    start_index = 0
    for i, line in enumerate(lines):
        if START_MARKER_RE.match(line.strip("\ufeff")):
            start_index = i + 1
            break

    end_index = len(lines)
    for i in range(start_index, len(lines)):
        if END_MARKER_RE.match(lines[i]):
            end_index = i
            break

    return "".join(lines[start_index:end_index]).strip()


def read_clean_gutenberg_book(path: str | Path) -> str:
    """Read a Gutenberg text file and return only the book content."""
    return strip_project_gutenberg_text(Path(path).read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    try:
        sys.stdout.write(read_clean_gutenberg_book(args.path))
    except BrokenPipeError:
        pass
