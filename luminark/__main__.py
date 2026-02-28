"""
LUMINARK CLI entry point.
Usage:
    python -m luminark "Your text here"
    python -m luminark --file path/to/file.txt
    python -m luminark --batch path/to/lines.txt
    python -m luminark --format markdown "Your text here"
"""

import sys
import argparse
from . import LuminarkGuardian
from .report import generate_text_report, generate_markdown_report, batch_to_csv


def main():
    parser = argparse.ArgumentParser(
        prog        = "luminark",
        description = "LUMINARK Ethical AI Guardian — CLI audit tool",
    )
    parser.add_argument("text", nargs="?", help="Text to analyze")
    parser.add_argument("--file",   "-f", help="Path to text file to analyze")
    parser.add_argument("--batch",  "-b", help="Path to file with one text per line (batch mode)")
    parser.add_argument("--format", "-o", choices=["text", "markdown", "json"], default="text",
                        help="Output format (default: text)")
    parser.add_argument("--out",    "-w", help="Write output to file instead of stdout")

    args = parser.parse_args()

    guardian = LuminarkGuardian()

    # ── Batch mode ──────────────────────────────────────────────────────
    if args.batch:
        with open(args.batch, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        if not texts:
            print("No texts found in file.", file=sys.stderr)
            sys.exit(1)

        print(f"Analyzing {len(texts)} texts…", file=sys.stderr)
        results = [guardian.analyze(t) for t in texts]
        csv_data = batch_to_csv(results)

        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(csv_data)
            print(f"Batch CSV saved to {args.out}", file=sys.stderr)
        else:
            print(csv_data)
        return

    # ── Single mode ─────────────────────────────────────────────────────
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            input_text = f.read()
    elif args.text:
        input_text = args.text
    elif not sys.stdin.isatty():
        input_text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)

    result = guardian.analyze(input_text)

    if args.format == "json":
        output = result.to_json()
    elif args.format == "markdown":
        output = generate_markdown_report(result)
    else:
        output = generate_text_report(result)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report saved to {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
