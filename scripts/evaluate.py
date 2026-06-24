from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Future Milestone 2+ planner evaluation entry point.")
    parser.add_argument("--config", required=False)
    parser.parse_args()
    raise SystemExit(
        "Planner evaluation is reserved for Milestone 2+. "
        "Milestone 1 currently supports dataset loading and visualization only."
    )


if __name__ == "__main__":
    main()
