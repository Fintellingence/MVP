import os
import argparse

from mvp import builder


def run(db_path, sym_path):
    yahoo = builder.Yahoo(db_path)
    yahoo.update(sym_path)


if __name__ == "__main__":
    root_dir = os.path.join(os.path.expanduser('~'), "FintelligenceData")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--db-path",
        type=str,
        default=os.path.join(root_dir, "Yahoo_D1.db"),
        help="path to database file",
    )
    p.add_argument(
        "--sym-path",
        type=str,
        default=os.path.join(root_dir, "all_symbols.txt"),
        help="path to symbols txt file",
    )
    args = p.parse_args()
    run(**vars(args))
