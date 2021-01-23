import os
import argparse

from mvp import builder


def run(db_path, sym_path, optional_csv_dir):
    mt = builder.MetaTrader()
    mt.update(db_path,sym_path,optional_csv_dir)


if __name__ == "__main__":
    root_dir = os.path.join(os.path.expanduser('~'), "FintelligenceData")
    p = argparse.ArgumentParser()
    p.add_argument(
            "--db-path",
            dest="db_path",
            type=str,
            default=os.path.join(root_dir, "MetaTrader_M1.db"),
            help="path to database file"
    )
    p.add_argument(
            "--sym-path",
            dest="sym_path",
            type=str,
            default=os.path.join(root_dir, "all_symbols.txt"),
            help="path to symbols txt file"
    )
    p.add_argument(
            "--csv-path",
            dest="optional_csv_dir",
            type=str,
            default=os.path.join(root_dir, "csv_files/"),
            help="path to csv files (if db does not exist yet)"
    )
    args = p.parse_args()
    run(**vars(args))
