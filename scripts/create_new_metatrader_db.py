import os
import argparse

from mvp import builder


def run(db_path, csv_dir_path):
    mt = builder.MetaTrader()
    mt.create_new_from_csv(db_path,csv_dir_path)


if __name__ == "__main__":
    root_dir = os.path.join(os.path.expanduser('~'), "FintelligenceData")
    p = argparse.ArgumentParser()
    p.add_argument(
            "--db_path",
            dest="db_path",
            type=str,
            default=os.path.join(root_dir, "MetaTrader_M1.db"),
            help="path to database file"
    )
    p.add_argument(
            "--csv_path",
            dest="csv_dir_path",
            type=str,
            default=os.path.join(root_dir, "csv_files/"),
            help="path to csv files"
    )
    args = p.parse_args()
    run(**vars(args))
