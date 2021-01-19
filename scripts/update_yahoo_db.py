import argparse

from mvp import builder


def run(db_path, sym_path):
    yahoo = builder.Yahoo(db_path)
    yahoo.update(sym_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-db_path",
        dest="db_path",
        type=str,
        default=builder.DEFT_D1_DB_PATH,
        help="path to database file",
    )
    p.add_argument(
        "-sym_path",
        dest="sym_path",
        type=str,
        default=builder.DEFT_SYMBOLS_PATH,
        help="path to symbols txt file",
    )
    args = p.parse_args()
    run(**vars(args))
