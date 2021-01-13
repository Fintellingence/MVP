import argparse

from mvp import builder


def run(db_path, company_path):
    yahoo = builder.Yahoo(db_path)
    yahoo.update(company_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("db_path", type=str)
    p.add_argument("company_path", type=str)
    args = p.parse_args()
    run(**vars(args))
