# Minimum Viable Product (MVP)

This MVP aims to create a minimum signal prediction as well as a parser for
finance data.

## Installation

In the MVP root directory, make
```
poetry install
```
to install the dependencies.

To use the package, build and install the wheels using
```
poetry build
cd dist/
pip3 install --user <name>.whl
```

## Development

### Adding new dependencies
Make `poetry add <name>`, e.g., `poetry add seaborn`.

### Git workflow
Remember,
1. `git clone` if you do not have this project in your local repository.
1. `git pull` if you already have the local snapshot.
2. `git checkout -b` if you will develop a new feature or fix something.
3. `git fetch` and `git checkout` if you will work in a exist branch.
4. `git add` and `git commit` to track your work locally
5. `git push <remote name> <branch name>` to save the work in Github
6. Is your work fished? **Make a pull request**

### Database
Database files in general have a prohibitive memory demand to store
remotely in the git repository. Thus, each one must provide a local
copy of the database files, which can be taken in the google drive.
Preferably, they should be located in `$HOME/FintelligenceData/` path
One may also use the functions in `database_builder.py` to set the
database files locally, though for the 1-minute data the .csv files
exported from MetaTrader are required. Try out:
```
python3 database_builder.py
```
