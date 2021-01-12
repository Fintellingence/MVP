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
7. Do not play around with `git rebase`

## How to update this repo?

By changing and creating PRs. If they are approved, they will be merged into `main`.

It's important to notice that, when we update the content, we must run the Setup again by the following command:

```
python3 setup.py install --user
```
