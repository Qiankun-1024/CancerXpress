# Build and Connect the Documentation

## Build Locally

The website uses MkDocs and the Markdown files in `tutorials/`. There is only
one copy of the quickstart, shared by GitHub and the website.

Use a separate documentation environment; neither TensorFlow nor model
parameters are required:

```bash
python3.12 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python -m mkdocs build --strict
python -m mkdocs serve
```

The static output is written to `site/`, which is ignored by Git.

## Connect Read the Docs

The repository includes `.readthedocs.yaml` for a Python 3.12 documentation
build. An account-level connection is still required:

1. Push the documentation and configuration files to the GitHub repository.
2. Sign in to Read the Docs and connect the GitHub account that owns the project.
3. Add/import `Qiankun-1024/CancerXpress` as a documentation project.
4. Confirm the repository's default branch and trigger the first build.
5. After the build succeeds, use the project URL assigned by Read the Docs.

No URL or successful deployment should be assumed before this process
completes. Do not store GitHub or Read the Docs tokens in documentation files.

Read the Docs installs only `docs/requirements.txt`; it does not install the
CancerXpress runtime, load private model files, or execute patient analyses.
The `READTHEDOCS_CANONICAL_URL` environment variable supplies the deployed
canonical URL automatically.

See the official
[MkDocs integration guide](https://docs.readthedocs.com/platform/stable/intro/mkdocs.html)
and [configuration guide](https://docs.readthedocs.com/platform/stable/config-file/).
