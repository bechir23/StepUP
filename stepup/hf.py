"""Push run artifacts (checkpoints, metrics, plots) to a Hugging Face model repo, so the heavy
files live in HF storage instead of the Colab/Drive disk, and read them back later.

Token resolution: explicit arg -> HF_TOKEN env -> the cached `huggingface_hub` login. In the
Colab notebook you `login(token=...)` once, then just pass `--hf-repo user/name`.
"""
import os
import pathlib

_ART = ("{name}_best.pt", "hist_{name}.parquet", "test_{name}.parquet", "verif_{name}.parquet",
        "acc_{name}.parquet", "curves_{name}.png", "embed_{name}.png")


def _token(token=None):
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _ensure(repo_id, token):
    from huggingface_hub import create_repo
    create_repo(repo_id, token=token, private=True, exist_ok=True, repo_type="model")


def push_files(repo_id, files, token=None):
    """Upload a list of files (by basename) to the HF model repo."""
    from huggingface_hub import HfApi
    tok = _token(token)
    _ensure(repo_id, tok)
    api = HfApi()
    for f in files:
        f = pathlib.Path(f)
        if f.exists():
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                            repo_id=repo_id, token=tok, repo_type="model")
    return repo_id


def push_model(repo_id, name, artifacts_dir, token=None):
    """Upload one model's artifacts (checkpoint + metrics + plots)."""
    art = pathlib.Path(artifacts_dir)
    return push_files(repo_id, [art / p.format(name=name) for p in _ART], token)


def pull(repo_id, dest, token=None):
    """Download the whole repo into `dest` to read checkpoints/results back."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id, local_dir=str(dest), token=_token(token), repo_type="model")
