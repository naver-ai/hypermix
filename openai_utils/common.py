__all__ = ["openai_request", "resolve_api_key"]

import json
import pathlib

import requests


def openai_request(api_key, url,
                   headers=None, data=None, method="get", file=None):
    if method == "get":
        requests_fn = requests.get
        data_key = "params"
    elif method == "post":
        requests_fn = requests.post
        data_key = "data"
    elif method == "delete":
        requests_fn = requests.delete
        data_key = "data"
    else:
        raise ValueError(f"unsupported method: {method}")

    headers = headers or {}
    headers["Authorization"] = f"Bearer {api_key}"

    kwargs = {"headers": headers}

    if data is not None:
        kwargs[data_key] = data

    if file is not None:
        kwargs["files"] = {"file": file}

    resp = requests_fn(url, **kwargs)

    if resp.status_code // 100 != 2:
        raise RuntimeError(f"OpenAI server failed. "
                           f"status code = {resp.status_code}, message: "
                           f"{json.dumps(json.loads(resp.content), indent=2)}")

    return json.loads(resp.content)


def resolve_api_key(args):
    if args.api_key is not None:
        return

    filename = "openai-key"
    dirs = ["./"]
    for directory in dirs:
        path = pathlib.Path(directory).joinpath(filename)
        if path.exists():
            with open(path) as f:
                args.api_key = f.read().strip()
            break

    if args.api_key is None:
        raise RuntimeError(f"provide api key via command line option or "
                           f"a 'openai-key' file in the current working "
                           f"directory.")
