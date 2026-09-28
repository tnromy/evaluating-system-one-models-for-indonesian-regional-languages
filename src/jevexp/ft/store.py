"""Durable key-value store for F5 runs (K60). MinIO/S3 on Colab, a local folder for tests and the laptop.

The store is the single source of truth: a run is finished iff `runs/<run_id>/done.json` exists, so any worker
can be killed at any time and restarted without redoing finished work.
"""
import hashlib
import json
import shutil
from pathlib import Path


class Store:
    def exists(self, key): raise NotImplementedError
    def get_bytes(self, key): raise NotImplementedError
    def put_bytes(self, key, data): raise NotImplementedError
    def put_file(self, key, path): raise NotImplementedError
    def get_file(self, key, path): raise NotImplementedError
    def delete(self, key): raise NotImplementedError
    def list(self, prefix): raise NotImplementedError  # keys under prefix, relative to the store root

    def get_json(self, key, default=None):
        return json.loads(self.get_bytes(key)) if self.exists(key) else default

    def put_json(self, key, obj):
        self.put_bytes(key, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8"))


class LocalStore(Store):
    def __init__(self, root):
        self.root = Path(root)

    def _p(self, key):
        return self.root / key

    def exists(self, key):
        return self._p(key).exists()

    def get_bytes(self, key):
        return self._p(key).read_bytes()

    def put_bytes(self, key, data):
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)  # atomic: a reader never sees half a file

    def put_file(self, key, path):
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, p)

    def get_file(self, key, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._p(key), path)

    def delete(self, key):
        p = self._p(key)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()

    def list(self, prefix):
        base = self._p(prefix)
        if not base.exists():
            return []
        return sorted(str(f.relative_to(self.root)) for f in base.rglob("*") if f.is_file()
                      and not f.name.endswith(".tmp"))


class S3Store(Store):
    """boto3 client (see the notebook, Cell 2) + bucket + prefix. Secrets never pass through here as text."""

    def __init__(self, client, bucket, prefix):
        self.s3, self.bucket, self.prefix = client, bucket, prefix.strip("/")

    def _k(self, key):
        return f"{self.prefix}/{key}"

    def exists(self, key):
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except self.s3.exceptions.ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def get_bytes(self, key):
        return self.s3.get_object(Bucket=self.bucket, Key=self._k(key))["Body"].read()

    def put_bytes(self, key, data):
        self.s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=data)

    def put_file(self, key, path):
        self.s3.upload_file(str(path), self.bucket, self._k(key))  # multipart for large checkpoints

    def get_file(self, key, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, self._k(key), str(path))

    def delete(self, key):
        keys = self.list(key) or [key]
        for k in keys:
            self.s3.delete_object(Bucket=self.bucket, Key=self._k(k))

    def list(self, prefix):
        out, token, full = [], None, self._k(prefix)
        while True:
            kw = {"Bucket": self.bucket, "Prefix": full}
            if token:
                kw["ContinuationToken"] = token
            r = self.s3.list_objects_v2(**kw)
            out += [o["Key"][len(self.prefix) + 1:] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                return sorted(out)
            token = r.get("NextContinuationToken")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- MinIO from config.json (pdf2md.ipynb pattern)
def normalize_endpoint(endpoint, port):
    """URL with the port only when it is not the scheme default: SigV4 signs the Host header, and ':443' on an
    https URL makes MinIO answer SignatureDoesNotMatch."""
    ep = str(endpoint or "").strip().rstrip("/")
    if not ep:
        raise ValueError("empty endpoint in config.json")
    if not ep.startswith(("http://", "https://")):
        ep = ("https://" if str(port) in ("443", "") else "http://") + ep
    if port and ":" not in ep.split("://", 1)[1]:
        default = "443" if ep.startswith("https://") else "80"
        if str(port) != default:
            ep = f"{ep}:{port}"
    return ep


def load_s3_config(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = raw.get("s3", raw) if isinstance(raw, dict) else {}
    out = {"endpoint": normalize_endpoint(cfg.get("endpoint") or cfg.get("host"), cfg.get("port")),
           "access_key": str(cfg.get("access_key") or cfg.get("accessKey") or "").strip(),
           "secret_key": str(cfg.get("secret_key") or cfg.get("secretKey") or "").strip(),
           "region": str(cfg.get("region") or "us-east-1").strip(),
           "addressing_style": str(cfg.get("addressing_style") or "path").strip(),
           "verify_ssl": bool(cfg.get("verify_ssl", True))}
    for k in ("access_key", "secret_key"):
        if not out[k]:
            raise ValueError(f"'{k}' is empty in {path}")
    return out


def config_fingerprint(cfg):
    """Shown instead of the access key, so notebook logs never contain credentials."""
    return hashlib.sha256(f"{cfg['endpoint']}|{cfg['access_key']}".encode()).hexdigest()[:12]


def s3_store(config_path, bucket="private", prefix="jev_f5"):
    import boto3
    from botocore.config import Config as BotoConfig
    cfg = load_s3_config(config_path)
    client = boto3.client("s3", endpoint_url=cfg["endpoint"], aws_access_key_id=cfg["access_key"],
                          aws_secret_access_key=cfg["secret_key"], region_name=cfg["region"],
                          verify=cfg["verify_ssl"],
                          config=BotoConfig(s3={"addressing_style": cfg["addressing_style"]},
                                            signature_version="s3v4",
                                            retries={"max_attempts": 5, "mode": "standard"}))
    client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/", MaxKeys=1)  # fails fast on bad credentials
    return S3Store(client, bucket, prefix), f"{cfg['endpoint']} (fingerprint {config_fingerprint(cfg)})"
