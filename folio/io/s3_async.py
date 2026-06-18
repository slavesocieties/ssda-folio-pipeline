"""Async streaming S3 layer built on aioboto3.

Design (Stage 0):
  * one lister coroutine paginates list_objects_v2 -> bounded key queue
  * N downloader coroutines get_object straight into RAM, decode, -> work queue
  * pipeline consumer pulls decoded images, runs GPU work
  * M uploader coroutines put_object crops + JSON sidecars back to S3

Bounded queues make the system backpressure-driven: if the GPU stalls,
downloaders block, and memory stays flat regardless of dataset size.
"""
from __future__ import annotations

import asyncio
import io
import json
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional, Tuple

import numpy as np

try:
    import aioboto3  # noqa
except Exception:  # pragma: no cover - optional at import time
    aioboto3 = None


def decode_image(data: bytes) -> Optional[np.ndarray]:
    """Decode JPEG/TIFF bytes to a BGR ndarray. Prefers PyTurboJPEG, falls
    back to OpenCV. Returns None on undecodable data (logged & skipped)."""
    import cv2
    try:
        from turbojpeg import TurboJPEG
        global _TJ
        if "_TJ" not in globals():
            _TJ = TurboJPEG()
        return _TJ.decode(data)  # BGR
    except Exception:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img


def encode_jpeg(image: np.ndarray, quality: int = 95) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


@dataclass
class DecodedItem:
    key: str
    image: np.ndarray


class S3Streamer:
    def __init__(self, cfg):
        if aioboto3 is None:
            raise RuntimeError("aioboto3 not installed; pip install aioboto3")
        self.cfg = cfg
        self._session = aioboto3.Session()
        # bound connect/read so a misconfigured run (no creds, wrong region)
        # fails fast instead of hanging on the IMDS credential probe.
        try:
            from botocore.config import Config
            self._bcfg = Config(connect_timeout=10, read_timeout=60,
                                retries={"max_attempts": 3, "mode": "standard"})
        except Exception:
            self._bcfg = None

    def _client(self):
        return self._session.client("s3", region_name=self.cfg.s3.region,
                                    config=self._bcfg)

    async def list_keys(self, queue: asyncio.Queue) -> None:
        s3cfg = self.cfg.s3
        async with self._client() as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=s3cfg.input_bucket, Prefix=s3cfg.input_prefix,
                PaginationConfig={"PageSize": s3cfg.list_page_size},
            ):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.lower().endswith((".jpg", ".jpeg", ".tif", ".tiff", ".png")):
                        await queue.put(key)   # blocks when full -> backpressure
        await queue.put(None)  # sentinel

    async def collect_done_stems(self) -> set:
        """Stems already present under the output prefix (for --resume): one
        listing, then in-memory membership checks (cheap vs per-key head_object)."""
        s3cfg = self.cfg.s3
        done: set = set()
        async with self._client() as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=s3cfg.output_bucket,
                                                  Prefix=s3cfg.output_prefix):
                for obj in page.get("Contents", []):
                    k = obj["Key"]
                    if not k.lower().endswith(".jpg"):
                        continue
                    stem = k.rsplit("/", 1)[-1][:-4]        # basename, drop .jpg
                    if stem.endswith(("-A", "-B")):
                        stem = stem[:-2]
                    done.add(stem)
        return done

    async def _download_worker(self, in_q: asyncio.Queue, out_q: asyncio.Queue,
                               s3) -> None:
        s3cfg = self.cfg.s3
        while True:
            key = await in_q.get()
            if key is None:
                await in_q.put(None)  # propagate sentinel to siblings
                in_q.task_done()
                break
            try:
                resp = await s3.get_object(Bucket=s3cfg.input_bucket, Key=key)
                data = await resp["Body"].read()
                img = decode_image(data)
                if img is not None:
                    await out_q.put(DecodedItem(key, img))
            except Exception as e:  # never let one bad object kill the run
                await out_q.put(DecodedItem(key, None))  # mark error downstream
            finally:
                in_q.task_done()

    async def stream_images(self) -> AsyncIterator[DecodedItem]:
        """Yield decoded images as they arrive, fully overlapped with I/O."""
        s3cfg = self.cfg.s3
        key_q: asyncio.Queue = asyncio.Queue(maxsize=s3cfg.list_page_size)
        dec_q: asyncio.Queue = asyncio.Queue(maxsize=s3cfg.decode_queue_size)

        async with self._client() as s3:
            lister = asyncio.create_task(self.list_keys(key_q))
            workers = [
                asyncio.create_task(self._download_worker(key_q, dec_q, s3))
                for _ in range(s3cfg.download_concurrency)
            ]
            done = 0
            n_workers = len(workers)
            # workers exit when they see the sentinel; we wait for all of them
            pending = asyncio.create_task(_join(workers))
            while True:
                get_task = asyncio.create_task(dec_q.get())
                finished, _ = await asyncio.wait(
                    {get_task, pending}, return_when=asyncio.FIRST_COMPLETED)
                if get_task in finished:
                    item = get_task.result()
                    yield item
                    dec_q.task_done()
                elif pending in finished:
                    get_task.cancel()
                    # drain whatever is left
                    while not dec_q.empty():
                        yield dec_q.get_nowait()
                    break
            await lister

    async def upload(self, key: str, image: np.ndarray, sidecar: dict,
                     review: bool = False) -> None:
        s3cfg = self.cfg.s3
        prefix = s3cfg.review_prefix if review else s3cfg.output_prefix
        body = encode_jpeg(image)
        async with self._client() as s3:
            await s3.put_object(Bucket=s3cfg.output_bucket,
                                Key=f"{prefix}{key}", Body=body,
                                ContentType="image/jpeg")
            await s3.put_object(Bucket=s3cfg.output_bucket,
                                Key=f"{prefix}{key}.json",
                                Body=json.dumps(sidecar).encode(),
                                ContentType="application/json")


async def _join(tasks: List[asyncio.Task]) -> None:
    await asyncio.gather(*tasks)
