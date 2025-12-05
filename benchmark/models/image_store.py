import io
import base64
import pyarrow as pa
import pyarrow.ipc as ipc

from PIL import Image


class ImageStore:
    def __init__(self, arrow_path: str):
        self.reader = ipc.RecordBatchFileReader(pa.memory_map(arrow_path, "r"))
        self.id_index = {}
        for bi in range(self.reader.num_record_batches):
            batch = self.reader.get_batch(bi)
            for ri, iid in enumerate(batch["image_id"].to_pylist()):
                self.id_index[iid] = (bi, ri)

    # Function used for passing image data to HF models
    def get_image(self, iid: str):
        bi, ri = self.id_index[iid]
        b = self.reader.get_batch(bi)["bytes"][ri].as_py()
        return Image.open(io.BytesIO(b)).convert("RGB")

    # Below functions are used to generate data urls for API models
    def get_bytes(self, iid: str):
        bi, ri = self.id_index[iid]
        batch = self.reader.get_batch(bi)
        return batch["bytes"][ri].as_py(), batch

    def get_bytes_and_mime(self, iid: str):
        b, batch = self.get_bytes(iid)
        if "mime" in batch.schema.names:
            mime = batch["mime"][self.id_index[iid][1]].as_py()
            if mime:
                return b, mime

        # fallback - detect via PIL
        with Image.open(io.BytesIO(b)) as im:
            fmt = (im.format or "JPEG").lower()
        mime = f"image/{fmt}"
        return b, mime

    def get_data_url(self, iid: str) -> str | None:
        b, mime = self.get_bytes_and_mime(iid)
        if b is None:
            return None
        if not mime:
            mime = "image/jpeg"
        b64 = base64.b64encode(b).decode("utf-8")
        return f"data:{mime};base64,{b64}"
