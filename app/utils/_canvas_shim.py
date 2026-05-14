"""Compatibility shim for streamlit-drawable-canvas on Streamlit >= 1.30.

The canvas library does `from streamlit.elements.image import image_to_url`,
which was removed from that module. In modern Streamlit the function lives in
`streamlit.elements.lib.image_utils` with a different signature (LayoutConfig
instead of width:int). We add a wrapper at the legacy import path that adapts
the old call signature so the canvas can register its background image with
Streamlit's MediaFileManager and the frontend gets a `/media/...` URL it can
actually load.

Import this module before importing `streamlit_drawable_canvas`.
"""
from __future__ import annotations


def _install_shim() -> None:
    import streamlit.elements.image as st_image_legacy

    if hasattr(st_image_legacy, "image_to_url"):
        return

    try:
        from streamlit.elements.lib.image_utils import image_to_url as _new_image_to_url
        from streamlit.elements.lib.layout_utils import LayoutConfig
    except ImportError:
        _new_image_to_url = None  # type: ignore[assignment]
        LayoutConfig = None  # type: ignore[assignment]

    def image_to_url(  # noqa: PLR0913 - matches the legacy signature
        image,
        width=-1,
        clamp=False,
        channels="RGB",
        output_format="auto",
        image_id="",
        allow_emoji=False,
    ) -> str:
        if _new_image_to_url is not None and LayoutConfig is not None:
            layout_config = LayoutConfig()
            return _new_image_to_url(
                image, layout_config, clamp, channels, output_format, image_id
            )

        # Last-resort fallback: emit a data URL.
        import base64
        import io

        from PIL import Image
        try:
            import numpy as np
        except ImportError:
            np = None

        if np is not None and hasattr(image, "shape"):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError(f"unsupported image type: {type(image)!r}")
        fmt = "PNG"
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{data}"

    st_image_legacy.image_to_url = image_to_url  # type: ignore[attr-defined]


_install_shim()
