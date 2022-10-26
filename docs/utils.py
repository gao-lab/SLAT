import os
from pathlib import Path
from typing import Union, Dict

def _get_thumbnails(root: Union[str, Path]) -> Dict[str, str]:
    r"""
    Get picture for thumbnails
    
    Parameters
    ----------
    root
        root path
    """
    res = {}
    root = Path(root)
    thumb_path = Path(__file__).parent / "_static" / "gallery_thumb"

    for fname in root.glob("**/*.ipynb"):
        path, name = os.path.split(str(fname)[:-6])    # for .ipynb only
        thumb_fname = f"tutorial_{name}.png"
        if (thumb_path / thumb_fname).is_file():
            res[str(fname)[:-6]] = f"_images/{thumb_fname}"

    res["**"] = "_static/SLAT_logo.png"

    return res