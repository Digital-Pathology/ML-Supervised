
import os
from typing import List

from histolab.tiler import ScoreTiler, GridTiler
from histolab.scorer import NucleiScorer
from histolab.slide import Slide
from tqdm import tqdm as loadingbar

from dataset.util import listdir_recursive


def tile_images(images, out_dir):
    tiler = ScoreTiler(
        scorer=NucleiScorer(),
        tile_size=(512, 512),
        n_tiles=100,
        suffix='.png'
    )
    if isinstance(images, str) and os.path.isdir(images):
        images = listdir_recursive(images)
    loading = loadingbar(images)
    for image_path in loading:
        loading.set_description(image_path)
        processed_path = os.path.join(
            out_dir,
            "tiles",
            os.path.basename(image_path)
        )
        slide = Slide(image_path, processed_path=processed_path)
        tiler.extract(slide,
                      report_path=os.path.join(out_dir, "tiling_reports"))
    return len(listdir_recursive(out_dir))
