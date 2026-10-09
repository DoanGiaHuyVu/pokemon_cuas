"""Conservative local match of a reviewed title-logo crop, never a menu classifier."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


def calibration_digest(calibration):
    return hashlib.sha256(json.dumps(calibration,sort_keys=True).encode()).hexdigest()


class TitleAnchor:
    def __init__(self,path): self.path=Path(path)

    def matches(self,observation):
        try:
            spec=json.loads(self.path.read_text())
            if spec['calibration_sha256']!=calibration_digest(observation['calibration']): return False
            reference=Path(spec['image_path'])
            if hashlib.sha256(reference.read_bytes()).hexdigest()!=spec['image_sha256']: return False
            with Image.open(reference) as ref,Image.open(observation['image_path']) as frame:
                crop=observation['calibration']['game_crop']
                if frame.size!=(crop['width'],crop['height']): return False
                box=spec['box']
                if len(box)!=4 or any(type(x)!=int for x in box) or not (0<=box[0]<box[2]<=frame.width and 0<=box[1]<box[3]<=frame.height): return False
                actual=frame.convert('RGB').crop(tuple(box))
                if ref.size!=actual.size: return False
                # Fixed tight tolerance: no provider claim can relax it.
                difference=ImageChops.difference(ref.convert('RGB'),actual)
                return sum(ImageStat.Stat(difference).mean)/3<=1.5
        except (OSError,ValueError,KeyError,TypeError):
            return False  # Missing/stale reference falls back to hosted vision.
