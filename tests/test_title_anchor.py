import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image,ImageDraw
from computer.title_anchor import TitleAnchor, calibration_digest


class AnchorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.image=self.root/'frame.png'; self.ref=self.root/'anchor.png'
        frame=Image.new('RGB',(512,768),'black')
        draw=ImageDraw.Draw(frame); draw.rectangle((80,100,240,160),fill='yellow'); draw.text((95,115),'TITLE',fill='blue')
        frame.save(self.image); frame.crop((70,90,250,170)).save(self.ref)
        self.calibration={'game_crop':{'width':512,'height':768},'verified':True}
        self.spec={'calibration_sha256':calibration_digest(self.calibration),'box':[70,90,250,170],
                   'image_path':str(self.ref),'image_sha256':hashlib.sha256(self.ref.read_bytes()).hexdigest()}
        self.path=self.root/'anchor.json'; self.path.write_text(json.dumps(self.spec))
        self.detector=TitleAnchor(self.path)
        self.obs={'calibration':self.calibration,'image_path':str(self.image)}

    def test_matching_static_logo_can_tolerate_animation_outside_it_but_not_menu_replacement(self):
        self.assertTrue(self.detector.matches(self.obs))
        with Image.open(self.image) as frame:
            changed=frame.copy(); ImageDraw.Draw(changed).rectangle((0,400,511,767),fill='white'); changed.save(self.image)
        self.assertTrue(self.detector.matches(self.obs))
        Image.new('RGB',(512,768),'gray').save(self.image)
        self.assertFalse(self.detector.matches(self.obs))

    def test_tampered_reference_stale_geometry_or_missing_profile_never_match(self):
        self.ref.write_bytes(b'changed'); self.assertFalse(self.detector.matches(self.obs))
        self.assertFalse(self.detector.matches(self.obs|{'calibration':{'game_crop':{'width':256,'height':384}}}))
        self.path.unlink(); self.assertFalse(self.detector.matches(self.obs))
