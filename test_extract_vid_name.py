from unittest import TestCase

from detect_ball import extract_vid_name


class TestExtractVidName(TestCase):
    def test_extract_vid_name(self):
        vid_name = extract_vid_name('../follow_up/datasets/efi/labels/efi_slomo_vid2_67.jpg')
        assert vid_name == 'efi_slomo_vid2'
