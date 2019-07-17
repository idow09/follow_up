from unittest import TestCase

from data_structures import Prediction, Label


class TestPrediction(TestCase):
    def test_calc_circle_iou_fully_intersecting(self):
        pred = Prediction(0, 0, 10, 0.58)
        lbl = Label(0, 0, 10)
        iou = pred.calc_circle_iou(lbl)
        self.assertAlmostEqual(iou, 1)

    def test_calc_circle_iou_non_intersecting(self):
        pred = Prediction(20, 0, 10, 0.58)
        lbl = Label(0, 0, 10)
        iou = pred.calc_circle_iou(lbl)
        self.assertAlmostEqual(iou, 0)

    def test_calc_circle_iou_semi_intersecting(self):
        pred = Prediction(10, 0, 10, 0.58)
        lbl = Label(0, 0, 10)
        iou = pred.calc_circle_iou(lbl)
        self.assertAlmostEqual(iou, 0.243, 3)

    def test_calc_circle_iou_not_even_close(self):
        pred = Prediction(50, 0, 10, 0.58)
        lbl = Label(0, 0, 10)
        iou = pred.calc_circle_iou(lbl)
        self.assertAlmostEqual(iou, 0)

    def test_calc_circle_iou_almost_fully_intersecting(self):
        pred = Prediction(0.2, 0, 10, 0.58)
        lbl = Label(0, 0, 10)
        iou = pred.calc_circle_iou(lbl)
        self.assertAlmostEqual(iou, 0.975, 3)

    def test_calc_circle_iou_one_inside_other(self):
        pred = Prediction(0, 0, 10, 0.58)
        lbl = Label(0, 0, 20)
        iou = pred.calc_circle_iou(lbl)
        self.assertAlmostEqual(iou, 0.25)

    def test_calc_circle_iou_one_inside_other_not_centered(self):
        pred = Prediction(0, 0.1, 10, 0.58)
        lbl = Label(0, 0, 20)
        iou = pred.calc_circle_iou(lbl)
        self.assertAlmostEqual(iou, 0.25)

    def test_calc_circle_iou_one_inside_other_not_centered_2(self):
        pred = Prediction(0, 0.1, 20, 0.58)
        lbl = Label(0, 0, 5)
        iou = pred.calc_circle_iou(lbl)
        self.assertAlmostEqual(iou, 0.0625)

    def test_calc_circle_iou_zero_radius(self):
        pred = Prediction(0, 0, 0, 0.58)
        lbl = Label(0, 0, 20)
        iou = pred.calc_circle_iou(lbl)
        self.assertAlmostEqual(iou, 0)

    def test_calc_circle_iou_zero_both_radius(self):
        pred = Prediction(0, 0, 0, 0.58)
        lbl = Label(0, 0, 0)
        iou = pred.calc_circle_iou(lbl)
        self.assertAlmostEqual(iou, 1)

    def test_calc_rect_iou(self):
        pred = Prediction(0, 0, 10, 0.58)
        lbl = Label(0, 5, 10)
        iou = pred.calc_rect_iou(lbl)
        self.assertAlmostEqual(iou, 0.6, 1)

    def test_calc_center_dist(self):
        pred = Prediction(13.88, 0, 10, 0.58)
        lbl = Label(0, 0, 10)
        pred.matched_label = lbl
        self.assertEquals(pred.center_dist, None)
        pred.calc_center_dist()
        self.assertEquals(pred.center_dist, 13.88)
