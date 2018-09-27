#!/usr/bin/env python

import skimage
import skimage.measure
import skimage.transform
import cv2
import unittest
import warnings

from assignment8 import *


class TestSolution(unittest.TestCase):
    
    def test_kozeny_carmen_plot(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.cla()
            plt.clf()
            kozeny_carmen_plot('poro_perm.dat')
            plt.savefig('poro_perm.png')

            gold_image = cv2.imread('poro_perm_gold.png')
            test_image = cv2.imread('poro_perm.png')

            test_image_resized = skimage.transform.resize(test_image, 
                                                          (gold_image.shape[0], gold_image.shape[1]), 
                                                          mode='constant')

            ssim = skimage.measure.compare_ssim(skimage.img_as_float(gold_image), test_image_resized, multichannel=True)
            assert ssim >= 0.75

    def test_contour_plot(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.cla()
            plt.clf()
            contour_plot('Nechelik.dat')
            plt.savefig('Nechelik.png')

            gold_image = cv2.imread('Nechelik_gold.png')
            test_image = cv2.imread('Nechelik.png')

            test_image_resized = skimage.transform.resize(test_image, 
                                                          (gold_image.shape[0], gold_image.shape[1]), 
                                                          mode='constant')

            ssim = skimage.measure.compare_ssim(skimage.img_as_float(gold_image), test_image_resized, multichannel=True)
            assert ssim >= 0.75

if __name__ == '__main__':
        unittest.main()
