#!/usr/bin/env python
# -*- coding: utf-8 -*-
from colormath.color_objects import CMYKColor, LabColor, HSVColor
from colormath.color_conversions import convert_color


class ColorPaletteConstraint(object):
    def __init__(self, idx):
        self.x = None
        self.idx = idx
        self._cmyk = None
        self._lab = None
        self._hsv = None

    @property
    def cmyk(self):
        if self._cmyk is None and self.x is not None:
            self._cmyk = convert_color(self.lab, CMYKColor)
            # self._cmyk = CMYKColor(
            #     self.x[0 + self.idx], self.x[1 + self.idx],
            #     self.x[2 + self.idx], self.x[3 + self.idx])
        return self._cmyk

    @property
    def lab(self):
        if self._lab is None and self.x is not None:
            self._lab = LabColor(*self.x[self.idx:self.idx+3])
            # self._lab = convert_color(self.cmyk, LabColor)
        return self._lab

    @property
    def hsv(self):
        if self._hsv is None and self.x is not None:
            self._hsv = convert_color(self.lab, HSVColor)
        return self._hsv


class LuminosityMinConstraint(ColorPaletteConstraint):
    def __init__(self, idx, minimum_luminosity):
        super().__init__(idx)
        self.minimum_luminosity = minimum_luminosity

    def __call__(self, x):
        self.x = x
        return self.lab.lab_l - self.minimum_luminosity


class LuminosityMaxConstraint(ColorPaletteConstraint):
    def __init__(self, idx, maximum_luminosity):
        super().__init__(idx)
        self.maximum_luminosity = maximum_luminosity

    def __call__(self, x):
        self.x = x
        return self.maximum_luminosity - self.lab.lab_l


class SaturationMinConstraint(ColorPaletteConstraint):
    def __init__(self, idx, minimum_saturation):
        super().__init__(idx)
        self.minimum_saturation = minimum_saturation

    def __call__(self, x):
        self.x = x
        return self.hsv.hsv_s - self.minimum_saturation


class SaturationMaxConstraint(ColorPaletteConstraint):
    def __init__(self, idx, maximum_saturation):
        super().__init__(idx)
        self.maximum_saturation = maximum_saturation

    def __call__(self, x):
        self.x = x
        return self.maximum_saturation - self.hsv.hsv_s


class HueBandConstraint(ColorPaletteConstraint):
    def __init__(self, idx, min_hue, max_hue):
        super().__init__(idx)
        self.max_hue = max_hue
        self.min_hue = min_hue

    def __call__(self, x):
        self.x = x
        return int(not self.min_hue <= self.hsv.hsv_h <= self.max_hue)


if __name__ == '__main__':
    f = LuminosityMinConstraint(0, 40)
    print(f)
    print(f([1, .50, .50, .50]))
