#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
from colormath.color_objects import CMYKColor, LabColor, HSVColor
from colormath.color_diff import delta_e_cmc
from colormath.color_conversions import convert_color
from scipy.optimize import minimize
import numpy as np
from constrained_color_triad.constraints import LuminosityMinConstraint
from constrained_color_triad.constraints import LuminosityMaxConstraint
from constrained_color_triad.constraints import SaturationMinConstraint
from constrained_color_triad.constraints import SaturationMaxConstraint
from constrained_color_triad.constraints import HueBandConstraint
from constrained_color_triad.scratch import OPTS, X0


class Densities(object):
    c = 1.45
    m = 1.45
    y = 1
    k = 1.7


class OptimalPalette(object):
    """Optimized color palette for charts and graphs.

    Color palette with high and low values optimized
    for even luminosity and saturation, and divergent hues.
    Luminosity and saturation can be constrained with min
    and max values. Hues can be constrained within min and
    max bounds.

    """

    def __init__(self, x0, constraints_obj):
        """
        Constructor of the OptimalPalette object.

        :param x0:
            Initial guess. n × 2 × 4 matrix of CMYK values between 0 and 100.
            Note that internally x is represented as Lab, not CMYK.
        :type x0:
            np.array
        :param constraints_obj:
            Dictionary of constraints.
        :type constraints_obj:
            iter
        """

        # Check that x0 is pairs of 4-valued arrays
        assert len(x0.size) % 4 == 8

        # make a list of LabColor objects
        self.labs = [convert_color(CMYKColor(*x), LabColor)
                     for x in x0.reshape((-1, 4)) / 100]

        # initial x is a string of L*a*b components of the
        # initial CMYK guesses
        self.x0 = np.array([h.get_value_tuple() for h in self.labs]).flatten()

        # Bounds are 0 to 100 for L, -100 to 100 for a, -100 to 100 for b
        self.bounds = np.vstack((
            np.array([[0, -100, -100] for _ in self.labs]).flatten(),
            np.array([[100, 100, 100] for _ in self.labs]).flatten(),
         )).T

        self.constraints = constraints_obj

    def objective(self, x):
        """
        Objective function to minimize.

        :param x: Value of x
        :type x: np.array
        :return: Value of objective function
        :rtype: float
        """

        l_arr = self.x0.reshape((-1, 2, 3))[:, :, 0]

        # Weighted sum of squares of differences across light ends
        # of hue spectra
        diff_lt = np.sum(np.power(np.diff(np.array(
            [x for x in itertools.combinations(l_arr[:, 0], 2)]),
            axis=1), 2)) / 100 ** 2 * 3

        # Weighted sum of squares of differences across dark ends
        # of hue spectra
        diff_dk = np.sum(np.power(np.diff(np.array(
            [x for x in itertools.combinations(l_arr[:, 1], 2)]),
            axis=1), 2)) / 100 ** 2 * 3

        # Weighted sum of squares of differences between light and dark ends
        # of hue spectra
        diff_hues = np.sum(np.power(np.diff(
            l_arr, axis=1), 2)) / 100 ** 2 * 3

        diff_cmc = np.sum(np.power(np.diff(np.array(
            [x for x in itertools.combinations(
                [delta_e_cmc(*hh)
                 for hh in itertools.combinations(self.labs, 2)], 2)]),
            axis=1), 2)) / 100 ** 2 * 3

        return (1e-6 / (diff_cmc + 1e-8) +
                diff_lt +
                diff_dk +
                0)

    def minimize(self):
        minimize(self.objective,
                 self.x0,
                 method='SLSQP',
                 bounds=self.bounds,
                 constraints=self.constraints,
                 options={'disp': True,
                          'iprint': 10,
                          'maxiter': 1000,
                          })


class SankeyConstraints(object):
    def __init__(self):
        pass

    def __iter__(self):
        return [
            {'type': 'eq',
             'fun': HueBandConstraint(0, 0, 210, 225)},
            {'type': 'eq',
             'fun': HueBandConstraint(0, 1, 210, 225)},
            {'type': 'eq',
             'fun': HueBandConstraint(20, 105, 120)},
            {'type': 'ineq',
             'fun': SaturationMaxConstraint(12, )},
             # 'fun': self.coal_sat_max},
            {'type': 'ineq',
             'fun': self.gas_sat_min},
            {'type': 'ineq',
             'fun': self.oil_sat_min},
            # {'type': 'ineq',
            #  'fun': gas_val_min},
            # {'type': 'ineq',
            #  'fun': oil_val_min},
            {'type': 'eq',
             'fun': self.g_min_l},
            {'type': 'eq',
             'fun': self.c_min_l},
            {'type': 'eq',
             'fun': self.o_min_l},
            {'type': 'eq',
             'fun': self.g_max_l},
            {'type': 'eq',
             'fun': self.c_max_l},
            {'type': 'eq',
             'fun': self.o_max_l},
        ]
    
    @staticmethod
    def get_cmyk(_n, _x):
        return CMYKColor(_x[0 + _n], _x[1 + _n], _x[2 + _n], _x[3 + _n])

    def g_min_l(self, _x):
        # Minimum luminosity of minimum gas greater than MIN_L
        _cmyk = self.get_cmyk(0, _x)
        _lab = convert_color(_cmyk, LabColor)
        return _lab.lab_l - OPTS['MIN_L']

    def c_min_l(self, _x):
        # Minimum luminosity of minimum coal greater than 90
        _cmyk = self.get_cmyk(8, _x)
        _lab = convert_color(_cmyk, LabColor)
        return _lab.lab_l - OPTS['MIN_L']
    
    def o_min_l(self, _x):
        # Minimum luminosity of minimum coal greater than 90
        _cmyk = self.get_cmyk(16, _x)
        _lab = convert_color(_cmyk, LabColor)
        return _lab.lab_l - OPTS['MIN_L']
        
    def g_max_l(self, _x):
        # Minimum luminosity of minimum gas less than MAX_L
        _cmyk = self.get_cmyk(4, _x)
        _lab = convert_color(_cmyk, LabColor)
        return OPTS['MAX_L'] - _lab.lab_l
        
    def c_max_l(self, _x):
        # Minimum luminosity of minimum coal less than MAX_L
        _cmyk = self.get_cmyk(12, _x)
        _lab = convert_color(_cmyk, LabColor)
        return OPTS['MAX_L'] - _lab.lab_l
        
    def o_max_l(self, _x):
        # Minimum luminosity of minimum coal less than MAX_L
        _cmyk = self.get_cmyk(20, _x)
        _lab = convert_color(_cmyk, LabColor)
        return OPTS['MAX_L'] - _lab.lab_l
        
    def gas_hue_min(self, _x):
        _cmyk = self.get_cmyk(4, _x)
        _hsv = convert_color(_cmyk, HSVColor)
        return _hsv.hsv_h - OPTS['GAS_H'][0]
        
    def gas_hue_max(self, _x):
        _cmyk = self.get_cmyk(4, _x)
        _hsv = convert_color(_cmyk, HSVColor)
        return OPTS['GAS_H'][1] - _hsv.hsv_h
        
    def oil_hue_min(self, _x):
        _cmyk = self.get_cmyk(20, _x)
        _hsv = convert_color(_cmyk, HSVColor)
        return _hsv.hsv_h - OPTS['OIL_H'][0]
        
    def oil_hue_max(self, _x):
        _cmyk = self.get_cmyk(20, _x)
        _hsv = convert_color(_cmyk, HSVColor)
        return OPTS['OIL_H'][1] - _hsv.hsv_h
        
    def coal_sat_max(self, _x):
        _cmyk = self.get_cmyk(12, _x)
        _hsv = convert_color(_cmyk, HSVColor)
        return OPTS['MAX_S'] - _hsv.hsv_s
        
    def gas_sat_min(self, _x):
        _cmyk = self.get_cmyk(4, _x)
        _hsv = convert_color(_cmyk, HSVColor)
        return _hsv.hsv_s - OPTS['MIN_S']
        
    def oil_sat_min(self, _x):
        _cmyk = self.get_cmyk(20, _x)
        _hsv = convert_color(_cmyk, HSVColor)
        return _hsv.hsv_s - OPTS['MIN_S']
        
    def gas_val_min(self, _x):
        _cmyk = self.get_cmyk(4, _x)
        _hsv = convert_color(_cmyk, HSVColor)
        return _hsv.hsv_v - OPTS['MIN_V']
        
    def oil_val_min(self, _x):
        _cmyk = self.get_cmyk(20, _x)
        _hsv = convert_color(_cmyk, HSVColor)
        return _hsv.hsv_v - OPTS['MIN_V']


if __name__ == '__main__':
    x0 = X0
    # x0 = np.random.rand(24)

    opt = OptimalPalette(X0, SankeyConstraints)

    """
    # print(x)
    n = 0
    print(get_cmyk(n, x.x))
    # hsv = convert_color(cmyk, HSVColor)
    n += 4
    cmyk = get_cmyk(n, x.x)
    print(cmyk)
    hsv = convert_color(cmyk, HSVColor)
    print(hsv)
    # lab = convert_color(cmyk, LabColor)
    # print(lab)
    n += 4
    print(get_cmyk(n, x.x))
    n += 4
    cmyk = get_cmyk(n, x.x)
    print(cmyk)
    hsv = convert_color(cmyk, HSVColor)
    print(hsv)
    # lab = convert_color(cmyk, LabColor)
    # print(lab)
    n += 4
    print(get_cmyk(n, x.x))
    n += 4
    cmyk = get_cmyk(n, x.x)
    print(cmyk)
    hsv = convert_color(cmyk, HSVColor)
    print(hsv)
    # lab = convert_color(cmyk, LabColor)
    # print(lab)
    """