'''Modelling functions from data.'''
# Standary library modules
import logging

# Third party modules available via "pip install ..."
import numpy as np
import scipy.optimize




logger = logging.getLogger(__name__)



class FitModel(object):
    names = False
    
    def fit(self, xdata, ydata, weights=None, stdevs=None):
        if weights is not None:
            assert stdevs is None
            sigma = weights
            abs_sigma = False
        elif stdevs is not None:
            assert weights is None
            sigma = stdevs
            abs_sigma = True
        else:
            sigma = None
            abs_sigma = False

        x = np.asarray(xdata)
        y = np.asarray(ydata)
        finite_x, finite_y = remove_invalid([x, y])
        self.params, self.cov_arr = scipy.optimize.curve_fit(self.func, finite_x, finite_y, sigma=sigma, absolute_sigma=abs_sigma)
        self.xdata = x
        self.ydata = y
        return self
        
    @property
    def param_names(self):
        if self.names:
            return self.names
        else:
            return [str(i) for i in range(len(self.params))]
            
    @property
    def param_stdevs(self):
        stdevs = []
        for i in range(len(self.params)):
            stdevs.append(np.sqrt(self.cov_arr[i, i]))
        return stdevs
        
    def y(self, x):
        return self.func(x, *self.params)
        
    @property
    def params_string(self):
        return ', '.join(['%s=%.2e' % (name, value) for name, value in
                          zip(self.param_names, self.params)])
                          
    @property
    def equation_fitted(self):
        eq = str(self.equation_general)        
        for i, param_name in enumerate(self.names):            
            eq = eq.replace(param_name, '{(%.3E)}' % self.params[i])
        return eq



class Linear(FitModel):
    name = 'Linear'
    names = ['m', 'c']
    equation_general = r'$mx+c$'

    def func(self, x, m, c):
        return m * x + c
        
        

class LinearThroughZero(FitModel):
    name = 'LinearThroughZero'
    names = ['m']
    equation_general = r'$mx$'

    def func(self, x, m):
        return m * x


        
class Sqrt(FitModel):
    name = 'Sqrt'
    names = ['m', 'c']
    equation_general = r'$m$.sqrt($x$) + $c$'
    
    def func(self, x, m, c):
        return m * np.sqrt(x) + c
 
 
   
class LogNatural(FitModel):
    name = 'LogNatural'
    names = ['m', 'c']
    equation_general = r'$m\log{x}+c$'

    def func(self, x, m, c):
        return m * np.log(x) + c



class Log10(FitModel):
    name = 'Log10'
    names = ['m', 'c']
    equation_general = r'$m\log[10]{x}+c$'

    def func(self, x, m, c):
        return m * np.log10(x) + c



class Exponential(FitModel):
    name = 'Exponential'
    names = ['A', 'c']
    equation_general = r'$Ae^x+c$'

    def func(self, x, m, c):
        return m * np.exp(x) + c



class Log10Log10(FitModel):
    name = 'Log10Log10'
    names = ['a', 'c']
    equation_general = r'$10^cx^a$'

    def func(self, x, m, c):
        return (x ** m) * (10 ** c)
        

        
# class Piecewise(object):
#     name = 'Piecewise'

#     def __init__(self, piece_model_class=Linear):
#         self.piece_model_class = piece_model_class
        
#     def fit(self, x, y): 
#         self.segments = []
#         arg_indexes = np.argsort(x)
#         x = np.array(x[arg_indexes])
#         y = np.array(y[arg_indexes])
#         for i in range(1, len(x)):
#             s = utils.NamedDict()
#             s.x = np.array([x[i - 1], x[i]])
#             s.y = np.array([y[i - 1], y[i]])
#             s.model = self.piece_model_class().fit(s.x, s.y)
#             self.segments.append(s)
#         return self
            
#     def y(self, x):
#         x = np.asarray(x)
#         y = np.ones_like(x) * np.nan
#         last = self.segments[-1]
#         for i in range(x.shape[0]):
#             for j, s in enumerate(self.segments):
#                 if x[i] >= s.x[0] and x[i] <= s.x[1]:
#                     y[i] = s.model.y(x[i])
#                     break
#             if np.isnan(y[i]):
#                 if x[i] < self.segments[0].x[1]:
#                     y[i] = self.segments[0].model.y(x[i])
#                 else:
#                     y[i] = self.segments[-1].model.y(x[i])
#         return y

#     @property
#     def equation_fitted(self):
#         return '%s(%s)' % (self.name, self.piece_model_class.name)
        
def remove_invalid(arrays):
    assert len(arrays) > 0
    n = len(arrays[0])
    for arr in arrays[1:]:
        assert len(arr) == n
    
    mask0 = np.isfinite(arrays[0])
    masks = [np.isfinite(arr) for arr in arrays[1:]]
    for mask in masks:
        mask0 = mask0 & mask
    return [np.array(arr[mask0]) for arr in arrays]