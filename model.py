# pure transition

import json
import numpy as np
import vector_tools as vt
from collections import OrderedDict
import scipy.optimize as opt

##
## dictionary functions
##

def load_json(fname):
    return json.load(open(fname), object_pairs_hook=OrderedDict)

def save_json(d, fname):
    json.dump(d, open(fname, 'w+'), indent=4)

def file_or_dict(fod):
    return vt.Bundle(load_json(fod)) if type(fod) is str else fod

def dict_to_vec(d):
    return np.array([x for x in d.values()])

def vec_to_dict(vec, names):
    return {k: v for k, v in zip(names, vec)}

##
## main economy
##

class Model:
    def __init__(m, alg=None, par=None, pol=None, var=None):
        if alg is not None:
            m.load_algpar(alg)
        if par is not None:
            m.load_params(par)
        if var is not None:
            m.load_eqvars(var)
        if pol is not None:
            m.load_policy(pol)

    def load_algpar(m, alg):
        m.alg = vt.Bundle(file_or_dict(alg))

    def load_params(m, par):
        m.par = vt.Bundle(file_or_dict(par))

    def load_eqvars(m, var):
        m.var = vt.Bundle(file_or_dict(var))

    def load_policy(m, pol):
        m.pol = vt.Bundle(file_or_dict(pol))

    def eqfunc(m):
        pass

    def solve(m, obj_args={}, **kwargs):
        def eqeval(x):
            names = list(m.var.keys())
            m.load_eqvars(vec_to_dict(x, names))
            return m.eqfunc(**obj_args)
        x0 = dict_to_vec(m.var)
        return opt.fsolve(eqeval, x0, **kwargs)
