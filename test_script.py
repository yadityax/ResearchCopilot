from frontend.render_utils import _wrap_raw_latex_lines as fwrap
import importlib.util, pathlib
import sys

try:
    spec=importlib.util.spec_from_file_location('hfapp', pathlib.Path('huggingface/app.py'))
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    sample='''h_t = f(UU_{hh} h_{t-1} + UU_{xh} x_t)\n\\frac{\\partial L}{\\partial h_t} = \\frac{\\partial L}{\\partial h_T} \\prod_{k=t+1}^{T} \\frac{\\partial h_k}{\\partial h_{k-1}}\nA_i = \\sum_{j=1}^{n} \\alpha_{ij} V_j'''
    print('---frontend---')
    print(fwrap(sample))
    print('---hf---')
    print(m._clean_report(sample))
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
