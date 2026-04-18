"""
Shared rendering utilities for Streamlit frontend.
"""
import re
import streamlit as st

_DISPLAY_MATH = re.compile(r'(\$\$.*?\$\$)', re.DOTALL)
_INLINE_MATH = re.compile(r'(\$[^$]+?\$)')
_EXISTING_DISPLAY = re.compile(r'(\$\$.*?\$\$)', re.DOTALL)
_LATEX_CMD_RE = re.compile(r'\\[a-zA-Z]+')
_LATEX_KEYWORDS_RE = re.compile(
    r'\\(frac|left|right|prod|sum|int|partial|nabla|sqrt|leq|geq|approx|propto|cdot|'
    r'text|mathcal|mathbb|begin|end|times|infty|'
    r'alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|'
    r'Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega|'
    r'exp|log|ln|sin|cos|tan|lim|max|min|argmax|argmin|'
    r'mathbf|mathrm|boldsymbol|hat|bar|tilde|vec|diag|softmax|relu|'
    r'equiv|neq|sim|cong|ll|gg|in|notin|subset|cup|cap|forall|exists)\b'
)
_MATH_RELATION_RE = re.compile(
    r'(=|\\approx|\\equiv|\\propto|\\leq|\\geq|\\neq|\\sim|\\rightarrow|\\Rightarrow|\\ll|\\gg)'
)
_SUBSCRIPT_RE = re.compile(r'[a-zA-Z0-9][_\^]\{')
_SUBSCRIPT_SIMPLE_RE = re.compile(r'[a-zA-Z0-9](?:_[a-zA-Z0-9]+|\^[a-zA-Z0-9]+)')
_BRACKET_DISPLAY_RE = re.compile(r'\\\[(.*?)\\\]', re.DOTALL)


def _is_raw_latex(line: str) -> bool:
    s = line.strip()
    if not s or '$' in s:
        return False
    if s.startswith(('#', '>', '|')):
        return False
    if s.startswith(('-', '*')) and not _LATEX_CMD_RE.search(s):
        return False
    kw = len(_LATEX_KEYWORDS_RE.findall(s))
    has_cmd = bool(_LATEX_CMD_RE.search(s))
    has_sub = bool(_SUBSCRIPT_RE.search(s) or _SUBSCRIPT_SIMPLE_RE.search(s))
    has_relation = bool(_MATH_RELATION_RE.search(s))
    n_words = len(s.split())
    if s.startswith('\\') and has_cmd:
        return True
    if n_words > 15 and not has_cmd:
        return False
    if kw >= 2:
        return True
    if kw >= 1 and (has_relation or has_sub):
        return True
    if has_cmd and has_relation:
        return True
    if has_sub and has_relation and n_words < 12:
        return True
    return False


def _is_latex_continuation(line: str) -> bool:
    """Continuation lines inside a multi-line display equation block."""
    s = line.strip()
    if not s or '$' in s:
        return False
    if s.startswith('&'):
        return True
    if s.endswith(r'\\'):
        return True
    if s.startswith((r'\\begin', r'\\end')):
        return True
    return False


def _normalize_display_expr(expr: str) -> str:
    """Normalize common over-escaped LaTeX sequences before st.latex."""
    # Convert commands like \\text, \\frac to \text, \frac.
    # Keep standalone line-break \\ intact.
    return re.sub(r'\\\\([A-Za-z])', r'\\\1', expr)


def _process_section(text: str) -> str:
    lines = text.split('\n')
    result = []
    i = 0
    while i < len(lines):
        if _is_raw_latex(lines[i]):
            block = [lines[i].strip()]
            i += 1
            # Keep only true continuation lines in the same display block.
            while i < len(lines) and _is_latex_continuation(lines[i]):
                block.append(lines[i].strip())
                i += 1
            result.append('\n$$\n' + '\n'.join(block) + '\n$$\n')
        else:
            result.append(lines[i])
            i += 1
    return '\n'.join(result)


def _wrap_raw_latex_lines(text: str) -> str:
    """Split on existing $$...$$ blocks, fix raw LaTeX in remaining sections."""
    # Normalize \[ ... \] display math to $$ ... $$ before further processing.
    text = _BRACKET_DISPLAY_RE.sub(lambda m: f"$$\n{m.group(1).strip()}\n$$", text)

    parts = _EXISTING_DISPLAY.split(text)
    out = []
    for part in parts:
        if part.startswith('$$') and part.endswith('$$'):
            out.append(part)
        else:
            out.append(_process_section(part))
    return ''.join(out)


def render_answer(text: str):
    if not text:
        return

    text = _wrap_raw_latex_lines(text)

    for part in _DISPLAY_MATH.split(text):
        if not part:
            continue
        if part.startswith('$$') and part.endswith('$$'):
            st.latex(_normalize_display_expr(part[2:-2].strip()))
            continue
        for segment in _INLINE_MATH.split(part):
            if not segment:
                continue
            if segment.startswith('$') and segment.endswith('$'):
                st.latex(segment[1:-1].strip())
            else:
                st.markdown(segment)
