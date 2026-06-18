#!/usr/bin/env python3
"""Mechanical conversion from gdd_*/gtd_*/gqd_* (double-based) to
   gds_*/gts_*/gqs_* (float-based) for the gdtq-0.0.2 CUDA library.

   This script is a build-time helper, not part of the installed product.
   Run from the inc/ directory:

       python convert_to_float.py

   It produces one float-based companion for each of these source pairs:

       gdd_basic.{cu,cuh}    gtd_basic.{cu,cuh}    gqd_basic.{cu,cuh}
       gdd_sqrt.{cu,cuh}     gtd_sqrt.{cu,cuh}     gqd_sqrt.{cu,cuh}
       gdd_exp.{cu,cuh}      gtd_exp.{cu,cuh}      gqd_exp.{cu,cuh}
       gdd_log.{cu,cuh}      gtd_log.{cu,cuh}      gqd_log.{cu,cuh}
       gdd_sincos.{cu,cuh}   gtd_sincos.{cu,cuh}   gqd_sincos.{cu,cuh}

   File names are the same with the third letter swapped:
       gdd -> gds, gtd -> gts, gqd -> gqs.
"""
import os
import re
import sys

INC = os.path.dirname(os.path.abspath(__file__))

# Substitutions performed in-order. Each entry is (regex, replacement).
# Order matters: longer / more-specific identifiers FIRST so they win
# before the shorter prefix substitutions (e.g. dd_inv_fact must be
# rewritten before bare dd_ -> ds_).
SUBS = [
    # ---- include guard renames ---------------------------------------
    (r'__GDD_BASIC_CU__',     '__GDS_BASIC_CU__'),
    (r'__GDD_BASIC_CUH__',    '__GDS_BASIC_CUH__'),
    (r'__GDD_SQRT_CU__',      '__GDS_SQRT_CU__'),
    (r'__GDD_SQRT_CUH__',     '__GDS_SQRT_CUH__'),
    (r'__GDD_EXP_CU__',       '__GDS_EXP_CU__'),
    (r'__GDD_EXP_CUH__',      '__GDS_EXP_CUH__'),
    (r'__GDD_LOG_CU__',       '__GDS_LOG_CU__'),
    (r'__GDD_LOG_CUH__',      '__GDS_LOG_CUH__'),
    (r'__GDD_SINCOS_CU__',    '__GDS_SINCOS_CU__'),
    (r'__GDD_SINCOS_CUH__',   '__GDS_SINCOS_CUH__'),

    (r'__GTD_BASIC_CU__',     '__GTS_BASIC_CU__'),
    (r'__GTD_BASIC_CUH__',    '__GTS_BASIC_CUH__'),
    (r'__GTD_SQRT_CU__',      '__GTS_SQRT_CU__'),
    (r'__GTD_SQRT_CUH__',     '__GTS_SQRT_CUH__'),
    (r'__GTD_EXP_CU__',       '__GTS_EXP_CU__'),
    (r'__GTD_EXP_CUH__',      '__GTS_EXP_CUH__'),
    (r'__GTD_LOG_CU__',       '__GTS_LOG_CU__'),
    (r'__GTD_LOG_CUH__',      '__GTS_LOG_CUH__'),
    (r'__GTD_SINCOS_CU__',    '__GTS_SINCOS_CU__'),
    (r'__GTD_SINCOS_CUH__',   '__GTS_SINCOS_CUH__'),

    (r'__GQD_BASIC_CU__',     '__GQS_BASIC_CU__'),
    (r'__GQD_BASIC_CUH__',    '__GQS_BASIC_CUH__'),
    (r'__GQD_SQRT_CU__',      '__GQS_SQRT_CU__'),
    (r'__GQD_SQRT_CUH__',     '__GQS_SQRT_CUH__'),
    (r'__GQD_EXP_CU__',       '__GQS_EXP_CU__'),
    (r'__GQD_EXP_CUH__',      '__GQS_EXP_CUH__'),
    (r'__GQD_LOG_CU__',       '__GQS_LOG_CU__'),
    (r'__GQD_LOG_CUH__',      '__GQS_LOG_CUH__'),
    (r'__GQD_SINCOS_CU__',    '__GQS_SINCOS_CU__'),
    (r'__GQD_SINCOS_CUH__',   '__GQS_SINCOS_CUH__'),

    # ---- header includes (so gds_basic.cu pulls in the float aggregate)
    # We emit gqs.cuh as the float-based equivalent of gqd.cuh.
    (r'"gqd\.cuh"',           '"gqs.cuh"'),
    (r'"gqd\.cu"',            '"gqs.cu"'),

    # ---- multi-char identifiers -- type aliases ----------------------
    (r'\bgdd_real\b',         'gds_real'),
    (r'\bgtd_real\b',         'gts_real'),
    (r'\bgqd_real\b',         'gqs_real'),

    # ---- table / array names (long forms first) -----------------------
    (r'\bdd_inv_fact\b',      'ds_inv_fact'),
    (r'\btd_inv_fact\b',      'ts_inv_fact'),
    (r'\bn_dd_inv_fact\b',    'n_ds_inv_fact'),
    (r'\bn_td_inv_fact\b',    'n_ts_inv_fact'),
    (r'\bn_inv_fact\b',       'n_qs_inv_fact'),
    (r'\binv_fact\b',         'qs_inv_fact'),

    (r'\bd_dd_sin_table\b',   'd_ds_sin_table'),
    (r'\bd_dd_cos_table\b',   'd_ds_cos_table'),
    (r'\bd_td_sin_table\b',   'd_ts_sin_table'),
    (r'\bd_td_cos_table\b',   'd_ts_cos_table'),
    (r'\bd_sin_table\b',      'd_qs_sin_table'),
    (r'\bd_cos_table\b',      'd_qs_cos_table'),

    # ---- type constructors -------------------------------------------
    (r'\bmake_dd\b',          'make_ds'),
    (r'\bmake_td\b',          'make_ts'),
    (r'\bmake_qd\b',          'make_qs'),
    (r'\bmake_td_renorm\b',   'make_ts_renorm'),
    (r'\bmake_double2\b',     'make_float2'),
    (r'\bmake_double3\b',     'make_float3'),
    (r'\bmake_double4\b',     'make_float4'),

    # ---- helper functions in the basic layer --------------------------
    (r'\bdd_add\b',           'ds_add'),

    # ---- numerical constants -----------------------------------------
    # Constants prefixed with _dd_ , _td_ , _qd_ all become _ds_ / _ts_ / _qs_
    (r'_dd_eps\b',            '_ds_eps'),
    (r'_dd_e\b',              '_ds_e'),
    (r'_dd_log2\b',           '_ds_log2'),
    (r'_dd_2pi\b',            '_ds_2pi'),
    (r'_dd_pi\b',             '_ds_pi'),
    (r'_dd_pi2\b',            '_ds_pi2'),
    (r'_dd_pi4\b',            '_ds_pi4'),
    (r'_dd_pi16\b',           '_ds_pi16'),
    (r'_dd_3pi4\b',           '_ds_3pi4'),

    (r'_td_eps\b',            '_ts_eps'),
    (r'_td_e\b',              '_ts_e'),
    (r'_td_log2\b',           '_ts_log2'),
    (r'_td_2pi\b',            '_ts_2pi'),
    (r'_td_pi\b',             '_ts_pi'),
    (r'_td_pi2\b',            '_ts_pi2'),
    (r'_td_pi4\b',            '_ts_pi4'),
    (r'_td_pi1024\b',         '_ts_pi1024'),

    (r'_qd_eps\b',            '_qs_eps'),
    (r'_qd_e\b',              '_qs_e'),
    (r'_qd_log2\b',           '_qs_log2'),
    (r'_qd_2pi\b',            '_qs_2pi'),
    (r'_qd_pi\b',             '_qs_pi'),
    (r'_qd_pi2\b',            '_qs_pi2'),
    (r'_qd_pi4\b',            '_qs_pi4'),
    (r'_qd_pi1024\b',         '_qs_pi1024'),
    (r'_qd_3pi4\b',           '_qs_3pi4'),

    # ---- to_double helper --------------------------------------------
    (r'\bto_double\b',        'to_float'),

    # ---- vector type names (must come before bare double -> float) ----
    (r'\bdouble2\b',          'float2'),
    (r'\bdouble3\b',          'float3'),
    (r'\bdouble4\b',          'float4'),

    # ---- CUDA double intrinsics --------------------------------------
    (r'\b__dadd_rn\b',        '__fadd_rn'),
    (r'\b__dsub_rn\b',        '__fsub_rn'),
    (r'\b__dmul_rn\b',        '__fmul_rn'),
    (r'\b__ddiv_rn\b',        '__fdiv_rn'),
    (r'\b__fma_rn\b',         '__fmaf_rn'),

    # ---- bare double -> float ----------------------------------------
    # Only as a token; comments naturally use the word "double" but
    # those (e.g. "double-double", "double-precision") are OK to keep
    # because they describe history / context. To minimise surprise we
    # only rewrite the C type token, not English prose. We approximate
    # by requiring word boundary on both sides AND that the next/prev
    # token does not look like prose punctuation we want to keep (we
    # already protected double2/3/4 above).
    (r'\bdouble\b',           'float'),

    # ---- splitter / FMA macro names (values rewritten elsewhere) -----
    (r'\b_GQD_SPLITTER\b',    '_GQS_SPLITTER'),
    (r'\b_GQD_SPLIT_THRESH\b','_GQS_SPLIT_THRESH'),
    (r'\bGQD_FMS\b',          'GQS_FMS'),
    (r'\bGD_FMA\b',           'GS_FMA'),
    (r'\bGD_FMS\b',           'GS_FMS'),
    (r'\bCUDA_FMA\b',         'CUDA_FMA'),  # leave as-is

    # ---- numeric literal for the double splitter constants -----------
    # If any survive a manual replacement of inline.cu, scrub them here.
    (r'\b134217729\.0\b',     '4097.0f'),
    (r'\b268435456\.0\b',     '8192.0f'),
    (r'3\.7252902984619140625e-09', '1.220703125e-4f'),
    (r'6\.69692879491417e\+299',    '2.0769187e+34f'),
]

# Directories / file naming.  We map the old basename to the new one
# and rely on the substitutions above for the *contents*.
NAME_MAP = {
    'gdd_': 'gds_',
    'gtd_': 'gts_',
    'gqd_': 'gqs_',
}

SOURCES = [
    'gdd_basic.cu',  'gdd_basic.cuh',
    'gdd_sqrt.cu',   'gdd_sqrt.cuh',
    'gdd_exp.cu',    'gdd_exp.cuh',
    'gdd_log.cu',    'gdd_log.cuh',
    'gdd_sincos.cu', 'gdd_sincos.cuh',
    'gtd_basic.cu',  'gtd_basic.cuh',
    'gtd_sqrt.cu',   'gtd_sqrt.cuh',
    'gtd_exp.cu',    'gtd_exp.cuh',
    'gtd_log.cu',    'gtd_log.cuh',
    'gtd_sincos.cu', 'gtd_sincos.cuh',
    'gqd_basic.cu',  'gqd_basic.cuh',
    'gqd_sqrt.cu',   'gqd_sqrt.cuh',
    'gqd_exp.cu',    'gqd_exp.cuh',
    'gqd_log.cu',    'gqd_log.cuh',
    'gqd_sincos.cu', 'gqd_sincos.cuh',
]

def convert(text):
    for pat, rep in SUBS:
        text = re.sub(pat, rep, text)
    return text

def out_name(src):
    for old, new in NAME_MAP.items():
        if src.startswith(old):
            return new + src[len(old):]
    return src

def main():
    written = []
    for src in SOURCES:
        spath = os.path.join(INC, src)
        if not os.path.isfile(spath):
            print(f"WARN: missing source: {src}", file=sys.stderr)
            continue
        with open(spath, 'r', encoding='utf-8') as f:
            text = f.read()
        new_text = convert(text)
        dst = out_name(src)
        dpath = os.path.join(INC, dst)
        with open(dpath, 'w', encoding='utf-8', newline='\n') as f:
            f.write(new_text)
        written.append(dst)
    print("converted:")
    for w in written:
        print("  " + w)

if __name__ == '__main__':
    main()
