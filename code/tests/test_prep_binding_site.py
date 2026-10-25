"""Local (CPU, no MLIP) checks of prep_binding_site on 2A3R / LDP chain A.

Run with:  python -m pytest code/tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "UMADock"))
import prep_binding_site as prep  # noqa: E402

PDB = ROOT / "data" / "pdbs" / "2A3R.pdb"


def _build(tmp_path, fill_gaps):
    bs = prep.prepare_binding_site(PDB, "LDP", chain="A", output_dir=tmp_path,
                                   write_pdb=False, fill_gaps=fill_gaps)
    xyz = np.loadtxt(bs["file_location"], skiprows=2, usecols=(1, 2, 3))
    return bs, xyz


@pytest.mark.parametrize("fill_gaps, n_close", [(True, 0), (False, 4)])
def test_one_residue_gaps_and_double_caps(tmp_path, fill_gaps, n_close):
    # Without gap filling, residues 106/108 and 146/148 are capped as separate
    # fragments and the NME/ACE caps overlap where 107 / 147 should be.
    bs, xyz = _build(tmp_path, fill_gaps)
    assert len(cKDTree(xyz).query_pairs(0.9)) == n_close
    assert bs["size"] == len(xyz)


def test_gap_residues_added(tmp_path):
    bs, _ = _build(tmp_path, True)
    assert len(bs["constraints"]) == 12   # 10 within 4 A + Ser107 + Lys147
    assert bs["charge"] == 0              # Lys147 brings +1 to the old -1 site
