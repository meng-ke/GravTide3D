from pathlib import Path

import numpy as np
import pytest

from gravtide3d import ExecutionConfig, gravitational_potential, tidal_eigensystem
from gravtide3d.storage.scratch import ScratchManager


VOXEL = (1e17, 1.1e17, 0.9e17)


def _cfg(**kwargs):
    return ExecutionConfig(workers=1, spectral_tile=(4, 4), eig_tile=(4, 5), **kwargs)


def _rho(multiplier=1.0):
    z, y, x = np.indices((6, 8, 10))
    return (2e-20 * (1 + 0.10 * multiplier * np.cos(2 * np.pi * z / 6)
                        + 0.04 * np.cos(2 * np.pi * x / 10))).astype(np.float32)


def test_scratch_manager_never_deletes_caller_directory(tmp_path):
    root = tmp_path / "scratch"
    root.mkdir()
    sentinel = root / "important.txt"
    sentinel.write_text("user data")
    with ScratchManager(root, cleanup=True) as manager:
        assert manager.path == root
    assert sentinel.read_text() == "user data"
    with ScratchManager(root) as manager:
        with pytest.raises(ValueError, match="user-provided"):
            manager.remove()
    assert sentinel.exists()
    with ScratchManager(cleanup=True) as manager:
        owned = manager.path
        assert owned.exists()
    assert not owned.exists()


def test_changed_file_does_not_resume_or_delete_user_outputs(tmp_path):
    density_path = tmp_path / "input.npy"
    np.save(density_path, _rho())
    out = tmp_path / "results"
    scratch = tmp_path / "user_scratch"
    scratch.mkdir()
    (out).mkdir()
    (out / "unrelated.txt").write_text("keep")
    (scratch / "unrelated.txt").write_text("keep")

    first = gravitational_potential(density_path, VOXEL, backend="cpu", memory="out_of_core",
                                    output=out, scratch=scratch, config=_cfg())
    before = np.asarray(first.phi).copy()
    assert (scratch / "unrelated.txt").read_text() == "keep"
    # Reuse the exact file path/shape but change density contents.
    np.save(density_path, _rho(1.5))
    for overwrite in (False, True):
        with pytest.raises(ValueError, match="different calculation"):
            gravitational_potential(
                density_path, VOXEL, backend="cpu", memory="out_of_core",
                output=out, scratch=scratch, config=_cfg(overwrite=overwrite),
            )
    assert (out / "unrelated.txt").read_text() == "keep"
    assert (scratch / "unrelated.txt").read_text() == "keep"
    assert np.array_equal(np.load(out / "phi3d.npy"), before)


def test_different_memory_array_cannot_reuse_existing_density(tmp_path):
    out = tmp_path / "results"
    scratch = tmp_path / "scratch"
    gravitational_potential(_rho(), VOXEL, backend="cpu", memory="out_of_core",
                            output=out, scratch=scratch, config=_cfg())
    with pytest.raises(ValueError, match="different calculation"):
        gravitational_potential(_rho(1.5), VOXEL, backend="cpu", memory="out_of_core",
                                output=out, scratch=scratch, config=_cfg())
    assert len(list(scratch.glob('run-*/density_input_*.npy'))) == 2


def test_tidal_correction_change_requires_overwrite(tmp_path):
    density_path = tmp_path / "input.npy"
    np.save(density_path, _rho())
    out = tmp_path / "ooc"
    first = tidal_eigensystem(density_path, VOXEL, correction="dc", backend="cpu",
                              memory="out_of_core", output=out,
                              scratch=tmp_path / "scratch", config=_cfg())
    before = np.asarray(first.eigenvalues).copy()
    with pytest.raises(ValueError, match="different correction"):
        tidal_eigensystem(density_path, VOXEL, correction="none", backend="cpu",
                          memory="out_of_core", output=out,
                          scratch=tmp_path / "scratch", config=_cfg())
    assert np.array_equal(np.load(out / "eigenvalues.npy"), before)
    raw = tidal_eigensystem(density_path, VOXEL, correction="none", backend="cpu",
                            memory="out_of_core", output=out,
                            scratch=tmp_path / "scratch", config=_cfg(overwrite=True))
    assert raw.correction == "none"
    assert not np.allclose(raw.eigenvalues, before, rtol=1e-5, atol=0.0)


def test_unverified_existing_results_are_not_relabelled(tmp_path):
    density_path = tmp_path / "input.npy"
    np.save(density_path, _rho())
    out = tmp_path / "output"
    out.mkdir()
    (out / "phi3d.npy").write_text("not a valid result")
    with pytest.raises(ValueError, match="no run signature"):
        gravitational_potential(density_path, VOXEL, memory="out_of_core",
                                output=out, scratch=tmp_path / "scratch", config=_cfg())
    assert not (out / "run_signature.json").exists()
    assert (out / "phi3d.npy").read_text() == "not a valid result"


def test_same_input_file_can_resume_and_keep_scratch_root_files(tmp_path):
    density_path = tmp_path / "input.npy"
    np.save(density_path, _rho())
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "my_data.txt").write_text("untouched")
    args = dict(density=density_path, voxel_size=VOXEL, memory="out_of_core",
                backend="cpu", output=tmp_path / "output", scratch=scratch, config=_cfg())
    first = gravitational_potential(**args)
    second = gravitational_potential(**args)
    assert np.array_equal(first.phi, second.phi)
    assert (scratch / "my_data.txt").read_text() == "untouched"


def test_potential_result_chaining_recovers_custom_scratch_workspace(tmp_path):
    density_path = tmp_path / "input.npy"
    np.save(density_path, _rho())
    scratch = tmp_path / "external_scratch_root"
    pot = gravitational_potential(
        density_path, VOXEL, memory="out_of_core", backend="cpu",
        output=tmp_path / "output", scratch=scratch, config=_cfg(),
        keep_spectrum=False,
    )
    assert pot.spectral_path is None
    tidal = tidal_eigensystem(pot, correction="dc", memory="out_of_core",
                              config=_cfg())
    assert tidal.eigenvalues.shape == _rho().shape + (3,)
    assert tidal.output_path == pot.output_path
