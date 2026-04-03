"""Tests for the GX external objective helpers."""

from desc.external.gx import _run_gx, _write_gx_input


def test_write_gx_input_forces_eik_geometry(tmp_path):
    """GX inputs should match DESC's plain-text geometry output."""
    template_path = tmp_path / "template.in"
    template_path.write_text(
        '[Geometry]\n'
        'geo_option = "nc"\n'
        'geo_file = "placeholder.nc"\n'
    )

    output_path = tmp_path / "gx.in"
    geo_path = tmp_path / "gx_geo.out"
    _write_gx_input(str(template_path), str(output_path), str(geo_path))

    data = output_path.read_text()
    assert 'geo_option = "eik"' in data
    assert f"geo_file = '{geo_path}'" in data


def test_run_gx_reports_child_logs(tmp_path):
    """GX launcher failures should surface the captured stdout/stderr."""
    exec_path = tmp_path / "fake_gx.sh"
    exec_path.write_text(
        "#!/bin/sh\n"
        'echo "starting gx"\n'
        'echo "fatal child error" >&2\n'
        "exit 2\n"
    )
    exec_path.chmod(0o755)

    try:
        _run_gx(str(tmp_path), str(exec_path))
    except RuntimeError as err:
        message = str(err)
    else:
        raise AssertionError("_run_gx should have raised RuntimeError")

    assert "exit status: 2" in message
    assert "starting gx" in message
    assert "fatal child error" in message
    assert str(tmp_path / "stdout.gx") in message
    assert str(tmp_path / "stderr.gx") in message
