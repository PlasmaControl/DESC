def test_SOLOVEV_successful_run(SOLOVEV):
    output = SOLOVEV['output']
    assert output.returncode == 0