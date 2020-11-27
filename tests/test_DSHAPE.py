
def test_DSHAPE_successful_run(DSHAPE):
    output = DSHAPE['output']
    assert output.returncode == 0
