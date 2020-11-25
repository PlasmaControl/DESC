import pytest
import subprocess
import os

@pytest.fixture(scope="session")
def DSHAPE(tmpdir_factory):
    max_time = 5*60 # 5 minute max time for DSHAPE run
    temp_dir = tmpdir_factory.mktemp('result').join('DSHAPE_out')
    input_path = 'tests//inputs//DSHAPE'
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd,'..')
    input_filename = os.path.join(exec_dir,input_path)
    print('Running DSHAPE test')
    print('exec_dir=',exec_dir)
    print('cwd=',cwd)
    
    # DSHAPE_run = subprocess.run(['ls'],
    #                      stdout = subprocess.PIPE, universal_newlines=True,
    #                      timeout = max_time, cwd=cwd)
    DSHAPE_run = subprocess.run(['python','-m', 'desc','-o',str(temp_dir),input_filename],
                         stdout = subprocess.PIPE, universal_newlines=True,
                         timeout = max_time, cwd=exec_dir)
    DSHAPE_out = {'output': DSHAPE_run, 'filepath':temp_dir}
    return DSHAPE_out

def pytest_collection_modifyitems(items):
    for item in items:
        if 'DSHAPE' in getattr(item, 'fixturenames', ()):
            item.add_marker('slow')