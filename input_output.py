import re
import warnings
import h5py
import numpy as np
from datetime import datetime
from netCDF4 import Dataset
from backend import sign


def read_input(fname):
    """Reads input from input file
    
    Args:
        fname (string): filename of input file
    
    Returns:
        inputs (dictionary): all the input parameters and options
    """
    
    # default values
    inputs = {
        'stell_sym'  : False,
        'NFP'        : 1,
        'Psi_lcfs'   : 1.0,
        'Mpol'       : np.atleast_1d(0),
        'Ntor'       : np.atleast_1d(0),
        'Mnodes'     : np.atleast_1d(0),
        'Nnodes'     : np.atleast_1d(0),
        'bdry_ratio' : np.atleast_1d(1.0),
        'pres_ratio' : np.atleast_1d(1.0),
        'zeta_ratio' : np.atleast_1d(1.0),
        'errr_ratio' : np.atleast_1d(1.0),
        'ftol'       : np.atleast_1d(1e-8),
        'xtol'       : np.atleast_1d(1e-8),
        'gtol'       : np.atleast_1d(1e-8),
        'nfev'       : np.atleast_1d(None),
        'verbose'    : 2,
        'errr_mode'  : 'force',
        'bdry_mode'  : 'spectral',
        'node_mode'  : 'cheb1',
        'out_fname'  : fname.split('.')[0]+'.output',
        'cP'         : np.atleast_1d(0.0),
        'cI'         : np.atleast_1d(0.0),
        'axis'       : np.atleast_2d((0,0.0,0.0)),
        'bdry'       : np.atleast_2d((0,0,0.0,0.0))
    }
    
    # file objects
    file = open(fname,'r')
    
    num_form = '-?\ *\d+\.?\d*(?:[Ee]\ *[-+]?\ *\d+)?'
    
    for line in file:
        
        # remove comments
        match = re.search('[!#]',line)
        if match:
            comment = match.start()
        else:
            comment = len(line)
        match = re.search('=',line)
        if match:
            equals = match.start()
        else:
            equals = len(line)
        command = (line.strip()+' ')[0:comment]
        
        argument = "".join((command.strip()+' ')[0:equals].split())
        numbers = [float(x) for x in re.findall(num_form,command)]
        words = command[equals+1:].split()
        
        # solver parameters
        match = re.search('stell_sym',argument,re.IGNORECASE)
        if match:
            inputs['stell_sym'] = int(numbers[0])
        match = re.search('NFP',argument,re.IGNORECASE)
        if match:
            inputs['NFP'] = int(numbers[0])
        match = re.search('Psi_lcfs',argument,re.IGNORECASE)
        if match:
            inputs['Psi_lcfs'] = numbers[0]
        match = re.search('Mpol',argument,re.IGNORECASE)
        if match:
            inputs['Mpol'] = np.array(numbers).astype(int)
        match = re.search('Ntor',argument,re.IGNORECASE)
        if match:
            inputs['Ntor'] = np.array(numbers).astype(int)
        match = re.search('Mnodes',argument,re.IGNORECASE)
        if match:
            inputs['Mnodes'] = np.array(numbers).astype(int)
        match = re.search('Nnodes',argument,re.IGNORECASE)
        if match:
            inputs['Nnodes'] = np.array(numbers).astype(int)
        match = re.search('bdry_ratio',argument,re.IGNORECASE)
        if match:
            inputs['bdry_ratio'] = np.array(numbers).astype(float)
        match = re.search('pres_ratio',argument,re.IGNORECASE)
        if match:
            inputs['pres_ratio'] = np.array(numbers).astype(float)
        match = re.search('zeta_ratio',argument,re.IGNORECASE)
        if match:
            inputs['zeta_ratio'] = np.array(numbers).astype(float)
        match = re.search('errr_ratio',argument,re.IGNORECASE)
        if match:
            inputs['errr_ratio'] = np.array(numbers).astype(float)
        match = re.search('ftol',argument,re.IGNORECASE)
        if match:
            inputs['ftol'] = np.array(numbers).astype(float)
        match = re.search('xtol',argument,re.IGNORECASE)
        if match:
            inputs['xtol'] = np.array(numbers).astype(float)
        match = re.search('gtol',argument,re.IGNORECASE)
        if match:
            inputs['gtol'] = np.array(numbers).astype(float)
        match = re.search('nfev',argument,re.IGNORECASE)
        if match:
            inputs['nfev'] = np.array([None if i==0 else i for i in numbers]).astype(int)
        match = re.search('verbose',argument,re.IGNORECASE)
        if match:
            inputs['verbose'] = int(numbers[0])
        match = re.search('errr_mode',argument,re.IGNORECASE)
        if match:
            inputs['errr_mode'] = words[0]
        match = re.search('bdry_mode',argument,re.IGNORECASE)
        if match:
            inputs['bdry_mode'] = words[0]
        match = re.search('node_mode',argument,re.IGNORECASE)
        if match:
            inputs['node_mode'] = words[0]
        match = re.search('out_fname',argument,re.IGNORECASE)
        if match:
            inputs['out_fname'] = words[0]
        
        # coefficient indicies
        match = re.search('l\s*:\s*'+num_form,command,re.IGNORECASE)
        if match:
            l = int(re.findall(num_form,match.group(0))[0])
        match = re.search('m\s*:\s*'+num_form,command,re.IGNORECASE)
        if match:
            m = int(re.findall(num_form,match.group(0))[0])
        match = re.search('n\s*:\s*'+num_form,command,re.IGNORECASE)
        if match:
            n = int(re.findall(num_form,match.group(0))[0])
        
        # profile coefficients
        match = re.search('cP\s*=\s*'+num_form,command,re.IGNORECASE)
        if match:
            cP = float(re.findall(num_form,match.group(0))[0])
            if inputs['cP'].size < l+1:
                inputs['cP'] = np.pad(inputs['cP'],(0,l+1-inputs['cP'].size),mode='constant')
            inputs['cP'][l] = cP
        match = re.search('cI\s*=\s*'+num_form,command,re.IGNORECASE)
        if match:
            cI = float(re.findall(num_form,match.group(0))[0])
            if inputs['cI'].size < l+1:
                inputs['cI'] = np.pad(inputs['cI'],(0,l+1-inputs['cI'].size),mode='constant')
            inputs['cI'][l] = cI
        
        # magnetic axis Fourier modes
        match = re.search('aR\s*=\s*'+num_form,command,re.IGNORECASE)
        if match:
            aR = float(re.findall(num_form,match.group(0))[0])
            axis_idx = np.where(inputs['axis'][:,0] == n)[0]
            if axis_idx.size == 0:
                axis_idx = np.atleast_1d(inputs['axis'].shape[0])
                inputs['axis'] = np.pad(inputs['axis'],((0,1),(0,0)),mode='constant')
                inputs['axis'][axis_idx[0],0] = n
            inputs['axis'][axis_idx[0],1] = aR
        match = re.search('aZ\s*=\s*'+num_form,command,re.IGNORECASE)
        if match:
            aZ = float(re.findall(num_form,match.group(0))[0])
            axis_idx = np.where(inputs['axis'][:,0] == n)[0]
            if axis_idx.size == 0:
                axis_idx = np.atleast_1d(inputs['axis'].shape[0])
                inputs['axis'] = np.pad(inputs['axis'],((0,1),(0,0)),mode='constant')
                inputs['axis'][axis_idx[0],0] = n
            inputs['axis'][axis_idx[0],2] = aZ
        
        # boundary Fourier modes
        match = re.search('bR\s*=\s*'+num_form,command,re.IGNORECASE)
        if match:
            bR = float(re.findall(num_form,match.group(0))[0])
            bdry_m = np.where(inputs['bdry'][:,0] == m)[0]
            bdry_n = np.where(inputs['bdry'][:,1] == n)[0]
            bdry_idx = np.where([i in bdry_m for i in bdry_n])[0]
            if bdry_idx.size == 0:
                bdry_idx = np.atleast_1d(inputs['bdry'].shape[0])
                inputs['bdry'] = np.pad(inputs['bdry'],((0,1),(0,0)),mode='constant')
                inputs['bdry'][bdry_idx[0],0] = m
                inputs['bdry'][bdry_idx[0],1] = n
            inputs['bdry'][bdry_idx[0],2] = bR
        match = re.search('bZ\s*=\s*'+num_form,command,re.IGNORECASE)
        if match:
            bZ = float(re.findall(num_form,match.group(0))[0])
            bdry_m = np.where(inputs['bdry'][:,0] == m)[0]
            bdry_n = np.where(inputs['bdry'][:,1] == n)[0]
            bdry_idx = np.where([i in bdry_m for i in bdry_n])[0]
            if bdry_idx.size == 0:
                bdry_idx = np.atleast_1d(inputs['bdry'].shape[0])
                inputs['bdry'] = np.pad(inputs['bdry'],((0,1),(0,0)),mode='constant')
                inputs['bdry'][bdry_idx[0],0] = m
                inputs['bdry'][bdry_idx[0],1] = n
            inputs['bdry'][bdry_idx[0],3] = bZ
    
    # error handling
    if np.any(inputs['Mpol'] == 0):
        raise Exception('Mpol is not assigned')
    if np.sum(inputs['bdry']) == 0:
        raise Exception('fixed-boundary surface is not assigned')
    arrs = ['Mpol','Ntor','Mnodes','Nnodes','bdry_ratio','pres_ratio','zeta_ratio','errr_ratio','ftol','xtol','gtol','nfev']
    arr_len = 0
    for a in arrs:
        arr_len = max(arr_len,len(inputs[a]))
    for a in arrs:
        if inputs[a].size == 1:
            inputs[a] = np.broadcast_to(inputs[a],arr_len,subok=True).copy()
        elif inputs[a].size != arr_len:
            raise Exception('continuation array inputs are not proper lengths')
    
    # unsupplied values
    if np.sum(inputs['Mnodes']) == 0:
        inputs['Mnodes'] = inputs['Mpol']
    if np.sum(inputs['Nnodes']) == 0:
        inputs['Nnodes'] = inputs['Ntor']
    if np.sum(inputs['axis']) == 0:
        axis_idx = np.where(inputs['bdry'][:,0] == 0)[0]
        inputs['axis'] = inputs['bdry'][axis_idx,1:]
    
    return inputs


def output_to_file(fname,equil):
    """Prints the equilibrium solution to a text file
    
    Args:
        fname (string): filename of output file
    
    Returns:
        None, creates a text file of the output
    """
    
    cR         = equil['cR']
    cZ         = equil['cZ']
    cL         = equil['cL']
    bdryR      = equil['bdryR']
    bdryZ      = equil['bdryZ']
    cP         = equil['cP']
    cI         = equil['cI']
    Psi_lcfs   = equil['Psi_lcfs']
    NFP        = equil['NFP']
    zern_idx   = equil['zern_idx']
    lambda_idx = equil['lambda_idx']
    bdry_idx   = equil['bdry_idx']
    
    # open file
    file = open(fname,'w+')
    file.seek(0)
    
    # scaling factors
    file.write('NFP = {:3d}\n'.format(NFP))
    file.write('Psi = {:16.8E}\n'.format(Psi_lcfs))

    # boundary paramters
    nbdry = len(bdry_idx)
    file.write('Nbdry = {:3d}\n'.format(nbdry))
    for k in range(nbdry):
        file.write('m: {:3d} n: {:3d} bR = {:16.8E} bZ = {:16.8E}\n'.format(int(bdry_idx[k,0]),int(bdry_idx[k,1]),bdryR[k],bdryZ[k]))
    
    # profile coefficients
    nprof = max(cP.size,cI.size)
    file.write('Nprof = {:3d}\n'.format(nprof))
    for k in range(nprof):
        if k >= cP.size:
            file.write('l: {:3d} cP = {:16.8E} cI = {:16.8E}\n'.format(k,0,cI[k]))
        elif k >= cI.size:
            file.write('l: {:3d} cP = {:16.8E} cI = {:16.8E}\n'.format(k,cP[k],0))
        else:
            file.write('l: {:3d} cP = {:16.8E} cI = {:16.8E}\n'.format(k,cP[k],cI[k]))
    
    # R & Z Fourier-Zernike coefficients
    nRZ = len(zern_idx)
    file.write('NRZ = {:5d}\n'.format(nRZ))
    for k, lmn in enumerate(zern_idx):
        file.write('l: {:3d} m: {:3d} n: {:3d} cR = {:16.8E} cZ = {:16.8E}\n'.format(lmn[0],lmn[1],lmn[2],cR[k],cZ[k]))
    
    # lambda Fourier coefficients
    nL = len(lambda_idx)
    file.write('NL = {:5d}\n'.format(nL))
    for k, mn in enumerate(lambda_idx):
        file.write('m: {:3d} n: {:3d} cL = {:16.8E}\n'.format(mn[0],mn[1],cL[k]))
    
    # close file
    file.truncate()
    file.close()
    
    return None


def read_desc(filename):
    """reads a previously generated DESC ascii output file"""

    equil = {}
    f = open(filename,'r')
    lines = list(f)
    equil['NFP'] = int(lines[0].strip('\n').split()[-1])
    equil['Psi_lcfs'] = float(lines[1].strip('\n').split()[-1])
    lines = lines[2:]

    Nbdry = int(lines[0].strip('\n').split()[-1])
    equil['bdry_idx'] = np.zeros((Nbdry,2),dtype=int)
    equil['r_bdry_coef'] = np.zeros(Nbdry)
    equil['z_bdry_coef'] = np.zeros(Nbdry)
    for i in range(Nbdry):
        equil['bdry_idx'][i,0] = int(lines[i+1].strip('\n').split()[1])
        equil['bdry_idx'][i,1] = int(lines[i+1].strip('\n').split()[3])
        equil['r_bdry_coef'][i] = float(lines[i+1].strip('\n').split()[6])
        equil['z_bdry_coef'][i] = float(lines[i+1].strip('\n').split()[9])
    lines = lines[Nbdry+1:]

    Nprof = int(lines[0].strip('\n').split()[-1])
    equil['pres_coef'] = np.zeros(Nprof)
    equil['iota_coef'] = np.zeros(Nprof)
    for i in range(Nprof):
        equil['pres_coef'][i] = float(lines[i+1].strip('\n').split()[4])
        equil['iota_coef'][i] = float(lines[i+1].strip('\n').split()[7])
    lines = lines[Nprof+1:]

    NRZ = int(lines[0].strip('\n').split()[-1])
    equil['zern_idx'] = np.zeros((NRZ,3),dtype=int)
    equil['r_coef'] = np.zeros(NRZ)
    equil['z_coef'] = np.zeros(NRZ)
    for i in range(NRZ):
        equil['zern_idx'][i,0] = int(lines[i+1].strip('\n').split()[1])
        equil['zern_idx'][i,1] = int(lines[i+1].strip('\n').split()[3])
        equil['zern_idx'][i,2] = int(lines[i+1].strip('\n').split()[5])
        equil['r_coef'][i] = float(lines[i+1].strip('\n').split()[8])
        equil['z_coef'][i] = float(lines[i+1].strip('\n').split()[11])
    lines = lines[NRZ+1:]

    NL = int(lines[0].strip('\n').split()[-1])
    equil['lambda_idx'] = np.zeros((NL,2),dtype=int)
    equil['lambda_coef'] = np.zeros(NL)
    for i in range(NL):
        equil['lambda_idx'][i,0] = int(lines[i+1].strip('\n').split()[1])
        equil['lambda_idx'][i,1] = int(lines[i+1].strip('\n').split()[3])
        equil['lambda_coef'][i] = float(lines[i+1].strip('\n').split()[6])
    lines = lines[NL+1:]
    
    return equil


def write_desc_h5(filename,equilibrium):
    """Writes a DESC equilibrium to a hdf5 format binary file"""

    f = h5py.File(filename,'w')
    equil = f.create_group('equilibrium')
    for key,val in equilibrium.items():
        equil.create_dataset(key,data=val)
    equil['zern_idx'].attrs.create('column_labels',['l','m','n'])
    equil['bdry_idx'].attrs.create('column_labels',['m','n'])
    equil['lambda_idx'].attrs.create('column_labels',['m','n'])
    f.close()


def load_from_file(fname):
    
    file = open(fname,'r')
    file.seek(0)
    num_form = '-?\ *\d+\.?\d*(?:[Ee]\ *[-+]?\ *\d+)?'
    
    # NFP
    line = file.readline()
    numbers = [int(x) for x in re.findall(num_form,line)]
    NFP = numbers[0]
    
    # Psi_total
    line = file.readline()
    numbers = [float(x) for x in re.findall(num_form,line)]
    Psi_total = numbers[0]
    
    # bdry
    line = file.readline()
    numbers = [int(x) for x in re.findall(num_form,line)]
    Nbdry = numbers[0]
    bdry = np.zeros((Nbdry,4))
    for i in range(Nbdry):
        line = file.readline()
        numbers = [float(x) for x in re.findall(num_form,line)]
        bdry[i,:] = np.array([numbers[0],numbers[1],numbers[2],numbers[3]])
    
    # cP & cI
    line = file.readline()
    numbers = [int(x) for x in re.findall(num_form,line)]
    Nprof = numbers[0]
    cP = np.zeros((Nprof,))
    cI = np.zeros((Nprof,))
    for i in range(Nprof):
        line = file.readline()
        numbers = [float(x) for x in re.findall(num_form,line)]
        cP[i] = numbers[1]
        cI[i] = numbers[2]
    
    # cR & cZ
    line = file.readline()
    numbers = [int(x) for x in re.findall(num_form,line)]
    NRZ = numbers[0]
    zern_idx = np.zeros((NRZ,3))
    cR = np.zeros((NRZ,))
    cZ = np.zeros((NRZ,))
    for i in range(NRZ):
        line = file.readline()
        numbers = [float(x) for x in re.findall(num_form,line)]
        zern_idx[i,:] = np.array([numbers[0],numbers[1],numbers[2]])
        cR[i] = numbers[3]
        cZ[i] = numbers[4]
    
    # cL
    line = file.readline()
    numbers = [int(x) for x in re.findall(num_form,line)]
    NL = numbers[0]
    lambda_idx = np.zeros((NL,2))
    cL = np.zeros((NL,))
    for i in range(NL):
        line = file.readline()
        numbers = [float(x) for x in re.findall(num_form,line)]
        lambda_idx[i,:] = np.array([numbers[0],numbers[1]])
        cL[i] = numbers[2]
    
    x = np.concatenate([cR,cZ,cL])
    file.close()
    return x,zern_idx,lambda_idx,NFP,Psi_total,cP,cI,bdry


# TODO: enable multi-line VMEC inputs
def vmec_to_desc_input(vmec_fname,desc_fname):
    """Converts a VMEC input file to an equivalent DESC input file
    
    Args:
        vmec_fname (string): filename of VMEC input file
        desc_fname (string): filename of DESC input file
    
    Returns:
        None, creates a DESC input file
    """
    
    # file objects
    vmec_file = open(vmec_fname,'r')
    desc_file = open(desc_fname,'w+')
    
    desc_file.seek(0)
    now = datetime.now()
    date = now.strftime('%m/%d/%Y')
    time = now.strftime('%H:%M:%S')
    desc_file.write('! This DESC input file was auto generated from the VMEC input file \n! '+vmec_fname+' on '+date+' at '+time+'.\n\n')
    desc_file.write('! solver parameters\n')
    
    num_form = '-?\ *\d+\.?\d*(?:[Ee]\ *[-+]?\ *\d+)?'
    Ntor = 99
    
    cP = np.array([0.0])
    cI = np.array([0.0])
    axis = np.array([[0,0,0.0]])
    bdry = np.array([[0,0,0.0,0.0]])
    
    for line in vmec_file:
        comment = line.find('!')
        command = (line.strip()+' ')[0:comment]
        
        # grid parameters
        if re.search('LRFP\ *=\ *T',command,re.IGNORECASE):
            warnings.warn('using poloidal flux instead of toroidal flux!')
        match = re.search('LASYM\ *=\ *[TF]',command,re.IGNORECASE)
        if match:
            if re.search('T',match.group(0),re.IGNORECASE):
                desc_file.write('stell_sym \t=   0\n')
            else:
                desc_file.write('stell_sym \t=   1\n')
        match = re.search('NFP\ *=\ *'+num_form,command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(num_form,match.group(0))]
            desc_file.write('NFP\t\t\t= {:3d}\n'.format(numbers[0]))
        match = re.search('MPOL\ *=\ *'+num_form,command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(num_form,match.group(0))]
            desc_file.write('Mpol\t\t= {:3d}\n'.format(numbers[0]))
        match = re.search('NTOR\ *=\ *'+num_form,command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(num_form,match.group(0))]
            desc_file.write('Ntor\t\t= {:3d}\n'.format(numbers[0]))
            Ntor = numbers[0]
        match = re.search('PHIEDGE\ *=\ *'+num_form,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(num_form,match.group(0))]
            desc_file.write('Psi_lcfs\t= {:16.8E}\n'.format(numbers[0]))
        
        # pressure profile parameters
        match = re.search('bPMASS_TYPE\ *=\ *\w*',command,re.IGNORECASE)
        if match:
            if not re.search(r'\bpower_series\b',match.group(0),re.IGNORECASE):
                warnings.warn('pressure is not a power series!')
        match = re.search('GAMMA\ *=\ *'+num_form,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(num_form,match.group(0))]
            if numbers[0] != 0:
                warnings.warn('GAMMA is not 0.0')
        match = re.search('BLOAT\ *=\ *'+num_form,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(num_form,match.group(0))]
            if numbers[0] != 1:
                warnings.warn('BLOAT is not 1.0')
        match = re.search('SPRES_PED\ *=\ *'+num_form,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(num_form,match.group(0))]
            if numbers[0] != 1:
                warnings.warn('SPRES_PED is not 1.0')
        match = re.search('PRES_SCALE\ *=\ *'+num_form,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(num_form,match.group(0))]
            desc_file.write('pres_scale\t= {:16.8E}\n'.format(numbers[0]))
        match = re.search('AM\ *=(\ *'+num_form+')*',command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(num_form,match.group(0))]
            for k in range(len(numbers)):
                l = 2*k
                if cP.size < l+1:
                    cP = np.pad(cP,(0,l+1-cP.size),mode='constant')
                cP[l] = numbers[k]
        
        # rotational transform paramters
        match = re.search('NCURR\ *=(\ *'+num_form+')*',command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(num_form,match.group(0))]
            if numbers[0] != 0:
                warnings.warn('not using rotational transform!')
        if re.search(r'\bPIOTA_TYPE\b',command,re.IGNORECASE):
            if not re.search(r'\bpower_series\b',command,re.IGNORECASE):
                warnings.warn('iota is not a power series!')
        match = re.search('AI\ *=(\ *'+num_form+')*',command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(num_form,match.group(0))]
            for k in range(len(numbers)):
                l = 2*k
                if cI.size < l+1:
                    cI = np.pad(cI,(0,l+1-cI.size),mode='constant')
                cI[l] = numbers[k]
        
        # magnetic axis paramters
        match = re.search('RAXIS\ *=(\ *'+num_form+')*',command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(num_form,match.group(0))]
            for k in range(len(numbers)):
                if k > Ntor:
                    l = -k+Ntor+1
                else:
                    l = k
                idx = np.where(axis[:,0]==l)[0]
                if np.size(idx) > 0:
                    axis[idx[0],1] = numbers[k]
                else:
                    axis = np.pad(axis,((0,1),(0,0)),mode='constant')
                    axis[-1,:] = np.array([l,numbers[k],0.0])
        match = re.search('ZAXIS\ *=(\ *'+num_form+')*',command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(num_form,match.group(0))]
            for k in range(len(numbers)):
                if k > Ntor:
                    l = k-Ntor-1
                else:
                    l = -k
                idx = np.where(axis[:,0]==l)[0]
                if np.size(idx) > 0:
                    axis[idx[0],2] = numbers[k]
                else:
                    axis = np.pad(axis,((0,1),(0,0)),mode='constant')
                    axis[-1,:] = np.array([l,0.0,numbers[k]])
        
        # boundary shape parameters
        # RBS * sin(m*t - n*p) = RBS * sin(m*t)*cos(n*p) - RBS * cos(m*t)*sin(n*p)
        match = re.search('RBS\(\ *'+num_form+'\ *,\ *'+num_form+'\ *\)\ *=\ *'+num_form,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(num_form,match.group(0))]
            n = int(numbers[0])
            m = int(numbers[1])
            n_sgn = sign(np.array([n]))[0]
            n *= n_sgn
            if sign(m) < 0:
                warnings.warn('m is negative!')
            RBS = numbers[2]
            if m != 0:
                m_idx = np.where(bdry[:,0]==-m)[0]
                n_idx = np.where(bdry[:,1]== n)[0]
                idx = np.where(np.isin(m_idx,n_idx))[0]
                if np.size(idx) > 0:
                    bdry[m_idx[idx[0]],2] = RBS
                else:
                    bdry = np.pad(bdry,((0,1),(0,0)),mode='constant')
                    bdry[-1,:] = np.array([-m,n,RBS,0.0])
            if n != 0:
                m_idx = np.where(bdry[:,0]== m)[0]
                n_idx = np.where(bdry[:,1]==-n)[0]
                idx = np.where(np.isin(m_idx,n_idx))[0]
                if np.size(idx) > 0:
                    bdry[m_idx[idx[0]],2] = -n_sgn*RBS
                else:
                    bdry = np.pad(bdry,((0,1),(0,0)),mode='constant')
                    bdry[-1,:] = np.array([m,-n,-n_sgn*RBS,0.0])
        # RBC * cos(m*t - n*p) = RBC * cos(m*t)*cos(n*p) + RBC * sin(m*t)*sin(n*p)
        match = re.search('RBC\(\ *'+num_form+'\ *,\ *'+num_form+'\ *\)\ *=\ *'+num_form,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(num_form,match.group(0))]
            n = int(numbers[0])
            m = int(numbers[1])
            n_sgn = sign(np.array([n]))[0]
            n *= n_sgn
            if sign(m) < 0:
                warnings.warn('m is negative!')
            RBC = numbers[2]
            m_idx = np.where(bdry[:,0]== m)[0]
            n_idx = np.where(bdry[:,1]== n)[0]
            idx = np.where(np.isin(m_idx,n_idx))[0]
            if np.size(idx) > 0:
                bdry[m_idx[idx[0]],2] = RBC
            else:
                bdry = np.pad(bdry,((0,1),(0,0)),mode='constant')
                bdry[-1,:] = np.array([m,n,RBC,0.0])
            if m != 0 and n != 0:
                m_idx = np.where(bdry[:,0]==-m)[0]
                n_idx = np.where(bdry[:,1]==-n)[0]
                idx = np.where(np.isin(m_idx,n_idx))[0]
                if np.size(idx) > 0:
                    bdry[m_idx[idx[0]],2] = n_sgn*RBC
                else:
                    bdry = np.pad(bdry,((0,1),(0,0)),mode='constant')
                    bdry[-1,:] = np.array([-m,-n,n_sgn*RBC,0.0])
        # ZBS * sin(m*t - n*p) = ZBS * sin(m*t)*cos(n*p) - ZBS * cos(m*t)*sin(n*p)
        match = re.search('ZBS\(\ *'+num_form+'\ *,\ *'+num_form+'\ *\)\ *=\ *'+num_form,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(num_form,match.group(0))]
            n = int(numbers[0])
            m = int(numbers[1])
            n_sgn = sign(np.array([n]))[0]
            n *= n_sgn
            if sign(m) < 0:
                warnings.warn('m is negative!')
            ZBS = numbers[2]
            if m != 0:
                m_idx = np.where(bdry[:,0]==-m)[0]
                n_idx = np.where(bdry[:,1]== n)[0]
                idx = np.where(np.isin(m_idx,n_idx))[0]
                if np.size(idx) > 0:
                    bdry[m_idx[idx[0]],3] = ZBS
                else:
                    bdry = np.pad(bdry,((0,1),(0,0)),mode='constant')
                    bdry[-1,:] = np.array([-m,n,0.0,ZBS])
            if n != 0:
                m_idx = np.where(bdry[:,0]== m)[0]
                n_idx = np.where(bdry[:,1]==-n)[0]
                idx = np.where(np.isin(m_idx,n_idx))[0]
                if np.size(idx) > 0:
                    bdry[m_idx[idx[0]],3] = -n_sgn*ZBS
                else:
                    bdry = np.pad(bdry,((0,1),(0,0)),mode='constant')
                    bdry[-1,:] = np.array([m,-n,0.0,-n_sgn*ZBS])
        # ZBC * cos(m*t - n*p) = ZBC * cos(m*t)*cos(n*p) + ZBC * sin(m*t)*sin(n*p)
        match = re.search('ZBC\(\ *'+num_form+'\ *,\ *'+num_form+'\ *\)\ *=\ *'+num_form,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(num_form,match.group(0))]
            n = int(numbers[0])
            m = int(numbers[1])
            n_sgn = sign(np.array([n]))[0]
            n *= n_sgn
            if sign(m) < 0:
                warnings.warn('m is negative!')
            ZBC = numbers[2]
            m_idx = np.where(bdry[:,0]== m)[0]
            n_idx = np.where(bdry[:,1]== n)[0]
            idx = np.where(np.isin(m_idx,n_idx))[0]
            if np.size(idx) > 0:
                bdry[m_idx[idx[0]],3] = ZBC
            else:
                bdry = np.pad(bdry,((0,1),(0,0)),mode='constant')
                bdry[-1,:] = np.array([m,n,0.0,ZBC])
            if m != 0 and n != 0:
                m_idx = np.where(bdry[:,0]==-m)[0]
                n_idx = np.where(bdry[:,1]==-n)[0]
                idx = np.where(np.isin(m_idx,n_idx))[0]
                if np.size(idx) > 0:
                    bdry[m_idx[idx[0]],3] = n_sgn*ZBC
                else:
                    bdry = np.pad(bdry,((0,1),(0,0)),mode='constant')
                    bdry[-1,:] = np.array([-m,-n,0.0,n_sgn*ZBC])
    
    desc_file.write('\n')
    desc_file.write('! pressure and rotational transform profiles\n')
    for k in range(max(cP.size,cI.size)):
        if k >= cP.size:
            desc_file.write('l: {:3d}\tcP = {:16.8E}\tcI = {:16.8E}\n'.format(k,0.0,cI[k]))
        elif k >= cI.size:
            desc_file.write('l: {:3d}\tcP = {:16.8E}\tcI = {:16.8E}\n'.format(k,cP[k],0.0))
        else:
            desc_file.write('l: {:3d}\tcP = {:16.8E}\tcI = {:16.8E}\n'.format(k,cP[k],cI[k]))
    
    desc_file.write('\n')
    desc_file.write('! magnetic axis initial guess\n')
    for k in range(np.shape(axis)[0]):
        desc_file.write('n: {:3d}\taR = {:16.8E}\taZ = {:16.8E}\n'.format(int(axis[k,0]),axis[k,1],axis[k,2]))
    
    desc_file.write('\n')
    desc_file.write('! fixed-boundary surface\n')
    for k in range(np.shape(bdry)[0]):
        desc_file.write('m: {:3d}\tn: {:3d}\tbR = {:16.8E}\tbZ = {:16.8E}\n'.format(int(bdry[k,0]),int(bdry[k,1]),bdry[k,2],bdry[k,3]))
    
    desc_file.truncate()
    
    # close files
    vmec_file.close()
    desc_file.close()
    
    return None


# TODO: add other fields including B, rmns, zmnc, lmnc, etc
def read_vmec_output(fname):
    """Reads VMEC data from wout nc file
    
    Args:
        fname (string): filename of VMEC output file
    
    Returns:
        vmec_data (dictionary): the VMEC data fields
    """
    
    file = Dataset(fname,mode='r')
    
    vmec_data = {
        'xm' : file.variables['xm'][:],
        'xn' : file.variables['xn'][:],
        'rmnc' : file.variables['rmnc'][:],
        'zmns' : file.variables['zmns'][:],
        'lmns' : file.variables['lmns'][:]
        }
    
    return vmec_data


def vmec_interpolate(Cmn,Smn,xm,xn,theta,phi):
    """Interpolates VMEC data on a flux surface
    
    Args:
        Cmn (ndarray, shape(MN,)): cos(mt-np) Fourier coefficients
        Smn (ndarray, shape(MN,)): sin(mt-np) Fourier coefficients
        xm (ndarray, shape(M,)): poloidal mode numbers
        xn (ndarray, shape(N,)): toroidal mode numbers
        theta (ndarray): poloidal angles
        phi (ndarray): toroidal angles
    
    Returns:
        VMEC data interpolated at the angles (theta,phi)
    """
    
    R_arr = []
    Z_arr = []
    dim = Cmn.shape
    
    for j in range(dim[1]):
        
        m = xm[j]
        n = xn[j]
        
        R = [[[Cmn[s,j]*np.cos(m*t - n*p) for p in phi] for t in theta] for s in range(dim[0])]
        Z = [[[Smn[s,j]*np.sin(m*t - n*p) for p in phi] for t in theta] for s in range(dim[0])]
        R_arr.append(R)
        Z_arr.append(Z)
    
    return np.sum(R_arr,axis=0), np.sum(Z_arr,axis=0)
