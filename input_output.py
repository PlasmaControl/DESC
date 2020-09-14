import re
import warnings
import h5py
import numpy as np
from datetime import datetime
from backend import unpack_x

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
    
    now = datetime.now()
    date = now.strftime('%m/%d/%Y')
    time = now.strftime('%H:%M:%S')
    desc_file.write('! This DESC input file was auto generated from the VMEC input file \n! '+vmec_fname+' on '+date+' at '+time+'.\n\n')
    
    number_format = '-?\ *\d+\.?\d*(?:[Ee]\ *[-+]?\ *\d+)?'
    Ntor = 99
    
    desc_file.seek(0)
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
        match = re.search('NFP\ *=\ *'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(number_format,match.group(0))]
            desc_file.write('NFP\t\t\t= {:3d}\n'.format(numbers[0]))
        match = re.search('MPOL\ *=\ *'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(number_format,match.group(0))]
            desc_file.write('Mpol\t\t= {:3d}\n'.format(numbers[0]))
        match = re.search('NTOR\ *=\ *'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(number_format,match.group(0))]
            desc_file.write('Ntor\t\t= {:3d}\n'.format(numbers[0]))
            Ntor = numbers[0]
        match = re.search('PHIEDGE\ *=\ *'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            desc_file.write('Psi_total\t= {:16.8E}\n'.format(numbers[0]))
        
        # pressure profile parameters
        match = re.search('bPMASS_TYPE\ *=\ *\w*',command,re.IGNORECASE)
        if match:
            if not re.search(r'\bpower_series\b',match.group(0),re.IGNORECASE):
                warnings.warn('pressure is not a power series!')
        match = re.search('GAMMA\ *=\ *'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            if numbers[0] != 0:
                warnings.warn('GAMMA is not 0.0')
        match = re.search('BLOAT\ *=\ *'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            if numbers[0] != 1:
                warnings.warn('BLOAT is not 1.0')
        match = re.search('SPRES_PED\ *=\ *'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            if numbers[0] != 1:
                warnings.warn('SPRES_PED is not 1.0')
        match = re.search('PRES_SCALE\ *=\ *'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            desc_file.write('pres_scale\t= {:16.8E}\n'.format(numbers[0]))
        match = re.search('AM\ *=(\ *'+number_format+')*',command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            for k in range(len(numbers)):
                desc_file.write('l: {:3d}\tcP = {:16.8E}\n'.format(int(2*k),numbers[k]))
        
        # rotational transform paramters
        match = re.search('NCURR\ *=(\ *'+number_format+')*',command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            if numbers[0] != 0:
                warnings.warn('not using rotational transform!')
        if re.search(r'\bPIOTA_TYPE\b',command,re.IGNORECASE):
            if not re.search(r'\bpower_series\b',command,re.IGNORECASE):
                warnings.warn('iota is not a power series!')
        match = re.search('AI\ *=(\ *'+number_format+')*',command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            for k in range(len(numbers)):
                desc_file.write('l: {:3d}\tcI = {:16.8E}\n'.format(k,numbers[k]))
        
        # magnetic axis paramters
        match = re.search('RAXIS\ *=(\ *'+number_format+')*',command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            for k in range(len(numbers)):
                if k > Ntor:
                    desc_file.write('n: {:3d}\taR = {:16.8E}\n'.format(-k+Ntor+1,numbers[k]))
                else:
                    desc_file.write('n: {:3d}\taR = {:16.8E}\n'.format(k,numbers[k]))
        match = re.search('ZAXIS\ *=(\ *'+number_format+')*',command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            for k in range(len(numbers)):
                if k > Ntor:
                    desc_file.write('n: {:3d}\taR = {:16.8E}\n'.format(k-Ntor-1,numbers[k]))
                else:
                    desc_file.write('n: {:3d}\taR = {:16.8E}\n'.format(-k,numbers[k]))
        
        # boundary shape parameters
        match = re.search('RBS\(\ *'+number_format+'\ *,\ *'+number_format+'\ *\)\ *=\ *'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            desc_file.write('m: {:3d}\tn: {:3d}\tbR = {:16.8E}\n'.format(int(-numbers[1]),int(numbers[0]),numbers[2]))
        match = re.search('RBC\(\ *'+number_format+'\ *,\ *'+number_format+'\ *\)\ *=\ *'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            desc_file.write('m: {:3d}\tn: {:3d}\tbR = {:16.8E}\n'.format(int(numbers[1]),int(numbers[0]),numbers[2]))
        match = re.search('ZBS\(\ *'+number_format+'\ *,\ *'+number_format+'\ *\)\ *=\ *'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            desc_file.write('m: {:3d}\tn: {:3d}\tbZ = {:16.8E}\n'.format(int(-numbers[1]),int(numbers[0]),numbers[2]))
        match = re.search('ZBC\(\ *'+number_format+'\ *,\ *'+number_format+'\ *\)\ *=\ *'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            desc_file.write('m: {:3d}\tn: {:3d}\tbZ = {:16.8E}\n'.format(int(numbers[1]),int(numbers[0]),numbers[2]))
    
    desc_file.truncate()
    
    # close files
    vmec_file.close()
    desc_file.close()
    
    return None

def read_input(fname):
    """Reads input from input file
    
    Args:
        fname (string): filename of input file
    
    Returns:
        inputs (dictionary): all the input parameters and options
    """
    
    # default values
    inputs = {
        'stell_sym' : False,
        'Mpol' : 0,
        'Ntor' : 0,
        'Mnodes' : 0,
        'Nnodes' : 0,
        'NFP' : 1,
        'Psi_total' : 1.0,
        'error_mode' : 'force',
        'bdry_mode' : 'spectral',
        'node_mode' : 'cheb1',
        'presfun_params' : np.array([0.0]),
        'iotafun_params' : np.array([0.0]),
        'axis' : np.array([[0,0.0,0.0]]),
        'bdry' : np.array([[0,0,0.0,0.0]])
    }
    
    # TODO: allow M,N arrays for continuation method
    
    # file objects
    file = open(fname,'r')
    
    number_format = '-?\ *\d+\.?\d*(?:[Ee]\ *[-+]?\ *\d+)?'
    pres_scale = 1.0
    
    for line in file:
        comment = line.find('!')
        command = (line.strip()+' ')[0:comment]
        
        # solver parameters
        match = re.search('stell_sym[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(number_format,match.group(0))]
            if numbers[0] == 1:
                inputs['stell_sym'] = True
        match = re.search('Mpol[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(number_format,match.group(0))]
            inputs['Mpol'] = numbers[0]
        match = re.search('Ntor[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(number_format,match.group(0))]
            inputs['Ntor'] = numbers[0]
        match = re.search('NFP[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(number_format,match.group(0))]
            inputs['NFP'] = numbers[0]
        match = re.search('Psi_total[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            inputs['Psi_total'] = numbers[0]
        match = re.search('error_mode[\ \t]*=[\ \t]*\w*',command,re.IGNORECASE)
        if match:
            words = match.group(0).split()
            inputs['error_mode'] = words[-1]
        match = re.search('bdry_mode[\ \t]*=[\ \t]*\w*',command,re.IGNORECASE)
        if match:
            words = match.group(0).split()
            inputs['bdry_mode'] = words[-1]
        match = re.search('node_mode[\ \t]*=[\ \t]*\w*',command,re.IGNORECASE)
        if match:
            words = match.group(0).split()
            inputs['node_mode'] = words[-1]
        
        # profile coefficients
        match = re.search('pres_scale[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            pres_scale = numbers[0]
        match = re.search('l:[\ \t]*'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(number_format,match.group(0))]
            l = numbers[0]
            match_prof = re.search('cP[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
            if match_prof:
                numbers = [float(x) for x in re.findall(number_format,match_prof.group(0))]
                cP = numbers[0]
                if inputs['presfun_params'].size < l+1:
                    inputs['presfun_params'] = np.pad(inputs['presfun_params'],(0,l+1-inputs['presfun_params'].size),mode='constant')
                inputs['presfun_params'][l] = cP
            match_prof = re.search('cI[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
            if match_prof:
                numbers = [float(x) for x in re.findall(number_format,match_prof.group(0))]
                cI = numbers[0]
                if inputs['iotafun_params'].size < l+1:
                    inputs['iotafun_params'] = np.pad(inputs['iotafun_params'],(0,l+1-inputs['iotafun_params'].size),mode='constant')
                inputs['iotafun_params'][l] = cI
        
        # magnetic axis Fourier modes
        match = re.search('n:[\ \t]*'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(number_format,match.group(0))]
            n = numbers[0]
            if np.sum(np.absolute(inputs['axis'])) != 0:
                inputs['axis'] = np.pad(inputs['axis'],((0,1),(0,0)),mode='constant')
            inputs['axis'][-1,0] = n
            match_axis = re.search('aR[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
            if match_axis:
                numbers = [float(x) for x in re.findall(number_format,match_axis.group(0))]
                aR = numbers[0]
                inputs['axis'][-1,1] = aR
            match_axis = re.search('aZ[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
            if match_axis:
                numbers = [float(x) for x in re.findall(number_format,match_axis.group(0))]
                aZ = numbers[0]
                inputs['axis'][-1,2] = aZ
        
        # boundary Fourier modes
        match_m = re.search('m:[\ \t]*'+number_format,command,re.IGNORECASE)
        match_n = re.search('n:[\ \t]*'+number_format,command,re.IGNORECASE)
        if match_m and match_n:
            numbers_m = [int(x) for x in re.findall(number_format,match_m.group(0))]
            numbers_n = [int(x) for x in re.findall(number_format,match_n.group(0))]
            m = numbers_m[0]
            n = numbers_n[0]
            if np.sum(np.absolute(inputs['bdry'])) != 0:
                inputs['bdry'] = np.pad(inputs['bdry'],((0,1),(0,0)),mode='constant')
            inputs['bdry'][-1,0] = m
            inputs['bdry'][-1,1] = n
            match_bdry = re.search('bR[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
            if match_bdry:
                numbers = [float(x) for x in re.findall(number_format,match_bdry.group(0))]
                bR = numbers[0]
                inputs['bdry'][-1,2] = bR
            match_bdry = re.search('bZ[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
            if match_bdry:
                numbers = [float(x) for x in re.findall(number_format,match_bdry.group(0))]
                bZ = numbers[0]
                inputs['bdry'][-1,3] = bZ
    
    inputs['presfun_params'] = pres_scale*inputs['presfun_params']
    if inputs['Mnodes'] == 0:
        inputs['Mnodes'] = inputs['Mpol']
    if inputs['Nnodes'] == 0:
        inputs['Nnodes'] = inputs['Ntor']
    
    return inputs

def output_to_file(fname,x,zern_idx,lambda_idx,NFP,Psi_total,presfun_params,iotafun_params,bdry):
    """Prints the equilibrium solution to a text file
    
    Args:
        fname (string): filename of output file
    
    Returns:
        None, creates a text file of the output
    """
    
    cR,cZ,cL = unpack_x(x,len(zern_idx))
    
    # open file
    file = open(fname,'w+')
    file.seek(0)
    
    # scaling factors
    file.write('NFP = {:3d}\n'.format(NFP))
    file.write('Psi = {:16.8E}\n'.format(Psi_total))
    

    # boundary paramters
    nbdry = len(bdry)
    file.write('Nbdry = {:3d}'.format(nbdry))
    file.write('\n')
    for k in range(nbdry):
        file.write('m: {:3d} n: {:3d} bR = {:16.8E} bZ = {:16.8E}\n'.format(int(bdry[k,0]),int(bdry[k,1]),bdry[k,2],bdry[k,3]))
    
    # profile coefficients
    nprof = max(presfun_params.size,iotafun_params.size)
    file.write('Nprof = {:3d}'.format(nprof))
    file.write('\n')
    for k in range(nprof):
        if k >= presfun_params.size:
            file.write('l: {:3d} cP = {:16.8E} cI = {:16.8E}\n'.format(k,0,iotafun_params[k]))
        elif k >= iotafun_params.size:
            file.write('l: {:3d} cP = {:16.8E} cI = {:16.8E}\n'.format(k,presfun_params[k],0))
        else:
            file.write('l: {:3d} cP = {:16.8E} cI = {:16.8E}\n'.format(k,presfun_params[k],iotafun_params[k]))
    
    # R & Z Fourier-Zernike coefficients
    nRZ = len(zern_idx)
    file.write('NRZ = {:5d}'.format(nRZ))
    file.write('\n')
    for k, lmn in enumerate(zern_idx):
        file.write('l: {:3d} m: {:3d} n: {:3d} cR = {:16.8E} cZ = {:16.8E}\n'.format(lmn[0],lmn[1],lmn[2],cR[k],cZ[k]))
    
    # lambda Fourier coefficients
    nL = len(lambda_idx)
    file.write('NL = {:5d}'.format(nL))
    file.write('\n')
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
    equil['Psi_total'] = float(lines[1].strip('\n').split()[-1])
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