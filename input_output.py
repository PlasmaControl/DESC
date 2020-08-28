import re
import warnings
import numpy as np
from datetime import datetime
from netCDF4 import Dataset
from backend import unpack_x

# TODO: fix the boundary mode conversion
# VMEC uses cos(mt-np) basis but DESC uses sin(mt)*sin(np)+cos(mt)*cos(np)
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
        'NFP' : 1,
        'Psi_total' : 1.0,
        'Mpol' : np.array([0]),
        'Ntor' : np.array([0]),
        'Mnodes' : np.array([0]),
        'Nnodes' : np.array([0]),
        'bdry_ratio' : np.array([1.0]),
        'pres_ratio' : np.array([1.0]),
        'errr_ratio' : np.array([1.0]),
        'errr_mode' : 'force',
        'bdry_mode' : 'spectral',
        'node_mode' : 'cheb1',
        'presfun_params' : np.array([0.0]),
        'iotafun_params' : np.array([0.0]),
        'axis' : np.array([[0,0.0,0.0]]),
        'bdry' : np.array([[0,0,0.0,0.0]])
    }
    
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
        match = re.search('NFP[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(number_format,match.group(0))]
            inputs['NFP'] = numbers[0]
        match = re.search('Psi_total[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            inputs['Psi_total'] = numbers[0]
        match = re.search('pres_scale[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            pres_scale = numbers[0]
        match = re.search('Mpol[\ \t]*=([\ \t]*'+number_format+')*',command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(number_format,match.group(0))]
            inputs['Mpol'] = np.array(numbers)
        match = re.search('Ntor[\ \t]*=([\ \t]*'+number_format+')*',command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(number_format,match.group(0))]
            inputs['Ntor'] = np.array(numbers)
        match = re.search('Mnodes[\ \t]*=([\ \t]*'+number_format+')*',command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(number_format,match.group(0))]
            inputs['Mnodes'] = np.array(numbers)
        match = re.search('Nnodes[\ \t]*=([\ \t]*'+number_format+')*',command,re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(number_format,match.group(0))]
            inputs['Nnodes'] = np.array(numbers)
        match = re.search('bdry_ratio[\ \t]*=([\ \t]*'+number_format+')*',command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            inputs['bdry_ratio'] = np.array(numbers)
        match = re.search('pres_ratio[\ \t]*=([\ \t]*'+number_format+')*',command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            inputs['pres_ratio'] = np.array(numbers)
        match = re.search('errr_ratio[\ \t]*=([\ \t]*'+number_format+')*',command,re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(number_format,match.group(0))]
            inputs['errr_ratio'] = np.array(numbers)
        match = re.search('errr_mode[\ \t]*=[\ \t]*\w*',command,re.IGNORECASE)
        if match:
            words = match.group(0).split()
            inputs['errr_mode'] = words[-1]
        match = re.search('bdry_mode[\ \t]*=[\ \t]*\w*',command,re.IGNORECASE)
        if match:
            words = match.group(0).split()
            inputs['bdry_mode'] = words[-1]
        match = re.search('node_mode[\ \t]*=[\ \t]*\w*',command,re.IGNORECASE)
        if match:
            words = match.group(0).split()
            inputs['node_mode'] = words[-1]
        
        # profile coefficients
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
        match_m = re.search('m:[\ \t]*'+number_format,command,re.IGNORECASE)
        match_n = re.search('n:[\ \t]*'+number_format,command,re.IGNORECASE)
        if match_n and not match_m:
            numbers = [int(x) for x in re.findall(number_format,match_n.group(0))]
            n = numbers[0]
            if np.sum(np.absolute(inputs['axis'][-1,:])) != 0:
                    inputs['axis'] = np.pad(inputs['axis'],((0,1),(0,0)),mode='constant')
            match_axis = re.search('aR[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
            if match_axis:
                numbers = [float(x) for x in re.findall(number_format,match_axis.group(0))]
                aR = numbers[0]
                inputs['axis'][-1,0] = n
                inputs['axis'][-1,1] = aR
            match_axis = re.search('aZ[\ \t]*=[\ \t]*'+number_format,command,re.IGNORECASE)
            if match_axis:
                numbers = [float(x) for x in re.findall(number_format,match_axis.group(0))]
                aZ = numbers[0]
                inputs['axis'][-1,0] = n
                inputs['axis'][-1,2] = aZ
        
        # boundary Fourier modes
        match_m = re.search('m:[\ \t]*'+number_format,command,re.IGNORECASE)
        match_n = re.search('n:[\ \t]*'+number_format,command,re.IGNORECASE)
        if match_m and match_n:
            numbers_m = [int(x) for x in re.findall(number_format,match_m.group(0))]
            numbers_n = [int(x) for x in re.findall(number_format,match_n.group(0))]
            m = numbers_m[0]
            n = numbers_n[0]
            if np.sum(np.absolute(inputs['bdry'][-1,:])) != 0:
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
    if np.sum(inputs['Mnodes']) == 0:
        inputs['Mnodes'] = inputs['Mpol']
    if np.sum(inputs['Nnodes']) == 0:
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
    M = np.amax(lambda_idx[:,0])
    N = np.amax(lambda_idx[:,1])
    
    # open file
    file = open(fname,'w+')
    file.seek(0)
    
    # grid parameters
    file.write('! grid parameters\n')
    file.write('M\t= {:3d}\n'.format(M))
    file.write('N\t= {:3d}\n'.format(N))
    file.write('NFP\t= {:3d}\n'.format(NFP))
    file.write('Psi\t= {:16.8E}\n'.format(Psi_total))
    
    # boundary shape
    file.write('\n')
    file.write('! boundary shape\n')
    for k in range(np.shape(bdry)[0]):
        file.write('\t\tm: {:3d}\tn: {:3d}\tbR = {:16.8E}\tbZ = {:16.8E}\n'.format(int(bdry[k,0]),int(bdry[k,1]),bdry[k,2],bdry[k,3]))
    
    # profiles
    file.write('\n')
    file.write('! profiles\n')
    for k in range(max(presfun_params.size,iotafun_params.size)):
        if k >= presfun_params.size:
            file.write('l: {:3d}\t\t\t\t\tcP = {:16.8E}\tcI = {:16.8E}\n'.format(k,0,iotafun_params[k]))
        elif k >= iotafun_params.size:
            file.write('l: {:3d}\t\t\t\t\tcP = {:16.8E}\tcI = {:16.8E}\n'.format(k,presfun_params[k],0))
        else:
            file.write('l: {:3d}\t\t\t\t\tcP = {:16.8E}\tcI = {:16.8E}\n'.format(k,presfun_params[k],iotafun_params[k]))
    
    # flux surfaces
    file.write('\n')
    file.write('! flux surfaces\n')
    for k, lmn in enumerate(zern_idx):
        file.write('l: {:3d}\tm: {:3d}\tn: {:3d}\tcR = {:16.8E}\tcZ = {:16.8E}\n'.format(lmn[0],lmn[1],lmn[2],cR[k],cZ[k]))
    
    # lambda
    file.write('\n')
    file.write('! lambda\n')
    for k, mn in enumerate(lambda_idx):
        file.write('\t\tm: {:3d}\tn: {:3d}\tcL = {:16.8E}\n'.format(mn[0],mn[1],cL[k]))
    
    # close file
    file.truncate()
    file.close()
    
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
    