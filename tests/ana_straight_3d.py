from scipy.constants import mu_0
from dataclasses import dataclass
from .ana_util import modes_gen
from .ana_base import ana_model
import numpy as np
# R not included in the ana_models, only passed on to the eqs
# only analytical model, not MHD equilibrium
@dataclass
class model_mirror1(ana_model):
    """axis-sym mirror with periodic zeta.(FourierZernike Basis)
        For definition, refer to _get_modes
        R is only passed as an argument to R_lmn, not included in the analytical formula.
    """
    Psi: float
    a0: float
    a1: float
    b1: float
    R: float

    def _get_modes(self):
        return {
            "R_lmn":{ (0,0,0):self.R, (1,1,0):self.a0, (3,1,0):self.a1, (3,1,1):self.b1 },
            "Z_lmn":{ (1,-1,0):-self.a0, (3,-1,0):-self.a1, (3,-1,1):-self.b1 },
        }

    @property
    def modes(self):
        return self._modes
    
    def B_vec_ana_cal(self, rtz):
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        zeta = rtz[:, 2]
        a0 = self.a0
        a1 = self.a1
        b1 = self.b1
        psi0 = self.Psi / np.pi / 2
        B0 = (-2*b1*psi0*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta))/(a0**2 + \
            4*a0*a1*(-1 + 3*rho**2) + a1**2*(4 - 24*rho**2 + 27*rho**4) + \
            2*b1*(a0*(-2 + 6*rho**2) + a1*(4 - 24*rho**2 + \
            27*rho**4))*np.cos(zeta) + b1**2*(4 - 24*rho**2 + \
            27*rho**4)*np.cos(zeta)**2)
        B1 = (2*psi0)/(a0**2 + 4*a0*a1*(-1 + 3*rho**2) + a1**2*(4 - 24*rho**2 + \
            27*rho**4) + 2*b1*(a0*(-2 + 6*rho**2) + a1*(4 - 24*rho**2 + \
            27*rho**4))*np.cos(zeta) + b1**2*(4 - 24*rho**2 + \
            27*rho**4)*np.cos(zeta)**2)
        B2 = (2*b1*psi0*rho*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta))/(a0**2 + \
            4*a0*a1*(-1 + 3*rho**2) + a1**2*(4 - 24*rho**2 + 27*rho**4) + \
            2*b1*(a0*(-2 + 6*rho**2) + a1*(4 - 24*rho**2 + \
            27*rho**4))*np.cos(zeta) + b1**2*(4 - 24*rho**2 + \
            27*rho**4)*np.cos(zeta)**2)
        return np.stack((B0,B1,B2), axis=1)
    
    def j_vec_ana_cal(self, rtz):
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        zeta = rtz[:, 2]
        a0 = self.a0
        a1 = self.a1
        b1 = self.b1
        psi0 = self.Psi / np.pi / 2
        mu0 = mu_0
        j0 = (psi0*rho*(-96*a0*a1 + 192*a1**2 + 96*b1**2 + 48*a0**2*b1**2 - \
            192*a0*a1*b1**2 + 192*a1**2*b1**2 + 48*b1**4 - 432*a1**2*rho**2 - \
            216*b1**2*rho**2 - 240*a0**2*b1**2*rho**2 + 1440*a0*a1*b1**2*rho**2 - \
            1920*a1**2*b1**2*rho**2 - 480*b1**4*rho**2 + 252*a0**2*b1**2*rho**4 - \
            3600*a0*a1*b1**2*rho**4 + 7488*a1**2*b1**2*rho**4 + 1872*b1**4*rho**4 \
            + 2808*a0*a1*b1**2*rho**6 - 12096*a1**2*b1**2*rho**6 - \
            3024*b1**4*rho**6 + 6804*a1**2*b1**2*rho**8 + 1701*b1**4*rho**8 + \
            b1*(4*a0**3*(-2 + 3*rho**2) + 12*a0**2*a1*(4 - 20*rho**2 + 21*rho**4) \
            + a1*(96*(4 - 9*rho**2) + 4*a1**2*(4 - 24*rho**2 + 27*rho**4)**2 + \
            3*b1**2*(2 - 3*rho**2)**2*(20 - 148*rho**2 + 333*rho**4)) + 3*a0*(-32 \
            + 4*a1**2*(-8 + 68*rho**2 - 174*rho**4 + 135*rho**6) + b1**2*(-40 + \
            308*rho**2 - 774*rho**4 + 603*rho**6)))*np.cos(zeta) + \
            2*b1**2*(b1**2*(4 - 24*rho**2 + 27*rho**4)**2 + 12*(4 + (-9 + 8*a0*a1 \
            - 16*a1**2)*rho**2 + 12*a1*(-2*a0 + 7*a1)*rho**4 + 18*(a0 - \
            8*a1)*a1*rho**6 + 81*a1**2*rho**8))*np.cos(2*zeta) + \
            24*a0*b1**3*np.cos(3*zeta) - 48*a1*b1**3*np.cos(3*zeta) - \
            108*a0*b1**3*rho**2*np.cos(3*zeta) + \
            192*a1*b1**3*rho**2*np.cos(3*zeta) + \
            234*a0*b1**3*rho**4*np.cos(3*zeta) - \
            360*a1*b1**3*rho**4*np.cos(3*zeta) - \
            189*a0*b1**3*rho**6*np.cos(3*zeta) + \
            432*a1*b1**3*rho**6*np.cos(3*zeta) - \
            243*a1*b1**3*rho**8*np.cos(3*zeta) - 16*b1**4*np.cos(4*zeta) + \
            96*b1**4*rho**2*np.cos(4*zeta) - 288*b1**4*rho**4*np.cos(4*zeta) + \
            432*b1**4*rho**6*np.cos(4*zeta) - \
            243*b1**4*rho**8*np.cos(4*zeta))*np.sin(theta))/(2.*mu0*(a0 + a1*(-2 \
            + 3*rho**2) + b1*(-2 + 3*rho**2)*np.cos(zeta))**2*(a0 + a1*(-2 + \
            9*rho**2) + b1*(-2 + 9*rho**2)*np.cos(zeta))**3)
        j1 = 0 * rho
        j2 = (psi0*rho*np.cos(theta)*(-96*a0*a1 + 192*a1**2 + 96*b1**2 + \
            48*a0**2*b1**2 - 192*a0*a1*b1**2 + 192*a1**2*b1**2 + 48*b1**4 - \
            432*a1**2*rho**2 - 216*b1**2*rho**2 - 240*a0**2*b1**2*rho**2 + \
            1440*a0*a1*b1**2*rho**2 - 1920*a1**2*b1**2*rho**2 - 480*b1**4*rho**2 \
            + 252*a0**2*b1**2*rho**4 - 3600*a0*a1*b1**2*rho**4 + \
            7488*a1**2*b1**2*rho**4 + 1872*b1**4*rho**4 + 2808*a0*a1*b1**2*rho**6 \
            - 12096*a1**2*b1**2*rho**6 - 3024*b1**4*rho**6 + \
            6804*a1**2*b1**2*rho**8 + 1701*b1**4*rho**8 + b1*(4*a0**3*(-2 + \
            3*rho**2) + 12*a0**2*a1*(4 - 20*rho**2 + 21*rho**4) + a1*(96*(4 - \
            9*rho**2) + 4*a1**2*(4 - 24*rho**2 + 27*rho**4)**2 + 3*b1**2*(2 - \
            3*rho**2)**2*(20 - 148*rho**2 + 333*rho**4)) + 3*a0*(-32 + \
            4*a1**2*(-8 + 68*rho**2 - 174*rho**4 + 135*rho**6) + b1**2*(-40 + \
            308*rho**2 - 774*rho**4 + 603*rho**6)))*np.cos(zeta) + \
            2*b1**2*(b1**2*(4 - 24*rho**2 + 27*rho**4)**2 + 12*(4 + (-9 + 8*a0*a1 \
            - 16*a1**2)*rho**2 + 12*a1*(-2*a0 + 7*a1)*rho**4 + 18*(a0 - \
            8*a1)*a1*rho**6 + 81*a1**2*rho**8))*np.cos(2*zeta) + \
            24*a0*b1**3*np.cos(3*zeta) - 48*a1*b1**3*np.cos(3*zeta) - \
            108*a0*b1**3*rho**2*np.cos(3*zeta) + \
            192*a1*b1**3*rho**2*np.cos(3*zeta) + \
            234*a0*b1**3*rho**4*np.cos(3*zeta) - \
            360*a1*b1**3*rho**4*np.cos(3*zeta) - \
            189*a0*b1**3*rho**6*np.cos(3*zeta) + \
            432*a1*b1**3*rho**6*np.cos(3*zeta) - \
            243*a1*b1**3*rho**8*np.cos(3*zeta) - 16*b1**4*np.cos(4*zeta) + \
            96*b1**4*rho**2*np.cos(4*zeta) - 288*b1**4*rho**4*np.cos(4*zeta) + \
            432*b1**4*rho**6*np.cos(4*zeta) - \
            243*b1**4*rho**8*np.cos(4*zeta)))/(2.*mu0*(a0 + a1*(-2 + 3*rho**2) + \
            b1*(-2 + 3*rho**2)*np.cos(zeta))**2*(a0 + a1*(-2 + 9*rho**2) + b1*(-2 \
            + 9*rho**2)*np.cos(zeta))**3)
        return np.stack((j0,j1,j2), axis=1)
    
    def gradp_vec_ana_cal(self,rtz):
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        zeta = rtz[:, 2]
        a0 = self.a0
        a1 = self.a1
        b1 = self.b1
        psi0 = self.Psi / np.pi / 2
        mu0 = mu_0
        f0 = -((psi0**2*rho*np.cos(theta)*(-96*a0*a1 + 192*a1**2 + 96*b1**2 + \
            48*a0**2*b1**2 - 192*a0*a1*b1**2 + 192*a1**2*b1**2 + 48*b1**4 - \
            432*a1**2*rho**2 - 216*b1**2*rho**2 - 240*a0**2*b1**2*rho**2 + \
            1440*a0*a1*b1**2*rho**2 - 1920*a1**2*b1**2*rho**2 - 480*b1**4*rho**2 \
            + 252*a0**2*b1**2*rho**4 - 3600*a0*a1*b1**2*rho**4 + \
            7488*a1**2*b1**2*rho**4 + 1872*b1**4*rho**4 + 2808*a0*a1*b1**2*rho**6 \
            - 12096*a1**2*b1**2*rho**6 - 3024*b1**4*rho**6 + \
            6804*a1**2*b1**2*rho**8 + 1701*b1**4*rho**8 + b1*(4*a0**3*(-2 + \
            3*rho**2) + 12*a0**2*a1*(4 - 20*rho**2 + 21*rho**4) + a1*(96*(4 - \
            9*rho**2) + 4*a1**2*(4 - 24*rho**2 + 27*rho**4)**2 + 3*b1**2*(2 - \
            3*rho**2)**2*(20 - 148*rho**2 + 333*rho**4)) + 3*a0*(-32 + \
            4*a1**2*(-8 + 68*rho**2 - 174*rho**4 + 135*rho**6) + b1**2*(-40 + \
            308*rho**2 - 774*rho**4 + 603*rho**6)))*np.cos(zeta) + \
            2*b1**2*(b1**2*(4 - 24*rho**2 + 27*rho**4)**2 + 12*(4 + (-9 + 8*a0*a1 \
            - 16*a1**2)*rho**2 + 12*a1*(-2*a0 + 7*a1)*rho**4 + 18*(a0 - \
            8*a1)*a1*rho**6 + 81*a1**2*rho**8))*np.cos(2*zeta) + \
            24*a0*b1**3*np.cos(3*zeta) - 48*a1*b1**3*np.cos(3*zeta) - \
            108*a0*b1**3*rho**2*np.cos(3*zeta) + \
            192*a1*b1**3*rho**2*np.cos(3*zeta) + \
            234*a0*b1**3*rho**4*np.cos(3*zeta) - \
            360*a1*b1**3*rho**4*np.cos(3*zeta) - \
            189*a0*b1**3*rho**6*np.cos(3*zeta) + \
            432*a1*b1**3*rho**6*np.cos(3*zeta) - \
            243*a1*b1**3*rho**8*np.cos(3*zeta) - 16*b1**4*np.cos(4*zeta) + \
            96*b1**4*rho**2*np.cos(4*zeta) - 288*b1**4*rho**4*np.cos(4*zeta) + \
            432*b1**4*rho**6*np.cos(4*zeta) - \
            243*b1**4*rho**8*np.cos(4*zeta)))/(mu0*(a0 + a1*(-2 + 3*rho**2) + \
            b1*(-2 + 3*rho**2)*np.cos(zeta))**3*(a0 + a1*(-2 + 9*rho**2) + b1*(-2 \
            + 9*rho**2)*np.cos(zeta))**4))
        f1 = -((b1*psi0**2*rho**2*(-2 + 3*rho**2)*(-96*a0*a1 + 192*a1**2 + \
            96*b1**2 + 48*a0**2*b1**2 - 192*a0*a1*b1**2 + 192*a1**2*b1**2 + \
            48*b1**4 - 432*a1**2*rho**2 - 216*b1**2*rho**2 - \
            240*a0**2*b1**2*rho**2 + 1440*a0*a1*b1**2*rho**2 - \
            1920*a1**2*b1**2*rho**2 - 480*b1**4*rho**2 + 252*a0**2*b1**2*rho**4 - \
            3600*a0*a1*b1**2*rho**4 + 7488*a1**2*b1**2*rho**4 + 1872*b1**4*rho**4 \
            + 2808*a0*a1*b1**2*rho**6 - 12096*a1**2*b1**2*rho**6 - \
            3024*b1**4*rho**6 + 6804*a1**2*b1**2*rho**8 + 1701*b1**4*rho**8 + \
            b1*(4*a0**3*(-2 + 3*rho**2) + 12*a0**2*a1*(4 - 20*rho**2 + 21*rho**4) \
            + a1*(96*(4 - 9*rho**2) + 4*a1**2*(4 - 24*rho**2 + 27*rho**4)**2 + \
            3*b1**2*(2 - 3*rho**2)**2*(20 - 148*rho**2 + 333*rho**4)) + 3*a0*(-32 \
            + 4*a1**2*(-8 + 68*rho**2 - 174*rho**4 + 135*rho**6) + b1**2*(-40 + \
            308*rho**2 - 774*rho**4 + 603*rho**6)))*np.cos(zeta) + \
            2*b1**2*(b1**2*(4 - 24*rho**2 + 27*rho**4)**2 + 12*(4 + (-9 + 8*a0*a1 \
            - 16*a1**2)*rho**2 + 12*a1*(-2*a0 + 7*a1)*rho**4 + 18*(a0 - \
            8*a1)*a1*rho**6 + 81*a1**2*rho**8))*np.cos(2*zeta) + \
            24*a0*b1**3*np.cos(3*zeta) - 48*a1*b1**3*np.cos(3*zeta) - \
            108*a0*b1**3*rho**2*np.cos(3*zeta) + \
            192*a1*b1**3*rho**2*np.cos(3*zeta) + \
            234*a0*b1**3*rho**4*np.cos(3*zeta) - \
            360*a1*b1**3*rho**4*np.cos(3*zeta) - \
            189*a0*b1**3*rho**6*np.cos(3*zeta) + \
            432*a1*b1**3*rho**6*np.cos(3*zeta) - \
            243*a1*b1**3*rho**8*np.cos(3*zeta) - 16*b1**4*np.cos(4*zeta) + \
            96*b1**4*rho**2*np.cos(4*zeta) - 288*b1**4*rho**4*np.cos(4*zeta) + \
            432*b1**4*rho**6*np.cos(4*zeta) - \
            243*b1**4*rho**8*np.cos(4*zeta))*np.sin(zeta))/(mu0*(a0 + a1*(-2 + \
            3*rho**2) + b1*(-2 + 3*rho**2)*np.cos(zeta))**3*(a0 + a1*(-2 + \
            9*rho**2) + b1*(-2 + 9*rho**2)*np.cos(zeta))**4))
        f2 = (psi0**2*rho*(-96*a0*a1 + 192*a1**2 + 96*b1**2 + 48*a0**2*b1**2 - \
            192*a0*a1*b1**2 + 192*a1**2*b1**2 + 48*b1**4 - 432*a1**2*rho**2 - \
            216*b1**2*rho**2 - 240*a0**2*b1**2*rho**2 + 1440*a0*a1*b1**2*rho**2 - \
            1920*a1**2*b1**2*rho**2 - 480*b1**4*rho**2 + 252*a0**2*b1**2*rho**4 - \
            3600*a0*a1*b1**2*rho**4 + 7488*a1**2*b1**2*rho**4 + 1872*b1**4*rho**4 \
            + 2808*a0*a1*b1**2*rho**6 - 12096*a1**2*b1**2*rho**6 - \
            3024*b1**4*rho**6 + 6804*a1**2*b1**2*rho**8 + 1701*b1**4*rho**8 + \
            b1*(4*a0**3*(-2 + 3*rho**2) + 12*a0**2*a1*(4 - 20*rho**2 + 21*rho**4) \
            + a1*(96*(4 - 9*rho**2) + 4*a1**2*(4 - 24*rho**2 + 27*rho**4)**2 + \
            3*b1**2*(2 - 3*rho**2)**2*(20 - 148*rho**2 + 333*rho**4)) + 3*a0*(-32 \
            + 4*a1**2*(-8 + 68*rho**2 - 174*rho**4 + 135*rho**6) + b1**2*(-40 + \
            308*rho**2 - 774*rho**4 + 603*rho**6)))*np.cos(zeta) + \
            2*b1**2*(b1**2*(4 - 24*rho**2 + 27*rho**4)**2 + 12*(4 + (-9 + 8*a0*a1 \
            - 16*a1**2)*rho**2 + 12*a1*(-2*a0 + 7*a1)*rho**4 + 18*(a0 - \
            8*a1)*a1*rho**6 + 81*a1**2*rho**8))*np.cos(2*zeta) + \
            24*a0*b1**3*np.cos(3*zeta) - 48*a1*b1**3*np.cos(3*zeta) - \
            108*a0*b1**3*rho**2*np.cos(3*zeta) + \
            192*a1*b1**3*rho**2*np.cos(3*zeta) + \
            234*a0*b1**3*rho**4*np.cos(3*zeta) - \
            360*a1*b1**3*rho**4*np.cos(3*zeta) - \
            189*a0*b1**3*rho**6*np.cos(3*zeta) + \
            432*a1*b1**3*rho**6*np.cos(3*zeta) - \
            243*a1*b1**3*rho**8*np.cos(3*zeta) - 16*b1**4*np.cos(4*zeta) + \
            96*b1**4*rho**2*np.cos(4*zeta) - 288*b1**4*rho**4*np.cos(4*zeta) + \
            432*b1**4*rho**6*np.cos(4*zeta) - \
            243*b1**4*rho**8*np.cos(4*zeta))*np.sin(theta))/(mu0*(a0 + a1*(-2 + \
            3*rho**2) + b1*(-2 + 3*rho**2)*np.cos(zeta))**3*(a0 + a1*(-2 + \
            9*rho**2) + b1*(-2 + 9*rho**2)*np.cos(zeta))**4)
        return np.stack((f0,f1,f2),axis=1)
        

@dataclass
class model_mirror_iota1(ana_model):
    """axis-sym mirror with periodic zeta.(FourierZernike Basis)
        With twisted field lines
        For definition, refer to _get_modes
        R is only passed as an argument to R_lmn, not included in the analytical formula.
    """
    Psi: float
    a0: float
    a1: float
    b1: float
    R: float
    i0: float
    i2: float
    
    def _get_modes(self):
        return {
            "R_lmn":{ (0,0,0):self.R, (1,1,0):self.a0, (3,1,0):self.a1, (3,1,1):self.b1 },
            "Z_lmn":{ (1,-1,0):-self.a0, (3,-1,0):-self.a1, (3,-1,1):-self.b1 },
            "iota": {"modes": (0,2), "params": (self.i0, self.i2)},
        }
    
    def B_vec_ana_cal(self, rtz):
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        zeta = rtz[:, 2]
        a0 = self.a0
        a1 = self.a1
        b1 = self.b1
        psi0 = self.Psi / np.pi / 2
        i0 = self.i0
        i2 = self.i2

        B0 = (-2*psi0*rho*((i0 + i2*rho**2)*(a0 + a1*(-2 + 3*rho**2) + b1*(-2 + \
            3*rho**2)*np.cos(zeta))*np.sin(theta) + b1*(-2 + \
            3*rho**2)*np.cos(theta)*np.sin(zeta)))/(a0**2 + 4*a0*a1*(-1 + \
            3*rho**2) + a1**2*(4 - 24*rho**2 + 27*rho**4) + 2*b1*(a0*(-2 + \
            6*rho**2) + a1*(4 - 24*rho**2 + 27*rho**4))*np.cos(zeta) + b1**2*(4 - \
            24*rho**2 + 27*rho**4)*np.cos(zeta)**2)
        B1 = (2*psi0)/(a0**2 + 4*a0*a1*(-1 + 3*rho**2) + a1**2*(4 - 24*rho**2 + \
            27*rho**4) + 2*b1*(a0*(-2 + 6*rho**2) + a1*(4 - 24*rho**2 + \
            27*rho**4))*np.cos(zeta) + b1**2*(4 - 24*rho**2 + \
            27*rho**4)*np.cos(zeta)**2)
        B2 = (-2*psi0*rho*((i0 + i2*rho**2)*np.cos(theta)*(a0 + a1*(-2 + 3*rho**2) \
            + b1*(-2 + 3*rho**2)*np.cos(zeta)) + b1*(2 - \
            3*rho**2)*np.sin(theta)*np.sin(zeta)))/(a0**2 + 4*a0*a1*(-1 + \
            3*rho**2) + a1**2*(4 - 24*rho**2 + 27*rho**4) + 2*b1*(a0*(-2 + \
            6*rho**2) + a1*(4 - 24*rho**2 + 27*rho**4))*np.cos(zeta) + b1**2*(4 - \
            24*rho**2 + 27*rho**4)*np.cos(zeta)**2)
        
        return np.stack( (B0,B1,B2), axis=1)
    
    def j_vec_ana_cal(self, rtz):
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        zeta = rtz[:, 2]
        a0 = self.a0
        a1 = self.a1
        b1 = self.b1
        psi0 = self.Psi / np.pi / 2
        i0 = self.i0
        i2 = self.i2
        mu0 = mu_0
        
        j0 = (psi0*rho*((-96*a0*a1 + 192*a1**2 + 96*b1**2 + 48*a0**2*b1**2 - \
            192*a0*a1*b1**2 + 192*a1**2*b1**2 + 48*b1**4 - 432*a1**2*rho**2 - \
            216*b1**2*rho**2 - 240*a0**2*b1**2*rho**2 + 1440*a0*a1*b1**2*rho**2 - \
            1920*a1**2*b1**2*rho**2 - 480*b1**4*rho**2 + 252*a0**2*b1**2*rho**4 - \
            3600*a0*a1*b1**2*rho**4 + 7488*a1**2*b1**2*rho**4 + 1872*b1**4*rho**4 \
            + 2808*a0*a1*b1**2*rho**6 - 12096*a1**2*b1**2*rho**6 - \
            3024*b1**4*rho**6 + 6804*a1**2*b1**2*rho**8 + 1701*b1**4*rho**8 + \
            b1*(4*a0**3*(-2 + 3*rho**2) + 12*a0**2*a1*(4 - 20*rho**2 + 21*rho**4) \
            + a1*(96*(4 - 9*rho**2) + 4*a1**2*(4 - 24*rho**2 + 27*rho**4)**2 + \
            3*b1**2*(2 - 3*rho**2)**2*(20 - 148*rho**2 + 333*rho**4)) + 3*a0*(-32 \
            + 4*a1**2*(-8 + 68*rho**2 - 174*rho**4 + 135*rho**6) + b1**2*(-40 + \
            308*rho**2 - 774*rho**4 + 603*rho**6)))*np.cos(zeta) + \
            2*b1**2*(b1**2*(4 - 24*rho**2 + 27*rho**4)**2 + 12*(4 + (-9 + 8*a0*a1 \
            - 16*a1**2)*rho**2 + 12*a1*(-2*a0 + 7*a1)*rho**4 + 18*(a0 - \
            8*a1)*a1*rho**6 + 81*a1**2*rho**8))*np.cos(2*zeta) + \
            24*a0*b1**3*np.cos(3*zeta) - 48*a1*b1**3*np.cos(3*zeta) - \
            108*a0*b1**3*rho**2*np.cos(3*zeta) + \
            192*a1*b1**3*rho**2*np.cos(3*zeta) + \
            234*a0*b1**3*rho**4*np.cos(3*zeta) - \
            360*a1*b1**3*rho**4*np.cos(3*zeta) - \
            189*a0*b1**3*rho**6*np.cos(3*zeta) + \
            432*a1*b1**3*rho**6*np.cos(3*zeta) - \
            243*a1*b1**3*rho**8*np.cos(3*zeta) - 16*b1**4*np.cos(4*zeta) + \
            96*b1**4*rho**2*np.cos(4*zeta) - 288*b1**4*rho**4*np.cos(4*zeta) + \
            432*b1**4*rho**6*np.cos(4*zeta) - \
            243*b1**4*rho**8*np.cos(4*zeta))*np.sin(theta) - \
            8*b1*np.cos(theta)*(a0 + a1*(-2 + 3*rho**2) + b1*(-2 + \
            3*rho**2)*np.cos(zeta))**2*(a0*i0*(-2 + 6*rho**2) + a0*i2*rho**2*(-4 \
            + 9*rho**2) + 2*a1*i2*rho**2*(4 - 18*rho**2 + 27*rho**4) + a1*i0*(4 - \
            12*rho**2 + 27*rho**4) + b1*(2*i2*rho**2*(4 - 18*rho**2 + 27*rho**4) \
            + i0*(4 - 12*rho**2 + \
            27*rho**4))*np.cos(zeta))*np.sin(zeta)))/(2.*mu0*(a0 + a1*(-2 + \
            3*rho**2) + b1*(-2 + 3*rho**2)*np.cos(zeta))**2*(a0 + a1*(-2 + \
            9*rho**2) + b1*(-2 + 9*rho**2)*np.cos(zeta))**3)
        j1 = (4*psi0*(a0**2*(i0 + 2*i2*rho**2) + 2*a0*a1*(i0*(-2 + 3*rho**2) + \
            i2*rho**2*(-4 + 9*rho**2)) + a1**2*(2*i2*rho**2*(4 - 18*rho**2 + \
            27*rho**4) + i0*(4 - 12*rho**2 + 27*rho**4)) + 2*b1*(a0*i0*(-2 + \
            3*rho**2) + a0*i2*rho**2*(-4 + 9*rho**2) + 2*a1*i2*rho**2*(4 - \
            18*rho**2 + 27*rho**4) + a1*i0*(4 - 12*rho**2 + \
            27*rho**4))*np.cos(zeta) + b1**2*(2*i2*rho**2*(4 - 18*rho**2 + \
            27*rho**4) + i0*(4 - 12*rho**2 + \
            27*rho**4))*np.cos(zeta)**2))/(mu0*(a0 + a1*(-2 + 3*rho**2) + b1*(-2 \
            + 3*rho**2)*np.cos(zeta))*(a0 + a1*(-2 + 9*rho**2) + b1*(-2 + \
            9*rho**2)*np.cos(zeta))**3)
        j2 = (psi0*rho*(np.cos(theta)*(-96*a0*a1 + 192*a1**2 + 96*b1**2 + \
            48*a0**2*b1**2 - 192*a0*a1*b1**2 + 192*a1**2*b1**2 + 48*b1**4 - \
            432*a1**2*rho**2 - 216*b1**2*rho**2 - 240*a0**2*b1**2*rho**2 + \
            1440*a0*a1*b1**2*rho**2 - 1920*a1**2*b1**2*rho**2 - 480*b1**4*rho**2 \
            + 252*a0**2*b1**2*rho**4 - 3600*a0*a1*b1**2*rho**4 + \
            7488*a1**2*b1**2*rho**4 + 1872*b1**4*rho**4 + 2808*a0*a1*b1**2*rho**6 \
            - 12096*a1**2*b1**2*rho**6 - 3024*b1**4*rho**6 + \
            6804*a1**2*b1**2*rho**8 + 1701*b1**4*rho**8 + b1*(4*a0**3*(-2 + \
            3*rho**2) + 12*a0**2*a1*(4 - 20*rho**2 + 21*rho**4) + a1*(96*(4 - \
            9*rho**2) + 4*a1**2*(4 - 24*rho**2 + 27*rho**4)**2 + 3*b1**2*(2 - \
            3*rho**2)**2*(20 - 148*rho**2 + 333*rho**4)) + 3*a0*(-32 + \
            4*a1**2*(-8 + 68*rho**2 - 174*rho**4 + 135*rho**6) + b1**2*(-40 + \
            308*rho**2 - 774*rho**4 + 603*rho**6)))*np.cos(zeta) + \
            2*b1**2*(b1**2*(4 - 24*rho**2 + 27*rho**4)**2 + 12*(4 + (-9 + 8*a0*a1 \
            - 16*a1**2)*rho**2 + 12*a1*(-2*a0 + 7*a1)*rho**4 + 18*(a0 - \
            8*a1)*a1*rho**6 + 81*a1**2*rho**8))*np.cos(2*zeta) + \
            24*a0*b1**3*np.cos(3*zeta) - 48*a1*b1**3*np.cos(3*zeta) - \
            108*a0*b1**3*rho**2*np.cos(3*zeta) + \
            192*a1*b1**3*rho**2*np.cos(3*zeta) + \
            234*a0*b1**3*rho**4*np.cos(3*zeta) - \
            360*a1*b1**3*rho**4*np.cos(3*zeta) - \
            189*a0*b1**3*rho**6*np.cos(3*zeta) + \
            432*a1*b1**3*rho**6*np.cos(3*zeta) - \
            243*a1*b1**3*rho**8*np.cos(3*zeta) - 16*b1**4*np.cos(4*zeta) + \
            96*b1**4*rho**2*np.cos(4*zeta) - 288*b1**4*rho**4*np.cos(4*zeta) + \
            432*b1**4*rho**6*np.cos(4*zeta) - 243*b1**4*rho**8*np.cos(4*zeta)) + \
            8*b1*(a0 + a1*(-2 + 3*rho**2) + b1*(-2 + \
            3*rho**2)*np.cos(zeta))**2*(a0*i0*(-2 + 6*rho**2) + a0*i2*rho**2*(-4 \
            + 9*rho**2) + 2*a1*i2*rho**2*(4 - 18*rho**2 + 27*rho**4) + a1*i0*(4 - \
            12*rho**2 + 27*rho**4) + b1*(2*i2*rho**2*(4 - 18*rho**2 + 27*rho**4) \
            + i0*(4 - 12*rho**2 + \
            27*rho**4))*np.cos(zeta))*np.sin(theta)*np.sin(zeta)))/(2.*mu0*(a0 + \
            a1*(-2 + 3*rho**2) + b1*(-2 + 3*rho**2)*np.cos(zeta))**2*(a0 + a1*(-2 \
            + 9*rho**2) + b1*(-2 + 9*rho**2)*np.cos(zeta))**3)
        
        return np.stack( (j0,j1,j2), axis=1)
    
    def gradp_vec_ana_cal(self, rtz):
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        zeta = rtz[:, 2]
        a0 = self.a0
        a1 = self.a1
        b1 = self.b1
        psi0 = self.Psi / np.pi / 2
        i0 = self.i0
        i2 = self.i2
        mu0 = mu_0

        F0 = -((psi0**2*rho*(np.cos(theta)*(-96*a0*a1 + 192*a1**2 + 96*b1**2 + \
            48*a0**2*b1**2 - 192*a0*a1*b1**2 + 192*a1**2*b1**2 + 48*b1**4 + \
            8*a0**4*i0**2 - 64*a0**3*a1*i0**2 + 192*a0**2*a1**2*i0**2 - \
            256*a0*a1**3*i0**2 + 128*a1**4*i0**2 + 96*a0**2*b1**2*i0**2 - \
            384*a0*a1*b1**2*i0**2 + 384*a1**2*b1**2*i0**2 + 48*b1**4*i0**2 - \
            432*a1**2*rho**2 - 216*b1**2*rho**2 - 240*a0**2*b1**2*rho**2 + \
            1440*a0*a1*b1**2*rho**2 - 1920*a1**2*b1**2*rho**2 - 480*b1**4*rho**2 \
            + 96*a0**3*a1*i0**2*rho**2 - 576*a0**2*a1**2*i0**2*rho**2 + \
            1152*a0*a1**3*i0**2*rho**2 - 768*a1**4*i0**2*rho**2 - \
            288*a0**2*b1**2*i0**2*rho**2 + 1728*a0*a1*b1**2*i0**2*rho**2 - \
            2304*a1**2*b1**2*i0**2*rho**2 - 288*b1**4*i0**2*rho**2 + \
            24*a0**4*i0*i2*rho**2 - 192*a0**3*a1*i0*i2*rho**2 + \
            576*a0**2*a1**2*i0*i2*rho**2 - 768*a0*a1**3*i0*i2*rho**2 + \
            384*a1**4*i0*i2*rho**2 + 288*a0**2*b1**2*i0*i2*rho**2 - \
            1152*a0*a1*b1**2*i0*i2*rho**2 + 1152*a1**2*b1**2*i0*i2*rho**2 + \
            144*b1**4*i0*i2*rho**2 + 252*a0**2*b1**2*rho**4 - \
            3600*a0*a1*b1**2*rho**4 + 7488*a1**2*b1**2*rho**4 + 1872*b1**4*rho**4 \
            + 576*a0**2*a1**2*i0**2*rho**4 - 2304*a0*a1**3*i0**2*rho**4 + \
            2304*a1**4*i0**2*rho**4 + 288*a0**2*b1**2*i0**2*rho**4 - \
            3456*a0*a1*b1**2*i0**2*rho**4 + 6912*a1**2*b1**2*i0**2*rho**4 + \
            864*b1**4*i0**2*rho**4 + 336*a0**3*a1*i0*i2*rho**4 - \
            2016*a0**2*a1**2*i0*i2*rho**4 + 4032*a0*a1**3*i0*i2*rho**4 - \
            2688*a1**4*i0*i2*rho**4 - 1008*a0**2*b1**2*i0*i2*rho**4 + \
            6048*a0*a1*b1**2*i0*i2*rho**4 - 8064*a1**2*b1**2*i0*i2*rho**4 - \
            1008*b1**4*i0*i2*rho**4 + 16*a0**4*i2**2*rho**4 - \
            128*a0**3*a1*i2**2*rho**4 + 384*a0**2*a1**2*i2**2*rho**4 - \
            512*a0*a1**3*i2**2*rho**4 + 256*a1**4*i2**2*rho**4 + \
            192*a0**2*b1**2*i2**2*rho**4 - 768*a0*a1*b1**2*i2**2*rho**4 + \
            768*a1**2*b1**2*i2**2*rho**4 + 96*b1**4*i2**2*rho**4 + \
            2808*a0*a1*b1**2*rho**6 - 12096*a1**2*b1**2*rho**6 - \
            3024*b1**4*rho**6 + 1728*a0*a1**3*i0**2*rho**6 - \
            3456*a1**4*i0**2*rho**6 + 2592*a0*a1*b1**2*i0**2*rho**6 - \
            10368*a1**2*b1**2*i0**2*rho**6 - 1296*b1**4*i0**2*rho**6 + \
            2016*a0**2*a1**2*i0*i2*rho**6 - 8064*a0*a1**3*i0*i2*rho**6 + \
            8064*a1**4*i0*i2*rho**6 + 1008*a0**2*b1**2*i0*i2*rho**6 - \
            12096*a0*a1*b1**2*i0*i2*rho**6 + 24192*a1**2*b1**2*i0*i2*rho**6 + \
            3024*b1**4*i0*i2*rho**6 + 240*a0**3*a1*i2**2*rho**6 - \
            1440*a0**2*a1**2*i2**2*rho**6 + 2880*a0*a1**3*i2**2*rho**6 - \
            1920*a1**4*i2**2*rho**6 - 720*a0**2*b1**2*i2**2*rho**6 + \
            4320*a0*a1*b1**2*i2**2*rho**6 - 5760*a1**2*b1**2*i2**2*rho**6 - \
            720*b1**4*i2**2*rho**6 + 6804*a1**2*b1**2*rho**8 + 1701*b1**4*rho**8 \
            + 1944*a1**4*i0**2*rho**8 + 5832*a1**2*b1**2*i0**2*rho**8 + \
            729*b1**4*i0**2*rho**8 + 5616*a0*a1**3*i0*i2*rho**8 - \
            11232*a1**4*i0*i2*rho**8 + 8424*a0*a1*b1**2*i0*i2*rho**8 - \
            33696*a1**2*b1**2*i0*i2*rho**8 - 4212*b1**4*i0*i2*rho**8 + \
            1440*a0**2*a1**2*i2**2*rho**8 - 5760*a0*a1**3*i2**2*rho**8 + \
            5760*a1**4*i2**2*rho**8 + 720*a0**2*b1**2*i2**2*rho**8 - \
            8640*a0*a1*b1**2*i2**2*rho**8 + 17280*a1**2*b1**2*i2**2*rho**8 + \
            2160*b1**4*i2**2*rho**8 + 5832*a1**4*i0*i2*rho**10 + \
            17496*a1**2*b1**2*i0*i2*rho**10 + 2187*b1**4*i0*i2*rho**10 + \
            3888*a0*a1**3*i2**2*rho**10 - 7776*a1**4*i2**2*rho**10 + \
            5832*a0*a1*b1**2*i2**2*rho**10 - 23328*a1**2*b1**2*i2**2*rho**10 - \
            2916*b1**4*i2**2*rho**10 + 3888*a1**4*i2**2*rho**12 + \
            11664*a1**2*b1**2*i2**2*rho**12 + 1458*b1**4*i2**2*rho**12 + \
            b1*(4*a0**3*(-2 + 3*rho**2 - 32*i2**2*rho**4 + 60*i2**2*rho**6 + \
            8*i0**2*(-2 + 3*rho**2) + 12*i0*i2*rho**2*(-4 + 7*rho**2)) + \
            12*a0**2*a1*(4 - 20*rho**2 + (21 + 64*i2**2)*rho**4 - \
            240*i2**2*rho**6 + 240*i2**2*rho**8 + 32*i0**2*(1 - 3*rho**2 + \
            3*rho**4) + 48*i0*i2*rho**2*(2 - 7*rho**2 + 7*rho**4)) + a1*(96*(4 - \
            9*rho**2) + 4*a1**2*(2 - 3*rho**2)**2*(4 - 36*rho**2 + (81 + \
            64*i2**2)*rho**4 - 288*i2**2*rho**6 + 432*i2**2*rho**8 + \
            24*i0*i2*rho**2*(4 - 16*rho**2 + 27*rho**4) + 8*i0**2*(4 - 12*rho**2 \
            + 27*rho**4)) + 3*b1**2*(2 - 3*rho**2)**2*(20 - 148*rho**2 + (333 + \
            64*i2**2)*rho**4 - 288*i2**2*rho**6 + 432*i2**2*rho**8 + \
            24*i0*i2*rho**2*(4 - 16*rho**2 + 27*rho**4) + 8*i0**2*(4 - 12*rho**2 \
            + 27*rho**4))) + 3*a0*(-32 + 4*a1**2*(-2 + 3*rho**2)*(4 - 28*rho**2 + \
            (45 + 64*i2**2)*rho**4 - 264*i2**2*rho**6 + 324*i2**2*rho**8 + \
            16*i0**2*(2 - 6*rho**2 + 9*rho**4) + 12*i0*i2*rho**2*(8 - 30*rho**2 + \
            39*rho**4)) + b1**2*(-2 + 3*rho**2)*(20 - 124*rho**2 + (201 + \
            64*i2**2)*rho**4 - 264*i2**2*rho**6 + 324*i2**2*rho**8 + 16*i0**2*(2 \
            - 6*rho**2 + 9*rho**4) + 12*i0*i2*rho**2*(8 - 30*rho**2 + \
            39*rho**4))))*np.cos(zeta) + 2*b1**2*(b1**2*(2 - 3*rho**2)**2*(4 - \
            36*rho**2 + (81 + 16*i2**2)*rho**4 - 72*i2**2*rho**6 + \
            108*i2**2*rho**8 + 6*i0*i2*rho**2*(4 - 16*rho**2 + 27*rho**4) + \
            i0**2*(8 - 24*rho**2 + 54*rho**4)) + 12*(4 - 9*rho**2 + \
            2*a0**2*(i0**2*(2 - 6*rho**2 + 6*rho**4) + 3*i0*i2*rho**2*(2 - \
            7*rho**2 + 7*rho**4) + i2**2*rho**4*(4 - 15*rho**2 + 15*rho**4)) + \
            a1**2*(2 - 3*rho**2)**2*(-4*rho**2 + (9 + 8*i2**2)*rho**4 - \
            36*i2**2*rho**6 + 54*i2**2*rho**8 + 3*i0*i2*rho**2*(4 - 16*rho**2 + \
            27*rho**4) + i0**2*(4 - 12*rho**2 + 27*rho**4)) + a0*a1*(-2 + \
            3*rho**2)*(-4*rho**2 + 2*(3 + 8*i2**2)*rho**4 - 66*i2**2*rho**6 + \
            81*i2**2*rho**8 + 4*i0**2*(2 - 6*rho**2 + 9*rho**4) + \
            3*i0*i2*rho**2*(8 - 30*rho**2 + 39*rho**4))))*np.cos(2*zeta) + \
            24*a0*b1**3*np.cos(3*zeta) - 48*a1*b1**3*np.cos(3*zeta) - \
            64*a0*b1**3*i0**2*np.cos(3*zeta) + 128*a1*b1**3*i0**2*np.cos(3*zeta) \
            - 108*a0*b1**3*rho**2*np.cos(3*zeta) + \
            192*a1*b1**3*rho**2*np.cos(3*zeta) + \
            288*a0*b1**3*i0**2*rho**2*np.cos(3*zeta) - \
            768*a1*b1**3*i0**2*rho**2*np.cos(3*zeta) - \
            192*a0*b1**3*i0*i2*rho**2*np.cos(3*zeta) + \
            384*a1*b1**3*i0*i2*rho**2*np.cos(3*zeta) + \
            234*a0*b1**3*rho**4*np.cos(3*zeta) - \
            360*a1*b1**3*rho**4*np.cos(3*zeta) - \
            576*a0*b1**3*i0**2*rho**4*np.cos(3*zeta) + \
            2304*a1*b1**3*i0**2*rho**4*np.cos(3*zeta) + \
            1008*a0*b1**3*i0*i2*rho**4*np.cos(3*zeta) - \
            2688*a1*b1**3*i0*i2*rho**4*np.cos(3*zeta) - \
            128*a0*b1**3*i2**2*rho**4*np.cos(3*zeta) + \
            256*a1*b1**3*i2**2*rho**4*np.cos(3*zeta) - \
            189*a0*b1**3*rho**6*np.cos(3*zeta) + \
            432*a1*b1**3*rho**6*np.cos(3*zeta) + \
            432*a0*b1**3*i0**2*rho**6*np.cos(3*zeta) - \
            3456*a1*b1**3*i0**2*rho**6*np.cos(3*zeta) - \
            2016*a0*b1**3*i0*i2*rho**6*np.cos(3*zeta) + \
            8064*a1*b1**3*i0*i2*rho**6*np.cos(3*zeta) + \
            720*a0*b1**3*i2**2*rho**6*np.cos(3*zeta) - \
            1920*a1*b1**3*i2**2*rho**6*np.cos(3*zeta) - \
            243*a1*b1**3*rho**8*np.cos(3*zeta) + \
            1944*a1*b1**3*i0**2*rho**8*np.cos(3*zeta) + \
            1404*a0*b1**3*i0*i2*rho**8*np.cos(3*zeta) - \
            11232*a1*b1**3*i0*i2*rho**8*np.cos(3*zeta) - \
            1440*a0*b1**3*i2**2*rho**8*np.cos(3*zeta) + \
            5760*a1*b1**3*i2**2*rho**8*np.cos(3*zeta) + \
            5832*a1*b1**3*i0*i2*rho**10*np.cos(3*zeta) + \
            972*a0*b1**3*i2**2*rho**10*np.cos(3*zeta) - \
            7776*a1*b1**3*i2**2*rho**10*np.cos(3*zeta) + \
            3888*a1*b1**3*i2**2*rho**12*np.cos(3*zeta) - 16*b1**4*np.cos(4*zeta) \
            + 16*b1**4*i0**2*np.cos(4*zeta) + 96*b1**4*rho**2*np.cos(4*zeta) - \
            96*b1**4*i0**2*rho**2*np.cos(4*zeta) + \
            48*b1**4*i0*i2*rho**2*np.cos(4*zeta) - \
            288*b1**4*rho**4*np.cos(4*zeta) + \
            288*b1**4*i0**2*rho**4*np.cos(4*zeta) - \
            336*b1**4*i0*i2*rho**4*np.cos(4*zeta) + \
            32*b1**4*i2**2*rho**4*np.cos(4*zeta) + \
            432*b1**4*rho**6*np.cos(4*zeta) - \
            432*b1**4*i0**2*rho**6*np.cos(4*zeta) + \
            1008*b1**4*i0*i2*rho**6*np.cos(4*zeta) - \
            240*b1**4*i2**2*rho**6*np.cos(4*zeta) - \
            243*b1**4*rho**8*np.cos(4*zeta) + \
            243*b1**4*i0**2*rho**8*np.cos(4*zeta) - \
            1404*b1**4*i0*i2*rho**8*np.cos(4*zeta) + \
            720*b1**4*i2**2*rho**8*np.cos(4*zeta) + \
            729*b1**4*i0*i2*rho**10*np.cos(4*zeta) - \
            972*b1**4*i2**2*rho**10*np.cos(4*zeta) + \
            486*b1**4*i2**2*rho**12*np.cos(4*zeta)) + 12*a0*b1*rho**2*(i0 + \
            i2*rho**2)*(2*a0**2 - 8*a0*a1 + 8*a1**2 + 4*b1**2 + 24*a0*a1*rho**2 - \
            48*a1**2*rho**2 - 24*b1**2*rho**2 + 54*a1**2*rho**4 + 27*b1**2*rho**4 \
            + 4*b1*(a0*(-2 + 6*rho**2) + a1*(4 - 24*rho**2 + \
            27*rho**4))*np.cos(zeta) + b1**2*(4 - 24*rho**2 + \
            27*rho**4)*np.cos(2*zeta))*np.sin(theta)*np.sin(zeta)))/(mu0*(a0 + \
            a1*(-2 + 3*rho**2) + b1*(-2 + 3*rho**2)*np.cos(zeta))**3*(a0 + a1*(-2 \
            + 9*rho**2) + b1*(-2 + 9*rho**2)*np.cos(zeta))**4))
        F1 = -((b1*psi0**2*rho**2*(192*a0*a1 - 384*a1**2 - 192*b1**2 - \
            96*a0**2*b1**2 + 384*a0*a1*b1**2 - 384*a1**2*b1**2 - 96*b1**4 - \
            16*a0**4*i0**2 + 128*a0**3*a1*i0**2 - 384*a0**2*a1**2*i0**2 + \
            512*a0*a1**3*i0**2 - 256*a1**4*i0**2 - 192*a0**2*b1**2*i0**2 + \
            768*a0*a1*b1**2*i0**2 - 768*a1**2*b1**2*i0**2 - 96*b1**4*i0**2 - \
            288*a0*a1*rho**2 + 1440*a1**2*rho**2 + 720*b1**2*rho**2 + \
            624*a0**2*b1**2*rho**2 - 3456*a0*a1*b1**2*rho**2 + \
            4416*a1**2*b1**2*rho**2 + 1104*b1**4*rho**2 + 48*a0**4*i0**2*rho**2 - \
            528*a0**3*a1*i0**2*rho**2 + 2016*a0**2*a1**2*i0**2*rho**2 - \
            3264*a0*a1**3*i0**2*rho**2 + 1920*a1**4*i0**2*rho**2 + \
            1008*a0**2*b1**2*i0**2*rho**2 - 4896*a0*a1*b1**2*i0**2*rho**2 + \
            5760*a1**2*b1**2*i0**2*rho**2 + 720*b1**4*i0**2*rho**2 - \
            48*a0**4*i0*i2*rho**2 + 384*a0**3*a1*i0*i2*rho**2 - \
            1152*a0**2*a1**2*i0*i2*rho**2 + 1536*a0*a1**3*i0*i2*rho**2 - \
            768*a1**4*i0*i2*rho**2 - 576*a0**2*b1**2*i0*i2*rho**2 + \
            2304*a0*a1*b1**2*i0*i2*rho**2 - 2304*a1**2*b1**2*i0*i2*rho**2 - \
            288*b1**4*i0*i2*rho**2 - 1296*a1**2*rho**4 - 648*b1**2*rho**4 - \
            1224*a0**2*b1**2*rho**4 + 11520*a0*a1*b1**2*rho**4 - \
            20736*a1**2*b1**2*rho**4 - 5184*b1**4*rho**4 + \
            648*a0**3*a1*i0**2*rho**4 - 4320*a0**2*a1**2*i0**2*rho**4 + \
            9504*a0*a1**3*i0**2*rho**4 - 6912*a1**4*i0**2*rho**4 - \
            2160*a0**2*b1**2*i0**2*rho**4 + 14256*a0*a1*b1**2*i0**2*rho**4 - \
            20736*a1**2*b1**2*i0**2*rho**4 - 2592*b1**4*i0**2*rho**4 + \
            120*a0**4*i0*i2*rho**4 - 1536*a0**3*a1*i0*i2*rho**4 + \
            6336*a0**2*a1**2*i0*i2*rho**4 - 10752*a0*a1**3*i0*i2*rho**4 + \
            6528*a1**4*i0*i2*rho**4 + 3168*a0**2*b1**2*i0*i2*rho**4 - \
            16128*a0*a1*b1**2*i0*i2*rho**4 + 19584*a1**2*b1**2*i0*i2*rho**4 + \
            2448*b1**4*i0*i2*rho**4 - 32*a0**4*i2**2*rho**4 + \
            256*a0**3*a1*i2**2*rho**4 - 768*a0**2*a1**2*i2**2*rho**4 + \
            1024*a0*a1**3*i2**2*rho**4 - 512*a1**4*i2**2*rho**4 - \
            384*a0**2*b1**2*i2**2*rho**4 + 1536*a0*a1*b1**2*i2**2*rho**4 - \
            1536*a1**2*b1**2*i2**2*rho**4 - 192*b1**4*i2**2*rho**4 + \
            756*a0**2*b1**2*rho**6 - 16416*a0*a1*b1**2*rho**6 + \
            46656*a1**2*b1**2*rho**6 + 11664*b1**4*rho**6 + \
            3240*a0**2*a1**2*i0**2*rho**6 - 13392*a0*a1**3*i0**2*rho**6 + \
            13824*a1**4*i0**2*rho**6 + 1620*a0**2*b1**2*i0**2*rho**6 - \
            20088*a0*a1*b1**2*i0**2*rho**6 + 41472*a1**2*b1**2*i0**2*rho**6 + \
            5184*b1**4*i0**2*rho**6 + 1728*a0**3*a1*i0*i2*rho**6 - \
            12960*a0**2*a1**2*i0*i2*rho**6 + 31104*a0*a1**3*i0*i2*rho**6 - \
            24192*a1**4*i0*i2*rho**6 - 6480*a0**2*b1**2*i0*i2*rho**6 + \
            46656*a0*a1*b1**2*i0*i2*rho**6 - 72576*a1**2*b1**2*i0*i2*rho**6 - \
            9072*b1**4*i0*i2*rho**6 + 72*a0**4*i2**2*rho**6 - \
            1008*a0**3*a1*i2**2*rho**6 + 4320*a0**2*a1**2*i2**2*rho**6 - \
            7488*a0*a1**3*i2**2*rho**6 + 4608*a1**4*i2**2*rho**6 + \
            2160*a0**2*b1**2*i2**2*rho**6 - 11232*a0*a1*b1**2*i2**2*rho**6 + \
            13824*a1**2*b1**2*i2**2*rho**6 + 1728*b1**4*i2**2*rho**6 + \
            8424*a0*a1*b1**2*rho**8 - 49896*a1**2*b1**2*rho**8 - \
            12474*b1**4*rho**8 + 7128*a0*a1**3*i0**2*rho**8 - \
            14256*a1**4*i0**2*rho**8 + 10692*a0*a1*b1**2*i0**2*rho**8 - \
            42768*a1**2*b1**2*i0**2*rho**8 - 5346*b1**4*i0**2*rho**8 + \
            9072*a0**2*a1**2*i0*i2*rho**8 - 41472*a0*a1**3*i0*i2*rho**8 + \
            46656*a1**4*i0*i2*rho**8 + 4536*a0**2*b1**2*i0*i2*rho**8 - \
            62208*a0*a1*b1**2*i0*i2*rho**8 + 139968*a1**2*b1**2*i0*i2*rho**8 + \
            17496*b1**4*i0*i2*rho**8 + 1080*a0**3*a1*i2**2*rho**8 - \
            8640*a0**2*a1**2*i2**2*rho**8 + 21600*a0*a1**3*i2**2*rho**8 - \
            17280*a1**4*i2**2*rho**8 - 4320*a0**2*b1**2*i2**2*rho**8 + \
            32400*a0*a1*b1**2*i2**2*rho**8 - 51840*a1**2*b1**2*i2**2*rho**8 - \
            6480*b1**4*i2**2*rho**8 + 20412*a1**2*b1**2*rho**10 + \
            5103*b1**4*rho**10 + 5832*a1**4*i0**2*rho**10 + \
            17496*a1**2*b1**2*i0**2*rho**10 + 2187*b1**4*i0**2*rho**10 + \
            20736*a0*a1**3*i0*i2*rho**10 - 45360*a1**4*i0*i2*rho**10 + \
            31104*a0*a1*b1**2*i0*i2*rho**10 - 136080*a1**2*b1**2*i0*i2*rho**10 - \
            17010*b1**4*i0*i2*rho**10 + 5832*a0**2*a1**2*i2**2*rho**10 - \
            28080*a0*a1**3*i2**2*rho**10 + 32832*a1**4*i2**2*rho**10 + \
            2916*a0**2*b1**2*i2**2*rho**10 - 42120*a0*a1*b1**2*i2**2*rho**10 + \
            98496*a1**2*b1**2*i2**2*rho**10 + 12312*b1**4*i2**2*rho**10 + \
            17496*a1**4*i0*i2*rho**12 + 52488*a1**2*b1**2*i0*i2*rho**12 + \
            6561*b1**4*i0*i2*rho**12 + 13608*a0*a1**3*i2**2*rho**12 - \
            31104*a1**4*i2**2*rho**12 + 20412*a0*a1*b1**2*i2**2*rho**12 - \
            93312*a1**2*b1**2*i2**2*rho**12 - 11664*b1**4*i2**2*rho**12 + \
            11664*a1**4*i2**2*rho**14 + 34992*a1**2*b1**2*i2**2*rho**14 + \
            4374*b1**4*i2**2*rho**14 + b1*(12*a0**2*a1*(-2 + 3*rho**2)*(4 - \
            20*rho**2 + (21 + 64*i2**2)*rho**4 - 264*i2**2*rho**6 + \
            324*i2**2*rho**8 + 24*i0*i2*rho**2*(4 - 16*rho**2 + 21*rho**4) + \
            4*i0**2*(8 - 30*rho**2 + 45*rho**4)) + 4*a0**3*(4 - 12*rho**2 + (9 + \
            64*i2**2)*rho**4 - 252*i2**2*rho**6 + 270*i2**2*rho**8 + \
            48*i0*i2*rho**2*(2 - 8*rho**2 + 9*rho**4) + 2*i0**2*(16 - 66*rho**2 + \
            81*rho**4)) + a1*(-2 + 3*rho**2)*(96*(4 - 9*rho**2) + 4*a1**2*(2 - \
            3*rho**2)**2*(4 - 36*rho**2 + (81 + 64*i2**2)*rho**4 - \
            288*i2**2*rho**6 + 432*i2**2*rho**8 + 24*i0*i2*rho**2*(4 - 16*rho**2 \
            + 27*rho**4) + 8*i0**2*(4 - 12*rho**2 + 27*rho**4)) + 3*b1**2*(2 - \
            3*rho**2)**2*(20 - 148*rho**2 + (333 + 64*i2**2)*rho**4 - \
            288*i2**2*rho**6 + 432*i2**2*rho**8 + 24*i0*i2*rho**2*(4 - 16*rho**2 \
            + 27*rho**4) + 8*i0**2*(4 - 12*rho**2 + 27*rho**4))) + 3*a0*(-2 + \
            3*rho**2)*(-32 + 4*a1**2*(-2 + 3*rho**2)*(4 - 28*rho**2 + (45 + \
            64*i2**2)*rho**4 - 276*i2**2*rho**6 + 378*i2**2*rho**8 + \
            96*i0*i2*rho**2*(1 - 4*rho**2 + 6*rho**4) + 2*i0**2*(16 - 54*rho**2 + \
            99*rho**4)) + b1**2*(-2 + 3*rho**2)*(20 - 124*rho**2 + (201 + \
            64*i2**2)*rho**4 - 276*i2**2*rho**6 + 378*i2**2*rho**8 + \
            96*i0*i2*rho**2*(1 - 4*rho**2 + 6*rho**4) + 2*i0**2*(16 - 54*rho**2 + \
            99*rho**4))))*np.cos(zeta) + 2*b1**2*(-2 + 3*rho**2)*(b1**2*(2 - \
            3*rho**2)**2*(4 - 36*rho**2 + (81 + 16*i2**2)*rho**4 - \
            72*i2**2*rho**6 + 108*i2**2*rho**8 + 6*i0*i2*rho**2*(4 - 16*rho**2 + \
            27*rho**4) + i0**2*(8 - 24*rho**2 + 54*rho**4)) + \
            6*(a0**2*(6*i0*i2*rho**2*(4 - 16*rho**2 + 21*rho**4) + i0**2*(8 - \
            30*rho**2 + 45*rho**4) + i2**2*rho**4*(16 - 66*rho**2 + 81*rho**4)) + \
            a0*a1*(-2 + 3*rho**2)*(-8*rho**2 + 4*(3 + 8*i2**2)*rho**4 - \
            138*i2**2*rho**6 + 189*i2**2*rho**8 + 48*i0*i2*rho**2*(1 - 4*rho**2 + \
            6*rho**4) + i0**2*(16 - 54*rho**2 + 99*rho**4)) + 2*(4 - 9*rho**2 + \
            a1**2*(2 - 3*rho**2)**2*(-4*rho**2 + (9 + 8*i2**2)*rho**4 - \
            36*i2**2*rho**6 + 54*i2**2*rho**8 + 3*i0*i2*rho**2*(4 - 16*rho**2 + \
            27*rho**4) + i0**2*(4 - 12*rho**2 + 27*rho**4)))))*np.cos(2*zeta) - \
            48*a0*b1**3*np.cos(3*zeta) + 96*a1*b1**3*np.cos(3*zeta) + \
            128*a0*b1**3*i0**2*np.cos(3*zeta) - 256*a1*b1**3*i0**2*np.cos(3*zeta) \
            + 288*a0*b1**3*rho**2*np.cos(3*zeta) - \
            528*a1*b1**3*rho**2*np.cos(3*zeta) - \
            816*a0*b1**3*i0**2*rho**2*np.cos(3*zeta) + \
            1920*a1*b1**3*i0**2*rho**2*np.cos(3*zeta) + \
            384*a0*b1**3*i0*i2*rho**2*np.cos(3*zeta) - \
            768*a1*b1**3*i0*i2*rho**2*np.cos(3*zeta) - \
            792*a0*b1**3*rho**4*np.cos(3*zeta) + \
            1296*a1*b1**3*rho**4*np.cos(3*zeta) + \
            2376*a0*b1**3*i0**2*rho**4*np.cos(3*zeta) - \
            6912*a1*b1**3*i0**2*rho**4*np.cos(3*zeta) - \
            2688*a0*b1**3*i0*i2*rho**4*np.cos(3*zeta) + \
            6528*a1*b1**3*i0*i2*rho**4*np.cos(3*zeta) + \
            256*a0*b1**3*i2**2*rho**4*np.cos(3*zeta) - \
            512*a1*b1**3*i2**2*rho**4*np.cos(3*zeta) + \
            1080*a0*b1**3*rho**6*np.cos(3*zeta) - \
            1944*a1*b1**3*rho**6*np.cos(3*zeta) - \
            3348*a0*b1**3*i0**2*rho**6*np.cos(3*zeta) + \
            13824*a1*b1**3*i0**2*rho**6*np.cos(3*zeta) + \
            7776*a0*b1**3*i0*i2*rho**6*np.cos(3*zeta) - \
            24192*a1*b1**3*i0*i2*rho**6*np.cos(3*zeta) - \
            1872*a0*b1**3*i2**2*rho**6*np.cos(3*zeta) + \
            4608*a1*b1**3*i2**2*rho**6*np.cos(3*zeta) - \
            567*a0*b1**3*rho**8*np.cos(3*zeta) + \
            1782*a1*b1**3*rho**8*np.cos(3*zeta) + \
            1782*a0*b1**3*i0**2*rho**8*np.cos(3*zeta) - \
            14256*a1*b1**3*i0**2*rho**8*np.cos(3*zeta) - \
            10368*a0*b1**3*i0*i2*rho**8*np.cos(3*zeta) + \
            46656*a1*b1**3*i0*i2*rho**8*np.cos(3*zeta) + \
            5400*a0*b1**3*i2**2*rho**8*np.cos(3*zeta) - \
            17280*a1*b1**3*i2**2*rho**8*np.cos(3*zeta) - \
            729*a1*b1**3*rho**10*np.cos(3*zeta) + \
            5832*a1*b1**3*i0**2*rho**10*np.cos(3*zeta) + \
            5184*a0*b1**3*i0*i2*rho**10*np.cos(3*zeta) - \
            45360*a1*b1**3*i0*i2*rho**10*np.cos(3*zeta) - \
            7020*a0*b1**3*i2**2*rho**10*np.cos(3*zeta) + \
            32832*a1*b1**3*i2**2*rho**10*np.cos(3*zeta) + \
            17496*a1*b1**3*i0*i2*rho**12*np.cos(3*zeta) + \
            3402*a0*b1**3*i2**2*rho**12*np.cos(3*zeta) - \
            31104*a1*b1**3*i2**2*rho**12*np.cos(3*zeta) + \
            11664*a1*b1**3*i2**2*rho**14*np.cos(3*zeta) + 32*b1**4*np.cos(4*zeta) \
            - 32*b1**4*i0**2*np.cos(4*zeta) - 240*b1**4*rho**2*np.cos(4*zeta) + \
            240*b1**4*i0**2*rho**2*np.cos(4*zeta) - \
            96*b1**4*i0*i2*rho**2*np.cos(4*zeta) + \
            864*b1**4*rho**4*np.cos(4*zeta) - \
            864*b1**4*i0**2*rho**4*np.cos(4*zeta) + \
            816*b1**4*i0*i2*rho**4*np.cos(4*zeta) - \
            64*b1**4*i2**2*rho**4*np.cos(4*zeta) - \
            1728*b1**4*rho**6*np.cos(4*zeta) + \
            1728*b1**4*i0**2*rho**6*np.cos(4*zeta) - \
            3024*b1**4*i0*i2*rho**6*np.cos(4*zeta) + \
            576*b1**4*i2**2*rho**6*np.cos(4*zeta) + \
            1782*b1**4*rho**8*np.cos(4*zeta) - \
            1782*b1**4*i0**2*rho**8*np.cos(4*zeta) + \
            5832*b1**4*i0*i2*rho**8*np.cos(4*zeta) - \
            2160*b1**4*i2**2*rho**8*np.cos(4*zeta) - \
            729*b1**4*rho**10*np.cos(4*zeta) + \
            729*b1**4*i0**2*rho**10*np.cos(4*zeta) - \
            5670*b1**4*i0*i2*rho**10*np.cos(4*zeta) + \
            4104*b1**4*i2**2*rho**10*np.cos(4*zeta) + \
            2187*b1**4*i0*i2*rho**12*np.cos(4*zeta) - \
            3888*b1**4*i2**2*rho**12*np.cos(4*zeta) + \
            1458*b1**4*i2**2*rho**14*np.cos(4*zeta))*np.sin(zeta))/(mu0*(a0 + \
            a1*(-2 + 3*rho**2) + b1*(-2 + 3*rho**2)*np.cos(zeta))**3*(a0 + a1*(-2 \
            + 9*rho**2) + b1*(-2 + 9*rho**2)*np.cos(zeta))**4))
        F2 = (psi0**2*rho*((-96*a0*a1 + 192*a1**2 + 96*b1**2 + 48*a0**2*b1**2 - \
            192*a0*a1*b1**2 + 192*a1**2*b1**2 + 48*b1**4 + 8*a0**4*i0**2 - \
            64*a0**3*a1*i0**2 + 192*a0**2*a1**2*i0**2 - 256*a0*a1**3*i0**2 + \
            128*a1**4*i0**2 + 96*a0**2*b1**2*i0**2 - 384*a0*a1*b1**2*i0**2 + \
            384*a1**2*b1**2*i0**2 + 48*b1**4*i0**2 - 432*a1**2*rho**2 - \
            216*b1**2*rho**2 - 240*a0**2*b1**2*rho**2 + 1440*a0*a1*b1**2*rho**2 - \
            1920*a1**2*b1**2*rho**2 - 480*b1**4*rho**2 + 96*a0**3*a1*i0**2*rho**2 \
            - 576*a0**2*a1**2*i0**2*rho**2 + 1152*a0*a1**3*i0**2*rho**2 - \
            768*a1**4*i0**2*rho**2 - 288*a0**2*b1**2*i0**2*rho**2 + \
            1728*a0*a1*b1**2*i0**2*rho**2 - 2304*a1**2*b1**2*i0**2*rho**2 - \
            288*b1**4*i0**2*rho**2 + 24*a0**4*i0*i2*rho**2 - \
            192*a0**3*a1*i0*i2*rho**2 + 576*a0**2*a1**2*i0*i2*rho**2 - \
            768*a0*a1**3*i0*i2*rho**2 + 384*a1**4*i0*i2*rho**2 + \
            288*a0**2*b1**2*i0*i2*rho**2 - 1152*a0*a1*b1**2*i0*i2*rho**2 + \
            1152*a1**2*b1**2*i0*i2*rho**2 + 144*b1**4*i0*i2*rho**2 + \
            252*a0**2*b1**2*rho**4 - 3600*a0*a1*b1**2*rho**4 + \
            7488*a1**2*b1**2*rho**4 + 1872*b1**4*rho**4 + \
            576*a0**2*a1**2*i0**2*rho**4 - 2304*a0*a1**3*i0**2*rho**4 + \
            2304*a1**4*i0**2*rho**4 + 288*a0**2*b1**2*i0**2*rho**4 - \
            3456*a0*a1*b1**2*i0**2*rho**4 + 6912*a1**2*b1**2*i0**2*rho**4 + \
            864*b1**4*i0**2*rho**4 + 336*a0**3*a1*i0*i2*rho**4 - \
            2016*a0**2*a1**2*i0*i2*rho**4 + 4032*a0*a1**3*i0*i2*rho**4 - \
            2688*a1**4*i0*i2*rho**4 - 1008*a0**2*b1**2*i0*i2*rho**4 + \
            6048*a0*a1*b1**2*i0*i2*rho**4 - 8064*a1**2*b1**2*i0*i2*rho**4 - \
            1008*b1**4*i0*i2*rho**4 + 16*a0**4*i2**2*rho**4 - \
            128*a0**3*a1*i2**2*rho**4 + 384*a0**2*a1**2*i2**2*rho**4 - \
            512*a0*a1**3*i2**2*rho**4 + 256*a1**4*i2**2*rho**4 + \
            192*a0**2*b1**2*i2**2*rho**4 - 768*a0*a1*b1**2*i2**2*rho**4 + \
            768*a1**2*b1**2*i2**2*rho**4 + 96*b1**4*i2**2*rho**4 + \
            2808*a0*a1*b1**2*rho**6 - 12096*a1**2*b1**2*rho**6 - \
            3024*b1**4*rho**6 + 1728*a0*a1**3*i0**2*rho**6 - \
            3456*a1**4*i0**2*rho**6 + 2592*a0*a1*b1**2*i0**2*rho**6 - \
            10368*a1**2*b1**2*i0**2*rho**6 - 1296*b1**4*i0**2*rho**6 + \
            2016*a0**2*a1**2*i0*i2*rho**6 - 8064*a0*a1**3*i0*i2*rho**6 + \
            8064*a1**4*i0*i2*rho**6 + 1008*a0**2*b1**2*i0*i2*rho**6 - \
            12096*a0*a1*b1**2*i0*i2*rho**6 + 24192*a1**2*b1**2*i0*i2*rho**6 + \
            3024*b1**4*i0*i2*rho**6 + 240*a0**3*a1*i2**2*rho**6 - \
            1440*a0**2*a1**2*i2**2*rho**6 + 2880*a0*a1**3*i2**2*rho**6 - \
            1920*a1**4*i2**2*rho**6 - 720*a0**2*b1**2*i2**2*rho**6 + \
            4320*a0*a1*b1**2*i2**2*rho**6 - 5760*a1**2*b1**2*i2**2*rho**6 - \
            720*b1**4*i2**2*rho**6 + 6804*a1**2*b1**2*rho**8 + 1701*b1**4*rho**8 \
            + 1944*a1**4*i0**2*rho**8 + 5832*a1**2*b1**2*i0**2*rho**8 + \
            729*b1**4*i0**2*rho**8 + 5616*a0*a1**3*i0*i2*rho**8 - \
            11232*a1**4*i0*i2*rho**8 + 8424*a0*a1*b1**2*i0*i2*rho**8 - \
            33696*a1**2*b1**2*i0*i2*rho**8 - 4212*b1**4*i0*i2*rho**8 + \
            1440*a0**2*a1**2*i2**2*rho**8 - 5760*a0*a1**3*i2**2*rho**8 + \
            5760*a1**4*i2**2*rho**8 + 720*a0**2*b1**2*i2**2*rho**8 - \
            8640*a0*a1*b1**2*i2**2*rho**8 + 17280*a1**2*b1**2*i2**2*rho**8 + \
            2160*b1**4*i2**2*rho**8 + 5832*a1**4*i0*i2*rho**10 + \
            17496*a1**2*b1**2*i0*i2*rho**10 + 2187*b1**4*i0*i2*rho**10 + \
            3888*a0*a1**3*i2**2*rho**10 - 7776*a1**4*i2**2*rho**10 + \
            5832*a0*a1*b1**2*i2**2*rho**10 - 23328*a1**2*b1**2*i2**2*rho**10 - \
            2916*b1**4*i2**2*rho**10 + 3888*a1**4*i2**2*rho**12 + \
            11664*a1**2*b1**2*i2**2*rho**12 + 1458*b1**4*i2**2*rho**12 + \
            b1*(4*a0**3*(-2 + 3*rho**2 - 32*i2**2*rho**4 + 60*i2**2*rho**6 + \
            8*i0**2*(-2 + 3*rho**2) + 12*i0*i2*rho**2*(-4 + 7*rho**2)) + \
            12*a0**2*a1*(4 - 20*rho**2 + (21 + 64*i2**2)*rho**4 - \
            240*i2**2*rho**6 + 240*i2**2*rho**8 + 32*i0**2*(1 - 3*rho**2 + \
            3*rho**4) + 48*i0*i2*rho**2*(2 - 7*rho**2 + 7*rho**4)) + a1*(96*(4 - \
            9*rho**2) + 4*a1**2*(2 - 3*rho**2)**2*(4 - 36*rho**2 + (81 + \
            64*i2**2)*rho**4 - 288*i2**2*rho**6 + 432*i2**2*rho**8 + \
            24*i0*i2*rho**2*(4 - 16*rho**2 + 27*rho**4) + 8*i0**2*(4 - 12*rho**2 \
            + 27*rho**4)) + 3*b1**2*(2 - 3*rho**2)**2*(20 - 148*rho**2 + (333 + \
            64*i2**2)*rho**4 - 288*i2**2*rho**6 + 432*i2**2*rho**8 + \
            24*i0*i2*rho**2*(4 - 16*rho**2 + 27*rho**4) + 8*i0**2*(4 - 12*rho**2 \
            + 27*rho**4))) + 3*a0*(-32 + 4*a1**2*(-2 + 3*rho**2)*(4 - 28*rho**2 + \
            (45 + 64*i2**2)*rho**4 - 264*i2**2*rho**6 + 324*i2**2*rho**8 + \
            16*i0**2*(2 - 6*rho**2 + 9*rho**4) + 12*i0*i2*rho**2*(8 - 30*rho**2 + \
            39*rho**4)) + b1**2*(-2 + 3*rho**2)*(20 - 124*rho**2 + (201 + \
            64*i2**2)*rho**4 - 264*i2**2*rho**6 + 324*i2**2*rho**8 + 16*i0**2*(2 \
            - 6*rho**2 + 9*rho**4) + 12*i0*i2*rho**2*(8 - 30*rho**2 + \
            39*rho**4))))*np.cos(zeta) + 2*b1**2*(b1**2*(2 - 3*rho**2)**2*(4 - \
            36*rho**2 + (81 + 16*i2**2)*rho**4 - 72*i2**2*rho**6 + \
            108*i2**2*rho**8 + 6*i0*i2*rho**2*(4 - 16*rho**2 + 27*rho**4) + \
            i0**2*(8 - 24*rho**2 + 54*rho**4)) + 12*(4 - 9*rho**2 + \
            2*a0**2*(i0**2*(2 - 6*rho**2 + 6*rho**4) + 3*i0*i2*rho**2*(2 - \
            7*rho**2 + 7*rho**4) + i2**2*rho**4*(4 - 15*rho**2 + 15*rho**4)) + \
            a1**2*(2 - 3*rho**2)**2*(-4*rho**2 + (9 + 8*i2**2)*rho**4 - \
            36*i2**2*rho**6 + 54*i2**2*rho**8 + 3*i0*i2*rho**2*(4 - 16*rho**2 + \
            27*rho**4) + i0**2*(4 - 12*rho**2 + 27*rho**4)) + a0*a1*(-2 + \
            3*rho**2)*(-4*rho**2 + 2*(3 + 8*i2**2)*rho**4 - 66*i2**2*rho**6 + \
            81*i2**2*rho**8 + 4*i0**2*(2 - 6*rho**2 + 9*rho**4) + \
            3*i0*i2*rho**2*(8 - 30*rho**2 + 39*rho**4))))*np.cos(2*zeta) + \
            24*a0*b1**3*np.cos(3*zeta) - 48*a1*b1**3*np.cos(3*zeta) - \
            64*a0*b1**3*i0**2*np.cos(3*zeta) + 128*a1*b1**3*i0**2*np.cos(3*zeta) \
            - 108*a0*b1**3*rho**2*np.cos(3*zeta) + \
            192*a1*b1**3*rho**2*np.cos(3*zeta) + \
            288*a0*b1**3*i0**2*rho**2*np.cos(3*zeta) - \
            768*a1*b1**3*i0**2*rho**2*np.cos(3*zeta) - \
            192*a0*b1**3*i0*i2*rho**2*np.cos(3*zeta) + \
            384*a1*b1**3*i0*i2*rho**2*np.cos(3*zeta) + \
            234*a0*b1**3*rho**4*np.cos(3*zeta) - \
            360*a1*b1**3*rho**4*np.cos(3*zeta) - \
            576*a0*b1**3*i0**2*rho**4*np.cos(3*zeta) + \
            2304*a1*b1**3*i0**2*rho**4*np.cos(3*zeta) + \
            1008*a0*b1**3*i0*i2*rho**4*np.cos(3*zeta) - \
            2688*a1*b1**3*i0*i2*rho**4*np.cos(3*zeta) - \
            128*a0*b1**3*i2**2*rho**4*np.cos(3*zeta) + \
            256*a1*b1**3*i2**2*rho**4*np.cos(3*zeta) - \
            189*a0*b1**3*rho**6*np.cos(3*zeta) + \
            432*a1*b1**3*rho**6*np.cos(3*zeta) + \
            432*a0*b1**3*i0**2*rho**6*np.cos(3*zeta) - \
            3456*a1*b1**3*i0**2*rho**6*np.cos(3*zeta) - \
            2016*a0*b1**3*i0*i2*rho**6*np.cos(3*zeta) + \
            8064*a1*b1**3*i0*i2*rho**6*np.cos(3*zeta) + \
            720*a0*b1**3*i2**2*rho**6*np.cos(3*zeta) - \
            1920*a1*b1**3*i2**2*rho**6*np.cos(3*zeta) - \
            243*a1*b1**3*rho**8*np.cos(3*zeta) + \
            1944*a1*b1**3*i0**2*rho**8*np.cos(3*zeta) + \
            1404*a0*b1**3*i0*i2*rho**8*np.cos(3*zeta) - \
            11232*a1*b1**3*i0*i2*rho**8*np.cos(3*zeta) - \
            1440*a0*b1**3*i2**2*rho**8*np.cos(3*zeta) + \
            5760*a1*b1**3*i2**2*rho**8*np.cos(3*zeta) + \
            5832*a1*b1**3*i0*i2*rho**10*np.cos(3*zeta) + \
            972*a0*b1**3*i2**2*rho**10*np.cos(3*zeta) - \
            7776*a1*b1**3*i2**2*rho**10*np.cos(3*zeta) + \
            3888*a1*b1**3*i2**2*rho**12*np.cos(3*zeta) - 16*b1**4*np.cos(4*zeta) \
            + 16*b1**4*i0**2*np.cos(4*zeta) + 96*b1**4*rho**2*np.cos(4*zeta) - \
            96*b1**4*i0**2*rho**2*np.cos(4*zeta) + \
            48*b1**4*i0*i2*rho**2*np.cos(4*zeta) - \
            288*b1**4*rho**4*np.cos(4*zeta) + \
            288*b1**4*i0**2*rho**4*np.cos(4*zeta) - \
            336*b1**4*i0*i2*rho**4*np.cos(4*zeta) + \
            32*b1**4*i2**2*rho**4*np.cos(4*zeta) + \
            432*b1**4*rho**6*np.cos(4*zeta) - \
            432*b1**4*i0**2*rho**6*np.cos(4*zeta) + \
            1008*b1**4*i0*i2*rho**6*np.cos(4*zeta) - \
            240*b1**4*i2**2*rho**6*np.cos(4*zeta) - \
            243*b1**4*rho**8*np.cos(4*zeta) + \
            243*b1**4*i0**2*rho**8*np.cos(4*zeta) - \
            1404*b1**4*i0*i2*rho**8*np.cos(4*zeta) + \
            720*b1**4*i2**2*rho**8*np.cos(4*zeta) + \
            729*b1**4*i0*i2*rho**10*np.cos(4*zeta) - \
            972*b1**4*i2**2*rho**10*np.cos(4*zeta) + \
            486*b1**4*i2**2*rho**12*np.cos(4*zeta))*np.sin(theta) - \
            12*a0*b1*rho**2*(i0 + i2*rho**2)*np.cos(theta)*(2*a0**2 - 8*a0*a1 + \
            8*a1**2 + 4*b1**2 + 24*a0*a1*rho**2 - 48*a1**2*rho**2 - \
            24*b1**2*rho**2 + 54*a1**2*rho**4 + 27*b1**2*rho**4 + 4*b1*(a0*(-2 + \
            6*rho**2) + a1*(4 - 24*rho**2 + 27*rho**4))*np.cos(zeta) + b1**2*(4 - \
            24*rho**2 + 27*rho**4)*np.cos(2*zeta))*np.sin(zeta)))/(mu0*(a0 + \
            a1*(-2 + 3*rho**2) + b1*(-2 + 3*rho**2)*np.cos(zeta))**3*(a0 + a1*(-2 \
            + 9*rho**2) + b1*(-2 + 9*rho**2)*np.cos(zeta))**4)
        return np.stack( (F0,F1,F2), axis=1)

@dataclass
class model_mirror_3d1(ana_model):
    """axis-sym mirror with periodic zeta.(FourierZernike Basis)
        With 3D shaping of the cross sections and zeta variation.
        For definition, refer to _get_modes
        R is only passed as an argument to R_lmn, not included in the analytical formula.
    """
    Psi: float
    a0: float
    a1: float
    b1: float
    R: float
    c1: float
    c2: float
    
    def _get_modes(self):
        return {
            "R_lmn":{ (0,0,0):self.R, (1,1,0):self.a0, (3,1,0):self.a1, (3,1,1):self.b1, (0,0,1):self.c1, (2,2,1):self.c2},
            "Z_lmn":{ (1,-1,0):-self.a0, (3,-1,0):-self.a1, (3,-1,1):-self.b1 },
            "iota": {"modes": (0,2), "params": (0,0)},
        }
    
    def B_vec_ana_cal(self, rtz):
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        zeta = rtz[:, 2]
        a0 = self.a0
        a1 = self.a1
        b1 = self.b1
        psi0 = self.Psi / np.pi / 2
        c1 = self.c1
        c2 = self.c2
        
        B0 = (-2*psi0*rho*(c1 + b1*rho*(-2 + 3*rho**2)*np.cos(theta) + \
            c2*rho**2*np.cos(2*theta))*np.sin(zeta))/(rho*np.cos(theta)*(a0 + \
            a1*(-2 + 3*rho**2) + b1*(-2 + \
            3*rho**2)*np.cos(zeta))*(2*c2*rho*np.cos(2*theta)*np.cos(zeta) + \
            np.cos(theta)*(a0 - 2*a1 + 9*a1*rho**2 + b1*(-2 + \
            9*rho**2)*np.cos(zeta))) + rho*(a0 + a1*(-2 + 9*rho**2) + b1*(-2 + \
            9*rho**2)*np.cos(zeta))*(a0 + a1*(-2 + 3*rho**2) + (b1*(-2 + \
            3*rho**2) + 4*c2*rho*np.cos(theta))*np.cos(zeta))*np.sin(theta)**2)
        B1 = (2*psi0*rho)/(rho*np.cos(theta)*(a0 + a1*(-2 + 3*rho**2) + b1*(-2 + \
            3*rho**2)*np.cos(zeta))*(2*c2*rho*np.cos(2*theta)*np.cos(zeta) + \
            np.cos(theta)*(a0 - 2*a1 + 9*a1*rho**2 + b1*(-2 + \
            9*rho**2)*np.cos(zeta))) + rho*(a0 + a1*(-2 + 9*rho**2) + b1*(-2 + \
            9*rho**2)*np.cos(zeta))*(a0 + a1*(-2 + 3*rho**2) + (b1*(-2 + \
            3*rho**2) + 4*c2*rho*np.cos(theta))*np.cos(zeta))*np.sin(theta)**2)
        B2 = (2*b1*psi0*rho**2*(-2 + \
            3*rho**2)*np.sin(theta)*np.sin(zeta))/(rho*np.cos(theta)*(a0 + a1*(-2 \
            + 3*rho**2) + b1*(-2 + \
            3*rho**2)*np.cos(zeta))*(2*c2*rho*np.cos(2*theta)*np.cos(zeta) + \
            np.cos(theta)*(a0 - 2*a1 + 9*a1*rho**2 + b1*(-2 + \
            9*rho**2)*np.cos(zeta))) + rho*(a0 + a1*(-2 + 9*rho**2) + b1*(-2 + \
            9*rho**2)*np.cos(zeta))*(a0 + a1*(-2 + 3*rho**2) + (b1*(-2 + \
            3*rho**2) + 4*c2*rho*np.cos(theta))*np.cos(zeta))*np.sin(theta)**2)
        
        return np.stack((B0,B1,B2), axis=1)
    
    def j_vec_ana_cal(self, rtz):
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        zeta = rtz[:, 2]
        a0 = self.a0
        a1 = self.a1
        b1 = self.b1
        psi0 = self.Psi / np.pi / 2
        c1 = self.c1
        c2 = self.c2
        mu0 = mu_0

        # to-do
        j0 = ((-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta))*((2*b1*psi0*rho**2*(-2 + 3*rho**2)*np.sin(theta)*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - 4*c2*rho**2*np.cos(2*theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(a0*rho*np.sin(theta) + a1*rho*(-2 + 3*rho**2)*np.sin(theta) + b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*np.sin(zeta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (2*b1*psi0*rho**2*(-2 + 3*rho**2)*(-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*np.sin(theta)*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(18*a1*rho*np.cos(theta) + 18*b1*rho*np.cos(theta)*np.cos(zeta) + 2*c2*np.cos(2*theta)*np.cos(zeta))) - (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-18*a1*rho*np.sin(theta) - 18*b1*rho*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*np.sin(zeta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 + (12*b1*psi0*rho**3*(-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*np.sin(theta)*np.sin(zeta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (4*b1*psi0*rho*(-2 + 3*rho**2)*(-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*np.sin(theta)*np.sin(zeta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*b1*psi0*rho**2*(-2 + 3*rho**2)*np.cos(theta)*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*np.sin(zeta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*psi0*rho*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(-6*b1*rho**2*np.cos(theta)*np.sin(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - 2*c2*rho*np.cos(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*psi0*rho*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - 4*c2*rho**2*np.cos(2*theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(a0*rho*np.sin(theta) + a1*rho*(-2 + 3*rho**2)*np.sin(theta) + b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (2*psi0*rho*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(18*a1*rho*np.cos(theta) + 18*b1*rho*np.cos(theta)*np.cos(zeta) + 2*c2*np.cos(2*theta)*np.cos(zeta))) - (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-18*a1*rho*np.sin(theta) - 18*b1*rho*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 + (2*psi0*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*psi0*rho*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(b1*rho*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta) + 2*c2*rho**2*np.sin(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))))/(mu0*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))) + ((-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*((2*psi0*rho*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(18*a1*rho*np.cos(theta) + 18*b1*rho*np.cos(theta)*np.cos(zeta) + 2*c2*np.cos(2*theta)*np.cos(zeta))) - (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-18*a1*rho*np.sin(theta) - 18*b1*rho*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (2*psi0)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*psi0*rho*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(-(c1*np.cos(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - c2*rho**2*np.cos(2*theta)*np.cos(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*b1*psi0*rho**2*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*b1**2*psi0*rho**3*(-2 + 3*rho**2)**2*np.sin(theta)**2*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(18*a1*rho*np.cos(theta) + 18*b1*rho*np.cos(theta)*np.cos(zeta) + 2*c2*np.cos(2*theta)*np.cos(zeta))) - (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-18*a1*rho*np.sin(theta) - 18*b1*rho*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*np.sin(zeta)**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (24*b1**2*psi0*rho**4*(-2 + 3*rho**2)*np.sin(theta)**2*np.sin(zeta)**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (6*b1**2*psi0*rho**2*(-2 + 3*rho**2)**2*np.sin(theta)**2*np.sin(zeta)**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*psi0*rho*(-6*b1*rho**2*np.cos(theta)*np.sin(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - 2*c2*rho*np.cos(2*theta)*np.sin(zeta))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*psi0*rho*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(18*a1*rho*np.cos(theta) + 18*b1*rho*np.cos(theta)*np.cos(zeta) + 2*c2*np.cos(2*theta)*np.cos(zeta))) - (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-18*a1*rho*np.sin(theta) - 18*b1*rho*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta))**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (2*psi0*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta))**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*b1*psi0*rho**2*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta)*(6*b1*rho**2*np.sin(theta)*np.sin(zeta) + b1*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*b1*psi0*rho**2*(-2 + 3*rho**2)*np.sin(theta)*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*np.sin(zeta)*(-(b1*rho*(-2 + 3*rho**2)*np.cos(theta)*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*np.sin(zeta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-6*b1*rho**2*np.cos(theta)*np.sin(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - 2*c2*rho*np.cos(2*theta)*np.sin(zeta)) + (-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(6*b1*rho**2*np.sin(theta)*np.sin(zeta) + b1*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(b1*rho*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta) + 2*c2*rho**2*np.sin(2*theta)*np.sin(zeta))))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (2*psi0*rho*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta))*(-(b1*rho*(-2 + 3*rho**2)*np.cos(theta)*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*np.sin(zeta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-6*b1*rho**2*np.cos(theta)*np.sin(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - 2*c2*rho*np.cos(2*theta)*np.sin(zeta)) + (-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(6*b1*rho**2*np.sin(theta)*np.sin(zeta) + b1*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(b1*rho*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta) + 2*c2*rho**2*np.sin(2*theta)*np.sin(zeta))))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2))/(mu0*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))) + ((a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*((-2*psi0*rho*((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - 4*c2*rho**2*np.cos(2*theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(a0*rho*np.sin(theta) + a1*rho*(-2 + 3*rho**2)*np.sin(theta) + b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (2*b1*psi0*rho**2*(-2 + 3*rho**2)*np.cos(zeta)*(-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*np.sin(theta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*psi0*rho*(-(c1*np.cos(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - c2*rho**2*np.cos(2*theta)*np.cos(zeta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*b1**2*psi0*rho**3*(-2 + 3*rho**2)**2*np.sin(theta)**2*((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - 4*c2*rho**2*np.cos(2*theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(a0*rho*np.sin(theta) + a1*rho*(-2 + 3*rho**2)*np.sin(theta) + b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*np.sin(zeta)**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 + (2*b1**2*psi0*rho**3*(-2 + 3*rho**2)**2*np.cos(theta)*np.sin(theta)*np.sin(zeta)**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*psi0*rho*((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - 4*c2*rho**2*np.cos(2*theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(a0*rho*np.sin(theta) + a1*rho*(-2 + 3*rho**2)*np.sin(theta) + b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta))**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 + (2*psi0*rho*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta))*(b1*rho*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta) + 2*c2*rho**2*np.sin(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*b1*psi0*rho**2*(-2 + 3*rho**2)*(-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*np.sin(theta)*np.sin(zeta)*(-(b1*rho*(-2 + 3*rho**2)*np.cos(theta)*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*np.sin(zeta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-6*b1*rho**2*np.cos(theta)*np.sin(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - 2*c2*rho*np.cos(2*theta)*np.sin(zeta)) + (-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(6*b1*rho**2*np.sin(theta)*np.sin(zeta) + b1*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(b1*rho*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta) + 2*c2*rho**2*np.sin(2*theta)*np.sin(zeta))))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 + (2*psi0*rho*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta))*(-(b1*rho*(-2 + 3*rho**2)*np.cos(theta)*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*np.sin(zeta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-6*b1*rho**2*np.cos(theta)*np.sin(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - 2*c2*rho*np.cos(2*theta)*np.sin(zeta)) + (-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(6*b1*rho**2*np.sin(theta)*np.sin(zeta) + b1*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(b1*rho*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta) + 2*c2*rho**2*np.sin(2*theta)*np.sin(zeta))))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2))/(mu0*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))))
        j1 = ((2*b1*psi0*rho**2*(-2 + 3*rho**2)*np.sin(theta)*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - 4*c2*rho**2*np.cos(2*theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(a0*rho*np.sin(theta) + a1*rho*(-2 + 3*rho**2)*np.sin(theta) + b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*np.sin(zeta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (2*b1*psi0*rho**2*(-2 + 3*rho**2)*(-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*np.sin(theta)*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(18*a1*rho*np.cos(theta) + 18*b1*rho*np.cos(theta)*np.cos(zeta) + 2*c2*np.cos(2*theta)*np.cos(zeta))) - (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-18*a1*rho*np.sin(theta) - 18*b1*rho*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*np.sin(zeta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 + (12*b1*psi0*rho**3*(-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*np.sin(theta)*np.sin(zeta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (4*b1*psi0*rho*(-2 + 3*rho**2)*(-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*np.sin(theta)*np.sin(zeta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*b1*psi0*rho**2*(-2 + 3*rho**2)*np.cos(theta)*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*np.sin(zeta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*psi0*rho*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(-6*b1*rho**2*np.cos(theta)*np.sin(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - 2*c2*rho*np.cos(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*psi0*rho*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - 4*c2*rho**2*np.cos(2*theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(a0*rho*np.sin(theta) + a1*rho*(-2 + 3*rho**2)*np.sin(theta) + b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (2*psi0*rho*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(18*a1*rho*np.cos(theta) + 18*b1*rho*np.cos(theta)*np.cos(zeta) + 2*c2*np.cos(2*theta)*np.cos(zeta))) - (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-18*a1*rho*np.sin(theta) - 18*b1*rho*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 + (2*psi0*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*psi0*rho*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(b1*rho*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta) + 2*c2*rho**2*np.sin(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))))/(mu0*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))))
        j2 = (b1*rho*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta)*((2*b1*psi0*rho**2*(-2 + 3*rho**2)*np.sin(theta)*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - 4*c2*rho**2*np.cos(2*theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(a0*rho*np.sin(theta) + a1*rho*(-2 + 3*rho**2)*np.sin(theta) + b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*np.sin(zeta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (2*b1*psi0*rho**2*(-2 + 3*rho**2)*(-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*np.sin(theta)*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(18*a1*rho*np.cos(theta) + 18*b1*rho*np.cos(theta)*np.cos(zeta) + 2*c2*np.cos(2*theta)*np.cos(zeta))) - (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-18*a1*rho*np.sin(theta) - 18*b1*rho*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*np.sin(zeta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 + (12*b1*psi0*rho**3*(-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*np.sin(theta)*np.sin(zeta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (4*b1*psi0*rho*(-2 + 3*rho**2)*(-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*np.sin(theta)*np.sin(zeta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*b1*psi0*rho**2*(-2 + 3*rho**2)*np.cos(theta)*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*np.sin(zeta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*psi0*rho*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(-6*b1*rho**2*np.cos(theta)*np.sin(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - 2*c2*rho*np.cos(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*psi0*rho*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - 4*c2*rho**2*np.cos(2*theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(a0*rho*np.sin(theta) + a1*rho*(-2 + 3*rho**2)*np.sin(theta) + b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (2*psi0*rho*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(18*a1*rho*np.cos(theta) + 18*b1*rho*np.cos(theta)*np.cos(zeta) + 2*c2*np.cos(2*theta)*np.cos(zeta))) - (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-18*a1*rho*np.sin(theta) - 18*b1*rho*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 + (2*psi0*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*psi0*rho*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(b1*rho*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta) + 2*c2*rho**2*np.sin(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))))/(mu0*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))) + ((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*((2*psi0*rho*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(18*a1*rho*np.cos(theta) + 18*b1*rho*np.cos(theta)*np.cos(zeta) + 2*c2*np.cos(2*theta)*np.cos(zeta))) - (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-18*a1*rho*np.sin(theta) - 18*b1*rho*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (2*psi0)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*psi0*rho*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(-(c1*np.cos(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - c2*rho**2*np.cos(2*theta)*np.cos(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*b1*psi0*rho**2*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*b1**2*psi0*rho**3*(-2 + 3*rho**2)**2*np.sin(theta)**2*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(18*a1*rho*np.cos(theta) + 18*b1*rho*np.cos(theta)*np.cos(zeta) + 2*c2*np.cos(2*theta)*np.cos(zeta))) - (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-18*a1*rho*np.sin(theta) - 18*b1*rho*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*np.sin(zeta)**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (24*b1**2*psi0*rho**4*(-2 + 3*rho**2)*np.sin(theta)**2*np.sin(zeta)**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (6*b1**2*psi0*rho**2*(-2 + 3*rho**2)**2*np.sin(theta)**2*np.sin(zeta)**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*psi0*rho*(-6*b1*rho**2*np.cos(theta)*np.sin(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - 2*c2*rho*np.cos(2*theta)*np.sin(zeta))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*psi0*rho*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(18*a1*rho*np.cos(theta) + 18*b1*rho*np.cos(theta)*np.cos(zeta) + 2*c2*np.cos(2*theta)*np.cos(zeta))) - (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-18*a1*rho*np.sin(theta) - 18*b1*rho*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta))**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (2*psi0*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta))**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*b1*psi0*rho**2*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta)*(6*b1*rho**2*np.sin(theta)*np.sin(zeta) + b1*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*b1*psi0*rho**2*(-2 + 3*rho**2)*np.sin(theta)*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*np.sin(zeta)*(-(b1*rho*(-2 + 3*rho**2)*np.cos(theta)*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*np.sin(zeta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-6*b1*rho**2*np.cos(theta)*np.sin(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - 2*c2*rho*np.cos(2*theta)*np.sin(zeta)) + (-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(6*b1*rho**2*np.sin(theta)*np.sin(zeta) + b1*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(b1*rho*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta) + 2*c2*rho**2*np.sin(2*theta)*np.sin(zeta))))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (2*psi0*rho*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta))*(-(b1*rho*(-2 + 3*rho**2)*np.cos(theta)*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*np.sin(zeta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-6*b1*rho**2*np.cos(theta)*np.sin(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - 2*c2*rho*np.cos(2*theta)*np.sin(zeta)) + (-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(6*b1*rho**2*np.sin(theta)*np.sin(zeta) + b1*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(b1*rho*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta) + 2*c2*rho**2*np.sin(2*theta)*np.sin(zeta))))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2))/(mu0*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))) + ((-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*((-2*psi0*rho*((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - 4*c2*rho**2*np.cos(2*theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(a0*rho*np.sin(theta) + a1*rho*(-2 + 3*rho**2)*np.sin(theta) + b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 - (2*b1*psi0*rho**2*(-2 + 3*rho**2)*np.cos(zeta)*(-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*np.sin(theta))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*psi0*rho*(-(c1*np.cos(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - c2*rho**2*np.cos(2*theta)*np.cos(zeta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*b1**2*psi0*rho**3*(-2 + 3*rho**2)**2*np.sin(theta)**2*((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - 4*c2*rho**2*np.cos(2*theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(a0*rho*np.sin(theta) + a1*rho*(-2 + 3*rho**2)*np.sin(theta) + b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*np.sin(zeta)**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 + (2*b1**2*psi0*rho**3*(-2 + 3*rho**2)**2*np.cos(theta)*np.sin(theta)*np.sin(zeta)**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) - (2*psi0*rho*((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) - 4*c2*rho**2*np.cos(2*theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*(a0*rho*np.sin(theta) + a1*rho*(-2 + 3*rho**2)*np.sin(theta) + b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 4*c2*rho*np.cos(zeta)*np.sin(2*theta)) + (-(a0*np.cos(theta)) - 6*a1*rho**2*np.cos(theta) - a1*(-2 + 3*rho**2)*np.cos(theta) - 6*b1*rho**2*np.cos(theta)*np.cos(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta))**2)/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 + (2*psi0*rho*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta))*(b1*rho*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta) + 2*c2*rho**2*np.sin(2*theta)*np.sin(zeta)))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))) + (2*b1*psi0*rho**2*(-2 + 3*rho**2)*(-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*np.sin(theta)*np.sin(zeta)*(-(b1*rho*(-2 + 3*rho**2)*np.cos(theta)*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*np.sin(zeta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-6*b1*rho**2*np.cos(theta)*np.sin(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - 2*c2*rho*np.cos(2*theta)*np.sin(zeta)) + (-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(6*b1*rho**2*np.sin(theta)*np.sin(zeta) + b1*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(b1*rho*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta) + 2*c2*rho**2*np.sin(2*theta)*np.sin(zeta))))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2 + (2*psi0*rho*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(-(c1*np.sin(zeta)) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - c2*rho**2*np.cos(2*theta)*np.sin(zeta))*(-(b1*rho*(-2 + 3*rho**2)*np.cos(theta)*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))*np.sin(zeta)) - (-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(-6*b1*rho**2*np.cos(theta)*np.sin(zeta) - b1*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - 2*c2*rho*np.cos(2*theta)*np.sin(zeta)) + (-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))*(6*b1*rho**2*np.sin(theta)*np.sin(zeta) + b1*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta)) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(b1*rho*(-2 + 3*rho**2)*np.sin(theta)*np.sin(zeta) + 2*c2*rho**2*np.sin(2*theta)*np.sin(zeta))))/(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta)))**2))/(mu0*(-((-(a0*rho*np.cos(theta)) - a1*rho*(-2 + 3*rho**2)*np.cos(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta))*(a0*np.cos(theta) + 6*a1*rho**2*np.cos(theta) + a1*(-2 + 3*rho**2)*np.cos(theta) + 6*b1*rho**2*np.cos(theta)*np.cos(zeta) + b1*(-2 + 3*rho**2)*np.cos(theta)*np.cos(zeta) + 2*c2*rho*np.cos(2*theta)*np.cos(zeta))) + (-(a0*np.sin(theta)) - 6*a1*rho**2*np.sin(theta) - a1*(-2 + 3*rho**2)*np.sin(theta) - 6*b1*rho**2*np.cos(zeta)*np.sin(theta) - b1*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta))*(-(a0*rho*np.sin(theta)) - a1*rho*(-2 + 3*rho**2)*np.sin(theta) - b1*rho*(-2 + 3*rho**2)*np.cos(zeta)*np.sin(theta) - 2*c2*rho**2*np.cos(zeta)*np.sin(2*theta))))
        
        return np.stack((j0,j1,j2), axis=1)
    def gradp_vec_ana_cal(self, rtz):
        return np.cross(self.j_vec_ana_cal(rtz), self.B_vec_ana_cal(rtz), axis=1)

@dataclass
class model_mirror_full1(ana_model):
    """axis-sym mirror with periodic zeta.(FourierZernike Basis)
        With 3D shaping of the cross sections, zeta variation and line twist.
        For definition, refer to _get_modes
        R is only passed as an argument to R_lmn, not included in the analytical formula.
    """
    Psi0: float
    a0: float
    a1: float
    b1: float
    c1: float
    c2: float
    i0: float
    i2: float
    R: float
    def _get_modes(self):
        return {
            "R_lmn":{ (0,0,0):self.R, (1,1,0):self.a0, (3,1,0):self.a1, (3,1,1):self.b1, (0,0,1):self.c1, (2,2,1):self.c2},
            "Z_lmn":{ (1,-1,0):-self.a0, (3,-1,0):-self.a1, (3,-1,1):-self.b1 },
            "iota": {"modes": (0,2), "params": (self.i0, self.i2)},
        }
    def j_vec_ana_cal(self, rtz):
        raise NotImplementedError("j not implemented for this model")
    def gradp_vec_ana_cal(self, rtz):
        raise NotImplementedError("gradp not implemented for this model")
    def B_vec_ana_cal(self, rtz):
        rho = rtz[:, 0]
        theta = rtz[:, 1]
        zeta = rtz[:, 2]
        a0 = self.a0
        a1 = self.a1
        b1 = self.b1
        psi0 = self.Psi0 / np.pi / 2
        c1 = self.c1
        c2 = self.c2
        i0 = self.i0
        i2 = self.i2

        B0 = (2*psi0*rho*(-(rho*(i0 + i2*rho**2)*(a0 + a1*(-2 + 3*rho**2) + \
            (b1*(-2 + 3*rho**2) + \
            4*c2*rho*np.cos(theta))*np.cos(zeta))*np.sin(theta)) - \
            c1*np.sin(zeta) - b1*rho*(-2 + 3*rho**2)*np.cos(theta)*np.sin(zeta) - \
            c2*rho**2*np.cos(2*theta)*np.sin(zeta)))/(rho*np.cos(theta)*(a0 + \
            a1*(-2 + 3*rho**2) + b1*(-2 + \
            3*rho**2)*np.cos(zeta))*(2*c2*rho*np.cos(2*theta)*np.cos(zeta) + \
            np.cos(theta)*(a0 - 2*a1 + 9*a1*rho**2 + b1*(-2 + \
            9*rho**2)*np.cos(zeta))) + rho*(a0 + a1*(-2 + 9*rho**2) + b1*(-2 + \
            9*rho**2)*np.cos(zeta))*(a0 + a1*(-2 + 3*rho**2) + (b1*(-2 + \
            3*rho**2) + 4*c2*rho*np.cos(theta))*np.cos(zeta))*np.sin(theta)**2)
        B1 = (2*psi0*rho)/(rho*np.cos(theta)*(a0 + a1*(-2 + 3*rho**2) + b1*(-2 + \
            3*rho**2)*np.cos(zeta))*(2*c2*rho*np.cos(2*theta)*np.cos(zeta) + \
            np.cos(theta)*(a0 - 2*a1 + 9*a1*rho**2 + b1*(-2 + \
            9*rho**2)*np.cos(zeta))) + rho*(a0 + a1*(-2 + 9*rho**2) + b1*(-2 + \
            9*rho**2)*np.cos(zeta))*(a0 + a1*(-2 + 3*rho**2) + (b1*(-2 + \
            3*rho**2) + 4*c2*rho*np.cos(theta))*np.cos(zeta))*np.sin(theta)**2)
        B2 = (2*psi0*rho*(-(rho*(i0 + i2*rho**2)*np.cos(theta)*(a0 + a1*(-2 + \
            3*rho**2) + b1*(-2 + 3*rho**2)*np.cos(zeta))) + b1*rho*(-2 + \
            3*rho**2)*np.sin(theta)*np.sin(zeta)))/(rho*np.cos(theta)*(a0 + \
            a1*(-2 + 3*rho**2) + b1*(-2 + \
            3*rho**2)*np.cos(zeta))*(2*c2*rho*np.cos(2*theta)*np.cos(zeta) + \
            np.cos(theta)*(a0 - 2*a1 + 9*a1*rho**2 + b1*(-2 + \
            9*rho**2)*np.cos(zeta))) + rho*(a0 + a1*(-2 + 9*rho**2) + b1*(-2 + \
            9*rho**2)*np.cos(zeta))*(a0 + a1*(-2 + 3*rho**2) + (b1*(-2 + \
            3*rho**2) + 4*c2*rho*np.cos(theta))*np.cos(zeta))*np.sin(theta)**2)
        
        return np.stack((B0,B1,B2), axis=1)