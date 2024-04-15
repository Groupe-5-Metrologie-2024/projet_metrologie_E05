from scipy.constants import mu_0
from scipy.constants import epsilon_0
import numpy as np
from scipy.integrate import quad
from scipy.special import iv as I
from scipy.special import kv as K
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

#Cuivre
b=1.58*10**(-3)
omega=2*np.pi*100*10**3
mur=1
mu0=mu_0
e0=epsilon_0
er=1
a=10.43*10**(-3)
n=1
rho_approx=820*10**(-9)


def f_re(xi,rho):
    gamma = b*(xi**2+1j*omega*mur*mu0/rho-omega**2*mu0*mur*e0*er)**(1/2)
    eta = (xi**2-omega**2*mu0*e0)**(1/2)
    return np.real(np.sinc(xi/2)**2*(mur*eta*b*I(0,eta*b)*I(1,gamma)-gamma*I(1,eta*b)*I(0,gamma))/(mur*eta*b*K(0,eta*b)*I(1,gamma)+gamma*K(1,eta*b)*I(0,gamma))*K(1,eta*a)**2)

def f_im(xi,rho):
    gamma = b*(xi**2+1j*omega*mur*mu0/rho-omega**2*mu0*mur*e0*er)**(1/2)
    eta = (xi**2-omega**2*mu0*e0)**(1/2)
    return np.imag(np.sinc(xi/2)**2*(mur*eta*b*I(0,eta*b)*I(1,gamma)-gamma*I(1,eta*b)*I(0,gamma))/(mur*eta*b*K(0,eta*b)*I(1,gamma)+gamma*K(1,eta*b)*I(0,gamma))*K(1,eta*a)**2)

def delta_R(rho,z):
    return -omega*mu0*a**2*n**2*(quad(f_im, -40,40, args=(rho,))[0])-np.real(z)

def delta_X(rho,z):
    return omega*mu0*a**2*n**2*(quad(f_re, -40,40, args=(rho,))[0])-np.imag(z)

def solve_z(z):
    i=0
    while True:
        x = fsolve(delta_R, [10**(-10+i)],args = (z,))
        y = fsolve(delta_X,x,args = (z,))
        if np.isclose(x,y):
            return x
        else:
            i+=1

print(solve_z(3308.8175810215957-1934.9982587670297j))
