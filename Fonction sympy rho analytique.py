import sympy as sp
import sympy.integrals.transforms as transform
from sympy.functions.special.bessel import besseli as I
from sympy.functions.special.bessel import besselk as K
from sympy.solvers import solve
from sympy import Eq
import numpy as np
from scipy.integrate import quad
from sympy import print_latex
from scipy.constants import mu_0
from scipy.constants import epsilon_0


b, xi, omega, mur, mu0, rho, e0, er, a ,n, r = sp.symbols("b,\\xi, \\omega, \\mu_r,\\mu_0,\\rho, \\epsilon_0,\\epsilon_r,a,n,\\infty")

gammaprint = b*(xi**2+1j*omega*mur*mu0/rho-omega**2*mu0*mur*e0*er)**(1/2)
etaprint = (xi**2-omega**2*mu0*e0)**(1/2)

def f(xi):
    gamma = b*(xi**2+1j*omega*mur*mu0/rho-omega**2*mu0*mur*e0*er)**(1/2)
    eta = (xi**2-omega**2*mu0*e0)**(1/2)
    return sp.sin(xi/2)**2/(xi/2)**2*(mur*eta*b*I(0,eta*b)*I(1,gamma)-gamma*I(1,eta*b)*I(0,gamma))/(mur*eta*b*K(0,eta*b)*I(1,gamma)+gamma*K(1,eta*b)*I(0,gamma))*K(1,eta*a)**2

z = 1j*omega*mu0*a**2*n**2*sp.Integral(f(xi), (xi,-r,r))
zprime = sp.diff(z, rho)

marde = z/zprime

rho_result = zprime.evalf(5,subs={b:1.58*10**(-3),omega:2*np.pi*100*10**3,mur:1,mu0:mu_0,e0:epsilon_0,er:1,a:10.43*10**(-3),n:1,rho:820*10**(-9),r:np.inf})
print(rho_result)


"""print_latex(z)
print("done")
print_latex(zprime)
print("done")
print_latex(marde)
print("eta")
print_latex(etaprint)
print("gamma")
print_latex(gammaprint)
print("gammaprime")
print_latex(sp.diff(gammaprint,rho))"""



