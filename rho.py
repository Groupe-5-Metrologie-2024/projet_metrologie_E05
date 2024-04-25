from scipy.constants import mu_0
from scipy.constants import epsilon_0
import numpy as np
from scipy.integrate import quad
from scipy.special import iv as I
from scipy.special import kv as K
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import csv
from scipy.interpolate import interp1d
from scipy.special import modstruve as L

def analyse_donnees(nomfichier):
    with open(nomfichier, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        x_1 = []
        x_2 = []
        t = []
        i = 0
        for i in range(23):
            next(csvreader)
        for row in csvreader:
            x_1.append(float(row[1]))
            x_2.append(float(row[2]))
            t.append(float(row[0]))
    return [x_1,x_2,t]


#Cuivre
b = 3.16e-3/2
omega = 2*np.pi*10**4
mu0 = mu_0
e0 = epsilon_0
er = 1
d = 15.62e-3/2-(9.84e-3/2+1e-3)
n=456
a = 15.62e-3/2-d/2
l=45.05e-3

rho_approx_cuivre = 1.7*10**(-8)
rho_approx_molyb = 5*10**(-8)
rho_approx_invar = 8.2*10**(-7)

def traitement_sinus(sinus,t) :
    # Générer une liste de valeurs de sinus pour la première fonction
    A = sinus

    T = t

    # Créer une fonction d'interpolation
    f = interp1d(T, A, kind='cubic')
#   
    # Générer une plage de temps pour évaluer la fonction
    x=np.linspace(0,t[len(t)-1],len(t)*10)
    
    # Calculer les valeurs interpolées
    valeurs_interp = f(x)
    #plt.plot(x,valeurs_interp)
    # Calculer la dérivée numérique
    dx = x[1] - x[0]

    derivee_interp = np.gradient(valeurs_interp, dx)

    # Trouver les zéros de la dérivée
    zeros_indices_0 = np.where(np.diff(np.sign(derivee_interp)))[0]
    zeros_indices = [zeros_indices_0[i] for i in range(len(zeros_indices_0)-1) if float(x[zeros_indices_0[i+1]]-x[zeros_indices_0[i]])>1.5/omega]

    # Afficher les zéros de la dérivée
    zeros_temps = x[zeros_indices]
    zeros_valeurs = valeurs_interp[zeros_indices]
    
    return zeros_valeurs,zeros_temps

def phase(output):
    phaseurs = []
    for i in range(len(output)-1):
        Valeurs, temps = traitement_sinus(output[i],output[2])
        max = [l for l in Valeurs if l>0]
        min = [l for l in Valeurs if l<0]
        voltage = (sum(max)/len(max)-sum(min)/len(min))/2

        t = list(Valeurs).index(max[int(len(max)/2)])
        dephasage = temps[t]
        phaseur = voltage*np.exp(-1j*omega*dephasage)      
        phaseurs.append(phaseur)
    return phaseurs
     
def f_integrale(xi,rho=rho_approx_cuivre,mur=1):
    gamma = b*(xi**2+1j*omega*mur*mu0/rho-omega**2*mu0*mur*e0*er)**(1/2)
    eta = (xi**2-omega**2*mu0*e0)**(1/2)
    return np.sinc(xi*l/(2*np.pi))**2*(mur*eta*b*I(0,eta*b)*I(1,gamma)-gamma*I(1,eta*b)*I(0,gamma))/(mur*eta*b*K(0,eta*b)*I(1,gamma)+gamma*K(1,eta*b)*I(0,gamma))*K(1,eta*a)**2

def f_integrale_struve(xi,rho=rho_approx_cuivre,mur=1):
    gamma = b*(xi**2+1j*omega*mur*mu0/rho-omega**2*mu0*mur*e0*er)**(1/2)
    eta = (xi**2-omega**2*mu0*e0)**(1/2)
    l_plus = a+d/2
    l_moins = a-d/2
    F = (np.pi/(2*eta))*l_plus*(K(1,l_plus*eta)*L(0,l_plus*eta)+L(1,l_plus*eta)*K(0,l_plus*eta)) - (np.pi/(2*eta))*l_moins*(K(1,l_moins*eta)*L(0,l_moins*eta)+L(1,l_moins*eta)*K(0,l_moins*eta))
    return (2/xi)**2*np.sin(l*xi/2)**2*(mur*eta*b*I(0,eta*b)*I(1,gamma)-gamma*I(1,eta*b)*I(0,gamma))/(mur*eta*b*K(0,eta*b)*I(1,gamma)+gamma*K(1,eta*b)*I(0,gamma))*F**2

def delta_Z(x,z):
        delta_Z = 1j*omega*mu0*a**2*n**2*quad(f_integrale, -80,80, args=(x[0],x[1],),complex_func=True,limit =1000,points=[0])[0]
        return (np.real(delta_Z-z),np.imag(delta_Z-z)) 

def delta_Z_struve(x,z):
        delta_Z = 2j*omega*mu0*(n/(d*l))**2*quad(f_integrale_struve, 0,300, args=(x[0],x[1],),complex_func=True,limit =1000,points=[0])[0]
        return (np.real(delta_Z-z),np.imag(delta_Z-z)) 

z1 = 2.5398+1j*omega*549.03e-6
zcuivre = 2.5698+1j*omega*548.16e-6
zmolyb = 2.5516+1j*omega*548.94e-6
zinvar = 2.5529 + 1j*omega*568.46e-6
zinvar_sans_aimant = 9.4555 + 1j*omega*3.1918e-3

z2 = 2.5912+1j*omega*548.94e-6
zcuivre_2= 2.6226+1j*omega*548.07e-6
zmolyb_2 = 2.6058 +1j*548.82e-6
zinvar_sans_aimant_2 = 9.5560+ 1j*3.2021e-3*omega
zinvar_2 = 2.6107+1j*omega*574.96e-6

sinus_cuivre = analyse_donnees(f"C:\\Users\\alexi\\OneDrive - polymtl.ca\\H24\\données\\Données_cuivre_10000.lvm")
sinus_calib = analyse_donnees(f"C:\\Users\\alexi\\OneDrive - polymtl.ca\\H24\\données\\Données_calib.lvm")
sinus_molyb = analyse_donnees(f"C:\\Users\\alexi\\OneDrive - polymtl.ca\\H24\\données\\Données_molyb.lvm")
sinus_invar = analyse_donnees(f"C:\\Users\\alexi\\OneDrive - polymtl.ca\\H24\\données\\Données_invar.lvm")
sinus_calib_2 = analyse_donnees(f"C:\\Users\\alexi\\OneDrive - polymtl.ca\\H24\\données\\Données_calib_2.lvm")

plt.scatter(sinus_cuivre[2],sinus_cuivre[0])
plt.xlim(max(sinus_cuivre[2])/2,max(sinus_cuivre[2])/2+2*np.pi/omega*4)
plt.show()

z_calib = z2-z1

phaseurs_calib = phase(sinus_calib)

phaseurs_cuivre = phase(sinus_cuivre)
poo_cuivre = -(phaseurs_cuivre[0])/phaseurs_cuivre[1]/15.32

phaseurs_molyb = phase(sinus_molyb)
poo_molyb = -(phaseurs_molyb[0])/phaseurs_molyb[1]/15.32

phaseurs_invar = phase(sinus_invar)
poo_invar = -(phaseurs_invar[0])/phaseurs_invar[1]/3

z_invar = 2*(1j*omega*550.63e-6)*poo_invar/(1-poo_invar)+z_calib
z_cuivre = 2*(2.5777+1j*omega*550.63e-6)*poo_cuivre/(1-poo_cuivre)+z_calib
z_molyb = 2*(2.53e-2+1j*omega*550.63e-6)*poo_molyb/(1-poo_molyb)+z_calib

x = fsolve(delta_Z_struve,[rho_approx_cuivre,1],args = (z_cuivre))[0]
print(x)