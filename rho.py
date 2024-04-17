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

def ouverture_fichier(nomfichier):
    with open(nomfichier, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        x_1 = []
        x_2 = []
        i = 0
        for i in range(23):
            next(csvreader)
        for row in csvreader:
            x_1.append(float(row[1]))
            x_2.append(float(row[2]))
    return [x_1,x_2]

#Cuivre
b=1.58*10**(-3)
omega=2*np.pi*100*10**3
mu0=mu_0
e0=epsilon_0
er=1
a=10.43*10**(-3)
n=200
rho_approx=820*10**(-9)


def traitement_sinus(sinus) :
    # Générer une liste de valeurs de sinus pour la première fonction
    A = sinus

    T = np.linspace(0,len(A)/100000, len(A))

    # Créer une fonction d'interpolation
    f = interp1d(T, A, kind='cubic')

    # Générer une plage de temps pour évaluer la fonction
    temps_interp = T

    # Calculer les valeurs interpolées
    valeurs_interp = f(temps_interp)

    # Calculer la dérivée numérique
    dx = temps_interp[1] - temps_interp[0]
    derivee_interp = np.gradient(valeurs_interp, dx)

    # Trouver les zéros de la dérivée
    zeros_indices = np.where(np.diff(np.sign(derivee_interp)))[0]

    # Afficher les zéros de la dérivée
    zeros_temps = temps_interp[zeros_indices]
    zeros_valeurs = valeurs_interp[zeros_indices]
    print(len(zeros_valeurs))
    print(len(zeros_temps))
    return zeros_valeurs,zeros_temps

def phase(output):
    phaseurs = []
    temps_ref = 10
    for i in range(len(output)):
        Valeurs, temps = traitement_sinus(output[i])
        max = [l for l in Valeurs if l>0]
        min = [l for l in Valeurs if l<0]
        voltage = sum(max)/len(max)-sum(min)/len(min)
        x = list(Valeurs).index(max[temps_ref])
        dephasage = temps[x]
        phaseur = voltage*np.exp(-1j*2*np.pi*10000*dephasage)
        phaseurs.append(phaseur)
        print(voltage)
    return phaseurs

def f_re(xi,rho,mur):
    gamma = b*(xi**2+1j*omega*mur*mu0/rho-omega**2*mu0*mur*e0*er)**(1/2)
    eta = (xi**2-omega**2*mu0*e0)**(1/2)
    return np.real(np.sinc(xi/2)**2*(mur*eta*b*I(0,eta*b)*I(1,gamma)-gamma*I(1,eta*b)*I(0,gamma))/(mur*eta*b*K(0,eta*b)*I(1,gamma)+gamma*K(1,eta*b)*I(0,gamma))*K(1,eta*a)**2)

def f_im(xi,rho,mur):
    gamma = b*(xi**2+1j*omega*mur*mu0/rho-omega**2*mu0*mur*e0*er)**(1/2)
    eta = (xi**2-omega**2*mu0*e0)**(1/2)
    return np.imag(np.sinc(xi/2)**2*(mur*eta*b*I(0,eta*b)*I(1,gamma)-gamma*I(1,eta*b)*I(0,gamma))/(mur*eta*b*K(0,eta*b)*I(1,gamma)+gamma*K(1,eta*b)*I(0,gamma))*K(1,eta*a)**2)

def delta_R(x,z):
    return -omega*mu0*a**2*n**2*(quad(f_im, -40,40, args=(x,1,))[0])-np.real(z)
def delta_X(x,z):
    return omega*mu0*a**2*n**2*(quad(f_re, -40,40, args=(x,1,))[0])-np.imag(z)


sinus_solenoide = ouverture_fichier("C:\\Users\\alexi\\OneDrive - polymtl.ca\\H24\\données.lvm")[0]
sinus_resistance = ouverture_fichier("C:\\Users\\alexi\\OneDrive - polymtl.ca\\H24\\données.lvm")[1]



t = np.linspace(0,0.1,10000)

phaseurs = phase([sinus_solenoide,sinus_resistance])
print(phaseurs)


deltaz= 327*phaseurs[0]/phaseurs[1]

print(deltaz)
print(fsolve(delta_Z,[1.7*10**(-8),1], args=(deltaz,)))