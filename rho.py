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

def analyse_donnees(nomfichier):
    with open(nomfichier, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        x_1 = []
        x_2 = []
        i = 0
        for i in range(23):
            next(csvreader)
        for row in csvreader:
            x_1.append(float(row[1]))
            x_2.append(float(row[3]))
    return [x_1,x_2]

#Cuivre
b = 1.59*10**(-3)
omega = 2*np.pi*10**4
mu0 = mu_0
e0 = epsilon_0
er = 1
a = 10.43*10**(-3)/2
n = 458
rho_approx_cuivre = 1.7*10**(-8)
rho_approx_molyb = 5.7*10**(-8)

def traitement_sinus(sinus) :
    # Générer une liste de valeurs de sinus pour la première fonction
    A = sinus

    T = np.linspace(0,0.05, len(A))

    # Créer une fonction d'interpolation
    f = interp1d(T, A, kind='cubic')
#   
    # Générer une plage de temps pour évaluer la fonction
    x=np.linspace(0,0.05,100000)
    
    # Calculer les valeurs interpolées
    valeurs_interp = f(x)
    plt.plot(x,valeurs_interp)
    # Calculer la dérivée numérique
    dx = x[1] - x[0]

    derivee_interp = np.gradient(valeurs_interp, dx)

    # Trouver les zéros de la dérivée
    zeros_indices_0 = np.where(np.diff(np.sign(derivee_interp)))[0]
    zeros_indices = [zeros_indices_0[i] for i in range(len(zeros_indices_0)-1) if float(x[zeros_indices_0[i+1]]-x[zeros_indices_0[i]])>0.000025]

    # Afficher les zéros de la dérivée
    zeros_temps = x[zeros_indices]
    zeros_valeurs = valeurs_interp[zeros_indices]
    
    return zeros_valeurs,zeros_temps

def phase(output):
    phaseurs = []
    for i in range(len(output)):
        Valeurs, temps = traitement_sinus(output[i])
        max = [l for l in Valeurs if l>0]
        min = [l for l in Valeurs if l<0]
        voltage = (sum(max)/len(max)-sum(min)/len(min))/2

        t = list(Valeurs).index(max[int(len(max)/2)])
        plt.scatter(temps,Valeurs)
        dephasage = temps[t]
        phaseur = voltage*np.exp(-1j*omega*dephasage)      
        phaseurs.append(phaseur)
    return phaseurs

def f_integrale(xi,rho=rho_approx_cuivre,mur=1):
    gamma = b*(xi**2+1j*omega*mur*mu0/rho-omega**2*mu0*mur*e0*er)**(1/2)
    eta = (xi**2-omega**2*mu0*e0)**(1/2)
    return np.sinc(xi/(2*np.pi))**2*(mur*eta*b*I(0,eta*b)*I(1,gamma)-gamma*I(1,eta*b)*I(0,gamma))/(mur*eta*b*K(0,eta*b)*I(1,gamma)+gamma*K(1,eta*b)*I(0,gamma))*K(1,eta*a)**2

def delta_Z(x,z):
        delta_Z = 1j*omega*mu0*a**2*n**2*quad(f_integrale, -80,80, args=(x[0],x[1],),complex_func=True,limit =100)[0]
        return (np.real(delta_Z-z),np.imag(delta_Z-z)) 


sinus_solenoide_cuivre = analyse_donnees("C:\\Users\\alexi\\OneDrive - polymtl.ca\\H24\\données\\Données_cuivre.lvm")[0]
sinus_resistance_cuivre = analyse_donnees("C:\\Users\\alexi\\OneDrive - polymtl.ca\\H24\\données\\Données_cuivre.lvm")[1]


phaseurs_cuivre = phase([sinus_solenoide_cuivre,sinus_resistance_cuivre])


z_cuivre_voulu = complex(delta_Z([rho_approx_cuivre,1],0)[0],delta_Z([rho_approx_cuivre,1],0)[0])
  
print(np.abs(z_cuivre_voulu),np.angle(z_cuivre_voulu))

x = np.linspace(0,0.05,10000)
plt.scatter(x,sinus_solenoide_cuivre)
t=np.linspace(0,0.05,100000)
y=np.exp(1j*omega*t)
plt.plot(t,y*phaseurs_cuivre[1])
plt.xlim([0.025,0.025+4/10000])
plt.show()

z_cuivre = 328*phaseurs_cuivre[0]/phaseurs_cuivre[1]/15.584

print(fsolve(delta_Z,[rho_approx_cuivre,1], args=(z_cuivre,)))