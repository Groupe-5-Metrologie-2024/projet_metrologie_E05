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
b = 1.59*10**(-3)
omega = 2*np.pi*10**4
mu0 = mu_0
e0 = epsilon_0
er = 1
d = 15.75*10**(-3)/2-(9.78*10**(-3)/2+1*10**(-3))
n = 458
a = 15.75*10**(-3)/2-d/2
l=50.15*10**(-3)-2.02*10**(-3)-2.08*10**(-3)

rho_approx_cuivre = 1.7*10**(-8)
rho_approx_molyb = 5.7*10**(-8)

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
        plt.scatter(temps,Valeurs)
        dephasage = temps[t]
        phaseur = voltage*np.exp(-1j*omega*dephasage)      
        phaseurs.append(phaseur)
    return phaseurs

def f_integrale(xi,rho=rho_approx_cuivre,mur=1):
    gamma = b*(xi**2+1j*omega*mur*mu0/rho-omega**2*mu0*mur*e0*er)**(1/2)
    eta = (xi**2-omega**2*mu0*e0)**(1/2)
    return np.sinc(xi/(2*np.pi))**2*(mur*eta*b*I(0,eta*b)*I(1,gamma)-gamma*I(1,eta*b)*I(0,gamma))/(mur*eta*b*K(0,eta*b)*I(1,gamma)+gamma*K(1,eta*b)*I(0,gamma))*K(1,eta*a)**2

def f_integrale_struve(xi,rho=rho_approx_cuivre,mur=1):
    gamma = b*(xi**2+1j*omega*mur*mu0/rho-omega**2*mu0*mur*e0*er)**(1/2)
    eta = (xi**2-omega**2*mu0*e0)**(1/2)
    l_plus = a+d/2
    l_moins = a-d/2
    F = (np.pi/(2*eta))*l_plus*(K(1,l_plus*eta)*L(0,l_plus*eta)+L(1,l_plus*eta)*K(0,l_plus*eta)) - (np.pi/(2*eta))*l_moins*(K(1,l_moins*eta)*L(0,l_moins*eta)+L(1,l_moins*eta)*K(0,l_moins*eta))
    return (2/xi)**2*np.sin(l*xi/2)**2*(mur*eta*b*I(0,eta*b)*I(1,gamma)-gamma*I(1,eta*b)*I(0,gamma))/(mur*eta*b*K(0,eta*b)*I(1,gamma)+gamma*K(1,eta*b)*I(0,gamma))*F**2

def delta_Z(x,z):
        delta_Z = 1j*omega*(n/d)**2*quad(f_integrale, -80,80, args=(x[0],x[1],),complex_func=True,limit =1000,points=[0])[0]
        return (np.real(delta_Z-z),np.imag(delta_Z-z)) 

def delta_Z_struve(x,z):
        delta_Z = 1j*omega*(n/d)**2*quad(f_integrale, -300,300, args=(x[0],x[1],),complex_func=True,limit =1000,points=[0])[0]
        return (np.real(delta_Z-z),np.imag(delta_Z-z)) 



sinus_cuivre = analyse_donnees("C:\\Users\\alexi\\OneDrive - polymtl.ca\\H24\\données\\Données_cuivre.lvm")
sinus_molyb = analyse_donnees("C:\\Users\\alexi\\OneDrive - polymtl.ca\\H24\\données\\Données_molyb.lvm")
sinis_calib = analyse_donnees("C:\\Users\\alexi\\OneDrive - polymtl.ca\\H24\\données\\Données_calib.lvm")
sinus_molyb_2 = analyse_donnees("C:\\Users\\alexi\\OneDrive - polymtl.ca\\H24\\données\\Données_molyb_2.lvm")

phaseurs_cuivre = phase(sinus_cuivre)
phaseurs_molyb = phase(sinus_molyb)
phaseurs_calib = phase(sinis_calib)
phaseurs_molyb_2 = phase(sinus_molyb_2)

x = np.linspace(0,0.05,10000)
t=np.linspace(0,0.05,100000)
y=np.exp(1j*omega*t)
plt.plot(t,y*phaseurs_cuivre[0])
plt.xlim([0.025,0.025+4/10000])

poo = phaseurs_cuivre[0]/phaseurs_cuivre[1]/15.584
z_cuivre = 2*(2.6030+1j*omega*555.63*10**(-6))*poo/(1-poo)

poo_molyb = phaseurs_molyb[0]/phaseurs_molyb[1]/15.584
z_molyb = 2*(2.5777+1j*omega*550.63*10**(-6))*poo_molyb/(1-poo_molyb)

poo_molyb_2 = phaseurs_molyb_2[0]/phaseurs_molyb_2[1]/15.584
z_molyb_2 = 2*(2.5777+1j*omega*550.63*10**(-6))*poo_molyb_2/(1-poo_molyb_2)

print(fsolve(delta_Z_struve,[rho_approx_cuivre,1],args=z_cuivre))