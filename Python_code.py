#Modules ---------------------------------------------------------------------------------------------------------------
import csv
import matplotlib.pyplot as plt
import statistics as stat
import scipy
import numpy as np
import os
import time
from scipy.interpolate import interp1d
from scipy.constants import mu_0
from scipy.constants import epsilon_0
from scipy.integrate import quad
from scipy.special import iv as I
from scipy.special import kv as K
from scipy.special import modstruve as L
from scipy.optimize import fsolve,curve_fit

print('')
print('')
print('#----------------- Menu d\'activation des programmes DEBUT')
print('')
print('')

# 1 = on
# 0 = off

Activation_Production_CSV_rho = 1
Activation_Graphique_résistivité_Température = 1


if Activation_Production_CSV_rho == 1 :
    print('Activation_Production_CSV_rho','|','on')
else :
    print('Activation_Production_CSV_rho','|','off')

if Activation_Graphique_résistivité_Température == 1 :
    print('Activation_Graphique_Resistivite_Temperature','|','on')
else :
    print('Activation_Graphique_Resistivite_Temperature','|','off')


print('')
print('')
print('#----------------- Menu d\'activation des programmes FIN')
print('')
print('')
#----------------- Modification generales DEBUT

#Copier et remplacer le chemin relatif vers le dossier Données.

Chemin_relatif = "Données"

#CODE 1
if Activation_Production_CSV_rho == 1 :
    Nom_materiau_utilisé = "invar"

#CODE 2
if Activation_Graphique_résistivité_Température == 1 :  
    Nom_materiau_utilisé = "invar"


#----- Données -----#
b = 3.16e-3/2
omega = 2*np.pi*10**3
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

z_calib = z2-z1

#----------------- Modification generales FIN


#fonctions DEBUT ---------------------------------------------------------------------------------------------------------------

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

def ouverture_fichier(nomfichier):
    with open(nomfichier, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        xc = []
        i = 0
        for row in csvreader:
            xc.append(float(row[1]))
            i += 1
            if i >= 100000:
                break
    return xc

def list_to_csv(input_list, file_name):

    # Obtenir le chemin du dossier actuel
    current_directory = os.getcwd()
    # Combiner le chemin du dossier avec le nom du fichier
    file_path = os.path.join(current_directory, file_name)
    
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in input_list:
            writer.writerow([row])

def attendre_fichier(nom_fichier, delai_attente=1):

    while not os.path.exists(nom_fichier):
        print(f"En attente du fichier {nom_fichier}...")
        time.sleep(delai_attente)

def conserver_premieres_lignes(fichier_entree, nombre_lignes_conservees, fichier_sortie):
    # Ouvrir le fichier d'entrée en mode lecture et le fichier de sortie en mode écriture
    with open(fichier_entree, 'r', newline='') as csv_entree, open(fichier_sortie, 'w', newline='') as csv_sortie:
        lecteur_csv = csv.reader(csv_entree)
        ecrivain_csv = csv.writer(csv_sortie)
        
        # Parcourir chaque ligne du fichier CSV
        for indice_ligne, ligne in enumerate(lecteur_csv, start=1):
            # Écrire la ligne dans le fichier de sortie si elle est avant ou à la ligne spécifiée
            if indice_ligne <= nombre_lignes_conservees:
                ecrivain_csv.writerow(ligne)
            else:
                # Si on a atteint la ligne spécifiée, sortir de la boucle
                break

def modifier_virgule_csv(fichier_entree, fichier_sortie):

    with open(fichier_entree, 'r', newline='') as csv_entree, open(fichier_sortie, 'w', newline='') as csv_sortie:
        lecteur_csv = csv.reader(csv_entree)
        ecrivain_csv = csv.writer(csv_sortie)
        
        # Parcourir chaque ligne du fichier CSV
        for ligne in lecteur_csv:
            ligne_modifiee = []
            # Parcourir chaque élément de la ligne
            for element in ligne:
                try:
                    # Tente de convertir l'élément en un nombre
                    nombre = float(element)
                    # Si la conversion réussit, ajoute une virgule devant le nombre
                    ligne_modifiee.append(',' + element)
                except ValueError:
                    # Si la conversion échoue, conserve l'élément tel quel
                    ligne_modifiee.append(element)
            
            # Écrire la ligne modifiée dans le fichier de sortie
            csv_sortie.write(','.join(ligne_modifiee) + '\n')

def supprimer_fichier(fichier):
    try:
        os.remove(fichier)
        print(f"Le fichier {fichier} a été supprimé avec succès.")
    except FileNotFoundError:
        print(f"Le fichier {fichier} n'existe pas.")
    except Exception as e:
        print(f"Une erreur s'est produite lors de la suppression du fichier : {e}")

def supprimer_premiere_ligne(fichier_entree, fichier_sortie):
    lignes_restantes = []
    with open(fichier_entree, 'r', newline='') as csv_entree:
        lecteur_csv = csv.reader(csv_entree)
        # Ignorer la première ligne
        next(lecteur_csv)
        # Collecter les lignes restantes
        for ligne in lecteur_csv:
            lignes_restantes.append(ligne)
    
    # Écrire les lignes restantes dans le fichier de sortie
    with open(fichier_sortie, 'w', newline='') as csv_sortie:
        ecrivain_csv = csv.writer(csv_sortie)
        ecrivain_csv.writerows(lignes_restantes)

def list_to_csv(input_list, file_name):

    # Obtenir le chemin du dossier actuel
    current_directory = os.getcwd()
    # Combiner le chemin du dossier avec le nom du fichier
    file_path = os.path.join(current_directory, file_name)
    
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in input_list:
            writer.writerow([row])

def phase(output):
    phaseurs = []
    res = []
    alpha = []
    for i in range(len(output)-1):
        def fit(x,V,phi):
            return V*np.cos(omega*x+phi)
        param = curve_fit(fit,output[2],output[i])
        y = np.array([fit(i,param[0][0],param[0][1]) for i in output[2]])
        std = np.std(output[i]-y)
        phaseurs.append(param[0][0]*np.exp(1j*param[0][1]))
        res.append(std)
        alpha.append(param[1])
    return phaseurs,res,alpha

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
        delta_Z = 1j*omega*mu0*a**2*n**2*quad(f_integrale, -80,80, args=(x[0],1,),complex_func=True,limit =1000,points=[0])[0]
        return np.real(delta_Z-z)

def delta_Z_struve(x,z=0):
        delta_Z = 2j*omega*mu0*(n/(d*l))**2*quad(f_integrale_struve, 0,300, args=(x[0],x[1],),complex_func=True,limit =1000,points=[0])[0]
        return (np.real(delta_Z-z),np.imag(delta_Z-z)) 

def delta_Z_struve_real(x,z=0):
    delta_Z = 2j*omega*mu0*(n/(d*l))**2*quad(f_integrale_struve, 0,200, args=(x,1,),complex_func=True,limit =1000,points=[0])[0]
    return np.imag(delta_Z-z)

#fonction partielles intégré
def f_integrale_struve_b(xi,b,rho=rho_approx_cuivre,mur=1):
    gamma = b*(xi**2+1j*omega*mur*mu0/rho-omega**2*mu0*mur*e0*er)**(1/2)
    eta = (xi**2-omega**2*mu0*e0)**(1/2)
    l_plus = a+d/2
    l_moins = a-d/2
    F = (np.pi/(2*eta))*l_plus*(K(1,l_plus*eta)*L(0,l_plus*eta)+L(1,l_plus*eta)*K(0,l_plus*eta)) - (np.pi/(2*eta))*l_moins*(K(1,l_moins*eta)*L(0,l_moins*eta)+L(1,l_moins*eta)*K(0,l_moins*eta))
    return (2/xi)**2*np.sin(l*xi/2)**2*(mur*eta*b*I(0,eta*b)*I(1,gamma)-gamma*I(1,eta*b)*I(0,gamma))/(mur*eta*b*K(0,eta*b)*I(1,gamma)+gamma*K(1,eta*b)*I(0,gamma))*F**2

def f_integrale_struve_a(xi,a,rho=rho_approx_cuivre,mur=1):
    gamma = b*(xi**2+1j*omega*mur*mu0/rho-omega**2*mu0*mur*e0*er)**(1/2)
    eta = (xi**2-omega**2*mu0*e0)**(1/2)
    l_plus = a+d/2
    l_moins = a-d/2
    F = (np.pi/(2*eta))*l_plus*(K(1,l_plus*eta)*L(0,l_plus*eta)+L(1,l_plus*eta)*K(0,l_plus*eta)) - (np.pi/(2*eta))*l_moins*(K(1,l_moins*eta)*L(0,l_moins*eta)+L(1,l_moins*eta)*K(0,l_moins*eta))
    return (2/xi)**2*np.sin(l*xi/2)**2*(mur*eta*b*I(0,eta*b)*I(1,gamma)-gamma*I(1,eta*b)*I(0,gamma))/(mur*eta*b*K(0,eta*b)*I(1,gamma)+gamma*K(1,eta*b)*I(0,gamma))*F**2

def f_integrale_struve_l(xi,l,rho=rho_approx_cuivre,mur=1):
    gamma = b*(xi**2+1j*omega*mur*mu0/rho-omega**2*mu0*mur*e0*er)**(1/2)
    eta = (xi**2-omega**2*mu0*e0)**(1/2)
    l_plus = a+d/2
    l_moins = a-d/2
    F = (np.pi/(2*eta))*l_plus*(K(1,l_plus*eta)*L(0,l_plus*eta)+L(1,l_plus*eta)*K(0,l_plus*eta)) - (np.pi/(2*eta))*l_moins*(K(1,l_moins*eta)*L(0,l_moins*eta)+L(1,l_moins*eta)*K(0,l_moins*eta))
    return (2/xi)**2*np.sin(l*xi/2)**2*(mur*eta*b*I(0,eta*b)*I(1,gamma)-gamma*I(1,eta*b)*I(0,gamma))/(mur*eta*b*K(0,eta*b)*I(1,gamma)+gamma*K(1,eta*b)*I(0,gamma))*F**2

def f_integrale_struve_d(xi,d,rho=rho_approx_cuivre,mur=1):
    gamma = b*(xi**2+1j*omega*mur*mu0/rho-omega**2*mu0*mur*e0*er)**(1/2)
    eta = (xi**2-omega**2*mu0*e0)**(1/2)
    l_plus = a+d/2
    l_moins = a-d/2
    F = (np.pi/(2*eta))*l_plus*(K(1,l_plus*eta)*L(0,l_plus*eta)+L(1,l_plus*eta)*K(0,l_plus*eta)) - (np.pi/(2*eta))*l_moins*(K(1,l_moins*eta)*L(0,l_moins*eta)+L(1,l_moins*eta)*K(0,l_moins*eta))
    return (2/xi)**2*np.sin(l*xi/2)**2*(mur*eta*b*I(0,eta*b)*I(1,gamma)-gamma*I(1,eta*b)*I(0,gamma))/(mur*eta*b*K(0,eta*b)*I(1,gamma)+gamma*K(1,eta*b)*I(0,gamma))*F**2

#fonction partielles deltaz
def delta_Z_struve_real_l(x=rho_approx_cuivre,l=l,z=0):
    integ = quad(f_integrale_struve_l, 0,200, args=(l,x,1,),complex_func=True,limit =1000,points=[0])
    delta_Z = 2j*omega*mu0*(n/(d*l))**2*integ[0]
    return np.imag(delta_Z-z)

def delta_Z_struve_real_d(x=rho_approx_cuivre,d=d,z=0):
    delta_Z = 2j*omega*mu0*(n/(d*l))**2*quad(f_integrale_struve_d, 0,200, args=(d,x,1,),complex_func=True,limit =1000,points=[0])[0]
    return np.imag(delta_Z-z)

def delta_Z_struve_real_a(x=rho_approx_cuivre,a=a,z=0):
    delta_Z = 2j*omega*mu0*(n/(d*l))**2*quad(f_integrale_struve_a, 0,200, args=(a,x,1,),complex_func=True,limit =1000,points=[0])[0]
    return np.imag(delta_Z-z)

def delta_Z_struve_real_b(x=rho_approx_cuivre,b=b,z=0):
    delta_Z = 2j*omega*mu0*(n/(d*l))**2*quad(f_integrale_struve_b, 0,200, args=(b,x,1,),complex_func=True,limit =1000,points=[0])[0]
    return np.imag(delta_Z-z)

#fonctions partielles invar
def delta_Z_struve_real_l_invar(x=rho_approx_cuivre,l=l,z=0):
    integ = quad(f_integrale_struve_l, 0,200, args=(l,x[0],x[1],),complex_func=True,limit =1000,points=[0])
    delta_Z = 2j*omega*mu0*(n/(d*l))**2*integ[0]
    return np.real(delta_Z-z),np.imag(delta_Z-z)

def delta_Z_struve_real_d_invar(x=rho_approx_cuivre,d=d,z=0):
    delta_Z = 2j*omega*mu0*(n/(d*l))**2*quad(f_integrale_struve_d, 0,200, args=(d,x[0],x[1],),complex_func=True,limit =1000,points=[0])[0]
    return np.real(delta_Z-z),np.imag(delta_Z-z)

def delta_Z_struve_real_a_invar(x=rho_approx_cuivre,a=a,z=0):
    delta_Z = 2j*omega*mu0*(n/(d*l))**2*quad(f_integrale_struve_a, 0,200, args=(a,x[0],x[1],),complex_func=True,limit =1000,points=[0])[0]
    return np.real(delta_Z-z),np.imag(delta_Z-z)

def delta_Z_struve_real_b_invar(x=rho_approx_cuivre,b=b,z=0):
    delta_Z = 2j*omega*mu0*(n/(d*l))**2*quad(f_integrale_struve_b, 0,200, args=(b,x[0],x[1],),complex_func=True,limit =1000,points=[0])[0]
    return np.real(delta_Z-z),np.imag(delta_Z-z)

def incertitude(z,phaseurs,alpha_res=0, alpha_sin = 0, mu =1):
    dx = 1e-10
    #dimensions
    incertitude_l = (fsolve(delta_Z_struve_real_l,[rho_approx_cuivre],args = (l+dx,z))-fsolve(delta_Z_struve_real_l,[rho_approx_cuivre],args = (l,z)))/dx*0.01e-3/np.sqrt(12)
    incertitude_b = (fsolve(delta_Z_struve_real_b,[rho_approx_cuivre],args = (b+dx,z))-fsolve(delta_Z_struve_real_b,[rho_approx_cuivre],args = (b,z)))/dx*0.01e-3/np.sqrt(12)
    incertitude_a = (fsolve(delta_Z_struve_real_a,[rho_approx_cuivre],args = (a+dx,z))-fsolve(delta_Z_struve_real_a,[rho_approx_cuivre],args = (a,z)))/dx*0.01e-3/np.sqrt(12)
    incertitude_d = (fsolve(delta_Z_struve_real_d,[rho_approx_cuivre],args = (d+dx,z))-fsolve(delta_Z_struve_real_d,[rho_approx_cuivre],args = (d,z)))/dx*0.01e-3/np.sqrt(12)
    incertitude_totale = incertitude_a**2+incertitude_b**2+incertitude_d**2+incertitude_l**2
    #Tension
    incertitude_mix = (fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z+1j*dx))-fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z)))/dx*alpha_res
    incertitude_totale += incertitude_mix**2
    return incertitude_totale[0]

def incertitude_invar(z,phaseurs,alpha_res=0, alpha_sin = 0, mu =1):
    dx = 1e-10
    #dimensions
    incertitude_l = (fsolve(delta_Z_struve_real_l_invar,[rho_approx_cuivre, 100],args = (l+dx,z))-fsolve(delta_Z_struve_real_l_invar,[rho_approx_cuivre, 100],args = (l,z)))/dx*0.01e-3/np.sqrt(12)
    incertitude_b = (fsolve(delta_Z_struve_real_b_invar,[rho_approx_cuivre, 100],args = (b+dx,z))-fsolve(delta_Z_struve_real_b_invar,[rho_approx_cuivre, 100],args = (b,z)))/dx*0.01e-3/np.sqrt(12)
    incertitude_a = (fsolve(delta_Z_struve_real_a_invar,[rho_approx_cuivre, 100],args = (a+dx,z))-fsolve(delta_Z_struve_real_a_invar,[rho_approx_cuivre, 100],args = (a,z)))/dx*0.01e-3/np.sqrt(12)
    incertitude_d = (fsolve(delta_Z_struve_real_d_invar,[rho_approx_cuivre, 100],args = (d+dx,z))-fsolve(delta_Z_struve_real_d_invar,[rho_approx_cuivre, 100],args = (d,z)))/dx*0.01e-3/np.sqrt(12)
    incertitude_totale = incertitude_a**2+incertitude_b**2+incertitude_d**2+incertitude_l**2
    #Tension
    incertitude_mix = (fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z+1j*dx))-fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z)))/dx*alpha_res
    incertitude_totale += incertitude_mix**2
    return incertitude_totale[0]

def incertitude_verif(z,alpha_res=0, alpha_sin = 0, mu =1):
    dx = 1e-10
    #dimensions
    incertitude_l = (fsolve(delta_Z_struve_real_l,[rho_approx_cuivre],args = (l+dx,z))-fsolve(delta_Z_struve_real_l,[rho_approx_cuivre],args = (l,z)))/dx*0.01e-3/np.sqrt(12)
    incertitude_b = (fsolve(delta_Z_struve_real_b,[rho_approx_cuivre],args = (b+dx,z))-fsolve(delta_Z_struve_real_b,[rho_approx_cuivre],args = (b,z)))/dx*0.01e-3/np.sqrt(12)
    incertitude_a = (fsolve(delta_Z_struve_real_a,[rho_approx_cuivre],args = (a+dx,z))-fsolve(delta_Z_struve_real_a,[rho_approx_cuivre],args = (a,z)))/dx*0.01e-3/np.sqrt(12)
    incertitude_d = (fsolve(delta_Z_struve_real_d,[rho_approx_cuivre],args = (d+dx,z))-fsolve(delta_Z_struve_real_d,[rho_approx_cuivre],args = (d,z)))/dx*0.01e-3/np.sqrt(12)
    incertitude_totale = incertitude_a**2+incertitude_b**2+incertitude_d**2+incertitude_l**2
    return incertitude_totale[0]

def degre2(x,a,b):
    return a*x**2+b

def TOTAL(Materiau):
    z = []
    try :
        if Materiau == "invar":
            for i in range(1,50):
                print(f'Calcul sur le fichier {i} en cours...')
                try :
                    sinus = analyse_donnees(f"{Chemin_relatif}/Données_invar_{i}.lvm")
                except :
                    print('Erreur Chemin, Tentative chemin Windows')
                    sinus = analyse_donnees(f"{Chemin_relatif}\\Données_invar_{i}.lvm")
                traité = phase(sinus)
                print('a')
                phaseurs = traité[0]
                erreur_res = traité[1]
                alpha_param = traité[2]
                poo = (phaseurs[0])/phaseurs[1]/(3)
                z = 2*(z2)*poo/(1-poo)-z_calib
                incertitude_G = np.sqrt((2/325.113*0.151421)**2+(2/325.113**2*2370*1.2824)**2)
                print('B')
                incertitude_z = np.sqrt((np.imag(2*(z2)/(1-poo**(-1))**2*1/phaseurs[0]*erreur_res[1]))**2+(np.imag(2*(z2)/(1-poo**(-1))**2*phaseurs[1]/phaseurs[0]**2*erreur_res[0]))**2)
                print('c')
                rho = fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z))
                print(z,phaseurs,incertitude_z,alpha_param)
                print('d')
                rho_incertitude = np.sqrt(incertitude(z,phaseurs, incertitude_z, alpha_param))
                print('e')
                print(rho_incertitude)
                print(rho[0])
                print(z)
                print([np.abs(rho[0]),rho_incertitude,i])
                z.append([np.abs(rho[0]),rho_incertitude,i])
                print('f')
            x = [i[2]*0.5+26.5+273.15 for i in z]
            y = [i[0] for i in z]
            err = [i[1] for i in z]
            param = curve_fit(degre2, x,y,sigma=err)
            y_fit = [degre2(i,param[0][0],param[0][1]) for i in x]
            res = [(y[i]-y_fit[i])/err[i] for i in range(len(x))]
        
        elif Materiau == "Cuivre":
            print(f'Calcul sur le fichier {i} en cours...')
            for i in range(1,50):
                try :
                    sinus = analyse_donnees(f"{Chemin_relatif}/Données_cuivre_{i}.lvm")
                except :
                    print('Erreur Chemin, Tentative chemin Windows')
                    sinus = analyse_donnees(f"{Chemin_relatif}\\Données_cuivre_{i}.lvm")
                traité = phase(sinus)
                phaseurs = traité[0]
                erreur_res = traité[1]
                alpha_param = traité[2]
                poo = -(phaseurs[0])/phaseurs[1]/(1+2*2370/325.113)
                z = 2*(z2)*poo/(1-poo)-z_calib
                incertitude_G = np.sqrt((2/325.113*0.151421)**2+(2/325.113**2*2370*1.2824)**2)
                incertitude_z = np.sqrt((np.imag(2*(z2)/(1-poo**(-1))**2*1/phaseurs[0]*erreur_res[1]))**2+(np.imag(2*(z2)/(1-poo**(-1))**2*phaseurs[1]/phaseurs[0]**2*erreur_res[0]))**2+(np.imag(2*(z2)/(1-poo**(-1)*(1+2*2370/325.113))**2*phaseurs[1]/phaseurs[0]*incertitude_G))**2)
                rho = fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z))
                rho_incertitude = np.sqrt(incertitude(z,phaseurs, incertitude_z, alpha_param))
                if np.abs(rho[0])<1e-5:
                    z.append([np.abs(rho[0]),rho_incertitude,i])
            x = [i[2]*0.5+25.5+273.15 for i in z]
            y = [i[0] for i in z]
            err = [2*i[1] for i in z]
            param = curve_fit(degre2, x,y,sigma=err)
            y_fit = [degre2(i,param[0][0],param[0][1]) for i in x]
            res = [(y[i]-y_fit[i])/err[i] for i in range(len(x))]
        
        elif Materiau == "Molyb":
            print(f'Calcul sur le fichier {i} en cours...')
            for i in range(1,50):
                try :
                    sinus = analyse_donnees(f"{Chemin_relatif}/Données_molyb_{i}.lvm")
                except :
                    print('Erreur Chemin, Tentative chemin Windows')
                    sinus = analyse_donnees(f"{Chemin_relatif}\\Données_molyb_{i}.lvm")
                traité = phase(sinus)
                phaseurs = traité[0]
                erreur_res = traité[1]
                alpha_param = traité[2]
                poo = (phaseurs[0])/phaseurs[1]/(1+2*2370/325.113)
                z = 2*(z2)*poo/(1-poo)-z_calib
                incertitude_G = np.sqrt((2/325.113*0.151421)**2+(2/325.113**2*2370*1.2824)**2)
                incertitude_z = np.sqrt((np.imag(2*(z2)/(1-poo**(-1))**2*1/phaseurs[0]*erreur_res[1]))**2+(np.imag(2*(z2)/(1-poo**(-1))**2*phaseurs[1]/phaseurs[0]**2*erreur_res[0]))**2+(np.imag(2*(z2)/(1-poo**(-1)*(1+2*2370/325.113))**2*phaseurs[1]/phaseurs[0]*incertitude_G))**2)
                rho = fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z))
                rho_incertitude = np.sqrt(incertitude(z,phaseurs, incertitude_z, alpha_param))
                print(rho_incertitude)
                if np.abs(rho[0])<1e-5:
                    z.append([np.abs(rho[0]),rho_incertitude,i])
            x = [i[2]*0.5+24.5+273.15 for i in z]
            y = [i[0] for i in z]
            err = [i[1] for i in z]
            param = curve_fit(degre2, x,y,sigma=err)
            y_fit = [degre2(i,param[0][0],param[0][1]) for i in x]
            res = [(y[i]-y_fit[i])/err[i] for i in range(len(x))]
                
    except :
        print('Erreur durant le calcul')

    return x,y_fit,y,err,param,res

#fonctions FIN ---------------------------------------------------------------------------------------------------------------
if Activation_Production_CSV_rho == 1 :
    print('')
    print('#CODE 1 DEBUT ---------------------------------------------------------------------------------------------------------------')
    print('')
    print('')
    
    a,b,c,d,e,f = TOTAL(Nom_materiau_utilisé)

    print('')
    print('')
    print('#CODE 1 FIN ---------------------------------------------------------------------------------------------------------------')


if Activation_Graphique_résistivité_Température == 1 :
    print('')
    print('#CODE 2 DEBUT ---------------------------------------------------------------------------------------------------------------')
    print('')
    print('')

    #Generation des graphiques -----------------------------------------------------------------------------------------------------------------------
    print('GENERATION GRAPHIQUE 1')
    try :
        if Nom_materiau_utilisé == "invar":
            plot1 = plt.subplot2grid((4,4),(0,0),colspan=4, rowspan=2)
            plot2 = plt.subplot2grid((3,4),(2,0),colspan=4, rowspan=1)
            plot1.plot(a,b,label = f"({e[0][0]:.2g}±{np.sqrt(e[1][0][0]):.2g})x+({e[0][1]:.2g}±{np.sqrt(e[1][1][1]):.2g})")
            plot1.errorbar(a,c,yerr=d, fmt=".k",elinewidth = 0.75,capsize=5)
            plot2.scatter(a,f)
            plot1.set_title("Valeurs mesurées de résistivité en fonction de la température de l'invar")
            plot2.set_ylabel("Résidus normalisés")
            plot1.set_ylabel("Résistivité (Ωm)")
            plot2.set_xlabel("Température (K)")
            plot1.grid(True)
            plot1.legend()
            plot2.grid(True)
            plt.show()
    
        elif Nom_materiau_utilisé == "molyb":
            plot1 = plt.subplot2grid((4,4),(0,0),colspan=4, rowspan=2)
            plot2 = plt.subplot2grid((3,4),(2,0),colspan=4, rowspan=1)
            plot1.plot(a,b,label = f"({e[0][0]:.2g}±{np.sqrt(e[1][0][0]):.2g})x+({e[0][1]:.2g}±{np.sqrt(e[1][1][1]):.2g})")
            plot1.errorbar(a,c,yerr=d, fmt=".k",elinewidth = 0.75,capsize=5)
            plot2.scatter(a,f)
            plot1.set_title("Valeurs mesurées de résistivité en fonction de la température du molybdène")
            plot2.set_ylabel("Résidus normalisés")
            plot1.set_ylabel("Résistivité (Ωm)")
            plot2.set_xlabel("Température (K)")
            plot1.grid(True)
            plot1.legend()
            plot2.grid(True)
            plt.show()
    
        elif Nom_materiau_utilisé == "cuivre":
            plot1 = plt.subplot2grid((4,4),(0,0),colspan=4, rowspan=2)
            plot2 = plt.subplot2grid((3,4),(2,0),colspan=4, rowspan=1)
            plot1.plot(a,b,label = f"({e[0][0]:.2g}±{np.sqrt(e[1][0][0]):.2g})x+({e[0][1]:.2g}±{np.sqrt(e[1][1][1]):.2g})")
            plot1.errorbar(a,c,yerr=d, fmt=".k",elinewidth = 0.75,capsize=5)
            plot2.scatter(a,f)
            plot1.set_title("Valeurs mesurées de résistivité en fonction de la température du cuivre OHFC")
            plot2.set_ylabel("Résidus normalisés")
            plot1.set_ylabel("Résistivité (Ωm)")
            plot2.set_xlabel("Température (K)")
            plot1.grid(True)
            plot1.legend()
            plot2.grid(True)
            plt.show()
    except :
        print('Erreur production graphiques')

    #-------------------------------------------------------------------------------------------------------------------------------------------------

    print('')
    print('')
    print('#CODE 2 FIN ---------------------------------------------------------------------------------------------------------------')