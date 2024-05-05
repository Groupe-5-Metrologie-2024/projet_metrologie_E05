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

#MATERIAUX POSSIBLES SONT : "cuivre" ; "molyb" ; "invar_avec" ; "invar_sans"     (le avec et sans réfère à l'utilisation ou non d'aimants)
#CODE 1
if Activation_Production_CSV_rho == 1 :
    Nom_materiau_utilisé = "invar_sans"

#CODE 2
if Activation_Graphique_résistivité_Température == 1 :  
    Nom_materiau_utilisé = "invar_sans"


#----- Données -----#
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

def phase(output):
    phaseurs = []
    alpha = []
    for i in range(len(output)-1):
        def fit(x,V,phi):
            return V*np.cos(omega*x+phi)
        param = curve_fit(fit,output[2],output[i])
        phaseurs.append(param[0][0]*np.exp(1j*param[0][1]))
        alpha.append(param[1])
    return phaseurs,alpha

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

def delta_Z_struve_real(x,z=0,mur=1):
    delta_Z = 2j*omega*mu0*(n/(d*l))**2*quad(f_integrale_struve, 0,200, args=(x,1,),complex_func=True,limit =1000,points=[0])[0]
    return np.imag(delta_Z-z)

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
def delta_Z_struve_real_l(x=rho_approx_cuivre,l=l,z=0,mur = 1):
    integ = quad(f_integrale_struve_l, 0,200, args=(l,x,mur,),complex_func=True,limit =1000,points=[0])
    delta_Z = 2j*omega*mu0*(n/(d*l))**2*integ[0]
    return np.imag(delta_Z-z)
def delta_Z_struve_real_d(x=rho_approx_cuivre,d=d,z=0,mur = 1):
    delta_Z = 2j*omega*mu0*(n/(d*l))**2*quad(f_integrale_struve_d, 0,200, args=(d,x,mur,),complex_func=True,limit =1000,points=[0])[0]
    return np.imag(delta_Z-z)
def delta_Z_struve_real_a(x=rho_approx_cuivre,a=a,z=0,mur = 1):
    delta_Z = 2j*omega*mu0*(n/(d*l))**2*quad(f_integrale_struve_a, 0,200, args=(a,x,mur,),complex_func=True,limit =1000,points=[0])[0]
    return np.imag(delta_Z-z)
def delta_Z_struve_real_b(x=rho_approx_cuivre,b=b,z=0,mur = 1):
    delta_Z = 2j*omega*mu0*(n/(d*l))**2*quad(f_integrale_struve_b, 0,200, args=(b,x,mur,),complex_func=True,limit =1000,points=[0])[0]
    return np.imag(delta_Z-z)

def incertitude(z,phaseurs_molyb,incertitude_g, alpha_sin = 0, mur =1):
    dx = 1e-10
    #dimensions
    incertitude_l = (fsolve(delta_Z_struve_real_l,[rho_approx_cuivre],args = (l+dx,z,mur))-fsolve(delta_Z_struve_real_l,[rho_approx_cuivre],args = (l,z,mur)))/dx*0.01e-3/np.sqrt(12)
    incertitude_b = (fsolve(delta_Z_struve_real_b,[rho_approx_cuivre],args = (b+dx,z,mur))-fsolve(delta_Z_struve_real_b,[rho_approx_cuivre],args = (b,z,mur)))/dx*0.01e-3/np.sqrt(12)
    incertitude_a = (fsolve(delta_Z_struve_real_a,[rho_approx_cuivre],args = (a+dx,z,mur))-fsolve(delta_Z_struve_real_a,[rho_approx_cuivre],args = (a,z,mur)))/dx*0.01e-3/np.sqrt(12)
    incertitude_d = (fsolve(delta_Z_struve_real_d,[rho_approx_cuivre],args = (d+dx,z,mur))-fsolve(delta_Z_struve_real_d,[rho_approx_cuivre],args = (d,z,mur)))/dx*0.01e-3/np.sqrt(12)
    incertitude_totale = incertitude_a**2+incertitude_b**2+incertitude_d**2+incertitude_l**2
    print(np.sqrt(incertitude_totale))
    #Gain
    incertitude_totale += ((fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z+1j*dx,mur))-fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z,mur)))/dx*incertitude_g)**2
    #Tension 1
    print(incertitude_totale**(1/2))
    poo_molyb = -(phaseurs_molyb[0])*(1+dx)/phaseurs_molyb[1]/(1+2*2370/325.113)
    z_dx = 2*(z2)*poo_molyb/(1-poo_molyb)-z_calib
    deriv_norme = (fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z_dx,mur))-fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z,mur)))/dx
    incertitude_totale+= deriv_norme**2*alpha_sin[0][1][1]
    poo_molyb = -(phaseurs_molyb[0])*np.exp(1j*dx)/phaseurs_molyb[1]/(1+2*2370/325.113)
    z_dx = 2*(z2)*poo_molyb/(1-poo_molyb)-z_calib
    deriv_phi = (fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z_dx,mur))-fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z,mur)))/dx
    incertitude_totale+= deriv_phi**2*alpha_sin[0][0][0]+2*deriv_phi*deriv_norme*alpha_sin[0][1][0]
    print(incertitude_totale**(1/2))
    #Tension 2
    poo_molyb = -(phaseurs_molyb[0])/phaseurs_molyb[1]*(1+dx)/(1+2*2370/325.113)
    z_dx = 2*(z2)*poo_molyb/(1-poo_molyb)-z_calib
    deriv_norme = (fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z_dx,mur))-fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z,mur)))/dx
    incertitude_totale+= deriv_norme**2*alpha_sin[1][1][1]
    poo_molyb = -(phaseurs_molyb[0])/phaseurs_molyb[1]*np.exp(1j*dx)/(1+2*2370/325.113)
    z_dx = 2*(z2)*poo_molyb/(1-poo_molyb)-z_calib
    deriv_phi = (fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z_dx,mur))-fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z,mur)))/dx
    incertitude_totale+= deriv_phi**2*alpha_sin[1][0][0]+2*deriv_phi*deriv_norme*alpha_sin[1][1][0]
    return incertitude_totale[0]

def TOTAL(Materiau):
    z = []
    try :
        if Materiau == "invar_avec" :
            for i in range(1,51):
                print(f'Calcul en cours sur le fichier {i}...')
                try :
                    sinus_inv = analyse_donnees(f"{Chemin_relatif}/Données_invar_avec_{i}.lvm")
                except :
                    print('Erreur Chemin, Tentative chemin Windows')
                    sinus_inv = analyse_donnees(f"{Chemin_relatif}\\Données_invar_avec_{i}.lvm")
                traité = phase(sinus_inv)
                phaseurs_inv = [-traité[0][1],traité[0][0]]
                alpha_param = traité[1]
                poo_inv = -(phaseurs_inv[0])/phaseurs_inv[1]/(1+2*2370/325.113)
                z_inv = 2*(z2)*poo_inv/(1-poo_inv)-z_calib
                incertitude_g = 0
                data = fsolve(delta_Z_struve,[rho_approx_invar,100],args = (z_inv))
                rho = data[0]
                mur = data[1]
                rho_incertitude = np.sqrt(incertitude(z_inv,phaseurs_inv, incertitude_g, alpha_param,mur))
                if np.abs(rho)<1e-5:
                    z.append([np.abs(rho),rho_incertitude,i])
            x = [i[2]*0.5+26.5+273.15 for i in z]
            y = [i[0] for i in z]
            err = [i[1] for i in z]
            def degre2(x,b,c):
                return b*x+c
            param = curve_fit(degre2, x,y,sigma=err)
            y_fit = [degre2(i,param[0][0],param[0][1]) for i in x]
            res = [(y[i]-y_fit[i])/err[i] for i in range(len(x))]
        
        if Materiau == "invar_sans" :
            for i in range(1,46):
                print(f'Calcul en cours sur le fichier {i}...')
                try :
                    sinus_inv = analyse_donnees(f"{Chemin_relatif}/Données_invar_sans_{i}.lvm")
                except :
                    print('Erreur Chemin, Tentative chemin Windows')
                    sinus_inv = analyse_donnees(f"{Chemin_relatif}\\Données_invar_avec_{i}.lvm")
                traité = phase(sinus_inv)
                phaseurs_inv = [-traité[0][1],traité[0][0]]
                alpha_param = traité[1]
                poo_inv = -(phaseurs_inv[0])/phaseurs_inv[1]/(1+2*2370/325.113)
                z_inv = 2*(z2)*poo_inv/(1-poo_inv)-z_calib
                incertitude_g = 0
                data = fsolve(delta_Z_struve,[rho_approx_invar,100],args = (z_inv))
                rho = data[0]
                mur = data[1]
                rho_incertitude = np.sqrt(incertitude(z_inv,phaseurs_inv, incertitude_g, alpha_param,mur))
                if np.abs(rho)<1e-5:
                    z.append([np.abs(rho),rho_incertitude,i])
            x = [i[2]*0.5+26.5+273.15 for i in z]
            y = [i[0] for i in z]
            err = [i[1] for i in z]
            def degre2(x,b,c):
                return b*x+c
            param = curve_fit(degre2, x,y,sigma=err)
            y_fit = [degre2(i,param[0][0],param[0][1]) for i in x]
            res = [(y[i]-y_fit[i])/err[i] for i in range(len(x))]

        elif Materiau == "cuivre":
            for i in range(1,52):
                print(f'Calcul sur le fichier {i} en cours...')
                try :
                    sinus_cu = analyse_donnees(f"{Chemin_relatif}/Données_cuivre_{i}.lvm")
                except :
                    print('Erreur Chemin, Tentative chemin Windows')
                    sinus_cu = analyse_donnees(f"{Chemin_relatif}\\Données_cuivre_{i}.lvm")
                traité = phase(sinus_molyb)
                phaseurs_cu = traité[0]
                alpha_param = traité[1]
                poo_cu = -(phaseurs_cu[0])/phaseurs_cu[1]/(1+2*2370/325.113)
                z_cu = 2*(z2)*poo_cu/(1-poo_cu)-z_calib
                incertitude_G = np.sqrt((2/325.113*0.151421)**2+(2/325.113**2*2370*1.2824)**2)
                incertitude_g = np.sqrt(+(np.imag(2*(z2)/(1-poo_cu**(-1)*(1+2*2370/325.113))**2*phaseurs_cu[1]/phaseurs_cu[0]*incertitude_G))**2)
                rho = fsolve(delta_Z_struve_real,[rho_approx_cuivre],args = (z_cu))
                rho_incertitude = np.sqrt(incertitude(z_molyb,phaseurs_cu, incertitude_g, alpha_param))
                if np.abs(rho[0])<1e-5:
                    z.append([np.abs(rho[0]),rho_incertitude,i])
            x = [i[2]*0.5+25.5+273.15 for i in z]
            y = [i[0] for i in z]
            err = [2*i[1] for i in z]
            def degre1(x,a,b):
                return a*x+b
            param = curve_fit(degre1, x,y,sigma=err)
            y_fit = [degre1(i,param[0][0],param[0][1]) for i in x]
            res = [(y[i]-y_fit[i])/err[i] for i in range(len(x))]
        
        elif Materiau == "molyb":
            for i in range(1,54):
                print(f'Calcul sur le fichier {i} en cours...')
                try :
                    sinus_molyb = analyse_donnees(f"{Chemin_relatif}/Données_molyb_{i}.lvm")
                except :
                    print('Erreur Chemin, Tentative chemin Windows')
                    sinus_molyb = analyse_donnees(f"{Chemin_relatif}\\Données_molyb_{i}.lvm")
                traité = phase(sinus_molyb)
                phaseurs_molyb = [-traité[0][1],traité[0][0]]
                alpha_param = traité[1]
                poo_molyb = -(phaseurs_molyb[0])/phaseurs_molyb[1]/(1+2*2370/325.113)
                z_molyb = 2*(z2)*poo_molyb/(1-poo_molyb)-z_calib
                incertitude_G = np.sqrt((2/325.113*0.151421)**2+(2/325.113**2*2370*1.2824)**2)
                incertitude_g = np.sqrt(+(np.imag(2*(z2)/(1-poo_molyb**(-1)*(1+2*2370/325.113))**2*phaseurs_molyb[1]/phaseurs_molyb[0]*incertitude_G))**2)
                rho = fsolve(delta_Z_struve_real,[rho_approx_molyb],args = (z_molyb))
                rho_incertitude = np.sqrt(incertitude(z_molyb,phaseurs_molyb, incertitude_g, alpha_param))
                if np.abs(rho[0])<1e-5:
                    z.append([np.abs(rho[0]),rho_incertitude,i])
            x = [i[2]*0.5+24.5+273.15 for i in z]
            y = [i[0] for i in z]
            err = [i[1] for i in z]
            def degre2(x,b,c):
                return b*x+c
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
        if Nom_materiau_utilisé == "invar_sans":

            plot1 = plt.subplot2grid((4,4),(0,0),colspan=4, rowspan=2)
            plot2 = plt.subplot2grid((3,4),(2,0),colspan=4, rowspan=1)
            plot1.plot(a,b,label = f"({e[0][0]:.2g}±{np.sqrt(e[1][0][0]):.2g})x+({e[0][1]:.2g}±{np.sqrt(e[1][1][1]):.2g})")
            plot1.errorbar(a,c,yerr=d,xerr = 0.01/np.sqrt(12), fmt=".k",elinewidth = 0.75,capsize=5)
            plot2.scatter(a,f)
            plot2.set_ylabel("Résidus normalisés")
            plot1.set_ylabel("Résistivité (Ωm)")
            plot2.set_xlabel("Température (K)")
            plot1.grid(True)
            plot1.legend()
            plot2.grid(True)
            plt.show()

        if Nom_materiau_utilisé == "invar_avec":

            plot1 = plt.subplot2grid((4,4),(0,0),colspan=4, rowspan=2)
            plot2 = plt.subplot2grid((3,4),(2,0),colspan=4, rowspan=1)
            plot1.plot(a,b,label = f"({e[0][0]:.2g}±{np.sqrt(e[1][0][0]):.2g})x+({e[0][1]:.2g}±{np.sqrt(e[1][1][1]):.2g})")
            plot1.errorbar(a,c,yerr=d,xerr = 0.01/np.sqrt(12), fmt=".k",elinewidth = 0.75,capsize=5)
            plot2.scatter(a,f)
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
            plot1.errorbar(a,c,yerr=d,xerr = 0.01/np.sqrt(12), fmt=".k",elinewidth = 0.75,capsize=5)
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
            plot1.errorbar(a,c,yerr=d,xerr = 0.01/np.sqrt(12), fmt=".k",elinewidth = 0.75,capsize=5)
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