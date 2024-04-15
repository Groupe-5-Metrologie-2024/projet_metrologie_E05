import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt



def traitement_sinus(sinus) :
    # Générer une liste de valeurs de sinus pour la première fonction
    A = sinus

    T = np.linspace(0,len(A)/10000, len(A))

    # Créer une fonction d'interpolation
    f = interp1d(T, A, kind='cubic')

    # Générer une plage de temps pour évaluer la fonction
    temps_interp = np.linspace(min(T), max(T), 1000)

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

    return zeros_valeurs,zeros_temps

def phase(output):
    phaseurs = []
    temps_ref = 20
    for i in range(len(output)):
        Valeurs, temps = traitement_sinus(output[i])
        max = [l for l in Valeurs if l>Valeurs[0]]
        min = [l for l in Valeurs if l<Valeurs[0]]
        voltage = sum(max)/len(max)-sum(min)/len(min)
        dephasage = max[temps_ref]
        phaseur = voltage*np.exp(-1j*2*np.pi*10000*dephasage)
        phaseurs.append(phaseur)
        print(voltage)
    return phaseurs

x = np.linspace(0.0001,1,10000)
y_1=np.sin(2*np.pi*10000*x)
y_2=2*np.sin(2*np.pi*10000*x+897325)

z_vect = phase([y_1,y_2])
z = 327*z_vect[1]/z_vect[0]


print(z)