import matplotlib.pyplot as plt
import numpy as np

def labview_graph(output, input=list, time=float, n=int):
    t = np.linspace(0,time, num=n)

    return output[0]