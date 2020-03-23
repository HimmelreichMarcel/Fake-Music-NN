# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 03:01:51 2018

@author: Marcel Himmelreich
"""

import matplotlib.pyplot as plt
import numpy as np

class Plotter():
    def __init__(self, network):
        self._network = network
        
    def plotEpochLoss(self,filename, title = "Training Result"):
        try:
            #Get History of fitted model as object dictionary
            H = self._network.getModel().history
            N = np.arange(0,len(H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, H["loss"], label="train_loss")
            plt.plot(N, H["val_loss"], label="test_loss")
            plt.plot(N, H["acc"], label="train_acc")
            plt.plot(N, H["val_acc"], label="test_acc")
            plt.title(title)
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig(filename)
            return plt.show()
        except Exception as n:
            print("Failed to plot epoch loss result")
            print(n)