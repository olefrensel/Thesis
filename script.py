# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:58:05 2023

@author: Ole Frensel
"""

# Packages used in the script
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Testing classes
class isotope:
    
    def __init__(self, sampleName, number, date, unit):
        self.sampleName = sampleName
        self.number = number
        self.date = date
        self.unit = unit

# Loading in the data
xls = pd.ExcelFile("C:/Users/frens/OneDrive/Documenten/Studie/BSc Thesis/Verzamel file veldmetingen en isotopen_JG3_JL2.xlsx")
data = pd.read_excel(xls, 'isotop C totaal (afbraak %)', header=None)        

# Testing how to access the data and doing some calculations to test if the correct values are given.
print(data.iloc[1, 0])

temp = data.iloc[0, 6]
temp = isotope(data.iloc[1, 1], data.iloc[2, 1], data.iloc[3, 1], data.iloc[4, 1])

B_enrich_low = data.iloc[121, 1]
delta_C_0 = data.iloc[26, 4]
delta_C_t = data.iloc[6, 3]

frac = 100 - np.exp(1000 * np.log((0.001*delta_C_t+1)/(0.001*delta_C_0+1)) / B_enrich_low) * 100
print(frac)


### Creating an array of the relative natural abundance of the international standards
standard_ratios = np.loadtxt("C:/Users/frens/OneDrive/Documenten/Studie/BSc Thesis/standard_isotope_ratios.csv", 
                             delimiter=";", skiprows=1, dtype=str)
#%%
print(standard_ratios[2][1])

#%%
### Fix docstring
def biodegrad(R, R0, enrich_fac, isotope_ratio = None, Rstd = None):
    """
    Function to calculate the percentage of biodegradation of a compound over time t. 
    
    Parameters
    ----------
    R : float
        isotope ratio of the sample at t = t (unitless)
    R0 : float
        isotope ratio of the sample at t = 0 (unitless)
    enrich_fac : float
        the enrichment factor (unitless)
    
    Optional Parameters
    -------------------
    isotope_ratio : string
        the elements of which the ratio is, i.e. 14N/15N (unitless)
    Rstd : float
        relative natural abundance of the international standards (unitless)
    
    Returns
    -------
    B : the percentage of biodegradation of the compound over time t (percentage, %)
    """
    # Either isotope_ratio or Rstd should be given a value, but not both. The following checks if that is the case.
    if isotope_ratio is not None and Rstd is not None:
        raise ValueError("Provide a value for either isotope_ratio or Rstd, not both.")
    elif isotope_ratio is None and Rstd is None:
        raise ValueError("Provide a value for either isotope_ratio or Rstd, not both.")
    
    # If the isotopes are given this searches in the list if there is a standard isotope ratio available.
    if isotope_ratio is not None:
        for i in range(len(standard_ratios)):
            if isotope_ratio == standard_ratios[i][0]:
                Rstd = float(standard_ratios[i][1])
                break
        # If an isotope_ratio (string) is given but it is not found in the list the following error should be given.    
        if Rstd is None:
            isotope_ratio_type = [row[0] for row in standard_ratios]
            raise ValueError(f"The ratio '{isotope_ratio}' is not in the list or is misspelled. The following standard isotope ratios are available: {isotope_ratio_type}.")
        
    
    dt = ((R / Rstd) - 1) * 1000
    d0 = ((R0 / Rstd) - 1) * 1000
    
    B = 100 - np.exp(1000 * np.log((0.001*dt+1)/(0.001*d0+1)) / enrich_fac) * 100
    
    return B

#%% Testing the function and seeing 

testing = ((-24.5/1000) + 1) * (98.89/1.11)
testing1 = ((-25.8/1000) + 1) * (98.89/1.11)
biodegrad(testing, testing1, -0.6, isotope_ratio = "12C/13C")
#%%
