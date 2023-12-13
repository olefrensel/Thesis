# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:04:45 2023

@author: Ole Frensel
"""

import pandas as pd
import numpy as np

#%% Defining the isotope class.
class isotope:
    
    def __init__(self, molecule, sampleName, number, date, unit, deltaC, deltaC0, enrichLow, enrichHigh):
        self.molecule = molecule
        self.sampleName = sampleName
        self.number = number
        self.date = date
        self.unit = unit
        self.deltaC = deltaC
        self.deltaC0 = deltaC0
        self.enrichLow = enrichLow
        self.enrichHigh = enrichHigh
        
    # Function to calculate the biodegradation percentage, adjusted for potential NaN values.
    def biodegradation(self):
        if isinstance(self.deltaC, (float, int)) and isinstance(self.deltaC0, (float, int)):
            
            if isinstance(self.enrichHigh, (float, int)):
                self.bio_high = 100 - np.exp(1000 * np.log((0.001*self.deltaC+1)/(0.001*self.deltaC0+1)) / self.enrichHigh) * 100
            else: self.bio_high = np.nan
            
            if isinstance(self.enrichLow, (float, int)):
                self.bio_low = 100 - np.exp(1000 * np.log((0.001*self.deltaC+1)/(0.001*self.deltaC0+1)) / self.enrichLow) * 100
            else: 
                self.bio_low = np.nan
            
        else:
            self.bio_low = np.nan
            self.bio_high = np.nan

#%% Reading in the data, and transforming it to my liking.        
df = pd.read_excel("C:/Users/frens/OneDrive/Documenten/Studie/BSc Thesis/data_test_small.xlsx",
                   header=None)
# Deleting the first row as I found it unnecessary
df = df.drop(0)
df.reset_index(drop=True, inplace=True)
# Swapping the x- and y-axes of the dataframe to be able to make the first column headers.
df = df.transpose()
# Set the first row as the header
df.columns = df.iloc[0]
# Drop the first row to avoid duplication as it is now used as the header and reset the index.
df = df[1:]
df.reset_index(drop=True, inplace=True)

#%%
print(df)
#%% Creating lists with items stored at various places in the datafile.
molecule_list = ['Carbon-13 (δ13C-Benzene)', 
                 'Carbon-13 (δ13C-Toluene)', 
                 'Carbon-13 (δ13C-Ethylbenzene)',
                 'Carbon-13 (δ13C-m,p-Xylene)',
                 'Carbon-13 (d13C-o-Xylene)',
                 'Carbon-13 (d13C-Styrene)',
                 'Carbon-13 (d13C-Cumene)',
                 'Carbon-13 (d13C-Mesitylene)',
                 'Carbon-13 (d13C-1,2,3-Trimethylbenzene)',
                 'Carbon-13 (d13C-1,2,4-Trimethylbenzene)',
                 'Carbon-13 (d13C-Indene)',
                 'Carbon-13 (d13C-Naphthaline)',
                 'Carbon-13 (d13C-1-Methyl-Naphthaline)',
                 'Carbon-13 (d13C-2-Methyl-Naphthaline)']

# The following lists correspond with the molecules above.
delta13C0 = [-28.7, -27.5, -27.6, -26.9, -26.1, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, -28.9, np.nan, np.nan]

# *** NOG EEN KEER KIJKEN NAAR WAARDES VOOR Ethylbenze ***
enrichment_low = [-0.6, -0.7, -0.6, -0.7, -0.7, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, -0.4, np.nan, np.nan]

enrichment_high = [-3.6, -6.7, -0.7, -2.7, -2.7, np.nan, np.nan, np.nan, np.nan,
                   np.nan, np.nan, -0.5, np.nan, np.nan]

#%% These assignments should eventually be put into the class itself, so it only takes the dataframe as input.
isotope_list = []

# Combining all the information of the datafile and storing them as class objects.

for mol in molecule_list:
    for index, row in df.iterrows():
        sample_name = row['sample name']
        molecule = mol
        number = row['lab. No./well No']
        date = row['sampling date']
        unit = row['unit']
        deltaC = row[mol]
        deltaC0 = delta13C0[molecule_list.index(mol)]
        enrich_low = enrichment_low[molecule_list.index(mol)]
        enrich_high = enrichment_high[molecule_list.index(mol)]
    
        iso_instance = isotope(molecule, sample_name, number, date, unit, 
                               deltaC, deltaC0, enrich_low, enrich_high)
        iso_instance.biodegradation()
        
        # Right now I'm storing the values in a list, but this should of course be a dataframe of some fort.
        isotope_list.append(iso_instance)
    
#%%
print(isotope_list[6].bio_low)

    
