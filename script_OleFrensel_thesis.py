# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:49:11 2024

@author: frens
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

c_file_path = "C:/Users/frens/OneDrive/Documenten/Studie/BSc Thesis/data_carbon.xlsx"
h_file_path = "C:/Users/frens/OneDrive/Documenten/Studie/BSc Thesis/data_hydrogen.xlsx"

#%%
class Isotope:
    
    def __init__(self, molecule, sampleName, number, date, unit, delta, delta0, enrichLow, enrichHigh, isotope_type):
        self.molecule = molecule
        self.sampleName = sampleName
        self.number = number
        self.date = date
        self.unit = unit
        self.delta = delta
        self.delta0 = delta0
        self.enrichLow = enrichLow
        self.enrichHigh = enrichHigh
        self.bio_low = np.nan
        self.bio_high = np.nan
        self.isotope_type = isotope_type
    
    # Biodegradation function, only tries the calculation if there are numbers available.
    def biodegradation(self):
        if isinstance(self.delta, (float, int)) and isinstance(self.delta0, (float, int)):
            
            if isinstance(self.enrichHigh, (float, int)):
                self.bio_high = 100 - np.exp(1000 * np.log((0.001*self.delta+1)/(0.001*self.delta0+1)) / self.enrichHigh) * 100
            else: 
                self.bio_high = np.nan
            
            if isinstance(self.enrichLow, (float, int)):
                self.bio_low = 100 - np.exp(1000 * np.log((0.001*self.delta+1)/(0.001*self.delta0+1)) / self.enrichLow) * 100
            else: 
                self.bio_low = np.nan
            
        else:
            self.bio_low = np.nan
            self.bio_high = np.nan
    
    @classmethod
    def create_from_dataframe(cls, df, isotope_type):
        isotope_list = []
        
        if isotope_type == 'hydrogen':
            molecule_list = ['Hydrogen-2 (δ2H-Benzene)',
                             'Hydrogen-2 (δ2H-Toluene)',
                             'Hydrogen-2 (δ2H-Ethylbenzene)',
                             'Hydrogen-2 (δ2H-m,p-Xylene)',
                             'Hydrogen-2 (δ2H-o-Xylene+Styrene)',
                             'Hydrogen-2 (δ2H-Cumene)',
                             'Hydrogen-2 (δ2H-Mesitylene)',
                             'Hydrogen-2 (δ2H-1,2,3-Trimethylbenzene)',
                             'Hydrogen-2 (δ2H-1,2,4-Trimethylbenzene)',
                             'Hydrogen-2 (δ2H-Indene)',
                             'Hydrogen-2 (δ2H-Naphthalene)']
            
            delta0_values = [-74, -115, -87, -92, -68, np.nan, np.nan, np.nan, np.nan,
                             np.nan, -8]
            
            enrichment_low_values = [-29, -17, -78, -19, -19, np.nan, np.nan, np.nan,
                                     np.nan, np.nan, -47]
            
            enrichment_high_values = [-79, -126, -189, -50, -50, np.nan, np.nan, np.nan,
                                      np.nan, np.nan, -100]
            
        elif isotope_type == 'carbon':
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

            delta0_values = [-28.7, -27.5, -27.6, -26.9, -26.1, np.nan, np.nan, np.nan, np.nan,
                             np.nan, np.nan, -28.9, np.nan, np.nan]

            enrichment_low_values = [-0.6, -0.7, -0.6, -0.7, -0.7, np.nan, np.nan, np.nan, np.nan,
                                      np.nan, np.nan, -0.4, np.nan, np.nan]

            enrichment_high_values = [-3.6, -6.7, -0.7, -2.7, -2.7, np.nan, np.nan, np.nan, np.nan,
                                       np.nan, np.nan, -0.5, np.nan, np.nan]
        
        else:
            raise ValueError("Invalid isotope type. Supported types are 'hydrogen' and 'carbon'.")
        
        for mol in molecule_list:
            for index, row in df.iterrows():
                sample_name = row['sample name']
                molecule = mol
                number = row['lab. No./well No']
                date = row['sampling date']
                unit = row['unit']
                delta = row[mol]
                delta0 = delta0_values[molecule_list.index(mol)]
                enrich_low = enrichment_low_values[molecule_list.index(mol)]
                enrich_high = enrichment_high_values[molecule_list.index(mol)]

                iso_instance = cls(molecule, sample_name, number, date, unit, 
                                   delta, delta0, enrich_low, enrich_high, isotope_type)
                iso_instance.biodegradation()
                isotope_list.append(iso_instance.__dict__)
        
        return pd.DataFrame(isotope_list)

#%% HYDROGEN
df_H = pd.read_excel(h_file_path, header=None)
df_H = df_H.drop(0)
df_H.reset_index(drop=True, inplace=True)
df_H = df_H.transpose()
df_H.columns = df_H.iloc[0]
df_H = df_H[1:]
df_H.reset_index(drop=True, inplace=True)

# Creating instances of the isotope class
hydrogen_dataframe = Isotope.create_from_dataframe(df_H, 'hydrogen')

# Printing the hydrogen dataframe
print(hydrogen_dataframe)

# Saving the hydrogen dataframe as csv
hydrogen_dataframe.to_csv("C:/Users/frens/OneDrive/Documenten/Studie/BSc Thesis/data_out/hydrogen_out.csv", index=False)

#%% CARBON
df_C = pd.read_excel(c_file_path, header=None)
df_C = df_C.drop(0)
df_C.reset_index(drop=True, inplace=True)
df_C = df_C.transpose()
df_C.columns = df_C.iloc[0]
df_C = df_C[1:]
df_C.reset_index(drop=True, inplace=True)

# Creating instances of the isotope class
carbon_dataframe = Isotope.create_from_dataframe(df_C, 'carbon')

# Printing the carbon dataframe
print(carbon_dataframe)

# Saving the carbon dataframe as csv
carbon_dataframe.to_csv("C:/Users/frens/OneDrive/Documenten/Studie/BSc Thesis/data_out/carbon_out.csv", index=False)

#%% PLOTTING
# Lists of every molecule I want to plot
H_list = ['Hydrogen-2 (δ2H-Toluene)',
          'Hydrogen-2 (δ2H-Ethylbenzene)']

C_list = ['Carbon-13 (δ13C-Toluene)', 
          'Carbon-13 (δ13C-Ethylbenzene)']

mol_list = ['Toluene',
            'Ethylbenzene']

markers = ['D',
           '^']


for i in range(len(H_list)):
    target_C_molecule = C_list[i]
    target_H_molecule = H_list[i]

    # Use boolean indexing to filter rows for the target molecule
    C_data = carbon_dataframe[carbon_dataframe['molecule'] == target_C_molecule]
    H_data = hydrogen_dataframe[hydrogen_dataframe['molecule'] == target_H_molecule]

    H_data['delta'] = pd.to_numeric(H_data['delta'], errors='coerce')
    C_data['delta'] = pd.to_numeric(C_data['delta'], errors='coerce')
    
    x = C_data['delta']
    y = H_data['delta']
    
    if target_H_molecule == 'Hydrogen-2 (δ2H-Naphthalene)':
        print(x)
        
    # HIER GAAT HET FOUT BIJ NAPHTHALENE
    # Remove NaN and inf values
    valid_indices = ~np.isnan(x) & ~np.isinf(x) & ~np.isnan(y) & ~np.isinf(y)
    x = x[valid_indices]
    y = y[valid_indices]
    
    if target_H_molecule == 'Hydrogen-2 (δ2H-Naphthalene)':
        print(x)
        
    # Perform linear regression
    coefficients = np.polyfit(x, y, 1)
    slope_coefficient = coefficients[0]
    polynomial = np.poly1d(coefficients)

    # Create a trendline
    trendline_x = np.linspace(min(x), max(x), 100)
    trendline_y = polynomial(trendline_x)

    # Plot the scatter plot
    plt.scatter(x, y, marker=markers[i])

    # Plot the trendline
    plt.plot(trendline_x, trendline_y, color='red', label='Linear Trendline')

    #plt.ylim(-80, 60)
    #plt.text(-25, 0, f"$\Lambda^{{H/C}}$ {round(slope_coefficient)}")
    plt.xlabel(f'$\delta^{{13}}$C {mol_list[i]}')
    plt.ylabel(f'$\delta^2$H {mol_list[i]}')
    plt.grid(axis='y')
    plt.legend()
    plt.show()