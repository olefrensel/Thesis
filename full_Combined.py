# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:49:11 2024

@author: Ole Frensel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
c_file_path = "C:/Users/frens/OneDrive/Documenten/Studie/BSc Thesis/data_carbon.xlsx"
h_file_path = "C:/Users/frens/OneDrive/Documenten/Studie/BSc Thesis/data_hydrogen.xlsx"
# The H and C constants need to be the names given in the original Excel document.
# The marker can be anything and the name 'Toluene' is also used for plotting.
data_dict = {
    'Toluene': {
        'H': 'Hydrogen-2 (δ2H-Toluene)',
        'C': 'Carbon-13 (δ13C-Toluene)',
        'marker': 'D'
    },
    'Benzene': {
        'H': 'Hydrogen-2 (δ2H-Benzene)',
        'C': 'Carbon-13 (δ13C-Benzene)',
        'marker': 'o'
    },
    'm,p-Xylene': {
        'H': 'Hydrogen-2 (δ2H-m,p-Xylene)',
        'C': 'Carbon-13 (δ13C-m,p-Xylene)',
        'marker': 's'
    },
    'Ethylbenzene': {
        'H': 'Hydrogen-2 (δ2H-Ethylbenzene)',
        'C': 'Carbon-13 (δ13C-Ethylbenzene)',
        'marker': '^'
    },
    'Naphthalene': {
        'H': 'Hydrogen-2 (δ2H-Naphthalene)',
        'C': 'Carbon-13 (d13C-Naphthaline)',
        'marker': '+'
    },
    'Indene': {
        'H': 'Hydrogen-2 (δ2H-Indene)',
        'C': 'Carbon-13 (d13C-Indene)',
        'marker': 'x'
    }
}


class Isotope:
    
    def __init__(self, molecule, sampleName, number, date, unit, delta, delta0, 
                 enrichLow, enrichHigh, isotope_type):
        self.molecule = molecule
        self.sampleName = sampleName
        self.number = number
        self.date = date
        self.unit = unit
        self.delta = delta
        self.delta0 = delta0
        self.enrichLow = enrichLow
        self.enrichHigh = enrichHigh
        self.isotope_type = isotope_type
    
    # Biodegradation function, only tries the calculation if there are numbers available.
    def biodegradation(self):
        """
        Function to calculate biodegradation percentages based on instance variables.
        
        This method calculates the low and high biodegradation percentages ('bio_low' and 'bio_high')
        using the low and high enrichment factors ('enrichLow' and 'enrichHigh').
        
        The calculated values are stored in the instance variables `bio_low` and `bio_high`.
        
        If the required instance variables are not available, the biodegradation values are set to NaN.
        
        Parameters
        ----------
        None
        """
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
    
    
def read_excel_file(file_path):
    """
    Function to read the data from an Excel file and transform it to the right format.
    
    Note: this function can only be used for Excel files with the exact same format,
    as otherwise the transformed file will NOT be in the right format.
    
    Parameters
    ----------
    file_path : str
        The file path to the Excel file.
        
    Returns
    -------
    DataFrame
        Transformed DataFrame containing the read data.
    """   
    df = pd.read_excel(file_path, header=None)
    df = df.drop(0)
    df.reset_index(drop=True, inplace=True)
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df[1:]
    df.reset_index(drop=True, inplace=True)
    
    return df


def plotting(c_df, h_df, target_molecule, data_dict):
    """
    Function for plotting the delta 13 C versus the delta 2 H of chosen molecule.
    
    Parameters
    ----------
    c_df : dataframe
        Dataframe containing all data for delta 13 carbon data.
    h_df : dataframe
        Dataframe containing all data for delta 2 hydrogen data.
    target_molecule : str
        The molecule for which the plot is generated.
    data_dict : dictonary
        Dictonary containing the different names for looking up purposes, as well as the markers.
    """
    target_C_molecule = data_dict[target_molecule]['C']
    target_H_molecule = data_dict[target_molecule]['H']
    
    # Use boolean indexing to filter rows for the target molecule
    C_data = c_df[c_df['molecule'] == target_C_molecule]
    H_data = h_df[h_df['molecule'] == target_H_molecule]

    H_data['delta'] = pd.to_numeric(H_data['delta'], errors='coerce')
    C_data['delta'] = pd.to_numeric(C_data['delta'], errors='coerce')
    
    # Set x and y values
    x = C_data['delta']
    y = H_data['delta']

    # Remove NaN and inf values
    valid_indices = ~np.isnan(x) & ~np.isinf(x) & ~np.isnan(y) & ~np.isinf(y)
    x = x[valid_indices]
    y = y[valid_indices]
    
    # Perform linear regression
    coefficients = np.polyfit(x, y, 1)
    #slope_coefficient = coefficients[0]
    polynomial = np.poly1d(coefficients)

    # Create a trendline
    trendline_x = np.linspace(min(x), max(x), 100)
    trendline_y = polynomial(trendline_x)

    # Plot the scatter plot
    plt.scatter(x, y, marker=data_dict[target_molecule]['marker'])

    # Plot the trendline
    plt.plot(trendline_x, trendline_y, color='red', label='Linear Trendline')

    #plt.text(-25, 0, f"$\Lambda^{{H/C}}$ {round(slope_coefficient)}")
    plt.xlabel(f'$\delta^{{13}}$C {target_molecule}')
    plt.ylabel(f'$\delta^2$H {target_molecule}')
    plt.grid(axis='y')
    plt.legend()
    plt.show()
    

def main():
    """
    Main function to process carbon and hydrogen isotope data.
    
    This function performs the following tasks:
    1. Reads data from Excel files for carbon and hydrogen isotope measurements.
    2. Creates a DataFrame of instances of the isotope class for both carbon and hydrogen.
    3. Saves the resulting dataframes as CSV files.
    4. Plots the data for specified molecules.

    Note: Adjust file paths and molecule data accordingly in the code.
    
    Parameters
    ----------
    None
    """
    # Read data from Excel files
    df_C = read_excel_file(c_file_path)
    df_H = read_excel_file(h_file_path)

    # Creating instances of the isotope class
    hydrogen_dataframe = Isotope.create_from_dataframe(df_H, 'hydrogen')
    carbon_dataframe = Isotope.create_from_dataframe(df_C, 'carbon')

    # Save dataframes as CSV
    carbon_dataframe.to_csv("C:/Users/frens/OneDrive/Documenten/Studie/BSc Thesis/data_out/carbon_out.csv", index=False)
    hydrogen_dataframe.to_csv("C:/Users/frens/OneDrive/Documenten/Studie/BSc Thesis/data_out/hydrogen_out.csv", index=False)
    
    # Plotting the data
    for molecule, properties in data_dict.items():
        plotting(carbon_dataframe, hydrogen_dataframe, molecule, data_dict)
        
#%%
if __name__ == "__main__":
    main()
