import os


data_dir = os.path.realpath(os.path.join("..", "..", "data"))

mass_shift_annotation_url = (
    "https://docs.google.com/spreadsheets/d/"
    "1-xh2XpSqdsa4yU-ATpDRxmpZEH6ht982jCCATFOpkyM/"
    "export?format=csv&gid=566878567"
)
# Living data analysis performed on 2020-11-18.
living_data_dir = os.path.join(
    data_dir,
    "external",
    "MSV000084314",
    "updates",
    "2020-11-18_mwang87_d115210a",
    "other",
)
# Global living data molecular networking performed on 2019-10-21.
# https://gnps.ucsd.edu/ProteoSAFe/status.jsp?task=4f69e11bfb544010b2c4225a255f17ba
global_network_dir = os.path.join(data_dir, "external", "global_network")
global_network_task_id = "4f69e11bfb544010b2c4225a255f17ba"
# Additional spectrum annotations to use instead of the living data ids.
filename_ids = None

# Criteria to form a suspect:
#
# - Identification ≤ 20 ppm.
max_ppm = 20
# - Identification ≥ 6 shared peaks.
min_shared_peaks = 6
# - Cosine ≥ 0.8.
min_cosine = 0.8
# - The spectrum with maximal precursor intensity is chosen as cluster
#   representative.

# Criteria to assign group delta m/z's:
#
# - Only delta m/z's that exceed 0.5 Da are considered.
min_delta_mz = 0.5
# - Delta m/z's are examined within each 1 m/z window separately
#   (centered around unit m/z's).
interval_width = 1.0
# - Delta m/z's are binned with 0.002 bin width.
bin_width = 0.002
# - Peaks with minimum height 10 are extracted from the delta m/z histograms.
peak_height = 10
# - Delta m/z's between the left and right bases of each peak and at maximum
#   0.01 Da distance from the peak m/z's are grouped.
max_dist = 0.01

# Suspect filtering:
#
# - Suspects whose delta m/z occurs less than 10 times are discarded.

# Formula and charge mapping for data cleaning and harmonization.
formulas = {
    "AC": "CH3COO",
    "Ac": "CH3COO",
    "ACN": "C2H3N",
    "AcN": "C2H3N",
    "C2H3O2": "CH3COO",
    "C2H3OO": "CH3COO",
    "EtOH": "C2H6O",
    "FA": "CHOO",
    "Fa": "CHOO",
    "Formate": "CHOO",
    "formate": "CHOO",
    "H3C2OO": "CH3COO",
    "HAc": "CH3COOH",
    "HCO2": "CHOO",
    "HCOO": "CHOO",
    "HFA": "CHOOH",
    "MeOH": "CH4O",
    "OAc": "CH3COO",
    "Oac": "CH3COO",
    "OFA": "CHOO",
    "OFa": "CHOO",
    "Ofa": "CHOO",
    "TFA": "CF3COOH",
}

charges = {
    # Positive, singly charged.
    "H": 1,
    "K": 1,
    "Li": 1,
    "Na": 1,
    "NH4": 1,
    # Positive, doubly charged.
    "Ca": 2,
    "Fe": 2,
    "Mg": 2,
    # Negative, singly charged.
    "AC": -1,
    "Ac": -1,
    "Br": -1,
    "C2H3O2": -1,
    "C2H3OO": -1,
    "CH3COO": -1,
    "CHO2": -1,
    "CHOO": -1,
    "Cl": -1,
    "FA": -1,
    "Fa": -1,
    "Formate": -1,
    "formate": -1,
    "H3C2OO": -1,
    "HCO2": -1,
    "HCOO": -1,
    "I": -1,
    "OAc": -1,
    "Oac": -1,
    "OFA": -1,
    "OFa": -1,
    "Ofa": -1,
    "OH": -1,
    # Neutral.
    "ACN": 0,
    "AcN": 0,
    "EtOH": 0,
    "H2O": 0,
    "HFA": 0,
    "i": 0,
    "MeOH": 0,
    "TFA": 0,
    # Misceallaneous.
    "Cat": 1,
}
