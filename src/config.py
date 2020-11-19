mass_shift_annotation_url = ('https://docs.google.com/spreadsheets/d/'
                             '1-xh2XpSqdsa4yU-ATpDRxmpZEH6ht982jCCATFOpkyM/'
                             'export?format=csv&gid=566878567')
living_data_base_url = 'MSV000084314/updates/2020-11-18_mwang87_d115210a/other'

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
