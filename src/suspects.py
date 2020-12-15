import ftplib
import logging
import math
import operator
import re
import time
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.signal as ssignal
import tqdm

import config


logger = logging.getLogger('suspect_list')

formulas = {
    'AC': 'CH3COO', 'Ac': 'CH3COO', 'ACN': 'C2H3N', 'AcN': 'C2H3N',
    'C2H3O2': 'CH3COO', 'C2H3OO': 'CH3COO', 'EtOH': 'C2H6O', 'FA': 'CHOO',
    'Fa': 'CHOO', 'Formate': 'CHOO', 'formate': 'CHOO', 'H3C2OO': 'CH3COO',
    'HAc': 'CH3COOH', 'HCO2': 'CHOO', 'HCOO': 'CHOO', 'HFA': 'CHOOH',
    'MeOH': 'CH4O', 'OAc': 'CH3COO', 'Oac': 'CH3COO', 'OFA': 'CHOO',
    'OFa': 'CHOO', 'Ofa': 'CHOO', 'TFA': 'CF3COOH'}

charges = {
    # Positive, singly charged.
    'H': 1, 'K': 1, 'Li': 1, 'Na': 1, 'NH4': 1,
    # Positive, doubly charged.
    'Ca': 2, 'Fe': 2, 'Mg': 2,
    # Negative, singly charged.
    'AC': -1, 'Ac': -1, 'Br': -1, 'C2H3O2': -1, 'C2H3OO': -1, 'CH3COO': -1,
    'CHO2': -1, 'CHOO': -1, 'Cl': -1, 'FA': -1, 'Fa': -1, 'Formate': -1,
    'formate': -1, 'H3C2OO': -1, 'HCO2': -1,  'HCOO': -1, 'I': -1, 'OAc': -1,
    'Oac': -1, 'OFA': -1, 'OFa': -1, 'Ofa': -1, 'OH': -1,
    # Neutral.
    'ACN': 0, 'AcN': 0, 'EtOH': 0, 'H2O': 0, 'HFA': 0, 'i': 0, 'MeOH': 0,
    'TFA': 0,
    # Misceallaneous.
    'Cat': 1}


def _download_cluster(msv_id: str, ftp_prefix: str, max_tries: int = 5) \
        -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame],
                 Optional[pd.DataFrame]]:
    """
    Download cluster information for the living data analysis with the given
    MassIVE identifier.

    Parameters
    ----------
    msv_id : str
        The MassIVE identifier of the dataset in the living data analysis.
    ftp_prefix : str
        The FTP prefix of the living data results.
    max_tries : int
        The maximum number of times to try downloading files.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple of the identifications, pairs, and clustering DataFrames.
    """
    tries_left = max_tries
    while tries_left > 0:
        try:
            identifications = pd.read_csv(
                f'{ftp_prefix}/IDENTIFICATIONS/'
                f'{msv_id}_identifications.tsv',
                sep='\t', usecols=[
                    'Compound_Name', 'Ion_Source', 'Instrument', 'IonMode',
                    'Adduct', 'Precursor_MZ', 'INCHI', 'SpectrumID', '#Scan#',
                    'MZErrorPPM', 'SharedPeaks'])
            identifications['Dataset'] = msv_id
            pairs = pd.read_csv(
                f'{ftp_prefix}/PAIRS/{msv_id}_pairs.tsv', sep='\t',
                usecols=['CLUSTERID1', 'CLUSTERID2', 'Cosine'])
            pairs['Dataset'] = msv_id
            clustering = pd.read_csv(
                f'{ftp_prefix}/CLUSTERINFO/{msv_id}_clustering.tsv',
                sep='\t', usecols=[
                    'cluster index', 'sum(precursor intensity)',
                    'parent mass', 'Original_Path', 'ScanNumber'])
            clustering['Dataset'] = msv_id

            return identifications, pairs, clustering
        except ValueError:
            logger.warning('Error while attempting to retrieve dataset %s',
                           msv_id)
            break
        except IOError:
            tries_left -= 1
            # Exponential back-off.
            time.sleep(np.random.uniform(
                high=2 ** (max_tries - tries_left) / 10))
    else:
        logger.warning('Failed to retrieve dataset %s after %d retries',
                       msv_id, max_tries)

    return None, None, None


def _filter_ids(ids: pd.DataFrame, max_ppm: float, min_shared_peaks: int) \
        -> pd.DataFrame:
    """
    Filter high-quality identifications according to the given maximum ppm
    deviation and minimum number of shared peaks.

    Clean the identifications metadata (instrument, ion source, ion mode,
    adduct).

    Arguments
    ---------
    ids : pd.DataFrame
        The tabular identifications retrieved from GNPS.
    max_ppm : float
        The maximum ppm deviation.
    min_shared_peaks : int
        The minimum number of shared peaks.

    Returns
    -------
    pd.DataFrame
        The identifications retained after filtering.
    """
    # Clean the identifications metadata.
    ids['Instrument'] = ids['Instrument'].replace({
        # Hybrid FT.
        'ESI-QFT': 'Hybrid FT', 'Hybrid Ft': 'Hybrid FT',
        'IT-FT/ion trap with FTMS': 'Hybrid FT', 'LC-ESI-ITFT': 'Hybrid FT',
        'LC-ESI-QFT': 'Hybrid FT', 'LTQ-FT-ICR': 'Hybrid FT',
        # Ion Trap.
        'CID; Velos': 'Ion Trap', 'IT/ion trap': 'Ion Trap',
        'Ger': 'Ion Trap',  'LCQ': 'Ion Trap', 'QqIT': 'Ion Trap',
        # qToF.
        ' impact HD': 'qTof', 'ESI-QTOF': 'qTof', 'LC-ESI-QTOF': 'qTof',
        'LC-Q-TOF/MS': 'qTof', 'Maxis HD qTOF': 'qTof', 'qToF': 'qTof',
        'Maxis II HD Q-TOF Bruker': 'qTof', 'Q-TOF': 'qTof', 'qTOF': 'qTof',
        # QQQ.
        'LC-APPI-QQ': 'QQQ', 'LC-ESI-QQ': 'QQQ', 'QqQ': 'QQQ',
        'Quattro_QQQ:25eV': 'QQQ', 'QqQ/triple quadrupole': 'QQQ',
        # Orbitrap.
        'HCD': 'Orbitrap', 'HCD; Lumos': 'Orbitrap', 'HCD; Velos': 'Orbitrap',
        'Q-Exactive Plus': 'Orbitrap',
        'Q-Exactive Plus Orbitrap Res 70k': 'Orbitrap',
        'Q-Exactive Plus Orbitrap Res 14k': 'Orbitrap'
        })
    ids['Ion_Source'] = ids['Ion_Source'].replace(
        {'CI': 'APCI', 'CI (MeOH)': 'APCI', 'ESI/APCI': 'APCI',
         'LC-APCI': 'APCI', 'in source ESI': 'ESI', 'LC-ESI-QFT': 'LC-ESI',
         'LC-ESIMS': 'LC-ESI', ' ': 'ESI', 'Positive': 'ESI'})
    ids['IonMode'] = (ids['IonMode'].str.strip().str.capitalize()
                      .str.split('-', 1).str[0])
    ids['Adduct'] = ids['Adduct'].apply(_clean_adduct)

    return (ids[(ids['MZErrorPPM'].abs() <= max_ppm) &
                (ids['SharedPeaks'] >= min_shared_peaks)]
            .dropna(subset=['Instrument', 'Ion_Source', 'IonMode', 'Adduct']))


def _clean_adduct(adduct: str) -> str:
    """
    Consistent encoding of adducts, including charge information.

    Parameters
    ----------
    adduct : str
        The original adduct string.

    Returns
    -------
    str
        The cleaned adduct string.
    """
    # Keep "]" for now to handle charge as "M+Ca]2"
    new_adduct = re.sub('[ ()\[]', '', adduct)
    # Find out whether the charge is specified at the end.
    charge, charge_sign = 0, None
    for i in reversed(range(len(new_adduct))):
        if new_adduct[i] in ('+', '-'):
            if charge_sign is None:
                charge, charge_sign = 1, new_adduct[i]
            else:
                # Keep increasing the charge for multiply charged ions.
                charge += 1
        elif new_adduct[i].isdigit():
            charge += int(new_adduct[i])
        else:
            # Only use charge if charge sign was detected;
            # otherwise no charge specified.
            if charge_sign is None:
                charge = 0
                # Special case to handle "M+Ca]2" -> missing sign, will remove
                # charge and try to calculate from parts later.
                if new_adduct[i] in (']', '/'):
                    new_adduct = new_adduct[:i + 1]
            else:
                # Charge detected: remove from str.
                new_adduct = new_adduct[:i + 1]
            break
    # Now remove trailing delimiters after charge detection.
    new_adduct = re.sub('[\]/]', '', new_adduct)

    # Unknown adduct.
    if new_adduct.lower() in map(str.lower, ['?', '??', '???', 'M', 'M+?',
                                             'M-?', 'unk', 'unknown']):
        return 'unknown'

    # Find neutral losses and additions.
    positive_parts, negative_parts = [], []
    for part in new_adduct.split('+'):
        pos_part, *neg_parts = part.split('-')
        positive_parts.append(_get_adduct_count(pos_part))
        for neg_part in neg_parts:
            negative_parts.append(_get_adduct_count(neg_part))
    mol = positive_parts[0]
    positive_parts = sorted(positive_parts[1:], key=operator.itemgetter(1))
    negative_parts = sorted(negative_parts, key=operator.itemgetter(1))
    # Handle weird Cat = [M]+ notation.
    if mol[1].lower() == 'Cat'.lower():
        mol = mol[0], 'M'
        charge, charge_sign = 1, '+'

    # Calculate the charge from the individual components.
    if charge_sign is None:
        charge = (sum([count * charges.get(adduct, 0)
                       for count, adduct in positive_parts])
                  + sum([count * -abs(charges.get(adduct, 0))
                         for count, adduct in negative_parts]))
        charge_sign = '-' if charge < 0 else '+' if charge > 0 else ''

    cleaned_adduct = ['[', f'{mol[0] if mol[0] > 1 else ""}{mol[1]}']
    if negative_parts:
        for count, adduct in negative_parts:
            cleaned_adduct.append(f'-{count if count > 1 else ""}{adduct}')
    if positive_parts:
        for count, adduct in positive_parts:
            cleaned_adduct.append(f'+{count if count > 1 else ""}{adduct}')
    cleaned_adduct.append(']')
    cleaned_adduct.append(
        f'{abs(charge) if abs(charge) > 1 else ""}{charge_sign}')
    return ''.join(cleaned_adduct)


def _get_adduct_count(adduct: str) -> Tuple[int, str]:
    """
    Split the adduct string in count and raw adduct.

    Parameters
    ----------
    adduct : str

    Returns
    -------
    Tuple[int, str]
        The count of the adduct and its raw value.
    """
    count, adduct = re.match('^(\d*)([A-Z]?.*)$', adduct).groups()
    count = int(count) if count else 1
    adduct = formulas.get(adduct, adduct)
    wrong_order = re.match('^([A-Z][a-z]*)(\d*)$', adduct)
    # Handle multimers: "M2" -> "2M".
    if wrong_order is not None:
        adduct, count_new = wrong_order.groups()
        count = int(count_new) if count_new else count
    return count, adduct


def _filter_pairs(pairs: pd.DataFrame, min_cosine: float) -> pd.DataFrame:
    """
    Only consider pairs with a cosine similarity that exceeds the given cosine
    threshold.

    Arguments
    ---------
    pairs : pd.DataFrame
        The tabular pairs retrieved from GNPS.
    min_cosine : float
        The minimum cosine used to retain high-quality pairs.

    Returns
    -------
    pd.DataFrame
        The pairs filtered by minimum cosine similarity.
    """
    return pairs[pairs['Cosine'] >= min_cosine]


def _filter_clusters(cluster_info: pd.DataFrame) -> pd.DataFrame:
    """
    For each cluster select as representative the scan with the highest
    precursor intensity.

    Arguments
    ---------
    cluster_info : pd.DataFrame
        The tabular cluster info retrieved from GNPS.

    Returns
    -------
    pd.DataFrame
        Clusters without duplicated spectra by keeping only the scan with the
        highest precursor intensity for each cluster.
    """
    cluster_info = (
        cluster_info.reindex(cluster_info.groupby(
            ['Dataset', 'cluster index'])['sum(precursor intensity)'].idxmax())
        .dropna().reset_index(drop=True)
        [['Dataset', 'cluster index', 'parent mass', 'ScanNumber',
          'Original_Path']])
    cluster_info['cluster index'] = cluster_info['cluster index'].astype(int)
    cluster_info['ScanNumber'] = cluster_info['ScanNumber'].astype(int)
    return cluster_info


def _generate_suspects(ids: pd.DataFrame, pairs: pd.DataFrame,
                       clusters: pd.DataFrame) -> pd.DataFrame:
    """
    Generate suspects from identifications and aligned spectra pairs.
    Provenance about the spectra pairs is added from the summary.

    Arguments
    ---------
    ids : pd.DataFrame
        The filtered identifications.
    pairs : pd.DataFrame
        The filtered pairs.
    clusters : pd.DataFrame
        The filtered clustering information.

    Returns
    -------
    pd.DataFrame
        A DataFrame with information about both spectra forming a suspect
        identification.
    """
    # Form suspects of library and unidentified spectra pairs.
    suspects = pd.concat([
        pd.merge(pairs, ids,
                 left_on=['Dataset', 'CLUSTERID1'],
                 right_on=['Dataset', '#Scan#'])
        .drop(columns=['CLUSTERID1'])
        .rename(columns={'CLUSTERID2': 'SuspectIndex'}),
        pd.merge(pairs, ids,
                 left_on=['Dataset', 'CLUSTERID2'],
                 right_on=['Dataset', '#Scan#'])
        .drop(columns=['CLUSTERID2'])
        .rename(columns={'CLUSTERID1': 'SuspectIndex'})],
        ignore_index=True, sort=False)

    # Add provenance information for the library and suspect scans.
    suspects = (suspects[['Dataset', 'INCHI', 'Compound_Name', 'Adduct',
                          'Ion_Source', 'Instrument', 'IonMode', 'Cosine',
                          'Precursor_MZ', 'SpectrumID', '#Scan#',
                          'SuspectIndex']]
                .rename(columns={'Compound_Name': 'CompoundName',
                                 'Ion_Source': 'IonSource',
                                 'Precursor_MZ': 'LibraryPrecursorMZ',
                                 'SpectrumID': 'LibraryID',
                                 '#Scan#': 'ClusterScanNr'}))
    suspects = (pd.merge(suspects, clusters,
                         left_on=['Dataset', 'SuspectIndex'],
                         right_on=['Dataset', 'cluster index'])
                .drop(columns=['SuspectIndex', 'cluster index'])
                .rename(columns={'parent mass': 'SuspectPrecursorMZ',
                                 'Original_Path': 'SuspectPath',
                                 'ScanNumber': 'SuspectScanNr'}))
    suspects['DeltaMZ'] = (suspects['SuspectPrecursorMZ'] -
                           suspects['LibraryPrecursorMZ'])
    suspects['GroupDeltaMZ'] = np.nan
    return suspects


def _group_mass_shifts(
        suspects: pd.DataFrame, mass_shift_annotations: pd.DataFrame,
        interval_width: float, bin_width: float, peak_height: float,
        max_dist: float) -> pd.DataFrame:
    """
    Group close mass shifts.

    Mass shifts are binned and the group delta m/z is detected by finding
    peaks in the histogram. Grouped mass shifts are assigned potential
    explanations from the given mass shift annotations. If no annotation can be
    found for a certain group, the rationale and atomic difference will be
    marked as "unspecified". Ungrouped suspects will have a missing rationale
    and atomic difference.

    Arguments
    ---------
    suspects : pd.DataFrame
        The suspects from which mass shifts are grouped.
    mass_shift_annotations : pd.DataFrame
        Mass shift explanations.
    interval_width : float
        The size of the interval in which mass shifts are binned, centered
        around unit masses.
    bin_width : float
        The bin width used to construct the histogram.
    peak_height : float
        The minimum height for a peak to be considered as a group.
    max_dist : float
        The maximum m/z difference that group members can have with the
        group's peak.

    Returns
    -------
    pd.DataFrame
        The suspects with grouped mass shifts and corresponding rationale (if
        applicable).
    """
    # Assign putative identifications to the mass shifts.
    for mz in np.arange(math.floor(suspects['DeltaMZ'].min()),
                        math.ceil(suspects['DeltaMZ'].max() + interval_width),
                        interval_width):
        suspects_interval = suspects[suspects['DeltaMZ'].between(
            mz - interval_width / 2, mz + interval_width / 2)]
        if len(suspects_interval) == 0:
            continue
        # Get peaks for frequent deltas in the histogram.
        bins = (np.linspace(mz - interval_width / 2,
                            mz + interval_width / 2,
                            int(interval_width / bin_width) + 1)
                + bin_width / 2)
        hist, _ = np.histogram(suspects_interval['DeltaMZ'], bins=bins)
        peaks_i, prominences = ssignal.find_peaks(
            hist, height=peak_height, distance=max_dist / bin_width,
            prominence=(None, None))
        if len(peaks_i) == 0:
            continue
        # Assign deltas to their closest peak.
        mask_peaks = np.unique(np.hstack(
            [suspects_interval.index[suspects_interval['DeltaMZ']
                                     .between(min_mz, max_mz)]
             for min_mz, max_mz in zip(bins[prominences['left_bases']],
                                       bins[prominences['right_bases']])]))
        mz_diffs = np.vstack([
            np.abs(suspects.loc[mask_peaks, 'DeltaMZ'] - peak)
            for peak in bins[peaks_i]])
        # Also make sure that delta assignments don't exceed the maximum
        # distance.
        # noinspection PyArgumentList
        mask_mz_diffs = mz_diffs.min(axis=0) < max_dist
        mz_diffs = mz_diffs[:, mask_mz_diffs]
        mask_peaks = mask_peaks[mask_mz_diffs]
        peak_assignments = mz_diffs.argmin(axis=0)
        # Assign putative explanations to the grouped mass shifts.
        for peak_i in zip(range(len(peaks_i))):
            mask_delta_mz = mask_peaks[peak_assignments == peak_i]
            delta_mz = suspects.loc[mask_delta_mz, 'DeltaMZ'].mean()
            delta_mz_std = suspects.loc[mask_delta_mz, 'DeltaMZ'].std()
            suspects.loc[mask_delta_mz, 'GroupDeltaMZ'] = delta_mz
            putative_id = mass_shift_annotations[
                (mass_shift_annotations['mz delta'].abs()
                 - abs(delta_mz)).abs() < delta_mz_std].sort_values(
                ['priority', 'atomic difference', 'rationale'])
            if len(putative_id) == 0:
                suspects.loc[mask_delta_mz, 'AtomicDifference'] = 'unspecified'
                suspects.loc[mask_delta_mz, 'Rationale'] = 'unspecified'
            else:
                suspects.loc[mask_delta_mz, 'AtomicDifference'] = '|'.join(
                    putative_id['atomic difference'].fillna('unspecified'))
                suspects.loc[mask_delta_mz, 'Rationale'] = '|'.join(
                    putative_id['rationale'].fillna('unspecified'))

    return (suspects.sort_values(['CompoundName', 'Adduct', 'GroupDeltaMZ'])
            .reset_index(drop=True)
            [['Dataset', 'INCHI', 'CompoundName', 'Adduct', 'IonSource',
              'Instrument', 'IonMode', 'DeltaMZ', 'GroupDeltaMZ',
              'AtomicDifference', 'Rationale', 'Cosine', 'LibraryPrecursorMZ',
              'LibraryID', 'ClusterScanNr', 'SuspectPrecursorMZ',
              'SuspectScanNr', 'SuspectPath']])


def generate_suspects() -> None:
    """
    Generate suspects from the GNPS living data results.

    Suspect (unfiltered and filtered, unique) metadata is exported to csv files
    in the data directory.

    Settings for the suspect generation are taken from the config file.
    """
    # Expert-based mass shift annotations.
    mass_shift_annotations = pd.read_csv(config.mass_shift_annotation_url)
    mass_shift_annotations['mz delta'] = (mass_shift_annotations['mz delta']
                                          .astype(float))
    mass_shift_annotations['priority'] = (mass_shift_annotations['priority']
                                          .astype(int))

    # Get all GNPS living data cluster results.
    ftp_prefix = f'ftp://massive.ucsd.edu/{config.living_data_base_url}'
    # Get the MassIVE IDs for all datasets included in the living data.
    ftp = ftplib.FTP('massive.ucsd.edu')
    ftp.login()
    ftp.cwd(f'{config.living_data_base_url}/CLUSTERINFO')
    msv_ids = [filename[:filename.find('_')] for filename in ftp.nlst()]
    # Generate the suspects.
    logger.info('Retrieve cluster information')
    ids_pairs_clusters = joblib.Parallel(n_jobs=5)(
        joblib.delayed(_download_cluster)(msv_id, ftp_prefix)
        for msv_id in tqdm.tqdm(msv_ids, desc='Datasets processed',
                                unit='dataset'))
    ids, pairs, clusters = [], [], []
    for i, p, c in ids_pairs_clusters:
        if i is not None and p is not None and c is not None:
            ids.append(i)
            pairs.append(p)
            clusters.append(c)
    ids = pd.concat(ids, ignore_index=True)
    pairs = pd.concat(pairs, ignore_index=True)
    clusters = pd.concat(clusters, ignore_index=True)
    # Compile suspects from the clustering data.
    logger.info('Compile suspect pairs')
    ids = _filter_ids(ids, config.max_ppm, config.min_shared_peaks)
    pairs = _filter_pairs(pairs, config.min_cosine)
    clusters = _filter_clusters(clusters)
    suspects_unfiltered = _generate_suspects(ids, pairs, clusters)
    suspects_unfiltered.to_csv('../../data/suspects_unfiltered.csv',
                               index=False)

    # Ignore suspects without a mass shift.
    suspects_grouped = suspects_unfiltered[
        suspects_unfiltered['DeltaMZ'].abs() > config.min_delta_mz].copy()
    # Group and assign suspects by observed mass shift.
    logger.info('Group suspects by mass shift and assign potential rationales')
    suspects_grouped = _group_mass_shifts(
        suspects_grouped, mass_shift_annotations, config.interval_width,
        config.bin_width, config.peak_height, config.max_dist)
    suspects_grouped.to_csv('../../data/suspects_grouped.csv', index=False)
    # Ignore ungrouped suspects.
    suspects_grouped = suspects_grouped.dropna(subset=['GroupDeltaMZ'])
    # Only use the top suspect (by cosine score) per combination of library
    # spectrum and grouped mass shift.
    suspects_unique = (
        suspects_grouped.sort_values(['Cosine'], ascending=False)
        .drop_duplicates(['CompoundName', 'Adduct', 'GroupDeltaMZ']))
    suspects_unique.to_csv('../../data/suspects_unique.csv', index=False)

    logger.info('%d suspects with non-zero mass differences collected '
                '(%d total)', len(suspects_grouped), len(suspects_unfiltered))
    logger.info('%d unique suspects after duplicate removal and filtering',
                len(suspects_unique))


if __name__ == '__main__':
    logging.basicConfig(
        format='{asctime} [{levelname}/{processName}] {message}',
        style='{', level=logging.INFO)
    logging.captureWarnings(True)
    logger.setLevel(logging.INFO)

    generate_suspects()
