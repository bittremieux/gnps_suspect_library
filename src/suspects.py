import ftplib
import logging
import math
import operator
import os
import re
import time
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pyteomics.mass.unimod as unimod
import scipy.signal as ssignal
import tqdm

import config


logger = logging.getLogger("suspect_library")


def generate_suspects() -> None:
    """
    Generate suspects from the GNPS living data results.

    Suspect (unfiltered and filtered, unique) metadata is exported to csv files
    in the data directory.

    Settings for the suspect generation are taken from the config file.
    """
    if config.filename_ids is not None:
        task_id = re.match(
            r"MOLECULAR-LIBRARYSEARCH-V2-([a-z0-9]{8})-"
            r"view_all_annotations_DB-main.tsv",
            "MOLECULAR-LIBRARYSEARCH-V2-8b9bb7c1-"
            "view_all_annotations_DB-main.tsv",
        ).group(1)
    else:
        task_id = re.match(
            r"MSV000084314/updates/\d{4}-\d{2}-\d{2}_.+_([a-z0-9]{8})/other",
            config.living_data_base_url,
        ).group(1)
    suspects_dir = os.path.join(config.data_dir, "interim")

    # Get the clustering data per individual dataset.
    clusters_individual = _generate_suspects_per_dataset(
        config.living_data_base_url, 5
    )
    logger.info(
        "%d spectrum annotations, %d spectrum pairs, %d clusters retrieved "
        "from living data results for individual datasets",
        len(clusters_individual[0]),
        len(clusters_individual[1]),
        len(clusters_individual[2]),
    )
    # Get the clustering data from the global analysis.
    clusters_global = _generate_suspects_global(
        config.global_network_dir, config.global_network_task_id
    )
    logger.info(
        "%d spectrum annotations, %d spectrum pairs, %d clusters retrieved "
        "from the global molecular network",
        len(clusters_global[0]),
        len(clusters_global[1]),
        len(clusters_global[2]),
    )
    # Merge the clustering data from both sources.
    ids = pd.concat(
        [clusters_individual[0], clusters_global[0]], ignore_index=True
    )
    if config.filename_ids is not None:
        extra_ids = _read_ids(config.filename_ids)
        ids = pd.concat([ids, extra_ids], ignore_index=True)
        logger.info(
            "%d additional spectrum annotations from external library "
            "searching included",
            len(extra_ids)
        )
        library_usis_to_include = set(extra_ids["LibraryUsi"])
    else:
        library_usis_to_include = None
    pairs = pd.concat(
        [clusters_individual[1], clusters_global[1]], ignore_index=True
    )
    clusters = pd.concat(
        [clusters_individual[2], clusters_global[2]], ignore_index=True
    )
    logger.info(
        "%d spectrum annotations, %d spectrum pairs, %d clusters retained "
        "before filtering",
        len(ids),
        len(pairs),
        len(clusters),
    )
    # Filter based on the defined acceptance criteria.
    ids = _filter_ids(ids, config.max_ppm, config.min_shared_peaks)
    pairs = _filter_pairs(pairs, config.min_cosine)
    clusters = _filter_clusters(clusters)
    logger.info(
        "%d spectrum annotations, %d spectrum pairs, %d clusters retained "
        "after filtering",
        len(ids),
        len(pairs),
        len(clusters),
    )

    # Generate suspects from all of the clustering data.
    suspects_unfiltered = _generate_suspects(ids, pairs, clusters)
    suspects_unfiltered.to_parquet(
        os.path.join(suspects_dir, f"suspects_{task_id}_unfiltered.parquet"),
        index=False,
    )
    logger.info(
        "%d candidate unfiltered suspects generated", len(suspects_unfiltered)
    )
    # Ignore suspects without a mass shift.
    suspects_grouped = suspects_unfiltered[
        suspects_unfiltered["DeltaMass"].abs() > config.min_delta_mz
    ].copy()
    # Group and assign suspects by observed mass shift.
    suspects_grouped = _group_mass_shifts(
        suspects_grouped,
        _get_mass_shift_annotations(config.mass_shift_annotation_url),
        config.interval_width,
        config.bin_width,
        config.peak_height,
        config.max_dist,
    )
    # Ignore ungrouped suspects.
    suspects_grouped = suspects_grouped.dropna(subset=["GroupDeltaMass"])
    # (Optionally) filter by the supplementary identifications.
    if library_usis_to_include is not None:
        suspects_grouped = suspects_grouped[
            suspects_grouped["LibraryUsi"].isin(library_usis_to_include)
        ]
    suspects_grouped.to_parquet(
        os.path.join(suspects_dir, f"suspects_{task_id}_grouped.parquet"),
        index=False,
    )
    logger.info(
        "%d (non-unique) suspects with non-zero mass differences collected",
        len(suspects_grouped),
    )

    # 1. Only use the top suspect (by cosine score) per combination of library
    #    spectrum and grouped mass shift.
    # 2. Avoid repeated occurrences of the same suspect with different adducts.
    suspects_unique = (
        suspects_grouped.sort_values("Cosine", ascending=False)
        .drop_duplicates(["CompoundName", "Adduct", "GroupDeltaMass"])
        .sort_values("Adduct", key=_get_adduct_n_elements)
        .drop_duplicates(["CompoundName", "SuspectUsi"])
        .sort_values(["CompoundName", "Adduct", "GroupDeltaMass"])
    )
    suspects_unique.to_parquet(
        os.path.join(suspects_dir, f"suspects_{task_id}_unique.parquet"),
        index=False,
    )
    logger.info(
        "%d unique suspects retained after duplicate removal and filtering",
        len(suspects_unique),
    )


def _generate_suspects_per_dataset(
    living_data_base_url: str, n_jobs: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get all individual dataset cluster results from the GNPS living data.

    Parameters
    ----------
    living_data_base_url : str
        The URL of the living data FTP location.
    n_jobs : int
        The maximum number of concurrently running FTP queries. Using too many
        parallel requests can lead to a temporary server ban.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple of the identifications, pairs, and cluster_info DataFrames for
        all datasets included in the living data analysis.
    """
    ftp_prefix = f"ftp://massive.ucsd.edu/{living_data_base_url}"
    # Get the MassIVE IDs for all datasets included in the living data.
    ftp = ftplib.FTP("massive.ucsd.edu")
    ftp.login()
    ftp.cwd(f"{living_data_base_url}/CLUSTERINFO")
    msv_ids = [filename[: filename.find("_")] for filename in ftp.nlst()]
    # Get cluster information for each dataset.
    logger.info("Retrieve cluster information for individual datasets")
    ids, pairs, clusters = [], [], []
    for i, p, c in joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_download_cluster)(msv_id, ftp_prefix)
        for msv_id in tqdm.tqdm(
            msv_ids, desc="Datasets processed", unit="dataset"
        )
    ):
        if i is not None and p is not None and c is not None:
            ids.append(i)
            pairs.append(p)
            clusters.append(c)
    return (
        pd.concat(ids, ignore_index=True),
        pd.concat(pairs, ignore_index=True),
        pd.concat(clusters, ignore_index=True),
    )


def _download_cluster(
    msv_id: str, ftp_prefix: str, max_tries: int = 5
) -> Tuple[
    Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]
]:
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
        A tuple of the identifications, pairs, and cluster_info DataFrames.
    """
    tries_left = max_tries
    while tries_left > 0:
        try:
            ids = pd.read_csv(
                f"{ftp_prefix}/IDENTIFICATIONS/"
                f"{msv_id}_identifications.tsv",
                sep="\t",
                usecols=[
                    "Compound_Name",
                    "Ion_Source",
                    "Instrument",
                    "IonMode",
                    "Adduct",
                    "Precursor_MZ",
                    "INCHI",
                    "SpectrumID",
                    "#Scan#",
                    "MZErrorPPM",
                    "SharedPeaks",
                ],
            )
            ids = ids.rename(
                columns={
                    "Compound_Name": "CompoundName",
                    "Ion_Source": "IonSource",
                    "Precursor_MZ": "LibraryPrecursorMass",
                    "INCHI": "InChI",
                    "SpectrumID": "LibraryUsi",
                    "#Scan#": "ClusterId",
                    "MZErrorPPM": "MzErrorPpm",
                }
            )
            ids["ClusterId"] = f"{msv_id}:scan:" + ids["ClusterId"].astype(str)
            ids["LibraryUsi"] = (
                "mzspec:GNPS:GNPS-LIBRARY:accession:" + ids["LibraryUsi"]
            )
            pairs = pd.read_csv(
                f"{ftp_prefix}/PAIRS/{msv_id}_pairs.tsv",
                sep="\t",
                usecols=["CLUSTERID1", "CLUSTERID2", "Cosine"],
            )
            pairs = pairs.rename(
                columns={
                    "CLUSTERID1": "ClusterId1",
                    "CLUSTERID2": "ClusterId2",
                }
            )
            for col in ("ClusterId1", "ClusterId2"):
                pairs[col] = f"{msv_id}:scan:" + pairs[col].astype(str)
            clust = pd.read_csv(
                f"{ftp_prefix}/CLUSTERINFO/{msv_id}_clustering.tsv",
                sep="\t",
                usecols=[
                    "cluster index",
                    "sum(precursor intensity)",
                    "parent mass",
                    "Original_Path",
                    "ScanNumber",
                ],
            )
            clust = clust.rename(
                columns={
                    "cluster index": "ClusterId",
                    "sum(precursor intensity)": "PrecursorIntensity",
                    "parent mass": "SuspectPrecursorMass",
                }
            )
            clust = clust.dropna(subset=["ScanNumber"])
            clust["ClusterId"] = f"{msv_id}:scan:" + clust["ClusterId"].astype(
                str
            )
            clust["ScanNumber"] = clust["ScanNumber"].astype(int)
            clust = clust[clust["ScanNumber"] >= 0]
            clust["SuspectUsi"] = (
                f"mzspec:{msv_id}:"
                + clust["Original_Path"].apply(os.path.basename)
                + ":scan:"
                + clust["ScanNumber"].astype(str)
            )
            clust = clust.drop(columns=["Original_Path", "ScanNumber"])

            return ids, pairs, clust
        except ValueError as e:
            logger.warning(
                "Error while attempting to retrieve dataset %s: %s", msv_id, e
            )
            break
        except IOError:
            tries_left -= 1
            # Exponential back-off.
            time.sleep(
                np.random.uniform(high=2 ** (max_tries - tries_left) / 10)
            )
    else:
        logger.warning(
            "Failed to retrieve dataset %s after %d retries", msv_id, max_tries
        )

    return None, None, None


def _generate_suspects_global(
    global_network_dir: str, task_id: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Transform cluster information for the global molecular network built on top
    of the living data analysis.

    Parameters
    ----------
    global_network_dir : str
        The directory with the output of the molecular networking job.
    task_id : str
        The GNPS task identifier of the molecular networking job.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple of the identifications, pairs, and cluster_info DataFrames.
    """
    filename_specnets = os.path.join(
        global_network_dir,
        "result_specnets_DB",
        os.listdir(os.path.join(global_network_dir, "result_specnets_DB"))[0],
    )
    filename_networkedges = os.path.join(
        global_network_dir,
        "networkedges",
        os.listdir(os.path.join(global_network_dir, "networkedges"))[0],
    )
    filename_clusterinfosummary = os.path.join(
        global_network_dir,
        "clusterinfosummary",
        os.listdir(os.path.join(global_network_dir, "clusterinfosummary"))[0],
    )
    ids = pd.read_csv(
        filename_specnets,
        sep="\t",
        usecols=[
            "Compound_Name",
            "Ion_Source",
            "Instrument",
            "IonMode",
            "Adduct",
            "Precursor_MZ",
            "INCHI",
            "SpectrumID",
            "#Scan#",
            "MZErrorPPM",
            "SharedPeaks",
        ],
    )
    ids = ids.rename(
        columns={
            "Compound_Name": "CompoundName",
            "Ion_Source": "IonSource",
            "Precursor_MZ": "LibraryPrecursorMass",
            "INCHI": "InChI",
            "SpectrumID": "LibraryUsi",
            "#Scan#": "ClusterId",
            "MZErrorPPM": "MzErrorPpm",
        }
    )
    ids["ClusterId"] = "GLOBAL_NETWORK:scan:" + ids["ClusterId"].astype(str)
    ids["LibraryUsi"] = (
        "mzspec:GNPS:GNPS-LIBRARY:accession:" + ids["LibraryUsi"]
    )
    pairs = pd.read_csv(
        filename_networkedges,
        sep="\t",
        names=["ClusterId1", "ClusterId2", "Cosine"],
        usecols=[0, 1, 4],
    )
    for col in ("ClusterId1", "ClusterId2"):
        pairs[col] = "GLOBAL_NETWORK:scan:" + pairs[col].astype(str)
    clust = pd.read_csv(
        filename_clusterinfosummary,
        sep="\t",
        usecols=[
            "cluster index",
            "sum(precursor intensity)",
            "parent mass",
            "number of spectra",
        ],
    )
    clust = clust.rename(
        columns={
            "cluster index": "ClusterId",
            "sum(precursor intensity)": "PrecursorIntensity",
            "parent mass": "SuspectPrecursorMass",
        }
    )
    # Fix because cleaning up didn't work.
    clust = clust[clust["number of spectra"] >= 3]
    clust[
        "SuspectUsi"
    ] = f"mzspec:GNPS:TASK-{task_id}-spectra/specs_ms.mgf:scan:" + clust[
        "ClusterId"
    ].astype(
        str
    )
    clust["ClusterId"] = "GLOBAL_NETWORK:scan:" + clust["ClusterId"].astype(
        str
    )
    clust = clust.drop(columns="number of spectra")
    return ids, pairs, clust


def _read_ids(filename: str) -> pd.DataFrame:
    """
    Read MS/MS spectrum annotations from a GNPS library searching results file
    to be used as the identifications.

    Parameters
    ----------
    filename : str
        The GNPS library searching file name.

    Returns
    -------
    pd.DataFrame
        The identifications DataFrame.
    """
    ids = pd.read_csv(
        filename,
        sep="\t",
        usecols=[
            "#Scan#",
            "Compound_Name",
            "Ion_Source",
            "Instrument",
            "Adduct",
            "LibMZ",
            "INCHI",
            "IonMode",
            "MZErrorPPM",
            "SharedPeaks",
            "SpectrumFile",
            "SpectrumID",
        ],
    )
    ids = ids.rename(
        columns={
            "Compound_Name": "CompoundName",
            "Ion_Source": "IonSource",
            "LibMZ": "LibraryPrecursorMass",
            "INCHI": "InChI",
            "MZErrorPPM": "MzErrorPpm",
        }
    )
    ids["LibraryUsi"] = (
        "mzspec:GNPS:GNPS-LIBRARY:accession:" + ids["SpectrumID"]
    )
    ids["ClusterId"] = (
        ids["SpectrumFile"].apply(lambda fn: os.path.splitext(fn)[0])
        + ":scan:"
        + ids["#Scan#"].astype(str)
    )
    ids = ids.drop(columns=["#Scan#", "SpectrumFile", "SpectrumID"])
    return ids


def _filter_ids(
    ids: pd.DataFrame, max_ppm: float, min_shared_peaks: int
) -> pd.DataFrame:
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
    ids["Instrument"] = ids["Instrument"].replace(
        {
            # Hybrid FT.
            "ESI-QFT": "Hybrid FT",
            "Hybrid Ft": "Hybrid FT",
            "IT-FT/ion trap with FTMS": "Hybrid FT",
            "LC-ESI-ITFT": "Hybrid FT",
            "LC-ESI-QFT": "Hybrid FT",
            "LTQ-FT-ICR": "Hybrid FT",
            # Ion Trap.
            "CID; Velos": "Ion Trap",
            "IT/ion trap": "Ion Trap",
            "Ger": "Ion Trap",
            "LCQ": "Ion Trap",
            "QqIT": "Ion Trap",
            # qToF.
            " impact HD": "qTof",
            "ESI-QTOF": "qTof",
            "LC-ESI-QTOF": "qTof",
            "LC-Q-TOF/MS": "qTof",
            "Maxis HD qTOF": "qTof",
            "qToF": "qTof",
            "Maxis II HD Q-TOF Bruker": "qTof",
            "Q-TOF": "qTof",
            "qTOF": "qTof",
            # QQQ.
            "BEqQ/magnetic and electric sectors with quadrupole": "QQQ",
            "LC-APPI-QQ": "QQQ",
            "LC-ESI-QQ": "QQQ",
            "QqQ": "QQQ",
            "Quattro_QQQ:10eV": "QQQ",
            "Quattro_QQQ:25eV": "QQQ",
            "QqQ/triple quadrupole": "QQQ",
            # Orbitrap.
            "HCD": "Orbitrap",
            "HCD; Lumos": "Orbitrap",
            "HCD; Velos": "Orbitrap",
            "Q-Exactive Plus": "Orbitrap",
            "Q-Exactive Plus Orbitrap Res 70k": "Orbitrap",
            "Q-Exactive Plus Orbitrap Res 14k": "Orbitrap",
        }
    )
    ids["IonSource"] = ids["IonSource"].replace(
        {
            "CI": "APCI",
            "CI (MeOH)": "APCI",
            "DI-ESI": "ESI",
            "ESI/APCI": "APCI",
            "LC-APCI": "APCI",
            "in source ESI": "ESI",
            "LC-ESI-QFT": "LC-ESI",
            "LC-ESIMS": "LC-ESI",
            " ": "ESI",
            "Positive": "ESI",
        }
    )
    ids["IonMode"] = (
        ids["IonMode"].str.strip().str.capitalize().str.split("-", 1).str[0]
    )
    ids["Adduct"] = ids["Adduct"].astype(str).apply(_clean_adduct)

    return ids[
        (ids["MzErrorPpm"].abs() <= max_ppm)
        & (ids["SharedPeaks"] >= min_shared_peaks)
    ].dropna(subset=["Instrument", "IonSource", "IonMode", "Adduct"])


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
    new_adduct = re.sub("[ ()\[]", "", adduct)
    # Find out whether the charge is specified at the end.
    charge, charge_sign = 0, None
    for i in reversed(range(len(new_adduct))):
        if new_adduct[i] in ("+", "-"):
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
                if new_adduct[i] in ("]", "/"):
                    new_adduct = new_adduct[: i + 1]
            else:
                # Charge detected: remove from str.
                new_adduct = new_adduct[: i + 1]
            break
    # Now remove trailing delimiters after charge detection.
    new_adduct = re.sub("[\]/]", "", new_adduct)

    # Unknown adduct.
    if new_adduct.lower() in map(
        str.lower, ["?", "??", "???", "M", "M+?", "M-?", "unk", "unknown"]
    ):
        return "unknown"

    # Find neutral losses and additions.
    positive_parts, negative_parts = [], []
    for part in new_adduct.split("+"):
        pos_part, *neg_parts = part.split("-")
        positive_parts.append(_get_adduct_count(pos_part))
        for neg_part in neg_parts:
            negative_parts.append(_get_adduct_count(neg_part))
    mol = positive_parts[0]
    positive_parts = sorted(positive_parts[1:], key=operator.itemgetter(1))
    negative_parts = sorted(negative_parts, key=operator.itemgetter(1))
    # Handle weird Cat = [M]+ notation.
    if mol[1].lower() == "Cat".lower():
        mol = mol[0], "M"
        charge, charge_sign = 1, "+"

    # Calculate the charge from the individual components.
    if charge_sign is None:
        charge = sum(
            [
                count * config.charges.get(adduct, 0)
                for count, adduct in positive_parts
            ]
        ) + sum(
            [
                count * -abs(config.charges.get(adduct, 0))
                for count, adduct in negative_parts
            ]
        )
        charge_sign = "-" if charge < 0 else "+" if charge > 0 else ""

    cleaned_adduct = ["[", f"{mol[0] if mol[0] > 1 else ''}{mol[1]}"]
    if negative_parts:
        for count, adduct in negative_parts:
            cleaned_adduct.append(f"-{count if count > 1 else ''}{adduct}")
    if positive_parts:
        for count, adduct in positive_parts:
            cleaned_adduct.append(f"+{count if count > 1 else ''}{adduct}")
    cleaned_adduct.append("]")
    cleaned_adduct.append(
        f"{abs(charge) if abs(charge) > 1 else ''}{charge_sign}"
    )
    return "".join(cleaned_adduct)


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
    count, adduct = re.match(r"^(\d*)([A-Z]?.*)$", adduct).groups()
    count = int(count) if count else 1
    adduct = config.formulas.get(adduct, adduct)
    wrong_order = re.match(r"^([A-Z][a-z]*)(\d*)$", adduct)
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
    return pairs[pairs["Cosine"] >= min_cosine]


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
        cluster_info.reindex(
            cluster_info.groupby("ClusterId")["PrecursorIntensity"].idxmax()
        )
        .dropna()
        .reset_index(drop=True)[
            ["ClusterId", "SuspectPrecursorMass", "SuspectUsi"]
        ]
    )
    return cluster_info


def _generate_suspects(
    ids: pd.DataFrame, pairs: pd.DataFrame, clusters: pd.DataFrame
) -> pd.DataFrame:
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
    suspects = pd.concat(
        [
            pd.merge(pairs, ids, left_on="ClusterId1", right_on="ClusterId")
            .drop(columns=["ClusterId", "ClusterId1"])
            .rename(columns={"ClusterId2": "ClusterId"}),
            pd.merge(pairs, ids, left_on="ClusterId2", right_on="ClusterId")
            .drop(columns=["ClusterId", "ClusterId2"])
            .rename(columns={"ClusterId1": "ClusterId"}),
        ],
        ignore_index=True,
        sort=False,
    )

    # Add provenance information for the library and suspect scans.
    suspects = pd.merge(suspects, clusters, on="ClusterId")
    suspects = suspects[
        [
            "InChI",
            "CompoundName",
            "Adduct",
            "IonSource",
            "Instrument",
            "IonMode",
            "Cosine",
            "LibraryUsi",
            "SuspectUsi",
            "LibraryPrecursorMass",
            "SuspectPrecursorMass",
        ]
    ]
    suspects["DeltaMass"] = (
        suspects["SuspectPrecursorMass"] - suspects["LibraryPrecursorMass"]
    )
    suspects["GroupDeltaMass"] = np.nan
    suspects["AtomicDifference"] = np.nan
    suspects["Rationale"] = np.nan
    return suspects


def _get_mass_shift_annotations(
    extra_annotations: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get explanations for delta masses from known molecular modifications.

    Modifications are sourced from Unimod and (optionally) an additional sheet.
    All modifications are considered both as additions (positive delta mass) and
    losses (negative delta mass).

    Parameters
    ----------
    extra_annotations : Optional[str]
        Optional URL to an additional delta mass annotations sheet.

    Returns
    -------
    pd.DataFrame
        A dataframe with as columns:
            - "mz delta": The mass difference in m/z.
            - "atomic difference": The atomic composition of the modification.
            - "rationale": A description of the modification.
            - "priority": Numerical priority to sort modifications with
                          (near-)identical delta masses.
    """
    # Unimod modifications.
    mass_shift_annotations = []
    for mod in unimod.Unimod().mods:
        composition = []
        if "C" in mod.composition:
            composition.append(f"{mod.composition['C']}C")
            del mod.composition["C"]
            if "H" in mod.composition:
                composition.append(f"{mod.composition['H']}H")
                del mod.composition["H"]
        for atom in sorted(mod.composition.keys()):
            composition.append(f"{mod.composition[atom]}{atom}")
        mass_shift_annotations.append(
            (mod.monoisotopic_mass, composition, mod.full_name, 5)
        )
    mass_shift_annotations = pd.DataFrame(
        mass_shift_annotations,
        columns=["mz delta", "atomic difference", "rationale", "priority"],
    )
    if extra_annotations is not None:
        mass_shift_annotations2 = pd.read_csv(
            extra_annotations,
            usecols=["mz delta", "atomic difference", "rationale", "priority"],
        )
        mass_shift_annotations2["atomic difference"] = mass_shift_annotations2[
            "atomic difference"
        ].str.split(",")
        mass_shift_annotations = pd.concat(
            [mass_shift_annotations, mass_shift_annotations2],
            ignore_index=True,
        )
    # Reversed modifications.
    mass_shift_annotations_rev = mass_shift_annotations.copy()
    mass_shift_annotations_rev["mz delta"] *= -1
    mass_shift_annotations_rev["rationale"] = (
        mass_shift_annotations_rev["rationale"] + " (reverse)"
    ).str.replace("unspecified (reverse)", "unspecified", regex=False)
    mass_shift_annotations_rev["atomic difference"] = (
        mass_shift_annotations_rev["atomic difference"]
        .fillna("")
        .apply(list)
        .apply(
            lambda row: [a[1:] if a.startswith("-") else f"-{a}" for a in row]
        )
    )
    mass_shift_annotations = pd.concat(
        [mass_shift_annotations, mass_shift_annotations_rev], ignore_index=True
    )
    mass_shift_annotations["atomic difference"] = mass_shift_annotations[
        "atomic difference"
    ].str.join(",")
    for col, t in (("mz delta", float), ("priority", int)):
        mass_shift_annotations[col] = mass_shift_annotations[col].astype(t)
    return mass_shift_annotations.sort_values("mz delta").reset_index(
        drop=True
    )


def _group_mass_shifts(
    suspects: pd.DataFrame,
    mass_shift_annotations: pd.DataFrame,
    interval_width: float,
    bin_width: float,
    peak_height: float,
    max_dist: float,
) -> pd.DataFrame:
    """
    Group similar mass shifts.

    Mass shifts are binned and the group delta mass is detected by finding
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
        The maximum mass difference that group members can have with the
        group's peak.

    Returns
    -------
    pd.DataFrame
        The suspects with grouped mass shifts and corresponding rationale (if
        applicable).
    """
    # Assign putative identifications to the mass shifts.
    for mz in np.arange(
        math.floor(suspects["DeltaMass"].min()),
        math.ceil(suspects["DeltaMass"].max() + interval_width),
        interval_width,
    ):
        suspects_interval = suspects[
            suspects["DeltaMass"].between(
                mz - interval_width / 2, mz + interval_width / 2
            )
        ]
        if len(suspects_interval) == 0:
            continue
        # Get peaks for frequent deltas in the histogram.
        bins = (
            np.linspace(
                mz - interval_width / 2,
                mz + interval_width / 2,
                int(interval_width / bin_width) + 1,
            )
            + bin_width / 2
        )
        hist, _ = np.histogram(suspects_interval["DeltaMass"], bins=bins)
        peaks_i, prominences = ssignal.find_peaks(
            hist,
            height=peak_height,
            distance=max_dist / bin_width,
            prominence=(None, None),
        )
        if len(peaks_i) == 0:
            continue
        # Assign deltas to their closest peak.
        mask_peaks = np.unique(
            np.hstack(
                [
                    suspects_interval.index[
                        suspects_interval["DeltaMass"].between(min_mz, max_mz)
                    ]
                    for min_mz, max_mz in zip(
                        bins[prominences["left_bases"]],
                        bins[prominences["right_bases"]],
                    )
                ]
            )
        )
        mz_diffs = np.vstack(
            [
                np.abs(suspects.loc[mask_peaks, "DeltaMass"] - peak)
                for peak in bins[peaks_i]
            ]
        )
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
            delta_mz = suspects.loc[mask_delta_mz, "DeltaMass"].mean()
            delta_mz_std = suspects.loc[mask_delta_mz, "DeltaMass"].std()
            suspects.loc[mask_delta_mz, "GroupDeltaMass"] = delta_mz
            putative_id = mass_shift_annotations[
                (mass_shift_annotations["mz delta"] - delta_mz).abs()
                < delta_mz_std
            ].sort_values(["priority", "atomic difference", "rationale"])
            if len(putative_id) == 0:
                for col in ("AtomicDifference", "Rationale"):
                    suspects.loc[mask_delta_mz, col] = "unspecified"
            else:
                for col in ("atomic difference", "rationale"):
                    putative_id[col] = putative_id[col].fillna("unspecified")
                # Only use reverse explanations if no other explanations match.
                n_reversed = sum(
                    [
                        rationale.endswith("(reverse)")
                        for rationale in putative_id["rationale"]
                    ]
                )
                if n_reversed < len(putative_id):
                    putative_id = putative_id[
                        ~putative_id["rationale"].str.endswith("(reverse)")
                    ]
                suspects.loc[mask_delta_mz, "AtomicDifference"] = "|".join(
                    putative_id["atomic difference"]
                )
                suspects.loc[mask_delta_mz, "Rationale"] = "|".join(
                    putative_id["rationale"]
                )

    suspects["DeltaMass"] = suspects["DeltaMass"].round(3)
    suspects["GroupDeltaMass"] = suspects["GroupDeltaMass"].round(3)
    return suspects.sort_values(
        ["CompoundName", "Adduct", "GroupDeltaMass"]
    ).reset_index(drop=True)[
        [
            "InChI",
            "CompoundName",
            "Adduct",
            "IonSource",
            "Instrument",
            "IonMode",
            "Cosine",
            "LibraryUsi",
            "SuspectUsi",
            "LibraryPrecursorMass",
            "SuspectPrecursorMass",
            "DeltaMass",
            "GroupDeltaMass",
            "AtomicDifference",
            "Rationale",
        ]
    ]


def _get_adduct_n_elements(adducts: pd.Series) -> pd.Series:
    """
    Determine how many components the adducts consist of.

    Parameters
    ----------
    adducts : pd.Series
        A Series with different adducts.

    Returns
    -------
    pd.Series
        A Series with the number of components for the corresponding adducts.
        Unknown adducts are assigned "infinity" components.
    """
    counts = []
    for adduct in adducts:
        if adduct == "unknown":
            counts.append(np.inf)
        else:
            n = sum(
                [
                    _get_adduct_count(split)[0]
                    for split in re.split(
                        r"[+-]",
                        adduct[adduct.find("[") + 1 : adduct.rfind("]")],
                    )
                ]
            )
            counts.append(n if n > 1 else np.inf)
    return pd.Series(counts)


if __name__ == "__main__":
    logging.basicConfig(
        format="{asctime} [{levelname}/{processName}] {message}",
        style="{",
        level=logging.INFO,
    )
    logging.captureWarnings(True)
    logger.setLevel(logging.INFO)

    generate_suspects()
