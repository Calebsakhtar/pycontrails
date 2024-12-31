"""Wind shear functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pycontrails.utils.types import ArrayScalarLike


def wind_shear_enhancement_factor(
    contrail_depth: npt.NDArray[np.float64],
    effective_vertical_resolution: float | npt.NDArray[np.float64],
    wind_shear_enhancement_exponent: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Calculate the multiplication factor to enhance the wind shear based on contrail depth.

    This factor accounts for any subgrid-scale that is not captured by the resolution
    of the meteorological datasets.

    Parameters
    ----------
    contrail_depth : npt.NDArray[np.float64]
        Contrail depth , [:math:`m`]. Expected to be positive and bounded away from 0.
    effective_vertical_resolution : float | npt.NDArray[np.float64]
        Vertical resolution of met data , [:math:`m`]
    wind_shear_enhancement_exponent : float | npt.NDArray[np.float64]
        Exponent used in calculation. Expected to be nonnegative.
        Discussed in paragraphs following eq. (39) in Schumann 2012 and referenced as `n`.
        When this parameter is 0, no enhancement occurs.

    Returns
    -------
    npt.NDArray[np.float64]
        Wind shear enhancement factor

    Notes
    -----
    Implementation based on eq (39) in :cite:`schumannContrailCirrusPrediction2012`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    ratio = effective_vertical_resolution / contrail_depth
    enhancement_factor = 0.5 * (1.0 + ratio**wind_shear_enhancement_exponent)

    hijacked_factor = 1.0
    return hijacked_factor


def wind_shear_normal(
    u_wind_top: ArrayScalarLike,
    u_wind_btm: ArrayScalarLike,
    v_wind_top: ArrayScalarLike,
    v_wind_btm: ArrayScalarLike,
    cos_a: ArrayScalarLike,
    sin_a: ArrayScalarLike,
    dz: float,
) -> ArrayScalarLike:
    r"""Calculate the total wind shear normal to an axis.

    The total wind shear is the vertical gradient of the horizontal velocity.

    Parameters
    ----------
    u_wind_top : ArrayScalarLike
        u wind speed in the top layer, [:math:`m \ s^{-1}`]
    u_wind_btm : ArrayScalarLike
        u wind speed in the bottom layer, [:math:`m \ s^{-1}`]
    v_wind_top : ArrayScalarLike
        v wind speed in the top layer, [:math:`m \ s^{-1}`]
    v_wind_btm : ArrayScalarLike
        v wind speed in the bottom layer, [:math:`m \ s^{-1}`]
    cos_a : ArrayScalarLike
        Cosine component of segment
    sin_a : ArrayScalarLike
        Sine component of segment
    dz : float
        Difference in altitude between measurements, [:math:`m`]

    Returns
    -------
    ArrayScalarLike
       Wind shear normal to axis, [:math:`s^{-1}`]
    """
    du_dz = (u_wind_top - u_wind_btm) / dz
    dv_dz = (v_wind_top - v_wind_btm) / dz
    dsn_dz = dv_dz * cos_a - du_dz * sin_a
    ds_dz = (du_dz**2 + dv_dz**2) ** 0.5

    # Set shear to the closest value 
    shear_max = np.max(ds_dz)

    shear_options = [2e-3, 4e-3, 6e-3]
    shear_distance = []

    for shear in shear_options:
        shear_distance.append(np.abs(shear_options - shear_max))

    shear_options = np.array(shear_options)
    shear_distance = np.array(shear_distance)

    index_min = np.argmin(shear_distance)
    chosen_shear = shear_options[index_min]
    if np.isnan(dsn_dz).any():
        chosen_shear = 2e-3
        nan_present = True
        print("Removed NaN!")
    else:
        nan_present = False

    mask = np.full(dsn_dz.shape, True)
    dsn_dz = np.where(mask, -chosen_shear, dsn_dz)
    min_shear = np.min(dsn_dz)

    if chosen_shear != 2e-3 or nan_present:
        print("NORMAL shear: " + str(min_shear))
    
    return dsn_dz


def wind_shear(
    u_wind_top: ArrayScalarLike,
    u_wind_btm: ArrayScalarLike,
    v_wind_top: ArrayScalarLike,
    v_wind_btm: ArrayScalarLike,
    dz: float,
) -> ArrayScalarLike:
    r"""Calculate the total wind shear.

    The total wind shear is the vertical gradient of the horizontal velocity.

    Parameters
    ----------
    u_wind_top : ArrayScalarLike
        u wind speed in the top layer, [:math:`m \ s^{-1}`]
    u_wind_btm : ArrayScalarLike
        u wind speed in the bottom layer, [:math:`m \ s^{-1}`]
    v_wind_top : ArrayScalarLike
        v wind speed in the top layer, [:math:`m \ s^{-1}`]
    v_wind_btm : ArrayScalarLike
        v wind speed in the bottom layer, [:math:`m \ s^{-1}`]
    dz : float
        Difference in altitude between measurements, [:math:`m`]

    Returns
    -------
    ArrayScalarLike
       Total wind shear, [:math:`s^{-1}`]
    """
    du_dz = (u_wind_top - u_wind_btm) / dz
    dv_dz = (v_wind_top - v_wind_btm) / dz

    ds_dz = (du_dz**2 + dv_dz**2) ** 0.5

    # Set shear to the closest value 
    shear_max = np.max(ds_dz)

    shear_options = [2e-3, 4e-3, 6e-3]
    shear_distance = []

    for shear in shear_options:
        shear_distance.append(np.abs(shear_options - shear_max))

    shear_options = np.array(shear_options)
    shear_distance = np.array(shear_distance)

    index_min = np.argmin(shear_distance)
    chosen_shear = shear_options[index_min]
    if np.isnan(ds_dz).any():
        chosen_shear = 2e-3
        nan_present = True
        # print("Removed NaN!")
    else:
        nan_present = False

    mask = np.full(ds_dz.shape, True)
    ds_dz = np.where(mask, chosen_shear, ds_dz)
    min_shear = np.min(ds_dz)

    if chosen_shear != 2e-3 or nan_present:
        print("Shear : " + str(min_shear))


    return ds_dz
