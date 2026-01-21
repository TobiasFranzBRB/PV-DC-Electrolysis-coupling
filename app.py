# app.py
# --------------------------------------------------------------------------------------
# Streamlit WebApp: PV Array ↔ Electrolyzer Coupling (PVGIS-driven, pvlib-based)
#
# Additions in this version:
#   - Significant runtime reduction for "Apply inputs" by vectorizing the full-year coupling:
#       * PV I–V curves are evaluated for all hours in one batched call (with safe fallback).
#       * PV@MPP (with DC resistances) is obtained from the same batched I–V matrix (no second loop).
#       * Electrolyzer V(I) at constant T_op is computed vectorized.
#       * Cooler duty is computed vectorized.
#   - LCOH (Levelized Cost of Hydrogen) calculation:
#       * Adds economics inputs in the sidebar.
#       * Shows LCOH as a top KPI (replacing “Annual HX heat removed (kWh)”).
#       * Adds an LCOH results section + pie chart after the polarization curve.
#
# Notes on LCOH formulation (as requested):
#   - CAPEX is built from material/area inventories (per stack) multiplied by total stack count,
#     plus cooling system cost and manufacturing cost per stack.
#   - CAPEX is levelized by straight-line annualization over the chosen lifetime (no discount rate).
#   - OPEX includes electricity only, based on simulated annual electricity delivered to the electrolyzer.
# --------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pvlib import iam, location, iotools, irradiance, temperature, pvsystem
from pvlib.ivtools.sdm import fit_cec_sam

# =========================
# Streamlit page config
# =========================
st.set_page_config(
    page_title="PV Array → direct DC coupling → Electrolyzer Array Coupling",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Locations dictionary
# =========================
LOCATIONS = {
    "Seville_ES": {"lat": 37.3891, "lon": -5.9845, "alt": 7, "tz": "Europe/Madrid", "label": "Seville, Spain"},
    "Magdeburg_DE": {"lat": 52.1200, "lon": 11.6270, "alt": 55, "tz": "Europe/Berlin", "label": "Magdeburg, Germany"},
    "Lubbock_US": {"lat": 33.5779, "lon": -101.8552, "alt": 1000, "tz": "America/Chicago", "label": "Lubbock, Texas, USA"},
    "Luderitz_NA": {"lat": -26.6480, "lon": 15.1590, "alt": 30, "tz": "Africa/Windhoek", "label": "Lüderitz, Namibia"},
    "Riyadh_SA": {"lat": 24.7136, "lon": 46.6753, "alt": 612, "tz": "Asia/Riyadh", "label": "Riyadh, Saudi Arabia"},
    "Newman_AU": {"lat": -23.3560, "lon": 119.7350, "alt": 545, "tz": "Australia/Perth", "label": "Newman, Western Australia"},
    "Calama_CL": {"lat": -22.4540, "lon": -68.9290, "alt": 2260, "tz": "America/Santiago", "label": "Calama, Chile"},
}

# =========================
# Fixed/default parameters
# =========================
CABLE_TEMP_C = 45.0
ARR_CABLE_TEMP_C = 45.0
CROSS_SEC_MM2 = 4.0
MODULE_DISPLAY_NAME = "Jinko Solar Tiger Neo 48HL4M-DV 460 Wp"
MODULE_DATASHEET_URL = "https://jinkosolar.eu/wp-content/uploads/2025/05/JKM450-475N-48HL4M-DV-Z3-EU-DE.pdf"
CABLE_MATERIAL = "Cu"

# PV module datasheet parameters
Voc_ref = 36.22
Isc_ref = 15.93
Vmp_ref = 30.51
Imp_ref = 15.08
alpha_Isc_pct_per_C = 0.045
beta_Voc_pct_per_C = -0.25
cells_in_series = 48
alpha_sc = (alpha_Isc_pct_per_C / 100.0) * Isc_ref
beta_voc = (beta_Voc_pct_per_C / 100.0) * Voc_ref
gamma_pmp = -0.29

# =========================
# Constants (thermal + electrochem)
# =========================
F = 96485.3329
R = 8.314462618

M_H2_KG_PER_MOL = 0.002016
M_H2O = 0.018015
M_KOH = 0.05611

CP_H2_GAS_MOL_J_PER_MOLK = 28.84
CP_H2O_VAP_MOL_J_PER_MOLK = 33.60

RHO_PTFE_KG_M3 = 2200.0
RHO_PES_KG_M3 = 1370.0

# Full-year IV point count (kept at high fidelity; loop overhead removed via batching)
IV_POINTS_YEAR = 240

# =========================
# Parameter containers
# =========================
@dataclass
class GeometryAndThermal:
    w_KOH: float = 0.30
    rho_KOH: float = 1290.0
    cp_KOH: float = 3800.0

    cp_H2O: float = 4180.0
    h_stack_air: float = 12.0

    f_cell: float = 1.1

    t_electrode: float = 0.5e-3
    t_endplates: float = 10.0e-3
    rho_electrode: float = 8000.0
    cp_electrode: float = 500.0

    t_gasket: float = 0.5e-3
    rho_gasket: float = RHO_PTFE_KG_M3


@dataclass
class ElectrochemParams:
    p_an: float = 1.0
    p_cat: float = 1.0
    p0_bar: float = 1.0

    b_mV_dec: float = 186.6
    j0_ref_A_cm2: float = 8.53e-6
    T_ref_j0_K: float = 373.15
    E_A_j0_kJ_mol: float = 60.0

    c_KOH_mol_L: float = 6.9
    t_sep: float = 1.0e-3
    t_gap: float = 2.0e-3
    tau_sep: float = 1.5
    eps_sep: float = 0.5
    R_contact: float = 60e-7  # Ohm*m2

    dH_evap_H2O: float = 40.7e3

    @property
    def b_V_ln(self) -> float:
        return (self.b_mV_dec * 1e-3) / np.log(10.0)

    @property
    def j0_ref_A_m2(self) -> float:
        return self.j0_ref_A_cm2 * 1e4

    @property
    def E_A_j0_J_mol(self) -> float:
        return self.E_A_j0_kJ_mol * 1e3


@dataclass
class StackDesign:
    n_cells: int
    A_active_m2: float
    A_cell_m2: float
    w_cell_m: float
    l_cell_m: float


# =========================
# Electrochem / thermal helper functions
# =========================
def beta_KOH(w_KOH: float, M_KOH_local: float = M_KOH) -> float:
    return w_KOH / (M_KOH_local * (1.0 - w_KOH))


def p_H2O_sat_KOH(T: float, w_KOH: float) -> float:
    beta = beta_KOH(w_KOH)
    term1 = (-0.01508 * beta - 0.0016788 * beta**2 + 2.25887e-5 * beta**3)
    bracket = (35.4462 - 3343.93 / T - 10.9 * np.log10(T) + 0.0041645 * T)
    term2 = (1.0 - 0.0012062 * beta + 5.6024e-4 * beta**2 - 7.8228e-6 * beta**3) * bracket
    return 10.0 ** (term1 + term2)


def kappa_KOH(c_mol_L: float, T: float) -> float:
    c = c_mol_L
    return 100.0 * (
        -2.041 * c
        - 0.0028 * c**2
        + 0.005332 * c * T
        + 207.2 * c / T
        + 0.001043 * c**3
        - 3e-7 * c**2 * T**2
    )


def reversible_voltage(T: float, ec: ElectrochemParams, geom: GeometryAndThermal) -> float:
    U_rev0 = 1.229 - 0.9e-3 * (T - 298.15)
    p_sat = p_H2O_sat_KOH(T, geom.w_KOH)
    p_O2_an = max(ec.p_an - p_sat, 1e-9)
    p_H2_cat = max(ec.p_cat - p_sat, 1e-9)
    p_ratio = np.sqrt(p_O2_an / ec.p0_bar) * (p_H2_cat / ec.p0_bar)
    return U_rev0 + (R * T) / (2.0 * F) * np.log(p_ratio)


def exchange_current_density(T: float, ec: ElectrochemParams) -> float:
    return ec.j0_ref_A_m2 * np.exp(ec.E_A_j0_J_mol / R * (1.0 / ec.T_ref_j0_K - 1.0 / T))


def activation_overpotential_vec(j_A_m2: np.ndarray, T: float, ec: ElectrochemParams) -> np.ndarray:
    j = np.maximum(np.asarray(j_A_m2, dtype=float), 1e-12)
    j0 = max(exchange_current_density(T, ec), 1e-20)
    eta = ec.b_V_ln * np.log(j / j0)
    return np.maximum(0.0, eta)


def separator_resistance(T: float, ec: ElectrochemParams) -> float:
    kappa = max(kappa_KOH(ec.c_KOH_mol_L, T), 1e-12)
    return (ec.t_sep * ec.tau_sep) / (kappa * ec.eps_sep)


def gap_resistance(T: float, ec: ElectrochemParams) -> float:
    kappa = max(kappa_KOH(ec.c_KOH_mol_L, T), 1e-12)
    return ec.t_gap / kappa


def thermoneutral_voltage_evap(T_stack: float, geom: GeometryAndThermal, ec: ElectrochemParams) -> float:
    p_sat = p_H2O_sat_KOH(T_stack, geom.w_KOH)
    p_O2_an = max(ec.p_an - p_sat, 1e-9)
    p_H2_cat = max(ec.p_cat - p_sat, 1e-9)
    term_p = p_sat * (1.0 / p_H2_cat + 1.0 / (2.0 * p_O2_an))
    return 1.48 + term_p * ec.dH_evap_H2O / (2.0 * F)


def cell_voltage_vec(j_A_m2: np.ndarray, T: float, geom: GeometryAndThermal, ec: ElectrochemParams) -> np.ndarray:
    U_rev = reversible_voltage(T, ec, geom)
    eta = activation_overpotential_vec(j_A_m2, T, ec)
    R_sep = separator_resistance(T, ec)
    R_gap = gap_resistance(T, ec)
    return U_rev + eta + (R_sep + 2.0 * R_gap + ec.R_contact) * np.asarray(j_A_m2, dtype=float)


def water_loss_rates(I_stack: float, T_stack: float, stack: StackDesign, geom: GeometryAndThermal, ec: ElectrochemParams):
    p_sat = p_H2O_sat_KOH(T_stack, geom.w_KOH)
    denom_an = max(ec.p_an - p_sat, 1e-9)
    denom_cat = max(ec.p_cat - p_sat, 1e-9)
    x_vap_an = p_sat / denom_an
    x_vap_cat = p_sat / denom_cat
    m_dot_react = I_stack / (2.0 * F) * M_H2O * stack.n_cells
    m_dot_evap = (I_stack * stack.n_cells * M_H2O) * (x_vap_an / (4.0 * F) + x_vap_cat / (2.0 * F))
    return m_dot_react, m_dot_evap, m_dot_react + m_dot_evap


def compute_stack_geometry_and_thermal(stack: StackDesign, geom: GeometryAndThermal, ec: ElectrochemParams):
    n_cells = stack.n_cells
    n_separator = n_cells
    n_electrodes = n_cells + 1

    h_stack = ec.t_sep * n_separator + geom.t_electrode * n_electrodes + 2.0 * geom.t_endplates + ec.t_gap * n_electrodes
    A_surface_stack = 2.0 * stack.A_cell_m2 + 2.0 * h_stack * (stack.w_cell_m + stack.l_cell_m)

    V_electrodes = n_electrodes * stack.A_cell_m2 * geom.t_electrode
    m_electrodes = V_electrodes * geom.rho_electrode
    C_th_electrodes = m_electrodes * geom.cp_electrode

    V_electrolyte_sep_pores = stack.A_active_m2 * ec.t_sep * ec.eps_sep
    V_electrolyte_gaps = 2 * stack.A_active_m2 * ec.t_gap
    V_electrolyte_total = n_separator * V_electrolyte_sep_pores + n_cells * V_electrolyte_gaps

    m_electrolyte = V_electrolyte_total * geom.rho_KOH
    C_th_electrolyte = m_electrolyte * geom.cp_KOH
    C_th_stack = 1.5 * (C_th_electrodes + C_th_electrolyte)

    A_cell_total = float(n_cells * stack.A_cell_m2)
    A_active_total = float(n_cells * stack.A_active_m2)
    A_separator_total = float(A_cell_total)

    V_separator_solid = float(A_separator_total * ec.t_sep * max(0.0, 1.0 - ec.eps_sep))
    m_separator = float(V_separator_solid * RHO_PES_KG_M3)

    A_gasket_total = float(0.1 * stack.A_cell_m2 * n_cells * 2.0)
    V_gasket = float(A_gasket_total * geom.t_gasket)
    m_gasket = float(V_gasket * geom.rho_gasket)

    return {
        "h_stack_m": h_stack,
        "A_surface_stack_m2": A_surface_stack,
        "n_electrodes": n_electrodes,
        "V_electrodes_m3": V_electrodes,
        "m_electrodes_kg": m_electrodes,
        "C_th_electrodes_JK": C_th_electrodes,
        "V_electrolyte_total_m3": V_electrolyte_total,
        "m_electrolyte_kg": m_electrolyte,
        "C_th_electrolyte_JK": C_th_electrolyte,
        "C_th_stack_JK": C_th_stack,
        "A_cell_total_m2": A_cell_total,
        "A_active_total_m2": A_active_total,
        "A_separator_total_m2": A_separator_total,
        "m_separator_kg": m_separator,
        "A_gasket_total_m2": A_gasket_total,
        "m_gasket_kg": m_gasket,
    }


def p_H2O_sat_magnus_Pa(T_C: float) -> float:
    return 610.78 * np.exp((17.1 * float(T_C)) / (235.0 + float(T_C)))


# =========================
# PV electrical: DC ohmic resistances
# =========================
def _rho_T(material: str, T_C: float) -> float:
    if material.lower() == "al":
        rho20, alpha = 2.82e-8, 0.0039
    else:
        rho20, alpha = 1.724e-8, 0.00393
    return rho20 * (1.0 + alpha * (T_C - 20.0))


def loop_R(one_way_m: float, N_parallel: int, *, module_imax_A: float = 16.0, design_factor: float = 1.25,
           j_cu_A_per_mm2: float = 4.0, T_C: float = 45.0, standard_sizes: list[float] | None = None):
    sizes = standard_sizes or [4, 6, 10, 16, 25, 35, 50, 70, 95, 120, 150, 185, 240, 300, 400, 500, 630, 800, 1000]
    rho = _rho_T("Cu", T_C)
    I_array_max = float(N_parallel) * float(module_imax_A)
    I_design = I_array_max * float(design_factor)
    A_required = I_design / max(j_cu_A_per_mm2, 1e-9)
    A_sel = next((s for s in sizes if s >= A_required - 1e-12), sizes[-1])
    A_m2 = A_sel * 1e-6
    R_loop = rho * (2.0 * one_way_m) / A_m2
    return float(R_loop), float(A_sel), float(A_required)


def string_wiring_R(Ns: int, seg_len_m: float, A_mm2: float, material="Cu", T_C=45.0) -> float:
    if Ns <= 1:
        return 0.0
    rho = _rho_T(material, T_C)
    A_m2 = A_mm2 * 1e-6
    total_one_way = (Ns - 1) * seg_len_m
    return rho * (2.0 * total_one_way) / A_m2


# =========================
# PVGIS fetch + effective irradiance (pre-soiling)
# =========================
@st.cache_data(show_spinner=True, ttl=24 * 3600)
def fetch_pvgis_and_effective_irradiance(site_lat, site_lon, site_alt, site_tz, start_year, surface_tilt, surface_azimuth):
    data, meta = iotools.get_pvgis_hourly(
        latitude=site_lat, longitude=site_lon,
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        start=start_year, end=start_year,
        map_variables=True, components=True, usehorizon=True, url="https://re.jrc.ec.europa.eu/api/v5_3/"
    )

    if "time" in data.columns:
        data = data.set_index("time")
    if data.index.tz is None:
        data = data.tz_localize("UTC")
    data = data.tz_convert(site_tz).sort_index()

    data["poa_global"] = data["poa_direct"] + data["poa_sky_diffuse"] + data["poa_ground_diffuse"]

    loc = location.Location(site_lat, site_lon, tz=site_tz, altitude=site_alt)
    sp = loc.get_solarposition(data.index)
    aoi = irradiance.aoi(surface_tilt, surface_azimuth, sp["apparent_zenith"], sp["azimuth"])

    b0 = 0.04
    iam_beam = iam.ashrae(aoi, b=b0)
    diffuse = iam.marion_diffuse("ashrae", surface_tilt=surface_tilt, b=b0)
    iam_sky, iam_ground = diffuse["sky"], diffuse["ground"]

    E_eff_base = (data["poa_direct"] * iam_beam +
                  data["poa_sky_diffuse"] * iam_sky +
                  data["poa_ground_diffuse"] * iam_ground).clip(lower=0).fillna(0.0)

    return data, meta, E_eff_base


# =========================
# PV Array I–V at a single time (used for IV plots and for the T_op solve hour)
# =========================
def array_iv_at_time(t, Ns, Np, Rstr, Rarr, IL_ser, I0_ser, Rs_ser, Rsh_ser, nNsVth_ser, curve_info_df, iv_points=220):
    try:
        Isc_mod = float(curve_info_df.at[t, "i_sc"])
    except Exception:
        return None
    if (not np.isfinite(Isc_mod)) or Isc_mod <= 0:
        return None

    IL_t = float(IL_ser.loc[t]); I0_t = float(I0_ser.loc[t])
    Rs_t = float(Rs_ser.loc[t]); Rsh_t = float(Rsh_ser.loc[t]); nVth_t = float(nNsVth_ser.loc[t])

    I_mod = np.linspace(0.0, Isc_mod, iv_points)
    V_mod = pvsystem.v_from_i(
        photocurrent=IL_t, saturation_current=I0_t,
        resistance_series=Rs_t, resistance_shunt=Rsh_t,
        nNsVth=nVth_t, current=I_mod, method="lambertw"
    )

    V_str = Ns * V_mod - I_mod * Rstr
    I_arr = Np * I_mod
    V_arr = V_str - I_arr * Rarr

    m = np.isfinite(V_arr) & np.isfinite(I_arr)
    if not np.any(m):
        return None
    V_arr = V_arr[m]; I_arr = I_arr[m]

    idx_pos = np.where(V_arr >= 0)[0]
    if idx_pos.size == 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.0]), 0.0, 0.0, 0.0

    k_last = idx_pos[-1]
    if k_last < len(V_arr) - 1 and V_arr[k_last] > 0 and V_arr[k_last + 1] < 0:
        V0, V1 = V_arr[k_last], V_arr[k_last + 1]
        I0p, I1p = I_arr[k_last], I_arr[k_last + 1]
        I_sc_eff = I0p - V0 * (I1p - I0p) / (V1 - V0)
        V_arr = np.concatenate([V_arr[:k_last + 1], [0.0]])
        I_arr = np.concatenate([I_arr[:k_last + 1], [float(I_sc_eff)]])
    else:
        V_arr[k_last] = max(0.0, V_arr[k_last])

    keep = V_arr >= 0
    V_arr = V_arr[keep]; I_arr = I_arr[keep]
    P_arr = V_arr * I_arr
    if P_arr.size == 0:
        return None
    i_max = int(np.nanargmax(P_arr))
    return V_arr, I_arr, P_arr, float(V_arr[i_max]), float(I_arr[i_max]), float(P_arr[i_max])


# =========================
# Electrolyzer voltage model (array) at constant T_op
# =========================
def v_elec_array_from_Iarray(I_array_A: np.ndarray, T_stack_K: float,
                             stack: StackDesign, geom: GeometryAndThermal, ec: ElectrochemParams,
                             n_series_stacks: int, n_parallel_strings: int) -> np.ndarray:
    I_array_A = np.asarray(I_array_A, dtype=float)
    I_per_stack = I_array_A / max(n_parallel_strings, 1)
    j_A_m2 = np.maximum(I_per_stack / max(stack.A_cell_m2, 1e-12), 1e-12)
    U_cell = cell_voltage_vec(j_A_m2, T_stack_K, geom, ec)
    return n_series_stacks * (stack.n_cells * U_cell)


def find_coupling_point_from_iv(V_pv: np.ndarray, I_pv: np.ndarray,
                                T_stack_K: float,
                                stack: StackDesign, geom: GeometryAndThermal, ec: ElectrochemParams,
                                n_series_stacks: int, n_parallel_strings: int):
    if V_pv is None or I_pv is None or len(I_pv) < 2:
        return 0.0, 0.0, 0.0, False

    V_el = v_elec_array_from_Iarray(I_pv, T_stack_K, stack, geom, ec, n_series_stacks, n_parallel_strings)
    delta = V_pv - V_el
    valid = np.isfinite(delta) & np.isfinite(V_pv) & np.isfinite(I_pv)
    Iv, Vv, dv = I_pv[valid], V_pv[valid], delta[valid]
    if len(Iv) < 2:
        return 0.0, 0.0, 0.0, False

    crossings = np.where(np.diff(np.sign(dv)) != 0)[0]
    if len(crossings) >= 1:
        cand = []
        for k0 in crossings:
            Iv0, Iv1 = Iv[k0], Iv[k0 + 1]
            d0, d1 = dv[k0], dv[k0 + 1]
            if (d1 - d0) == 0:
                I_op = float(Iv0)
            else:
                I_op = float(Iv0 - d0 * (Iv1 - Iv0) / (d1 - d0))
            I_op = float(np.clip(I_op, min(Iv0, Iv1), max(Iv0, Iv1)))
            # linear interp on PV segment
            if (Iv1 - Iv0) != 0:
                V_op = float(Vv[k0] + (Vv[k0 + 1] - Vv[k0]) * (I_op - Iv0) / (Iv1 - Iv0))
            else:
                V_op = float(Vv[k0])
            P_op = float(I_op * V_op)
            cand.append((P_op, I_op, V_op))
        P_op, I_op, V_op = max(cand, key=lambda x: x[0])
        return I_op, V_op, P_op, True

    k = int(np.nanargmin(np.abs(dv)))
    I_op = float(Iv[k]); V_op = float(Vv[k])
    return I_op, V_op, float(I_op * V_op), False


# =========================
# Solve operating temperature ONCE from steady thermal balance at max irradiance hour
# =========================
def solve_operating_temperature_once(
    V_pv: np.ndarray,
    I_pv: np.ndarray,
    T_air_K: float,
    stack: StackDesign, geom: GeometryAndThermal, ec: ElectrochemParams, thermo: dict,
    n_series_stacks: int, n_parallel_strings: int,
    T_min_K: float | None = None,
    T_max_K: float | None = None,
) -> float:
    A_surface = float(thermo["A_surface_stack_m2"])
    h_air = float(geom.h_stack_air)
    T_refill_K = float(T_air_K)

    if T_min_K is None:
        T_min_K = max(273.15, T_air_K)
    if T_max_K is None:
        T_max_K = 473.15

    def thermal_balance(T_K: float) -> float:
        I_op, _, _, _ = find_coupling_point_from_iv(
            V_pv, I_pv, T_K, stack, geom, ec, n_series_stacks, n_parallel_strings
        )
        if I_op <= 0:
            return float(T_air_K - T_K)

        I_stack = I_op / max(n_parallel_strings, 1)
        j_A_m2 = max(I_stack / max(stack.A_cell_m2, 1e-12), 1e-12)
        U_cell = float(cell_voltage_vec(np.array([j_A_m2]), T_K, geom, ec)[0])
        U_th = thermoneutral_voltage_evap(T_K, geom, ec)

        Q_dot_stack = (U_cell - U_th) * I_stack * stack.n_cells
        Q_dot_env = A_surface * h_air * (T_K - T_air_K)

        _, _, m_dot_loss = water_loss_rates(I_stack, T_K, stack, geom, ec)
        Q_refill = m_dot_loss * geom.cp_H2O * (T_refill_K - T_K)

        return float(Q_dot_stack - Q_dot_env + Q_refill)

    lo = float(T_min_K)
    hi = float(T_max_K)
    f_lo = thermal_balance(lo)
    f_hi = thermal_balance(hi)

    if np.sign(f_lo) == np.sign(f_hi):
        for hi_try in [493.15, 523.15, 553.15]:
            f_hi_try = thermal_balance(hi_try)
            if np.sign(f_lo) != np.sign(f_hi_try):
                hi, f_hi = float(hi_try), float(f_hi_try)
                break

    if np.sign(f_lo) == np.sign(f_hi):
        grid = np.linspace(lo, hi, 25)
        vals = np.array([thermal_balance(Tg) for Tg in grid], dtype=float)
        return float(grid[int(np.nanargmin(np.abs(vals)))])

    for _ in range(70):
        mid = 0.5 * (lo + hi)
        f_mid = thermal_balance(mid)
        if not np.isfinite(f_mid):
            break
        if abs(f_mid) < 1e-3:
            return float(mid)
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return float(0.5 * (lo + hi))


# =========================
# Batched full-year PV I–V matrix (fast path) with safe fallback
# =========================
def _batched_v_from_i(IL_v, I0_v, Rs_v, Rsh_v, nNsVth_v, I_mod_mat):
    # Fast path: rely on pvlib broadcasting. If that fails, fall back to a row-wise loop.
    try:
        V_mod_mat = pvsystem.v_from_i(
            photocurrent=IL_v[:, None], saturation_current=I0_v[:, None],
            resistance_series=Rs_v[:, None], resistance_shunt=Rsh_v[:, None],
            nNsVth=nNsVth_v[:, None], current=I_mod_mat, method="lambertw"
        )
        return np.asarray(V_mod_mat, dtype=float)
    except Exception:
        V_rows = []
        for i in range(len(IL_v)):
            V_i = pvsystem.v_from_i(
                photocurrent=float(IL_v[i]), saturation_current=float(I0_v[i]),
                resistance_series=float(Rs_v[i]), resistance_shunt=float(Rsh_v[i]),
                nNsVth=float(nNsVth_v[i]), current=np.asarray(I_mod_mat[i], dtype=float),
                method="lambertw"
            )
            V_rows.append(np.asarray(V_i, dtype=float))
        return np.vstack(V_rows)


def compute_array_iv_matrices_year(
    IL: pd.Series, I0: pd.Series, Rs: pd.Series, Rsh: pd.Series, nNsVth: pd.Series,
    curve_info_df: pd.DataFrame,
    Ns: int, Np: int, R_string: float, R_array: float,
    npts: int = IV_POINTS_YEAR,
):
    Isc = curve_info_df["i_sc"].astype(float).values
    Isc = np.where(np.isfinite(Isc) & (Isc > 0), Isc, 0.0)

    frac = np.linspace(0.0, 1.0, npts)
    I_mod = Isc[:, None] * frac[None, :]

    IL_v = IL.astype(float).values
    I0_v = I0.astype(float).values
    Rs_v = Rs.astype(float).values
    Rsh_v = Rsh.astype(float).values
    nNsVth_v = nNsVth.astype(float).values

    V_mod = _batched_v_from_i(IL_v, I0_v, Rs_v, Rsh_v, nNsVth_v, I_mod)

    V_str = Ns * V_mod - I_mod * float(R_string)
    I_arr = Np * I_mod
    V_arr = V_str - I_arr * float(R_array)

    # emulate truncation at V>=0 by forcing the first negative point to V=0 and then NaN beyond
    neg = np.isfinite(V_arr) & (V_arr < 0)
    first_neg = np.where(neg.any(axis=1), neg.argmax(axis=1), -1)

    rows = np.where(first_neg > 0)[0]
    if rows.size:
        k1 = first_neg[rows]
        k0 = k1 - 1
        V0 = V_arr[rows, k0]
        V1 = V_arr[rows, k1]
        I0r = I_arr[rows, k0]
        I1r = I_arr[rows, k1]
        denom = (V1 - V0)
        denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
        I_sc_eff = I0r - V0 * (I1r - I0r) / denom
        V_arr[rows, k1] = 0.0
        I_arr[rows, k1] = I_sc_eff

    # Cut tails (vectorized)
    kcut = np.where(first_neg >= 0, first_neg, npts - 1)
    col = np.arange(npts)[None, :]
    mask_cut = col > kcut[:, None]
    V_arr = np.where(mask_cut, np.nan, V_arr)
    I_arr = np.where(mask_cut, np.nan, I_arr)

    V_arr = np.where(np.isfinite(V_arr) & (V_arr >= 0), V_arr, np.nan)
    I_arr = np.where(np.isfinite(I_arr), I_arr, np.nan)
    return V_arr, I_arr


# =========================
# LCOH (economics) helpers
# =========================
def compute_lcoh_breakdown(
    *,
    annual_h2_kg: float,
    annual_electricity_kwh: float,
    sec_kwh_per_kg: float = np.nan,
    total_stacks: int,
    thermo_per_stack: dict,
    electrolyte_cost_eur_per_kg: float,
    electrode_cost_eur_per_kg: float,
    gasket_cost_eur_per_m2: float,
    separator_cost_eur_per_m2: float,
    cooling_system_cost_eur: float,
    manufacturing_cost_eur_per_stack: float,
    electricity_cost_eur_per_kwh: float,
    lifetime_years: float,
) -> dict:
    """Compute a simple LCOH with straight-line CAPEX levelization (no discount rate).

    Notes:
      - CAPEX is built from stack-material contributions (electrolyte, electrodes, gaskets, separators),
        manufacturing per stack, plus a lump-sum cooling-system cost.
      - CAPEX is levelized by dividing total CAPEX by lifetime years.
      - OPEX is electricity only.
    """
    lifetime = max(float(lifetime_years), 1e-9)
    annual_h2 = max(float(annual_h2_kg), 1e-12)
    annual_e_kwh = max(float(annual_electricity_kwh), 0.0)

    # --- CAPEX totals (EUR) ---
    m_electrolyte_stack = float(thermo_per_stack.get("m_electrolyte_kg", 0.0))
    m_electrodes_stack = float(thermo_per_stack.get("m_electrodes_kg", 0.0))
    A_gasket_stack = float(thermo_per_stack.get("A_gasket_total_m2", 0.0))
    A_separator_stack = float(thermo_per_stack.get("A_separator_total_m2", 0.0))

    cap_electrolyte = float(total_stacks) * m_electrolyte_stack * float(electrolyte_cost_eur_per_kg)
    cap_electrodes = float(total_stacks) * m_electrodes_stack * float(electrode_cost_eur_per_kg)
    cap_gasket = float(total_stacks) * A_gasket_stack * float(gasket_cost_eur_per_m2)
    cap_separator = float(total_stacks) * A_separator_stack * float(separator_cost_eur_per_m2)
    cap_manufacturing = float(total_stacks) * float(manufacturing_cost_eur_per_stack)
    cap_cooling = float(cooling_system_cost_eur)

    capex_components = {
        "Electrolyte (CAPEX)": cap_electrolyte,
        "Electrodes (CAPEX)": cap_electrodes,
        "Gaskets (CAPEX)": cap_gasket,
        "Separators (CAPEX)": cap_separator,
        "Manufacturing (CAPEX)": cap_manufacturing,
        "Cooling system (CAPEX)": cap_cooling,
    }
    capex_total = float(sum(capex_components.values()))
    capex_annual = capex_total / lifetime  # used for LCOH computation, but not shown as a headline value

    # Total (non-annualized) CAPEX contributions for reporting
    capex_components_total_eur = {k.replace(" (CAPEX)", ""): float(v) for k, v in capex_components.items()}
    capex_components_share = {
        k: (float(v) / capex_total if capex_total > 0 else np.nan) for k, v in capex_components_total_eur.items()
    }

    # --- OPEX (electricity) ---
    opex_electricity_annual = annual_e_kwh * float(electricity_cost_eur_per_kwh)
    opex_electricity_total = opex_electricity_annual * lifetime

    # --- LCOH ---
    annual_total_cost = capex_annual + opex_electricity_annual
    lcoh = annual_total_cost / annual_h2

    # Annualized cost breakdown (EUR/yr) for contribution calculations (not a headline output)
    annualized = {k: (float(v) / lifetime) for k, v in capex_components.items()}
    annualized["Electricity (OPEX)"] = float(opex_electricity_annual)

    # LCOH contributions (EUR/kg)
    perkg = {k: (float(v) / annual_h2) for k, v in annualized.items()}

    return {
        "lcoh_eur_per_kg": float(lcoh),
        "capex_total_eur": float(capex_total),
        "capex_components_total_eur": capex_components_total_eur,
        "capex_components_share": capex_components_share,
        "opex_electricity_annual_eur": float(opex_electricity_annual),
        "opex_electricity_total_eur": float(opex_electricity_total),
        "total_cost_lifetime_eur": float(capex_total + opex_electricity_total),
        "annualized_costs_eur_per_year": annualized,
        "perkg_costs_eur_per_kg": perkg,
        "sec_kwh_per_kg": float(sec_kwh_per_kg),
        "annual_h2_kg": float(annual_h2),
        "annual_electricity_kwh": float(annual_e_kwh),
        "lifetime_years": float(lifetime),
    }


def make_lcoh_pie(costs: dict[str, float], title: str) -> go.Figure:
    """Generic donut chart helper for cost breakdowns."""
    labels = list(costs.keys())
    values = [float(costs[k]) for k in labels]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.45)])
    fig.update_layout(title=title, height=380, margin=dict(t=70, b=30, l=10, r=10))
    return fig


# =========================
# PV electricity price helper (undiscounted LCOE-style average price)
# =========================
def compute_pv_levelized_electricity_price(
    *,
    nameplate_kwp: float,
    capex_eur_per_kwp: float,
    opex_eur_per_kwp_year: float,
    lifetime_years: int,
    degradation_pct_per_year: float,
    year1_energy_kwh: float,
) -> dict:
    """Compute an average electricity price (€/kWh) from PV CAPEX/OPEX and lifetime energy.

    Assumptions:
      - No discount rate (simple levelized average).
      - 'year1_energy_kwh' is the simulated annual PV DC energy delivered to the electrolyzer in year 1.
      - Annual energy degrades geometrically by 'degradation_pct_per_year'.
      - Annual OPEX is constant per kWp-year.
    """
    nameplate_kwp = float(max(nameplate_kwp, 0.0))
    capex_eur_per_kwp = float(max(capex_eur_per_kwp, 0.0))
    opex_eur_per_kwp_year = float(max(opex_eur_per_kwp_year, 0.0))
    lifetime_years = int(max(lifetime_years, 0))
    degradation = float(max(degradation_pct_per_year, 0.0)) / 100.0
    year1_energy_kwh = float(max(year1_energy_kwh, 0.0))

    pv_capex_eur = capex_eur_per_kwp * nameplate_kwp
    pv_opex_total_eur = opex_eur_per_kwp_year * nameplate_kwp * lifetime_years

    if lifetime_years <= 0 or year1_energy_kwh <= 0.0:
        return {
            "lcoe_eur_per_kwh": float("nan"),
            "pv_capex_eur": float(pv_capex_eur),
            "pv_opex_total_eur": float(pv_opex_total_eur),
            "pv_total_energy_kwh": 0.0,
            "lifetime_years": int(lifetime_years),
            "degradation_pct_per_year": float(degradation_pct_per_year),
        }

    if degradation <= 0.0:
        pv_total_energy_kwh = year1_energy_kwh * lifetime_years
    else:
        r = max(0.0, 1.0 - degradation)
        # sum_{k=0}^{L-1} r^k = (1 - r^L) / (1 - r)
        pv_total_energy_kwh = year1_energy_kwh * (1.0 - r**lifetime_years) / max(1.0 - r, 1e-12)

    lcoe_eur_per_kwh = (pv_capex_eur + pv_opex_total_eur) / max(pv_total_energy_kwh, 1e-12)

    return {
        "lcoe_eur_per_kwh": float(lcoe_eur_per_kwh),
        "pv_capex_eur": float(pv_capex_eur),
        "pv_opex_total_eur": float(pv_opex_total_eur),
        "pv_total_energy_kwh": float(pv_total_energy_kwh),
        "lifetime_years": int(lifetime_years),
        "degradation_pct_per_year": float(degradation_pct_per_year),
    }


# =========================
# Sidebar credit
# =========================
with st.sidebar:
    st.markdown(
        """
        <div class="brand-badge">
            Built by: <a href="https://x.com/tobias_franzbrb" target="_blank">Tobias Franz</a>
        </div>
        <style>
        .brand-badge {
            font-size: 0.9rem;
            padding: 6px 10px;
            border-radius: 8px;
            border: 1px solid rgba(0,0,0,0.08);
            margin: 0.25rem 0 0.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Debounced sidebar form: Inputs
# =========================
if "committed" not in st.session_state:
    st.session_state.committed = dict(
        site_key="Seville_ES",
        year=2023,
        sm=6, sd=25,
        wm=12, wd=25,
        SURFACE_TILT=0.0,
        SURFACE_AZIMUTH=180.0,
        N_SERIES=37,
        N_PARALLEL=63,
        SEG_LEN_M=0.8,
        ARR_CABLE_ONE_WAY_M=25.0,
        SOILING_PCT=10,

        N_CELLS_PER_STACK=455,
        ELEC_N_SERIES_STACKS=1,
        ELEC_N_PARALLEL_STRINGS=1,
        FARADAIC_EFF_H2=1.0,

        TAFEL_SLOPE_MV_DEC=186.6,
        J0_REF_A_CM2=8.53e-6,
        T_REF_J0_C=90.0,
        T_SEP_MM=1.0,
        EPS_SEP=0.5,
        TAU_SEP=1.5,
        T_GAP_MM=2.0,
        A_ACTIVE_M2=0.151,

        R_CONTACT_mOHM_CM2=60.0,
        T_ELECTRODE_MM=0.5,
        T_GASKET_MM=0.5,

        COOLER_T_OUT_C=50.0,

        # Economics (LCOH) defaults
        # PV electricity price (derived from PV CAPEX/OPEX & lifetime)
        PV_CAPEX_EUR_PER_KWP=150.0,
        PV_LIFETIME_YEARS=20,
        PV_DEGRADATION_PCT_PER_YEAR=0.4,
        PV_OPEX_EUR_PER_KWP_YEAR=2.0,
        ELECTROLYTE_COST_EUR_PER_KG=5.0,
        ELECTRODE_COST_EUR_PER_KG=5.0,
        GASKET_COST_EUR_PER_M2=30.0,
        SEPARATOR_COST_EUR_PER_M2=30.0,
        COOLING_SYSTEM_COST_EUR=5000.0,
        MANUFACTURING_COST_EUR_PER_STACK=5000.0,
        LIFETIME_YEARS=10.0,
    )

with st.sidebar.form("inputs_form", border=True):
    st.title("Inputs")
    apply_clicked = st.form_submit_button("Apply inputs", type="primary", use_container_width=True)

    st.subheader("General")
    _keys = list(LOCATIONS.keys())
    site_key_idx = _keys.index(st.session_state.committed["site_key"])
    site_key_form = st.selectbox(
        "Location", _keys, index=site_key_idx,
        format_func=lambda k: LOCATIONS[k]["label"],
        key="form_site_key"
    )

    year_form = st.slider("Year", min_value=2005, max_value=2023,
                          value=int(st.session_state.committed["year"]), step=1, key="form_year")

    def md_picker_form(label, cm, cd, year_val):
        colm, cold = st.columns([2, 1])
        month = colm.selectbox(f"{label} — Month", list(range(1, 13)),
                               index=int(cm) - 1, key=f"form_{label}_m")
        day = cold.number_input(f"{label} — Day", min_value=1, max_value=31,
                                value=int(cd), step=1, key=f"form_{label}_d")
        import calendar
        maxd = calendar.monthrange(int(year_val), int(month))[1]
        day = min(int(day), maxd)
        return int(month), int(day)

    sm_form, sd_form = md_picker_form("Summer Day",
                                      st.session_state.committed["sm"],
                                      st.session_state.committed["sd"],
                                      year_form)
    wm_form, wd_form = md_picker_form("Winter Day",
                                      st.session_state.committed["wm"],
                                      st.session_state.committed["wd"],
                                      year_form)

    st.subheader("PV Array")
    SURFACE_TILT_form = st.number_input("Surface tilt (deg)", min_value=0.0, max_value=90.0,
                                        value=float(st.session_state.committed["SURFACE_TILT"]), step=1.0)
    SURFACE_AZIMUTH_form = st.number_input("Surface azimuth (deg, pvlib conv.)", min_value=0.0, max_value=360.0,
                                           value=float(st.session_state.committed["SURFACE_AZIMUTH"]), step=1.0)
    N_SERIES_form = st.number_input("Modules per string (Ns)", min_value=1,
                                    value=int(st.session_state.committed["N_SERIES"]), step=1)
    N_PARALLEL_form = st.number_input("Strings in parallel (Np)", min_value=1,
                                      value=int(st.session_state.committed["N_PARALLEL"]), step=1)
    SEG_LEN_M_form = st.number_input("Inter-module jumper one-way length (m)", min_value=0.0,
                                     value=float(st.session_state.committed["SEG_LEN_M"]), step=0.1)
    ARR_CABLE_ONE_WAY_M_form = st.number_input("Array cable one-way length (m)", min_value=0.0,
                                               value=float(st.session_state.committed["ARR_CABLE_ONE_WAY_M"]), step=1.0)
    SOILING_PCT_form = st.slider("Soiling losses (%)", min_value=0, max_value=20,
                                 value=int(st.session_state.committed["SOILING_PCT"]), step=1)

    st.subheader("Electrolyzer Array")
    N_CELLS_PER_STACK_form = st.number_input("Number of cells per stack", min_value=1,
                                             value=int(st.session_state.committed["N_CELLS_PER_STACK"]), step=10)
    ELEC_N_SERIES_STACKS_form = st.number_input("Stacks in series per string", min_value=1,
                                                value=int(st.session_state.committed["ELEC_N_SERIES_STACKS"]), step=1)
    ELEC_N_PARALLEL_STRINGS_form = st.number_input("Electrolyzer strings in parallel", min_value=1,
                                                   value=int(st.session_state.committed["ELEC_N_PARALLEL_STRINGS"]), step=1)
    FARADAIC_EFF_H2_form = st.number_input("Faradaic efficiency to H₂ (0..1)", min_value=0.0, max_value=1.0,
                                           value=float(st.session_state.committed["FARADAIC_EFF_H2"]), step=0.01)

    st.markdown("**Electrochemistry & Geometry (from dynamic thermal model)**")
    TAFEL_SLOPE_MV_DEC_form = st.number_input("Tafel slope b (mV/dec)", min_value=1.0,
                                              value=float(st.session_state.committed["TAFEL_SLOPE_MV_DEC"]), step=1.0)
    J0_REF_A_CM2_form = st.number_input("Exchange current density j₀,ref (A/cm²)", min_value=0.0,
                                        value=float(st.session_state.committed["J0_REF_A_CM2"]), format="%.8e")
    T_REF_J0_C_form = st.number_input("Reference temperature for j₀ (°C)", min_value=-20.0, max_value=200.0,
                                      value=float(st.session_state.committed["T_REF_J0_C"]), step=1.0)

    T_SEP_MM_form = st.number_input("Separator thickness t_sep (mm)", min_value=0.01,
                                    value=float(st.session_state.committed["T_SEP_MM"]), step=0.05)
    EPS_SEP_form = st.number_input("Separator porosity ε_sep", min_value=0.05, max_value=0.95,
                                   value=float(st.session_state.committed["EPS_SEP"]), step=0.05)
    TAU_SEP_form = st.number_input("Separator tortuosity τ_sep", min_value=1.0, max_value=10.0,
                                   value=float(st.session_state.committed["TAU_SEP"]), step=0.1)
    T_GAP_MM_form = st.number_input("Electrode gap thickness t_gap (mm)", min_value=0.01,
                                    value=float(st.session_state.committed["T_GAP_MM"]), step=0.05)

    A_ACTIVE_M2_form = st.number_input("Active cell area A_active (m²)", min_value=1e-4,
                                       value=float(st.session_state.committed["A_ACTIVE_M2"]), step=0.005)

    R_CONTACT_mOHM_CM2_form = st.number_input(
        "Contact resistance R_contact (mΩ·cm²)",
        min_value=0.0,
        value=float(st.session_state.committed["R_CONTACT_mOHM_CM2"]),
        step=0.05
    )
    T_ELECTRODE_MM_form = st.number_input(
        "Electrode plate thickness t_electrode (mm)",
        min_value=0.01,
        value=float(st.session_state.committed["T_ELECTRODE_MM"]),
        step=0.05
    )
    T_GASKET_MM_form = st.number_input(
        "Gasket thickness t_gasket (mm) [PTFE]",
        min_value=0.01,
        value=float(st.session_state.committed["T_GASKET_MM"]),
        step=0.05
    )
    COOLER_T_OUT_C_form = st.number_input(
        "Condenser / cooler outlet temperature (°C)",
        min_value=0.0,
        max_value=120.0,
        value=float(st.session_state.committed["COOLER_T_OUT_C"]),
        step=1.0
    )

    st.subheader("Economics (LCOH)")
    st.markdown("**PV electricity price model (used for electricity OPEX)**")
    PV_CAPEX_EUR_PER_KWP_form = st.number_input("PV array CAPEX (€/kWp)", min_value=0.0,
                                               value=float(st.session_state.committed["PV_CAPEX_EUR_PER_KWP"]),
                                               step=10.0, format="%.1f")
    PV_LIFETIME_YEARS_form = st.number_input("PV array lifetime (years)", min_value=1,
                                             value=int(st.session_state.committed["PV_LIFETIME_YEARS"]),
                                             step=1)
    PV_DEGRADATION_PCT_PER_YEAR_form = st.number_input("PV output degradation (%/year)", min_value=0.0, max_value=10.0,
                                                       value=float(st.session_state.committed["PV_DEGRADATION_PCT_PER_YEAR"]),
                                                       step=0.05, format="%.2f")
    PV_OPEX_EUR_PER_KWP_YEAR_form = st.number_input("PV array OPEX (€/kWp·year)", min_value=0.0,
                                                   value=float(st.session_state.committed["PV_OPEX_EUR_PER_KWP_YEAR"]),
                                                   step=0.5, format="%.2f")
    LIFETIME_YEARS_form = st.number_input("Electrolysis lifetime (years)", min_value=1.0,
                                          value=float(st.session_state.committed["LIFETIME_YEARS"]),
                                          step=1.0)
    ELECTROLYTE_COST_EUR_PER_KG_form = st.number_input("Electrolyte cost (€/kg)", min_value=0.0,
                                                       value=float(st.session_state.committed["ELECTROLYTE_COST_EUR_PER_KG"]),
                                                       step=0.5)
    ELECTRODE_COST_EUR_PER_KG_form = st.number_input("Electrode cost (€/kg)", min_value=0.0,
                                                     value=float(st.session_state.committed["ELECTRODE_COST_EUR_PER_KG"]),
                                                     step=0.5)
    SEPARATOR_COST_EUR_PER_M2_form = st.number_input("Separator cost (€/m²)", min_value=0.0,
                                                     value=float(st.session_state.committed["SEPARATOR_COST_EUR_PER_M2"]),
                                                     step=1.0)
    GASKET_COST_EUR_PER_M2_form = st.number_input("Gasket cost (€/m²)", min_value=0.0,
                                                  value=float(st.session_state.committed["GASKET_COST_EUR_PER_M2"]),
                                                  step=1.0)
    COOLING_SYSTEM_COST_EUR_form = st.number_input("Cooling system cost (EUR)", min_value=0.0,
                                                   value=float(st.session_state.committed["COOLING_SYSTEM_COST_EUR"]),
                                                   step=1000.0)
    MANUFACTURING_COST_EUR_PER_STACK_form = st.number_input("Manufacturing cost per stack (EUR/stack)", min_value=0.0,
                                                            value=float(st.session_state.committed["MANUFACTURING_COST_EUR_PER_STACK"]),
                                                            step=500.0)

if apply_clicked:
    st.session_state.committed = dict(
        site_key=site_key_form,
        year=int(year_form),
        sm=int(sm_form), sd=int(sd_form),
        wm=int(wm_form), wd=int(wd_form),

        SURFACE_TILT=float(SURFACE_TILT_form),
        SURFACE_AZIMUTH=float(SURFACE_AZIMUTH_form),
        N_SERIES=int(N_SERIES_form),
        N_PARALLEL=int(N_PARALLEL_form),
        SEG_LEN_M=float(SEG_LEN_M_form),
        ARR_CABLE_ONE_WAY_M=float(ARR_CABLE_ONE_WAY_M_form),
        SOILING_PCT=int(SOILING_PCT_form),

        N_CELLS_PER_STACK=int(N_CELLS_PER_STACK_form),
        ELEC_N_SERIES_STACKS=int(ELEC_N_SERIES_STACKS_form),
        ELEC_N_PARALLEL_STRINGS=int(ELEC_N_PARALLEL_STRINGS_form),
        FARADAIC_EFF_H2=float(FARADAIC_EFF_H2_form),

        TAFEL_SLOPE_MV_DEC=float(TAFEL_SLOPE_MV_DEC_form),
        J0_REF_A_CM2=float(J0_REF_A_CM2_form),
        T_REF_J0_C=float(T_REF_J0_C_form),
        T_SEP_MM=float(T_SEP_MM_form),
        EPS_SEP=float(EPS_SEP_form),
        TAU_SEP=float(TAU_SEP_form),
        T_GAP_MM=float(T_GAP_MM_form),
        A_ACTIVE_M2=float(A_ACTIVE_M2_form),

        R_CONTACT_mOHM_CM2=float(R_CONTACT_mOHM_CM2_form),
        T_ELECTRODE_MM=float(T_ELECTRODE_MM_form),

        T_GASKET_MM=float(T_GASKET_MM_form),
        COOLER_T_OUT_C=float(COOLER_T_OUT_C_form),
        PV_CAPEX_EUR_PER_KWP=float(PV_CAPEX_EUR_PER_KWP_form),
        PV_LIFETIME_YEARS=int(PV_LIFETIME_YEARS_form),
        PV_DEGRADATION_PCT_PER_YEAR=float(PV_DEGRADATION_PCT_PER_YEAR_form),
        PV_OPEX_EUR_PER_KWP_YEAR=float(PV_OPEX_EUR_PER_KWP_YEAR_form),
        LIFETIME_YEARS=float(LIFETIME_YEARS_form),
        ELECTROLYTE_COST_EUR_PER_KG=float(ELECTROLYTE_COST_EUR_PER_KG_form),
        ELECTRODE_COST_EUR_PER_KG=float(ELECTRODE_COST_EUR_PER_KG_form),
        SEPARATOR_COST_EUR_PER_M2=float(SEPARATOR_COST_EUR_PER_M2_form),
        GASKET_COST_EUR_PER_M2=float(GASKET_COST_EUR_PER_M2_form),
        COOLING_SYSTEM_COST_EUR=float(COOLING_SYSTEM_COST_EUR_form),
        MANUFACTURING_COST_EUR_PER_STACK=float(MANUFACTURING_COST_EUR_PER_STACK_form),
    )

# =========================
# Use committed values
# =========================
c = st.session_state.committed
site_key = c["site_key"]
year = c["year"]
sm, sd = c["sm"], c["sd"]
wm, wd = c["wm"], c["wd"]

SURFACE_TILT = c["SURFACE_TILT"]
SURFACE_AZIMUTH = c["SURFACE_AZIMUTH"]
N_SERIES = c["N_SERIES"]
N_PARALLEL = c["N_PARALLEL"]
SEG_LEN_M = c["SEG_LEN_M"]
ARR_CABLE_ONE_WAY_M = c["ARR_CABLE_ONE_WAY_M"]
SOILING_PCT = c["SOILING_PCT"]

N_CELLS_PER_STACK = c["N_CELLS_PER_STACK"]
ELEC_N_SERIES_STACKS = c["ELEC_N_SERIES_STACKS"]
ELEC_N_PARALLEL_STRINGS = c["ELEC_N_PARALLEL_STRINGS"]
FARADAIC_EFF_H2 = c["FARADAIC_EFF_H2"]

TAFEL_SLOPE_MV_DEC = c["TAFEL_SLOPE_MV_DEC"]
J0_REF_A_CM2 = c["J0_REF_A_CM2"]
T_REF_J0_C = c["T_REF_J0_C"]
T_SEP_MM = c["T_SEP_MM"]
EPS_SEP = c["EPS_SEP"]
TAU_SEP = c["TAU_SEP"]
T_GAP_MM = c["T_GAP_MM"]
A_ACTIVE_M2 = c["A_ACTIVE_M2"]

R_CONTACT_mOHM_CM2 = c["R_CONTACT_mOHM_CM2"]
T_ELECTRODE_MM = c["T_ELECTRODE_MM"]

T_GASKET_MM = c["T_GASKET_MM"]
COOLER_T_OUT_C = c["COOLER_T_OUT_C"]
COOLER_T_OUT_K = COOLER_T_OUT_C + 273.15

# Economics
PV_CAPEX_EUR_PER_KWP = c["PV_CAPEX_EUR_PER_KWP"]
PV_LIFETIME_YEARS = c["PV_LIFETIME_YEARS"]
PV_DEGRADATION_PCT_PER_YEAR = c["PV_DEGRADATION_PCT_PER_YEAR"]
PV_OPEX_EUR_PER_KWP_YEAR = c["PV_OPEX_EUR_PER_KWP_YEAR"]
LIFETIME_YEARS = c["LIFETIME_YEARS"]
ELECTROLYTE_COST_EUR_PER_KG = c["ELECTROLYTE_COST_EUR_PER_KG"]
ELECTRODE_COST_EUR_PER_KG = c["ELECTRODE_COST_EUR_PER_KG"]
GASKET_COST_EUR_PER_M2 = c["GASKET_COST_EUR_PER_M2"]
SEPARATOR_COST_EUR_PER_M2 = c["SEPARATOR_COST_EUR_PER_M2"]
COOLING_SYSTEM_COST_EUR = c["COOLING_SYSTEM_COST_EUR"]
MANUFACTURING_COST_EUR_PER_STACK = c["MANUFACTURING_COST_EUR_PER_STACK"]

WINTER_DAY = f"{year}-{wm:02d}-{wd:02d}"
SUMMER_DAY = f"{year}-{sm:02d}-{sd:02d}"

cfg = LOCATIONS[site_key]
SITE_LAT, SITE_LON, SITE_ALT, SITE_TZ = cfg["lat"], cfg["lon"], cfg["alt"], cfg["tz"]
SITE_NAME = cfg.get("label", site_key)

# Header
st.title("PV Array → direct DC coupling → Electrolyzer Array")
st.caption(f"Site: **{SITE_NAME}** · Year: **{year}** · Summer Day: **{SUMMER_DAY}** · Winter Day: **{WINTER_DAY}**")

# =========================
# Cable resistances
# =========================
R_string = string_wiring_R(N_SERIES, SEG_LEN_M, CROSS_SEC_MM2, material=CABLE_MATERIAL, T_C=CABLE_TEMP_C)
R_array, ARR_CABLE_CROSS_SEC_MM2, ARR_CABLE_CSA_REQUIRED = loop_R(
    one_way_m=ARR_CABLE_ONE_WAY_M, N_parallel=N_PARALLEL, module_imax_A=16.0,
    design_factor=1.25, j_cu_A_per_mm2=4.0, T_C=ARR_CABLE_TEMP_C
)

# =========================
# CEC module parameters (fit once)
# =========================
I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = fit_cec_sam(
    celltype="monoSi",
    v_mp=Vmp_ref, i_mp=Imp_ref, v_oc=Voc_ref, i_sc=Isc_ref,
    alpha_sc=alpha_sc, beta_voc=beta_voc, gamma_pmp=gamma_pmp,
    cells_in_series=cells_in_series, temp_ref=25
)

# =========================
# Full-year coupling (cached) — economics inputs intentionally NOT included (fast LCOH updates)
# =========================
@st.cache_data(show_spinner=True, ttl=6 * 3600)
def compute_year_coupling_and_outputs(
    site_lat, site_lon, site_alt, site_tz,
    year,
    surface_tilt, surface_azimuth,
    N_SERIES, N_PARALLEL, R_string, R_array,
    SOILING_PCT,
    # electrolyzer inputs
    N_CELLS_PER_STACK, ELEC_N_SERIES_STACKS, ELEC_N_PARALLEL_STRINGS, FARADAIC_EFF_H2,
    TAFEL_SLOPE_MV_DEC, J0_REF_A_CM2, T_REF_J0_C, T_SEP_MM, EPS_SEP, TAU_SEP, T_GAP_MM, A_ACTIVE_M2,
    R_CONTACT_mOHM_CM2, T_ELECTRODE_MM,
    T_GASKET_MM,
    COOLER_T_OUT_C,
):
    data, meta, E_eff_base = fetch_pvgis_and_effective_irradiance(
        site_lat, site_lon, site_alt, site_tz, year, surface_tilt, surface_azimuth
    )
    E_eff = E_eff_base * (1.0 - SOILING_PCT / 100.0)

    sapm = temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    data["cell_temperature"] = temperature.sapm_cell(
        poa_global=data["poa_global"],
        temp_air=data["temp_air"],
        wind_speed=data["wind_speed"],
        **sapm
    )

    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_cec(
        effective_irradiance=E_eff,
        temp_cell=data["cell_temperature"],
        alpha_sc=alpha_sc,
        a_ref=a_ref, I_L_ref=I_L_ref, I_o_ref=I_o_ref, R_sh_ref=R_sh_ref, R_s=R_s,
        Adjust=Adjust,
        EgRef=1.121, dEgdT=-0.0002677
    )

    curve_info_local = pvsystem.singlediode(
        photocurrent=IL, saturation_current=I0,
        resistance_series=Rs, resistance_shunt=Rsh, nNsVth=nNsVth,
        method="lambertw"
    )

    # Build electrolyzer physical parameters
    geom = GeometryAndThermal(
        t_electrode=float(T_ELECTRODE_MM) * 1e-3,
        t_gasket=float(T_GASKET_MM) * 1e-3,
    )
    ec = ElectrochemParams(
        b_mV_dec=float(TAFEL_SLOPE_MV_DEC),
        j0_ref_A_cm2=float(J0_REF_A_CM2),
        T_ref_j0_K=float(T_REF_J0_C + 273.15),
        t_sep=float(T_SEP_MM * 1e-3),
        eps_sep=float(EPS_SEP),
        tau_sep=float(TAU_SEP),
        t_gap=float(T_GAP_MM * 1e-3),
        # preserved original mapping:
        R_contact=float(R_CONTACT_mOHM_CM2) * 1e-7,
    )

    A_active = float(A_ACTIVE_M2)
    A_cell = float(A_active * geom.f_cell)
    w_cell = float(np.sqrt(A_cell))
    stack = StackDesign(
        n_cells=int(N_CELLS_PER_STACK),
        A_active_m2=A_active,
        A_cell_m2=A_cell,
        w_cell_m=w_cell,
        l_cell_m=w_cell,
    )
    thermo = compute_stack_geometry_and_thermal(stack, geom, ec)

    # Max effective irradiance hour (IAM+soiling)
    t_max = E_eff.astype(float).fillna(0.0).idxmax()
    T_air_max_K = float(data.loc[t_max, "temp_air"] + 273.15)

    res_max = array_iv_at_time(
        t_max, N_SERIES, N_PARALLEL, R_string, R_array,
        IL, I0, Rs, Rsh, nNsVth, curve_info_local, iv_points=260
    )
    if res_max is None:
        T_op_K = float(T_air_max_K)
    else:
        V_pv_max, I_pv_max, *_ = res_max
        T_op_K = solve_operating_temperature_once(
            V_pv=V_pv_max, I_pv=I_pv_max, T_air_K=T_air_max_K,
            stack=stack, geom=geom, ec=ec, thermo=thermo,
            n_series_stacks=int(ELEC_N_SERIES_STACKS),
            n_parallel_strings=int(ELEC_N_PARALLEL_STRINGS),
        )

    # ---------- FAST full-year coupling (batched I–V) ----------
    V_arr_mat, I_arr_mat = compute_array_iv_matrices_year(
        IL, I0, Rs, Rsh, nNsVth, curve_info_local,
        Ns=int(N_SERIES), Np=int(N_PARALLEL),
        R_string=float(R_string), R_array=float(R_array),
        npts=IV_POINTS_YEAR
    )

    # PV@MPP (with DC resistances) from same matrix
    P_arr_mat = V_arr_mat * I_arr_mat
    Pmp_fullR_W = pd.Series(np.nanmax(P_arr_mat, axis=1), index=data.index).fillna(0.0)

    # Electrolyzer array voltage on same current grid
    V_el_mat = v_elec_array_from_Iarray(
        I_arr_mat, float(T_op_K), stack, geom, ec,
        n_series_stacks=int(ELEC_N_SERIES_STACKS),
        n_parallel_strings=int(ELEC_N_PARALLEL_STRINGS),
    )

    # Operating points per hour (small loop; expensive parts already vectorized)
    nT = len(data.index)
    I_op = np.zeros(nT, dtype=float)
    V_op = np.zeros(nT, dtype=float)
    P_op = np.zeros(nT, dtype=float)

    for i in range(nT):
        Iv = I_arr_mat[i, :]
        Vv = V_arr_mat[i, :]
        Ve = V_el_mat[i, :]
        valid = np.isfinite(Iv) & np.isfinite(Vv) & np.isfinite(Ve)
        if not np.any(valid):
            continue
        Iv = Iv[valid]; Vv = Vv[valid]; dv = (Vv - Ve[valid])
        if Iv.size < 2:
            continue

        crossings = np.where(np.diff(np.sign(dv)) != 0)[0]
        if crossings.size:
            bestP, bestI, bestV = -1.0, 0.0, 0.0
            for k0 in crossings:
                I0c, I1c = Iv[k0], Iv[k0 + 1]
                d0, d1 = dv[k0], dv[k0 + 1]
                if (d1 - d0) == 0:
                    I_c = float(I0c)
                else:
                    I_c = float(I0c - d0 * (I1c - I0c) / (d1 - d0))
                I_c = float(np.clip(I_c, min(I0c, I1c), max(I0c, I1c)))
                if (I1c - I0c) != 0:
                    V_c = float(Vv[k0] + (Vv[k0 + 1] - Vv[k0]) * (I_c - I0c) / (I1c - I0c))
                else:
                    V_c = float(Vv[k0])
                P_c = I_c * V_c
                if P_c > bestP:
                    bestP, bestI, bestV = P_c, I_c, V_c
            I_op[i], V_op[i], P_op[i] = bestI, bestV, bestP
        else:
            k = int(np.nanargmin(np.abs(dv)))
            I_op[i], V_op[i] = float(Iv[k]), float(Vv[k])
            P_op[i] = I_op[i] * V_op[i]

    coupling_results = pd.DataFrame(
        {"I_op_A": I_op, "V_op_V": V_op, "P_op_W": P_op},
        index=data.index
    ).fillna(0.0)

    # Electrolyzer derived quantities at constant T_op
    I_array = coupling_results["I_op_A"].astype(float).values
    I_stack = I_array / max(int(ELEC_N_PARALLEL_STRINGS), 1)
    j_A_m2 = np.maximum(I_stack / max(stack.A_cell_m2, 1e-12), 1e-12)
    j_A_cm2 = j_A_m2 / 1e4

    U_cell = cell_voltage_vec(j_A_m2, float(T_op_K), geom, ec)
    U_stack = stack.n_cells * U_cell
    U_string = int(ELEC_N_SERIES_STACKS) * U_stack

    coupling_results["j_cell_Acm2"] = j_A_cm2
    coupling_results["U_cell_V"] = U_cell
    coupling_results["U_stack_V"] = U_stack
    coupling_results["U_string_V"] = U_string

    # PV breakdown at operating point
    idx = coupling_results.index
    I_module = (coupling_results["I_op_A"] / max(int(N_PARALLEL), 1)).astype(float).values
    mask = I_module > 0

    V_module = np.zeros(len(idx), dtype=float)
    if np.any(mask):
        try:
            V_mod_vals = pvsystem.v_from_i(
                photocurrent=IL.values[mask],
                saturation_current=I0.values[mask],
                resistance_series=Rs.values[mask],
                resistance_shunt=Rsh.values[mask],
                nNsVth=nNsVth.values[mask],
                current=I_module[mask],
                method="lambertw"
            )
            V_module[mask] = np.where(np.isfinite(V_mod_vals), np.asarray(V_mod_vals, dtype=float), 0.0)
        except Exception:
            for k in np.where(mask)[0]:
                V_module[k] = float(pvsystem.v_from_i(
                    photocurrent=float(IL.values[k]),
                    saturation_current=float(I0.values[k]),
                    resistance_series=float(Rs.values[k]),
                    resistance_shunt=float(Rsh.values[k]),
                    nNsVth=float(nNsVth.values[k]),
                    current=float(I_module[k]),
                    method="lambertw"
                ))

    V_string = np.maximum(0.0, (int(N_SERIES) * V_module - I_module * float(R_string)))
    V_array = np.maximum(0.0, (V_string - coupling_results["I_op_A"].astype(float).values * float(R_array)))

    coupling_results["Single_PV_module_voltage"] = V_module
    coupling_results["Single_PV_module_current"] = I_module
    coupling_results["Single_PV_module_power"] = V_module * I_module
    coupling_results["PV_string_current"] = I_module
    coupling_results["PV_string_voltage"] = V_string
    coupling_results["PV_string_power"] = V_string * I_module
    coupling_results["PV_array_voltage"] = V_array
    coupling_results["PV_array_current"] = coupling_results["I_op_A"].astype(float).values
    coupling_results["PV_array_power"] = V_array * coupling_results["I_op_A"].astype(float).values

    coupling_results["Single_Stack_voltage"] = coupling_results["U_stack_V"].astype(float).values
    coupling_results["Single_Stack_current"] = I_stack
    coupling_results["Single_Stack_Power"] = coupling_results["Single_Stack_voltage"] * coupling_results["Single_Stack_current"]

    coupling_results["Electrolyzer_string_voltage"] = coupling_results["U_string_V"].astype(float).values
    coupling_results["Elecgtrolyzer_string_current"] = I_stack
    coupling_results["Electrolyzer_string_power"] = coupling_results["Electrolyzer_string_voltage"] * coupling_results["Elecgtrolyzer_string_current"]

    coupling_results["Electrolyzer_array_voltage"] = coupling_results["U_string_V"].astype(float).values
    coupling_results["Electrolyzer_array_current"] = coupling_results["I_op_A"].astype(float).values
    coupling_results["Electrolyzer_array_power"] = coupling_results["Electrolyzer_array_voltage"] * coupling_results["Electrolyzer_array_current"]

    # Energies
    E_mpp_fullR_kWh = float(Pmp_fullR_W.sum() / 1000.0)
    E_coupled_kWh = float(coupling_results["P_op_W"].sum() / 1000.0)

    # H2 production
    ts = coupling_results.index
    dt_s = (ts.to_series().shift(-1) - ts.to_series()).dt.total_seconds()
    dt_s.iloc[-1] = dt_s.median() if np.isfinite(dt_s.median()) else 3600.0
    dt_s = dt_s.values

    total_stacks = int(ELEC_N_SERIES_STACKS * ELEC_N_PARALLEL_STRINGS)
    mol_H2 = (I_stack * stack.n_cells * total_stacks * dt_s) / (2.0 * F) * float(FARADAIC_EFF_H2)
    kg_H2_total = float(np.nansum(mol_H2 * M_H2_KG_PER_MOL))
    SEC_kWh_per_kg = (E_coupled_kWh / kg_H2_total) if kg_H2_total > 0 else np.nan

    # Cooler duty (vectorized)
    T_out_K = float(COOLER_T_OUT_C) + 273.15
    Q_cooler_W = np.zeros(len(coupling_results), dtype=float)
    m_cond_kg_s = np.zeros(len(coupling_results), dtype=float)

    if float(T_op_K) > T_out_K:
        p_sat_in_Pa = float(p_H2O_sat_KOH(float(T_op_K), geom.w_KOH) * 1e5)
        p_sat_out_Pa = float(p_H2O_sat_magnus_Pa(float(COOLER_T_OUT_C)))
        p_total_Pa = float(max(ec.p_cat, 1e-9)) * 1e5

        y_in = float(np.clip(p_sat_in_Pa / p_total_Pa, 0.0, 0.999999))
        y_out = float(np.clip(p_sat_out_Pa / p_total_Pa, 0.0, 0.999999))
        r_in = y_in / max(1.0 - y_in, 1e-12)
        r_out = y_out / max(1.0 - y_out, 1e-12)
        r_cond = max(0.0, r_in - r_out)

        dT = float(T_op_K - T_out_K)

        n_H2_mol_s = (I_stack * stack.n_cells * total_stacks / (2.0 * F)) * float(FARADAIC_EFF_H2)
        n_H2O_in = n_H2_mol_s * r_in
        n_cond = n_H2_mol_s * r_cond

        Q_sens = (n_H2_mol_s * CP_H2_GAS_MOL_J_PER_MOLK + n_H2O_in * CP_H2O_VAP_MOL_J_PER_MOLK) * dT
        Q_lat = n_cond * float(ec.dH_evap_H2O)

        Q = np.maximum(0.0, Q_sens + Q_lat)
        mcond = np.maximum(0.0, n_cond) * M_H2O

        on = I_stack > 0
        Q_cooler_W[on] = Q[on]
        m_cond_kg_s[on] = mcond[on]

    coupling_results["cooler_Qdot_W"] = Q_cooler_W
    coupling_results["cooler_mcond_kg_s"] = m_cond_kg_s
    coupling_results["T_air_C"] = data["temp_air"].astype(float).values

    E_cooler_kWh = float(np.sum(Q_cooler_W * dt_s) / 3.6e6)
    m_cond_total_kg = float(np.sum(m_cond_kg_s * dt_s))
    Q_cooler_max_W = float(np.nanmax(Q_cooler_W)) if len(Q_cooler_W) else 0.0

    util_pct = (100.0 * E_coupled_kWh / E_mpp_fullR_kWh) if E_mpp_fullR_kWh > 0 else np.nan

    # maxima dict
    _added_cols = [
        "Single_PV_module_voltage", "Single_PV_module_current", "Single_PV_module_power",
        "PV_string_current", "PV_string_voltage", "PV_string_power",
        "PV_array_voltage", "PV_array_current", "PV_array_power",
        "Single_Stack_voltage", "Single_Stack_current", "Single_Stack_Power",
        "Electrolyzer_string_voltage", "Elecgtrolyzer_string_current", "Electrolyzer_string_power",
        "Electrolyzer_array_voltage", "Electrolyzer_array_current", "Electrolyzer_array_power",
    ]
    maxima = {col: float(np.nanmax(coupling_results[col].values)) if col in coupling_results.columns else np.nan
              for col in _added_cols}

    j_vals = coupling_results["j_cell_Acm2"].astype(float).values
    j_pos = j_vals[np.isfinite(j_vals) & (j_vals > 0)]
    max_current_density = float(np.nanmax(j_vals)) if len(j_vals) else np.nan
    min_current_density = float(np.nanmin(j_pos)) if len(j_pos) else 0.0

    summary = {
        "T_op_K": float(T_op_K),
        "t_max": str(pd.to_datetime(t_max)),
        "E_mpp_fullR_kWh": float(E_mpp_fullR_kWh),
        "E_coupled_kWh": float(E_coupled_kWh),
        "util_pct": float(util_pct),
        "SEC_kWh_per_kg": float(SEC_kWh_per_kg),
        "kg_H2_total": float(kg_H2_total),
        "Max_electrolyzer_array_power_W": float(np.nanmax(coupling_results["P_op_W"].values)) if len(coupling_results) else 0.0,
        "max_current_density": float(max_current_density),
        "min_current_density": float(min_current_density),
        "E_cooler_kWh": float(E_cooler_kWh),
        "Q_cooler_max_W": float(Q_cooler_max_W),
        "m_cond_total_kg": float(m_cond_total_kg),
        "COOLER_T_OUT_C": float(COOLER_T_OUT_C),
        "total_stacks": int(total_stacks),
    }

    return data, curve_info_local, (IL, I0, Rs, Rsh, nNsVth), Pmp_fullR_W, coupling_results, maxima, summary, thermo, geom, ec, stack


with st.spinner("Computing coupling, operating temperature, and annual results..."):
    data2, curve_info2, pv_params2, Pmp_fullR_W, coupling_results, maxima, summary, thermo, geom, ec, stack = compute_year_coupling_and_outputs(
        SITE_LAT, SITE_LON, SITE_ALT, SITE_TZ,
        year,
        SURFACE_TILT, SURFACE_AZIMUTH,
        N_SERIES, N_PARALLEL, R_string, R_array,
        SOILING_PCT,
        N_CELLS_PER_STACK, ELEC_N_SERIES_STACKS, ELEC_N_PARALLEL_STRINGS, FARADAIC_EFF_H2,
        TAFEL_SLOPE_MV_DEC, J0_REF_A_CM2, T_REF_J0_C, T_SEP_MM, EPS_SEP, TAU_SEP, T_GAP_MM, A_ACTIVE_M2,
        R_CONTACT_mOHM_CM2, T_ELECTRODE_MM,
        T_GASKET_MM,
        COOLER_T_OUT_C,
    )

# =========================
# Unpack key results
# =========================
E_mpp_fullR_kWh = summary["E_mpp_fullR_kWh"]
E_coupled_kWh = summary["E_coupled_kWh"]
util_pct = summary["util_pct"]
SEC_kWh_per_kg = summary["SEC_kWh_per_kg"]
kg_H2_total = summary["kg_H2_total"]
Max_electrolyzer_array_power = summary["Max_electrolyzer_array_power_W"]
max_current_density = summary["max_current_density"]
min_current_density = summary["min_current_density"]

T_op_C = summary["T_op_K"] - 273.15
E_cooler_kWh = summary["E_cooler_kWh"]
Q_cooler_max_kW = summary["Q_cooler_max_W"] / 1000.0
m_cond_total_kg = summary["m_cond_total_kg"]
total_stacks = int(summary["total_stacks"])

# =========================
# LCOH (computed outside the heavy cache)
# =========================
# PV-derived electricity price (average €/kWh over PV lifetime, no discounting)
pv_nameplate_kwp = 0.46 * float(N_SERIES) * float(N_PARALLEL)  # matches nameplate shown in the configuration summary
pv_elec_price = compute_pv_levelized_electricity_price(
    nameplate_kwp=pv_nameplate_kwp,
    capex_eur_per_kwp=float(PV_CAPEX_EUR_PER_KWP),
    opex_eur_per_kwp_year=float(PV_OPEX_EUR_PER_KWP_YEAR),
    lifetime_years=int(PV_LIFETIME_YEARS),
    degradation_pct_per_year=float(PV_DEGRADATION_PCT_PER_YEAR),
    year1_energy_kwh=float(E_coupled_kWh),
)
electricity_price_eur_per_kwh = pv_elec_price["lcoe_eur_per_kwh"]

lcoh = compute_lcoh_breakdown(
    electrolyte_cost_eur_per_kg=ELECTROLYTE_COST_EUR_PER_KG,
    electrode_cost_eur_per_kg=ELECTRODE_COST_EUR_PER_KG,
    gasket_cost_eur_per_m2=GASKET_COST_EUR_PER_M2,
    separator_cost_eur_per_m2=SEPARATOR_COST_EUR_PER_M2,
    cooling_system_cost_eur=COOLING_SYSTEM_COST_EUR,
    lifetime_years=LIFETIME_YEARS,
    electricity_cost_eur_per_kwh=electricity_price_eur_per_kwh,
    manufacturing_cost_eur_per_stack=MANUFACTURING_COST_EUR_PER_STACK,
    thermo_per_stack=thermo,
    total_stacks=total_stacks,
    annual_h2_kg=kg_H2_total,
    annual_electricity_kwh=E_coupled_kWh,
    sec_kwh_per_kg=SEC_kWh_per_kg,
)

LCOH_EUR_PER_KG = lcoh["lcoh_eur_per_kg"]

# ======================================================
# Plotly helpers (kept structure)
# ======================================================
def _hourly_times_for_day(index, day_str):
    m = index.strftime("%Y-%m-%d") == day_str
    return index[m]


def _coupling_points_for_day(day_str):
    times = _hourly_times_for_day(coupling_results.index, day_str)
    return coupling_results.loc[times, ["V_op_V", "I_op_A"]].dropna()


def _power_series_for_day(day_str):
    times = _hourly_times_for_day(curve_info2.index, day_str)
    pmp = Pmp_fullR_W.reindex(times).fillna(0.0)
    pcpl = coupling_results.loc[times, "P_op_W"].fillna(0.0)
    return pmp, pcpl


def _mk_iv_figure_pair(winter_day, summer_day):
    MPP_MARKER = "circle"
    LINESTYLES = ["solid", "dash", "dashdot", "dot"]
    BLUES = ['#2c7bb6', '#3a89c9', '#4f9bd4', '#66addf']
    GREENS = ['#1a9850', '#4daf4a', '#66bd63', '#99d594']
    ORANGES = ['#fdae61', '#f98e52', '#f46d43', '#f2703d']
    REDS = ['#d73027', '#d7191c']

    IL, I0, Rs, Rsh, nNsVth = pv_params2

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Array I-U — {winter_day} (Winter)  |  Ns={N_SERIES}, Np={N_PARALLEL}",
            f"Array I-U — {summer_day} (Summer)  |  Ns={N_SERIES}, Np={N_PARALLEL}",
        ),
        horizontal_spacing=0.08
    )

    def _class_color(i_class: str, counters: dict) -> str:
        if i_class == 'blue':
            c = BLUES[counters['blue'] % len(BLUES)]; counters['blue'] += 1; return c
        if i_class == 'green':
            c = GREENS[counters['green'] % len(GREENS)]; counters['green'] += 1; return c
        if i_class == 'orange':
            c = ORANGES[counters['orange'] % len(ORANGES)]; counters['orange'] += 1; return c
        c = REDS[counters['red'] % len(REDS)]; counters['red'] += 1; return c

    def _class_from_Imax(Imax_val, q1, q2, q3):
        if Imax_val <= q1: return 'blue'
        if Imax_val <= q2: return 'green'
        if Imax_val <= q3: return 'orange'
        return 'red'

    def _add_iv_panel(day_str, ccol):
        times_day = _hourly_times_for_day(curve_info2.index, day_str)
        hour_Imax, iv_cache = {}, {}
        imax_mpp, vmax_pv_curves, vmax_el_curve = 0.0, 0.0, 0.0

        for t in times_day:
            res = array_iv_at_time(
                t, N_SERIES, N_PARALLEL, R_string, R_array,
                IL, I0, Rs, Rsh, nNsVth, curve_info2, iv_points=220
            )
            if res is None:
                continue
            V_arr, I_arr, _, Vmp, Imp, _ = res
            iv_cache[t] = (V_arr, I_arr, Vmp, Imp)
            h = int(pd.to_datetime(t).hour)
            if len(I_arr):
                hour_Imax[h] = max(hour_Imax.get(h, 0.0), float(np.nanmax(I_arr)))
            if len(V_arr):
                vmax_pv_curves = max(vmax_pv_curves, float(np.nanmax(V_arr)))
            if np.isfinite(Imp):
                imax_mpp = max(imax_mpp, float(Imp))

        if not hour_Imax:
            fig.add_annotation(text=f"No sun on {day_str}", xref=f"x{ccol}", yref=f"y{ccol}",
                               showarrow=False, x=0.5, y=0.5)
            fig.update_xaxes(title_text="Array Voltage U (V)", row=1, col=ccol, range=[0.0, 1.0])
            fig.update_yaxes(title_text="Array Current I (A)", row=1, col=ccol, autorange=True)
            return 0.0

        vals = np.array(list(hour_Imax.values()), dtype=float)
        q1, q2, q3 = np.quantile(vals, [0.25, 0.50, 0.75])
        class_counters = {'blue': 0, 'green': 0, 'orange': 0, 'red': 0}

        for t in sorted(iv_cache.keys()):
            V_arr, I_arr, Vmp, Imp = iv_cache[t]
            h = int(pd.to_datetime(t).hour)
            i_class = _class_from_Imax(hour_Imax.get(h, 0.0), q1, q2, q3)
            color = _class_color(i_class, class_counters)
            dash = LINESTYLES[h % len(LINESTYLES)]
            label = f"{h:02d}:00"

            fig.add_trace(go.Scatter(
                x=V_arr, y=I_arr, mode="lines",
                name=label, legendgroup=f"hours_{day_str}",
                line=dict(width=1.9, color=color, dash=dash),
                hovertemplate="U=%{x:.1f} V<br>I=%{y:.1f} A<extra></extra>",
                showlegend=True
            ), row=1, col=ccol)

            fig.add_trace(go.Scatter(
                x=[Vmp], y=[Imp], mode="markers",
                name="PV MPP", legendgroup=f"mpp_{day_str}",
                showlegend=False,
                marker=dict(size=7, symbol=MPP_MARKER, color=color, line=dict(width=0.8, color="black")),
                hovertemplate="Vmp=%{x:.1f} V<br>Imp=%{y:.1f} A<extra></extra>"
            ), row=1, col=ccol)

        Imax_day = max(hour_Imax.values()) if hour_Imax else 0.0
        if Imax_day > 0:
            I_grid = np.linspace(0.0, 1.05 * Imax_day, 300)
            V_el = v_elec_array_from_Iarray(
                I_grid, summary["T_op_K"], stack, geom, ec,
                n_series_stacks=int(ELEC_N_SERIES_STACKS),
                n_parallel_strings=int(ELEC_N_PARALLEL_STRINGS),
            )
            if np.size(V_el):
                vmax_el_curve = max(vmax_el_curve, float(np.nanmax(V_el)))

            fig.add_trace(go.Scatter(
                x=V_el, y=I_grid, mode="lines",
                name=f"Electrolyzer array (T_op={T_op_C:.1f}°C)",
                legendgroup=f"el_{day_str}",
                line=dict(width=2.2, color="red"),
                hovertemplate="U=%{x:.1f} V<br>I=%{y:.1f} A<extra></extra>",
                showlegend=True
            ), row=1, col=ccol)

            cp = _coupling_points_for_day(day_str)
            if not cp.empty:
                fig.add_trace(go.Scatter(
                    x=cp["V_op_V"], y=cp["I_op_A"], mode="markers",
                    name="Coupling points", legendgroup=f"cp_{day_str}",
                    marker=dict(symbol="x", size=7, color="red"),
                    hovertemplate="U=%{x:.1f} V<br>I=%{y:.1f} A<extra></extra>",
                    showlegend=True
                ), row=1, col=ccol)

        x_max = 1.1 * max(vmax_pv_curves, vmax_el_curve, 1e-9)
        fig.update_xaxes(title_text="Array Voltage U (V)", row=1, col=ccol, range=[0.0, x_max])
        return imax_mpp

    imax_mpp_w = _add_iv_panel(winter_day, 1)
    imax_mpp_s = _add_iv_panel(summer_day, 2)

    imax_common = 1.1 * max(imax_mpp_w, imax_mpp_s, 1e-9)
    for ccol in (1, 2):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)", zeroline=False, row=1, col=ccol)
        fig.update_yaxes(title_text="Array Current I (A)", range=[0.0, imax_common], row=1, col=ccol)

    fig.update_layout(
        height=560,
        plot_bgcolor="white",
        hovermode="closest",
        margin=dict(t=90, r=320, b=60, l=70),
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            tracegroupgap=6,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            title="Hour / Curves"
        ),
    )

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        name="PV MPP",
        marker=dict(size=7, symbol=MPP_MARKER, color="white", line=dict(width=0.8, color="black"))
    ), row=1, col=1)

    return fig


def _mk_power_figure_pair(winter_day, summer_day):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Powers vs Time — {winter_day}", f"Powers vs Time — {summer_day}"),
        horizontal_spacing=0.08
    )

    def _times_for_day(day_str):
        m = curve_info2.index.strftime("%Y-%m-%d") == day_str
        return curve_info2.index[m]

    pmp_w = Pmp_fullR_W.reindex(_times_for_day(winter_day)).fillna(0.0)
    pmp_s = Pmp_fullR_W.reindex(_times_for_day(summer_day)).fillna(0.0)
    y_max_kw = 1.1 * max(float(pmp_w.max()), float(pmp_s.max())) / 1000.0 if (len(pmp_w) or len(pmp_s)) else None

    def _add_panel(day_str, ccol):
        pmp_watts, pcpl_watts = _power_series_for_day(day_str)
        fig.add_trace(go.Scatter(
            x=pmp_watts.index, y=(pmp_watts / 1000.0).values, mode="lines+markers",
            name="PV @ MPP (with DC R)", line=dict(width=2), marker=dict(size=6, color="blue"),
            hovertemplate="Time=%{x|%H:%M}<br>P=%{y:.2f} kW<extra></extra>"
        ), row=1, col=ccol)
        fig.add_trace(go.Scatter(
            x=pcpl_watts.index, y=(pcpl_watts / 1000.0).values, mode="markers",
            name="Coupled power", marker=dict(symbol="x", size=8, color="red"),
            hovertemplate="Time=%{x|%H:%M}<br>P=%{y:.2f} kW<extra></extra>"
        ), row=1, col=ccol)
        fig.update_xaxes(title_text="Time", row=1, col=ccol)
        if y_max_kw is not None and y_max_kw > 0:
            fig.update_yaxes(title_text="Power (kW)", range=[0, y_max_kw], row=1, col=ccol)
        else:
            fig.update_yaxes(title_text="Power (kW)", row=1, col=ccol)

    _add_panel(winter_day, 1)
    _add_panel(summer_day, 2)
    fig.update_layout(height=480, legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0))
    return fig


def _mk_h2_hourly_figure_pair():
    KG_PER_COULOMB = M_H2_KG_PER_MOL / (2.0 * F)
    h2_rate_kg_per_h = (
        coupling_results["Single_Stack_current"].astype(float)
        * (stack.n_cells * total_stacks) * KG_PER_COULOMB * 3600.0 * FARADAIC_EFF_H2
    )

    mask_w = h2_rate_kg_per_h.index.strftime("%Y-%m-%d") == WINTER_DAY
    mask_s = h2_rate_kg_per_h.index.strftime("%Y-%m-%d") == SUMMER_DAY

    vals_for_limit = np.concatenate([
        h2_rate_kg_per_h[mask_w].values if mask_w.any() else np.array([0.0]),
        h2_rate_kg_per_h[mask_s].values if mask_s.any() else np.array([0.0])
    ])
    ymax_hourly = max(1e-6, 1.1 * float(np.nanmax(vals_for_limit)) if vals_for_limit.size else 1.0)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Hourly H₂ production — {WINTER_DAY}", f"Hourly H₂ production — {SUMMER_DAY}"),
        horizontal_spacing=0.08
    )
    fig.add_trace(go.Bar(x=h2_rate_kg_per_h.index[mask_w], y=h2_rate_kg_per_h[mask_w].values, name="Winter day (kg/h)"),
                  row=1, col=1)
    fig.update_yaxes(range=[0, ymax_hourly], row=1, col=1, title_text="kg/h")

    fig.add_trace(go.Bar(x=h2_rate_kg_per_h.index[mask_s], y=h2_rate_kg_per_h[mask_s].values, name="Summer day (kg/h)"),
                  row=1, col=2)
    fig.update_yaxes(range=[0, ymax_hourly], row=1, col=2, title_text="kg/h")

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0))
    return fig


def _mk_h2_daily_figure():
    KG_PER_COULOMB = M_H2_KG_PER_MOL / (2.0 * F)

    ts = coupling_results.index
    dt_s = (ts.to_series().shift(-1) - ts.to_series()).dt.total_seconds()
    dt_s.iloc[-1] = dt_s.median() if np.isfinite(dt_s.median()) else 3600.0
    dt_s = dt_s.values

    h2_kg_interval = (
        coupling_results["Single_Stack_current"].astype(float).values
        * (stack.n_cells * total_stacks) * dt_s * KG_PER_COULOMB * FARADAIC_EFF_H2
    )
    h2_kg_series = pd.Series(h2_kg_interval, index=coupling_results.index)
    h2_daily_kg = h2_kg_series.resample("D").sum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h2_daily_kg.index, y=h2_daily_kg.values, mode="lines+markers", name="kg/day",
                             line=dict(width=2), marker=dict(size=5)))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="kg/day")
    fig.update_layout(height=420, legend=dict(orientation="h"))
    return fig


def _mk_polarization_figure_interactive():
    j_vals = coupling_results["j_cell_Acm2"].astype(float).values
    j_pos = j_vals[np.isfinite(j_vals) & (j_vals > 0)]
    j_min_seen = float(np.nanmin(j_pos)) if j_pos.size else 0.0
    j_max_seen = float(np.nanmax(j_vals)) if j_vals.size else 0.1
    if not np.isfinite(j_max_seen) or j_max_seen <= 0:
        j_max_seen = 0.1

    j_max_plot = 1.2 * j_max_seen
    j_grid = np.linspace(0.0, j_max_plot, 600)
    j_grid_A_m2 = np.maximum(j_grid * 1e4, 1e-12)
    U_grid = cell_voltage_vec(j_grid_A_m2, summary["T_op_K"], geom, ec)

    j_lo = max(0.0, j_min_seen)
    j_hi = max(j_lo, j_max_seen)
    U_lo = float(cell_voltage_vec(np.array([max(j_lo * 1e4, 1e-12)]), summary["T_op_K"], geom, ec)[0])
    U_hi = float(cell_voltage_vec(np.array([max(j_hi * 1e4, 1e-12)]), summary["T_op_K"], geom, ec)[0])

    U_span = float(np.nanmax(U_grid) - np.nanmin(U_grid)) if np.isfinite(np.nanmax(U_grid)) else 1.0
    dy = max(0.04, 0.08 * U_span)
    dx = 0.01 * j_max_plot

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=j_grid, y=U_grid, mode="lines",
                             name=f"Cell polarization (T_op={T_op_C:.1f}°C)", line=dict(width=3)))

    fig.add_vrect(x0=j_lo, x1=j_hi, fillcolor="orange", opacity=0.25, line_width=0,
                  annotation_text="Operating range", annotation_position="top left")
    fig.add_vline(x=j_lo, line_width=2, line_dash="dash", line_color="orange")
    fig.add_vline(x=j_hi, line_width=2, line_dash="dash", line_color="red")
    fig.add_trace(go.Scatter(
        x=[j_lo, j_hi], y=[U_lo, U_hi],
        mode="markers", name="Range endpoints",
        marker=dict(size=9, color="white", line=dict(width=1, color="black"))
    ))
    fig.add_annotation(x=j_lo + dx, y=U_lo - dy, text=f"j_min = {j_lo:.3g} A/cm²", showarrow=False, xanchor="left", yanchor="top")
    fig.add_annotation(x=j_hi - dx, y=U_hi - dy, text=f"j_max = {j_hi:.3g} A/cm²", showarrow=False, xanchor="right", yanchor="top")

    fig.update_xaxes(title_text="Current density j (A/cm²)", range=[0.0, j_max_plot * 1.02],
                     showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)", zeroline=False)
    fig.update_yaxes(title_text="Cell voltage U_cell (V)",
                     showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)", zeroline=False)
    fig.update_layout(height=460, plot_bgcolor="white", legend=dict(orientation="h", yanchor="bottom", y=-0.1, x=0))
    return fig


def _mk_cell_jU_day_pair():
    def _get_cell_day_series(day_str):
        mask = coupling_results.index.strftime("%Y-%m-%d") == day_str
        return coupling_results.loc[mask, ["j_cell_Acm2", "U_cell_V"]].copy()

    df_w = _get_cell_day_series(WINTER_DAY)
    df_s = _get_cell_day_series(SUMMER_DAY)

    def _safe_max(series):
        if series is None or series.empty:
            return 0.0
        vals = pd.to_numeric(series.replace([np.inf, -np.inf], np.nan), errors="coerce").dropna()
        return float(vals.max()) if not vals.empty else 0.0

    j_color, u_color = "red", "blue"
    j_max = max(_safe_max(df_w.get("j_cell_Acm2")), _safe_max(df_s.get("j_cell_Acm2")))
    u_max_data = max(_safe_max(df_w.get("U_cell_V")), _safe_max(df_s.get("U_cell_V")))
    u_lower = float(np.nanmin(coupling_results["U_cell_V"].astype(float).values)) if len(coupling_results) else 1.2
    u_upper = 1.1 * max(u_max_data, u_lower)
    j_upper = 1.1 * j_max if j_max > 0 else 1.0

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        subplot_titles=(f"Electrolyzer cell — {WINTER_DAY} (Winter)", f"Electrolyzer cell — {SUMMER_DAY} (Summer)"),
        horizontal_spacing=0.08
    )

    def _add_day(df, col, showlegend):
        if df is None or df.empty:
            fig.add_annotation(text="No coupling data", xref=f"x{col}", yref=f"y{col}",
                               x=0.5, y=0.5, showarrow=False)
            return
        fig.add_trace(go.Scatter(
            x=df.index, y=df["j_cell_Acm2"].astype(float),
            mode="lines+markers", name="j (A/cm²)",
            line=dict(width=2, color=j_color), marker=dict(size=6, color=j_color),
            hovertemplate="Time=%{x|%H:%M}<br>j=%{y:.4f} A/cm²<extra></extra>",
            showlegend=showlegend,
        ), row=1, col=col, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df.index, y=df["U_cell_V"].astype(float),
            mode="lines+markers", name="U_cell (V)",
            line=dict(width=2, color=u_color), marker=dict(size=6, color=u_color),
            hovertemplate="Time=%{x|%H:%M}<br>U=%{y:.3f} V<extra></extra>",
            showlegend=showlegend,
        ), row=1, col=col, secondary_y=True)

    _add_day(df_w, 1, True)
    _add_day(df_s, 2, False)

    fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=1, col=1)
    fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=1, col=2)

    for col in (1, 2):
        fig.update_yaxes(title_text="Cell current density j (A/cm²)", range=[0, j_upper],
                         row=1, col=col, secondary_y=False,
                         tickfont=dict(color=j_color), title_font=dict(color=j_color))
        fig.update_yaxes(title_text="Cell voltage U_cell (V)", range=[u_lower, u_upper],
                         row=1, col=col, secondary_y=True,
                         tickfont=dict(color=u_color), title_font=dict(color=u_color))

    fig.update_layout(height=460, plot_bgcolor="white",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0))
    return fig


def _mk_ambient_temp_year_figure():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coupling_results.index,
        y=coupling_results["T_air_C"].astype(float).values,
        mode="lines",
        name="Ambient temperature"
    ))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="T_air (°C)")
    fig.update_layout(height=380, plot_bgcolor="white", legend=dict(orientation="h"))
    return fig


def _mk_cooler_power_figure_pair(winter_day, summer_day, cooler_t_out_c: float):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Cooler duty to {cooler_t_out_c:.0f}°C — {winter_day}", f"Cooler duty to {cooler_t_out_c:.0f}°C — {summer_day}"),
        horizontal_spacing=0.08
    )

    def _series_for_day(day_str):
        times = _hourly_times_for_day(coupling_results.index, day_str)
        return coupling_results.loc[times, "cooler_Qdot_W"].fillna(0.0) / 1000.0

    s_w = _series_for_day(winter_day)
    s_s = _series_for_day(summer_day)
    y_max = 1.1 * max(float(s_w.max() if len(s_w) else 0.0), float(s_s.max() if len(s_s) else 0.0))

    for ccol, s in [(1, s_w), (2, s_s)]:
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines+markers",
            name="HX duty (kW)", line=dict(width=2), marker=dict(size=6),
            hovertemplate="Time=%{x|%H:%M}<br>Q=%{y:.2f} kW<extra></extra>",
            showlegend=(ccol == 1)
        ), row=1, col=ccol)
        fig.update_xaxes(title_text="Time", row=1, col=ccol)
        fig.update_yaxes(title_text="Heat removed (kW)", range=[0, max(0.1, y_max)], row=1, col=ccol)

    fig.update_layout(height=420, plot_bgcolor="white",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0))
    return fig


# =========================
# TOP KPIs — LCOH replaces Annual HX heat removed
# =========================
top1, top2, top3 = st.columns(3)
top1.metric("PV DC energy @ MPP (kWh)", f"{E_mpp_fullR_kWh:,.1f}")
top2.metric("PV DC delivered to electrolyzer (kWh)", f"{E_coupled_kWh:,.1f}")
top3.metric("Utilization vs. MPP (%)", f"{util_pct:,.2f}")

top4, top5, top6 = st.columns(3)
top4.metric("Max electrolyzer array power (kW)", f"{Max_electrolyzer_array_power/1000.0:,.2f}")
top5.metric("Specific electricity consumption (kWh/kg H₂)", f"{SEC_kWh_per_kg:,.2f}")
top6.metric("Total H₂ produced (kg)", f"{kg_H2_total:,.3f}")

extra1, extra2, extra3 = st.columns(3)
extra1.metric("Operating temperature T_op (°C)", f"{T_op_C:,.1f}")
extra2.metric("Levelized cost of H₂ (€/kg)", f"{LCOH_EUR_PER_KG:,.2f}")
extra3.metric("Max HX duty (kW)", f"{Q_cooler_max_kW:,.2f}")

st.caption(
    f"Operating temperature solved once at max irradiance hour ({summary['t_max']}) from steady thermal balance, then held constant for the full year."
)

# =========================
# Render interactive figures
# =========================
st.markdown("## IV Curves — Winter & Summer Day")
st.plotly_chart(_mk_iv_figure_pair(WINTER_DAY, SUMMER_DAY), use_container_width=True)

st.markdown("## Power vs Time — Winter & Summer Day")
st.plotly_chart(_mk_power_figure_pair(WINTER_DAY, SUMMER_DAY), use_container_width=True)

st.markdown("## Hydrogen Production — Hourly Winter & Summer Day")
st.plotly_chart(_mk_h2_hourly_figure_pair(), use_container_width=True)

st.markdown("## Daily Hydrogen Production over the year")
st.plotly_chart(_mk_h2_daily_figure(), use_container_width=True)

st.markdown("## Electrolyzer-Cell Polarization Curve (U–j) at operating temperature")
st.plotly_chart(_mk_polarization_figure_interactive(), use_container_width=True)

# =========================
# LCOH section (after polarization curve, per request)
# =========================
# =========================
# LCOH section (after polarization curve, per request)
# =========================
st.markdown("## Levelized Cost of Hydrogen (LCOH)")

l1, l2, l3, l4, l5 = st.columns(5)
l1.metric("LCOH (€/kg H₂)", f"{lcoh['lcoh_eur_per_kg']:,.2f}")
l2.metric("Total CAPEX (€)", f"{lcoh['capex_total_eur']:,.0f}")
l3.metric("PV electricity price (€/kWh)", f"{electricity_price_eur_per_kwh:,.4f}")
l4.metric("Electricity OPEX (€/yr)", f"{lcoh['opex_electricity_annual_eur']:,.0f}")
l5.metric("OPEX over lifetime (€)", f"{lcoh['opex_electricity_total_eur']:,.0f}")


capex_df = (
    pd.DataFrame({
        "CAPEX (€)": pd.Series(lcoh["capex_components_total_eur"]),
        "Share of CAPEX (%)": pd.Series({
            k: 100.0 * v for k, v in lcoh["capex_components_share"].items()
        }),
    })
    .sort_values("CAPEX (€)", ascending=False)
)

perkg_df = (
    pd.DataFrame({
        "Contribution to LCOH (€/kg H₂)": pd.Series(lcoh["perkg_costs_eur_per_kg"]),
        "Cost category": ["OPEX" if "Electricity" in k else "CAPEX" for k in lcoh["perkg_costs_eur_per_kg"].keys()],
    })
    .sort_values("Contribution to LCOH (€/kg H₂)", ascending=False)
)

tab1, tab2 = st.tabs(["Total CAPEX breakdown", "LCOH contributions (€/kg)"])
with tab1:
    st.dataframe(capex_df, use_container_width=True)
with tab2:
    st.dataframe(perkg_df, use_container_width=True)

p1, p2 = st.columns(2)
with p1:
    st.plotly_chart(make_lcoh_pie(lcoh["capex_components_total_eur"], "Total CAPEX breakdown (€)"), use_container_width=True)
with p2:
    st.plotly_chart(make_lcoh_pie(lcoh["perkg_costs_eur_per_kg"], "Contributions to LCOH (€/kg H₂)"), use_container_width=True)

st.caption(
    "LCOH = (annualized CAPEX + annual electricity OPEX) / annual H₂ production. "
    "CAPEX is levelized by straight-line over the selected lifetime (no discount rate). "
    "The CAPEX values shown above are totals (not annualized)."
)

st.markdown("## Electrolyzer Cell — j & U over the day")
st.plotly_chart(_mk_cell_jU_day_pair(), use_container_width=True)

st.markdown(f"## Cooler / Condenser duty (H₂ outlet → {COOLER_T_OUT_C:.0f}°C)")
st.plotly_chart(_mk_cooler_power_figure_pair(WINTER_DAY, SUMMER_DAY, COOLER_T_OUT_C), use_container_width=True)
st.write(f"- Annual HX heat removed: **{E_cooler_kWh:,.1f} kWh**")
st.write(f"- Annual condensed water (from H₂ stream): **{m_cond_total_kg:,.1f} kg**")

st.markdown("## Ambient temperature over the year")
st.plotly_chart(_mk_ambient_temp_year_figure(), use_container_width=True)

# =========================
# DASHBOARD (bottom) — Maxima & configuration summary + thermal inventory
# =========================
st.markdown("### PV Maxima (at coupled operation)")
pv_mod, pv_str, pv_arr = st.columns(3)
pv_mod.write(f"**Module**  \nU_max = {maxima['Single_PV_module_voltage']:.1f} V  \nI_max = {maxima['Single_PV_module_current']:.1f} A  \nP_max = {maxima['Single_PV_module_power']/1000.0:.2f} kW")
pv_str.write(f"**String**  \nU_max = {maxima['PV_string_voltage']:.1f} V  \nI_max = {maxima['PV_string_current']:.1f} A  \nP_max = {maxima['PV_string_power']/1000.0:.2f} kW")
pv_arr.write(f"**Array**  \nU_max = {maxima['PV_array_voltage']:.1f} V  \nI_max = {maxima['PV_array_current']:.1f} A  \nP_max = {maxima['PV_array_power']/1000.0:.2f} kW")

st.markdown("### Electrolyzer Maxima")
el_stack, el_string, el_array = st.columns(3)
el_stack.write(f"**Single Stack**  \nU_max = {maxima['Single_Stack_voltage']:.1f} V  \nI_max = {maxima['Single_Stack_current']:.1f} A  \nP_max = {maxima['Single_Stack_Power']/1000.0:.2f} kW")
el_string.write(f"**String**  \nU_max = {maxima['Electrolyzer_string_voltage']:.1f} V  \nI_max = {maxima['Elecgtrolyzer_string_current']:.1f} A  \nP_max = {maxima['Electrolyzer_string_power']/1000.0:.2f} kW")
el_array.write(f"**Array**  \nU_max = {maxima['Electrolyzer_array_voltage']:.1f} V  \nI_max = {maxima['Electrolyzer_array_current']:.1f} A  \nP_max = {maxima['Electrolyzer_array_power']/1000.0:.2f} kW")

st.markdown("### Configuration Summary")
cfg1, cfg2 = st.columns(2)

with cfg1:
    st.markdown("**PV Array**")
    st.write(f"- **PV module:** [{MODULE_DISPLAY_NAME}]({MODULE_DATASHEET_URL})")
    num_modules = N_SERIES * N_PARALLEL
    st.write(f"- Number of PV modules: **{num_modules}**")
    st.write(f"- PV string: **{N_SERIES} modules in series**")
    st.write(f"- PV array: **{N_PARALLEL} strings in parallel**")
    nameplate_kwp = 0.46 * N_SERIES * N_PARALLEL
    st.write(f"- **Max potential power: {nameplate_kwp:,.1f} kWp**")
    st.write(f"- Surface tilt: **{SURFACE_TILT:.1f}°**, azimuth: **{SURFACE_AZIMUTH:.1f}°**")
    st.write(f"- Inter-module jumper length (one-way): **{SEG_LEN_M:.2f} m**")
    st.write(f"- Array cable one-way length: **{ARR_CABLE_ONE_WAY_M:.1f} m**")
    st.write("**DC Cable**")
    st.write(f"  • String jumper R: **{R_string:.5f} Ω** per string")
    st.write(f"  • Array cable R: **{R_array:.5f} Ω** at array level")
    st.write(f"  • Array cable CSA selected: **{ARR_CABLE_CROSS_SEC_MM2:.1f} mm²** (required: **{ARR_CABLE_CSA_REQUIRED:.1f} mm²**)")

with cfg2:
    st.markdown("**Electrolyzer Topology & Inputs (thermal-model based)**")
    st.write(f"- Cells per stack: **{N_CELLS_PER_STACK}**")
    st.write(f"- Stacks in series per string: **{ELEC_N_SERIES_STACKS}**")
    st.write(f"- Electrolyzer strings in parallel: **{ELEC_N_PARALLEL_STRINGS}**")
    st.write(f"- Faradaic efficiency: **{FARADAIC_EFF_H2:.2f}**")
    st.write(f"- Active cell area A_active: **{A_ACTIVE_M2:.4f} m²**  |  A_cell = A_active·f_cell: **{stack.A_cell_m2:.4f} m²** (f_cell={geom.f_cell:.2f})")
    st.write(f"- Tafel slope b: **{TAFEL_SLOPE_MV_DEC:.1f} mV/dec**")
    st.write(f"- j₀,ref: **{J0_REF_A_CM2:.3e} A/cm²** at **{T_REF_J0_C:.1f} °C**")
    st.write(f"- Separator: t_sep=**{T_SEP_MM:.2f} mm**, ε=**{EPS_SEP:.2f}**, τ=**{TAU_SEP:.2f}** (material: PES, ρ≈{RHO_PES_KG_M3:.0f} kg/m³)")
    st.write(f"- Gap: t_gap=**{T_GAP_MM:.2f} mm**")
    st.write(f"- Contact resistance R_contact: **{R_CONTACT_mOHM_CM2:.3f} mΩ·cm²**")
    st.write(f"- Electrode plate thickness t_electrode: **{T_ELECTRODE_MM:.3f} mm**")
    st.write(f"- Gasket thickness t_gasket: **{T_GASKET_MM:.3f} mm** (material: PTFE, ρ≈{RHO_PTFE_KG_M3:.0f} kg/m³)")
    st.write(f"- Condenser/cooler outlet temperature: **{COOLER_T_OUT_C:.1f} °C**")

    st.write("**Electrolyzer Operating Range (coupled)**")
    st.write(f"  • Minimum cell current density: **{min_current_density:,.5f} A/cm²**")
    st.write(f"  • Maximum cell current density: **{max_current_density:,.5f} A/cm²**")

st.markdown("### Electrolyzer Stack Dimensions & Material Inventory")
t1, t2, t3, t4 = st.columns(4)
t1.write(
    f"**Stack dimensions**  \n"
    f"- Cell width: **{stack.w_cell_m:.3f} m**  \n"
    f"- Cell length: **{stack.l_cell_m:.3f} m**  \n"
    f"- Stack height: **{thermo['h_stack_m']:.3f} m**  \n"
    f"- Total cell area: **{thermo['A_cell_total_m2']:.2f} m²**  \n"
    f"- Total active area: **{thermo['A_active_total_m2']:.2f} m²**  \n"
    f"- Outer Surface Area stack: **{thermo['A_surface_stack_m2']:.2f} m²**"
)
t2.write(
    f"**Steel Electrodes**  \n"
    f"- Number of Electrodes: **{stack.n_cells+1:.0f}**  \n"
    f"- Total Electrode area: **{thermo['A_cell_total_m2']:.2f} m²**  \n"
    f"- Total mass of electrodes: **{thermo['m_electrodes_kg']:.2f} kg**  \n"
)
t3.write(
    f"**Electrolyte**  \n"
    f"- Electrolyte Volume: **{thermo['V_electrolyte_total_m3']:.2f} m³**  \n"
    f"- Electrolyte Mass: **{thermo['m_electrolyte_kg']:.2f} kg**  \n"
)
t4.write(
    f"**Separator & Gasket**  \n"
    f"- Total separator area: **{thermo['A_separator_total_m2']:.2f} m²**  \n"
    f"- Separator (PES) mass (solid fraction): **{thermo['m_separator_kg']:.2f} kg**  \n"
    f"- Total gasket area: **{thermo['A_gasket_total_m2']:.2f} m²**  \n"
    f"- Gasket (PTFE) mass: **{thermo['m_gasket_kg']:.2f} kg**"
)



