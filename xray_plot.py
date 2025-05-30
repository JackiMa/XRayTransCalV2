#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
X-ray Data Visualization Module
Provides functions for plotting X-ray attenuation and transmission related charts
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datetime
import warnings
import base64
from io import BytesIO
from typing import Optional, Dict, List, Tuple, Union, Any

# 导入模型
from xray_model import Element, Elements, DEFAULT_INTERPOLATION_POINTS

# 绘图默认设置
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150  # 显示DPI适中即可
plt.rcParams['savefig.dpi'] = 300  # PDF矢量格式，DPI不需要太高
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False  # Fix for minus sign display issue

# 临时绘图目录
TMP_PLOTS_DIR = 'tmp_plots'
if not os.path.exists(TMP_PLOTS_DIR):
    os.makedirs(TMP_PLOTS_DIR)

def save_base64_plot(fig):
    """Save plot to base64 string for embedding in web pages"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_str

# --- Helper function for tick/grid styling ---
def _apply_tick_grid_styles(ax):
    """Applies the requested major/minor tick and grid styles."""
    ax.minorticks_on() # Ensure minor ticks are enabled
    ax.grid(True, which="major", axis="both", ls="--", color='gray', linewidth=0.7, alpha=0.7) # Dashed major grid
    ax.grid(True, which="minor", axis="both", ls=":", color='gray', linewidth=0.5, alpha=0.3) # Dotted minor grid
     # Thicker frame/spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')

def plot_element_cross_sections(element: Element,
                              e_min: float = None,
                              e_max: float = None,
                              points: int = DEFAULT_INTERPOLATION_POINTS,
                              show_plot: bool = False,
                              save_path: Optional[str] = None,
                              return_fig: bool = False,
                              y_min: float = None,
                              y_max: float = None,
                              x_scale: str = 'log', # New: 'linear' or 'log'
                              y_scale: str = 'log'  # New: 'linear' or 'log'
                              ) -> Optional[matplotlib.figure.Figure]:
    """Plot mass attenuation coefficients for an element with selectable scales."""

    # --- Energy range validation (remains the same) ---
    if e_min is None: e_min = element.energy_min
    if e_max is None: e_max = element.energy_max
    if e_min < element.energy_min: e_min = element.energy_min
    if e_max > element.energy_max: e_max = element.energy_max
    if e_min >= e_max:
        warnings.warn(f"Invalid energy range for {element.symbol}: e_min={e_min} >= e_max={e_max}")
        return None

    # --- Create energy points (use linear for calculation, scale axis later) ---
    if x_scale == 'log':
        # Ensure e_min and e_max are positive for logspace
        if e_min <= 0:
            e_min_log = element.energy_min if hasattr(element, 'energy_min') and element.energy_min > 0 else 1e-6 # Use a small positive number if needed
            warnings.warn(f"e_min was <= 0 for log scale, adjusting to {e_min_log}")
        else:
            e_min_log = e_min
        if e_max <= e_min_log: # Check again after potential adjustment
             warnings.warn(f"Cannot create logspace: adjusted e_min {e_min_log} >= e_max {e_max}")
             return None
        energies = np.logspace(np.log10(e_min_log), np.log10(e_max), points)
    else:
        energies = np.linspace(e_min, e_max, points)

    # --- Prepare figure and axes ---
    fig, ax = plt.subplots(figsize=(10 * 1.1, 6 * 1.1)) # Slightly larger figure for bigger fonts
    base_fontsize = 22 # Increased base font size

    # --- Define columns and labels (remains the same) ---
    cols_to_plot = {
        'Total_w_coh (cm2/g)': ('Total Attenuation (with coherent)', 'black', '--'), # Added color/style
        'Total_wo_coh (cm2/g)': ('Total Attenuation (without coherent)', 'black', ':'), # Added color/style
        'Coherent (cm2/g)': ('Coherent Scattering', None, '-'), # Default color, solid line
        'Incoherent (cm2/g)': ('Incoherent Scattering', None, '-'),
        'Photoelectric (cm2/g)': ('Photoelectric Absorption', None, '-'),
        'Nuclear (cm2/g)': ('Pair Production (Nuclear)', None, '-'),
        'Electron (cm2/g)': ('Pair Production (Electron)', None, '-')
    }

    # --- Plotting loop ---
    results_valid = False
    for col, (label, color, linestyle) in cols_to_plot.items():
        # Ensure element data is loaded before trying to get cross section
        if element.data is None:
             warnings.warn(f"Element {element.symbol} data is not loaded. Cannot plot {col}.")
             continue # Skip this column if data isn't loaded
        values = element.get_cross_section(energies, col)
        # 替换 inf、nan、负值为 np.nan
        values[values <= 1e-10] = np.nan
        if values is not None and np.any(np.isfinite(values)):
             # Use standard plot, scales set later
             ax.plot(energies, values, label=label, linewidth=1.5 if linestyle != '--' else 2.0,
                     color=color, linestyle=linestyle) # Apply color/style
             results_valid = True

    # --- Set scales (remains the same) ---
    try:
        ax.set_xscale(x_scale)
    except ValueError:
        warnings.warn(f"Invalid x_scale value: '{x_scale}'. Using 'linear'.")
        ax.set_xscale('linear')
    try:
        ax.set_yscale(y_scale)
    except ValueError:
        warnings.warn(f"Invalid y_scale value: '{y_scale}'. Using 'linear'.")
        ax.set_yscale('linear')

    # --- Aesthetics and Labels ---

    if not results_valid:
         ax.text(0.5, 0.5, 'No valid data available for plotting',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes, fontsize=base_fontsize)
         # Avoid redundant warning if loop already warned
         # warnings.warn(f"Element {element.symbol} has no valid cross section data for plotting.")
    else:
        ax.set_xlabel('Photon Energy (MeV)', fontsize=base_fontsize)
        ax.set_ylabel('Mass Attenuation Coefficient (cm²/g)', fontsize=base_fontsize)
        title_name = element.name if hasattr(element, 'name') and element.name else element.symbol
        title_z = f" (Z={element.z})" if hasattr(element, 'z') and element.z else ""
        ax.set_title(f'{title_name}{title_z} Attenuation Coefficients', fontsize=base_fontsize * 1.1)
        ax.legend(loc='best', fontsize=base_fontsize * 0.75)
        _apply_tick_grid_styles(ax) # Apply new grid/tick styles
        ax.tick_params(axis='both', which='major', labelsize=base_fontsize * 0.75)
        if y_min is not None and y_max is not None: ax.set_ylim(y_min, y_max)
    plt.tight_layout()

    # --- Saving and Showing ---
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            try: os.makedirs(save_dir)
            except OSError as e: warnings.warn(f"Cannot create directory {save_dir}: {e}")
        try:
            fig.savefig(save_path, dpi=300)
            print(f"Chart saved to: {save_path}")
        except Exception as e:
            warnings.warn(f"Error saving chart to {save_path}: {e}")

    if show_plot:
        plt.show()

    if return_fig:
        return fig
    else:
        plt.close(fig)
        return None

def plot_compound_components(elements: Elements,
                             formula: str,
                             e_min: float,
                             e_max: float,
                             points: int = DEFAULT_INTERPOLATION_POINTS,
                             save_path: Optional[str] = None,
                             show_plot: bool = False,
                             return_fig: bool = False,
                             x_scale: str = 'log', # New
                             y_scale: str = 'log',   # New
                             y_min: float = None,
                             y_max: float = None
                             ) -> Optional[matplotlib.figure.Figure]:
    """Plot the contribution of each element to the mass attenuation coefficient of a compound"""

    # --- Create energy points ---
    if e_min <= 0 and x_scale == 'log':
        warnings.warn("e_min <= 0 for log scale, using default minimum energy from elements if available.")
        valid_elements = [el for el in elements.elements.values() if el.data is not None and hasattr(el, 'energy_min') and el.energy_min > 0]
        if not valid_elements:
             warnings.warn("No valid element data to determine minimum energy.")
             # Try to find *any* loaded element min energy > 0 as a fallback
             all_elements = [el for el in elements.elements.values() if hasattr(el, 'energy_min') and el.energy_min > 0]
             if not all_elements: e_min = 1e-6 # Absolute fallback
             else: e_min = min(el.energy_min for el in all_elements)
        else:
             e_min = min(el.energy_min for el in valid_elements)

    if e_min >= e_max:
        warnings.warn(f"Invalid energy range for {formula}: e_min={e_min} >= e_max={e_max}")
        return None

    if x_scale == 'log':
        energies = np.logspace(np.log10(e_min), np.log10(e_max), points)
    else:
        energies = np.linspace(e_min, e_max, points)

    # --- Calculate data --- # Now returns total, element dict, effect dict
    compound_results = elements.calculate_compound_cross_section(formula, energies, with_coherent=True)

    # --- Prepare figure ---
    fig, ax = plt.subplots(figsize=(10 * 1.1, 6 * 1.1)) # Larger figure
    base_fontsize = 22 # Increased base font size

    # --- Handle calculation errors ---
    if compound_results is None:
        total_mu_rho, element_contributions = None, None # Ensure variables exist
        warnings.warn(f"Cannot calculate valid data for compound {formula}, cannot plot.")
        ax.text(0.5, 0.5, f'Cannot plot compound: {formula}\n(Calc error)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=base_fontsize)
        # Still return/save/show the empty plot with the error message
    else:
        total_mu_rho, element_contributions, _ = compound_results # Unpack (ignore effects here)

        # --- Plotting ---
        plot_successful = False
        # Plot element contributions
        if element_contributions:
            for symbol, contribution in element_contributions.items():
                if contribution is not None and np.any(np.isfinite(contribution)):
                     ax.plot(energies, contribution, label=f'{symbol} contribution', linewidth=1.5) # Use plot
                     plot_successful = True
        # Plot total coefficient (ensure it's black and dashed)
        if total_mu_rho is not None and np.any(np.isfinite(total_mu_rho)):
             ax.plot(energies, total_mu_rho, label='Total (with coherent)', linestyle='--', color='black', linewidth=2.0) # Use plot
             plot_successful = True

        # --- Set scales ---
        if plot_successful:
            try: ax.set_xscale(x_scale) # Use provided scale
            except ValueError: warnings.warn(f"Invalid x_scale: '{x_scale}'. Using 'linear'."); ax.set_xscale('linear')
            try: ax.set_yscale(y_scale) # Use provided scale
            except ValueError: warnings.warn(f"Invalid y_scale: '{y_scale}'. Using 'linear'."); ax.set_yscale('linear')

            # --- Aesthetics and Labels ---
            ax.set_xlabel('Photon Energy (MeV)', fontsize=base_fontsize)
            ax.set_ylabel('Mass Attenuation Coefficient (cm²/g)', fontsize=base_fontsize)
            ax.set_title(f'Element Contributions to {formula} Attenuation', fontsize=base_fontsize * 1.1)
            ax.legend(loc='best', fontsize=base_fontsize * 0.75)
            _apply_tick_grid_styles(ax) # Apply new grid/tick styles
            ax.tick_params(axis='both', which='major', labelsize=base_fontsize * 0.75)
            if y_min is not None and y_max is not None: ax.set_ylim(y_min, y_max)
        else:
            # Handles cases where calculation worked but results were all NaN/inf
            ax.text(0.5, 0.5, f'Cannot plot compound: {formula}\n(Invalid values)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=base_fontsize)


    plt.tight_layout()

    # --- Saving and Showing ---
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            try: os.makedirs(save_dir)
            except OSError as e: warnings.warn(f"Cannot create directory {save_dir}: {e}")
        try:
            fig.savefig(save_path, dpi=300)
            print(f"Chart saved to: {save_path}")
        except Exception as e:
            warnings.warn(f"Error saving chart to {save_path}: {e}")

    if show_plot:
        plt.show()

    if return_fig:
        return fig
    else:
        plt.close(fig)
        return None

def plot_compound_effect_contributions(elements: Elements,
                                       formula: str,
                                       e_min: float,
                                       e_max: float,
                                       points: int = DEFAULT_INTERPOLATION_POINTS,
                                       save_path: Optional[str] = None,
                                       show_plot: bool = False,
                                       return_fig: bool = False,
                                       y_min: float = None,
                                       y_max: float = None,
                                       x_scale: str = 'log',
                                       y_scale: str = 'log' # Now selectable
                                      ) -> Optional[matplotlib.figure.Figure]:
    """Plots the absolute contribution (cm²/g) of each interaction effect."""

    # --- Energy points ---
    if e_min <= 0 and x_scale == 'log':
        warnings.warn("e_min <= 0 for log scale, using default minimum energy.")
        valid_elements = [el for el in elements.elements.values() if el.data is not None and hasattr(el, 'energy_min') and el.energy_min > 0]
        e_min = min(el.energy_min for el in valid_elements) if valid_elements else 1e-6
    if e_min >= e_max: warnings.warn(f"Invalid energy range for {formula}."); return None

    if x_scale == 'log': energies = np.logspace(np.log10(e_min), np.log10(e_max), points)
    else: energies = np.linspace(e_min, e_max, points)

    # --- Calculate data (including effects) ---
    compound_results = elements.calculate_compound_cross_section(formula, energies, with_coherent=True)

    fig, ax = plt.subplots(figsize=(10 * 1.1, 6 * 1.1))
    base_fontsize = 22

    if compound_results is None:
        warnings.warn(f"Cannot calculate effect contributions for {formula}.")
        ax.text(0.5, 0.5, f'Cannot plot effect contributions: {formula}\n(Calc error)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=base_fontsize)
        total_mu_rho_w_coh, _, effect_contributions = None, None, None
        plot_successful = False
    else:
        # Unpack total_with_coherent and effect dict
        total_mu_rho_w_coh, _, effect_contributions = compound_results

        # --- Plotting Lines ---
        plot_successful = False
        effect_plot_params = { # Define colors/styles for effects
            'Photoelectric':  ('Photoelectric Absorption', None, '-'),
            'Coherent': ('Coherent Scattering', None, '-'), 
            'Incoherent': ('Incoherent Scattering', None, '-'),
            'Nuclear': ('Pair Production (Nuclear)', None, '-'),
            'Electron': ('Pair Production (Electron)', None, '-')
        }

        for effect_key, (label, color, linestyle) in effect_plot_params.items():
            data = effect_contributions.get(effect_key)
            data[data <= 1e-10] = np.nan
            ax.plot(energies, data, label=label, color=color, linestyle=linestyle, linewidth=1.5)
            plot_successful = True

        # Plot Total (with coherent) line
        if total_mu_rho_w_coh is not None and np.any(np.isfinite(total_mu_rho_w_coh)):
             ax.plot(energies, total_mu_rho_w_coh, label='Total (with coherent)', color='black', linestyle='--', linewidth=2.0)
             plot_successful = True # Ensure success if total is plotted

    # --- Set scales --- Always done after potential plotting
    if plot_successful:
        try: ax.set_xscale(x_scale)
        except ValueError: warnings.warn(f"Invalid x_scale: '{x_scale}'. Using 'linear'."); ax.set_xscale('linear')
        try: ax.set_yscale(y_scale) # Use selectable y_scale
        except ValueError: warnings.warn(f"Invalid y_scale: '{y_scale}'. Using 'linear'."); ax.set_yscale('linear')

        # --- Aesthetics and Labels ---
        ax.set_xlabel('Photon Energy (MeV)', fontsize=base_fontsize)
        ax.set_ylabel('Contribution to μ/ρ (cm²/g)', fontsize=base_fontsize) # Updated Label
        ax.set_title(f'Interaction Effect Contributions for {formula}', fontsize=base_fontsize * 1.1)
        ax.legend(loc='best', fontsize=base_fontsize * 0.75) # Legend inside plot now
        _apply_tick_grid_styles(ax)
        ax.tick_params(axis='both', which='major', labelsize=base_fontsize * 0.75)
        if y_min is not None and y_max is not None: ax.set_ylim(y_min, y_max)
    elif total_mu_rho_w_coh is not None: # If calc worked but no valid data points
         ax.text(0.5, 0.5, f'Cannot plot effect contributions: {formula}\n(No valid data points)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=base_fontsize)

    plt.tight_layout() # Legend is inside, no need for rect adjustment

    # --- Saving and Showing logic remains the same, remove bbox_inches='tight' ---
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            try: os.makedirs(save_dir)
            except OSError as e: warnings.warn(f"Cannot create directory {save_dir}: {e}")
        try:
            fig.savefig(save_path, dpi=300) # Remove bbox_inches
            print(f"Chart saved to: {save_path}")
        except Exception as e: warnings.warn(f"Error saving chart: {e}")

    if show_plot: plt.show()

    if return_fig: return fig
    else: plt.close(fig); return None

def plot_compound_transmission(elements: Elements,
                               formula: str,
                               e_min: float,
                               e_max: float,
                               density: float,
                               thickness: float,
                               points: int = DEFAULT_INTERPOLATION_POINTS,
                               save_path: Optional[str] = None,
                               show_plot: bool = False,
                               return_fig: bool = False,
                               y_min: float = None,
                               y_max: float = None,
                               x_scale: str = 'log',  # New
                               y_scale: str = 'linear' # Default linear for transmission %
                               ) -> Optional[matplotlib.figure.Figure]:
    """Plot the transmission rate of X-rays through a compound material"""

    # --- Create energy points ---
    if e_min <= 0 and x_scale == 'log':
        warnings.warn("e_min <= 0 for log scale, using default minimum energy.")
        valid_elements = [el for el in elements.elements.values() if el.data is not None and hasattr(el, 'energy_min') and el.energy_min > 0]
        if not valid_elements:
            all_elements = [el for el in elements.elements.values() if hasattr(el, 'energy_min') and el.energy_min > 0]
            if not all_elements: e_min = 1e-6
            else: e_min = min(el.energy_min for el in all_elements)
        else: e_min = min(el.energy_min for el in valid_elements)

    if e_min >= e_max:
        warnings.warn(f"Invalid energy range for {formula}: e_min={e_min} >= e_max={e_max}")
        return None

    if x_scale == 'log':
        energies = np.logspace(np.log10(e_min), np.log10(e_max), points)
    else:
        energies = np.linspace(e_min, e_max, points)

    # --- Calculate data --- Use the original compound transmission function
    transmission = elements.calculate_compound_transmission(formula, energies, density, thickness, with_coherent=True)

    # --- Prepare figure ---
    fig, ax = plt.subplots(figsize=(10 * 1.1, 6 * 1.1)) # Larger figure
    base_fontsize = 22 # Increased base font size

    # --- Handle calculation errors ---
    if transmission is None or np.all(np.isnan(transmission)):
         warnings.warn(f"Cannot calculate valid transmission rates for compound {formula}, cannot plot.")
         ax.text(0.5, 0.5, f'Cannot plot transmission for {formula}\n(Calc error)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=base_fontsize)
    else:
        # --- Plotting ---
        ax.plot(energies, transmission, label='Transmission Rate (%)', linewidth=1.5) # Use plot

        # --- Set scales ---
        try: ax.set_xscale(x_scale)
        except ValueError: warnings.warn(f"Invalid x_scale: '{x_scale}'. Using 'linear'."); ax.set_xscale('linear')
        try: ax.set_yscale(y_scale)
        except ValueError: warnings.warn(f"Invalid y_scale: '{y_scale}'. Using 'linear'."); ax.set_yscale('linear')

        # --- Aesthetics and Labels ---
        ax.set_xlabel('Photon Energy (MeV)', fontsize=base_fontsize)
        ax.set_ylabel('Transmission Rate (%)', fontsize=base_fontsize)
        ax.set_title(f'{formula}, {density:.2f} g/cm³, {thickness:.2f} cm Transmission', fontsize=base_fontsize * 1.1)
        ax.legend(loc='best', fontsize=base_fontsize * 0.75)
        _apply_tick_grid_styles(ax) # Apply new grid/tick styles
        ax.tick_params(axis='both', which='major', labelsize=base_fontsize * 0.75)
        if y_min is not None and y_max is not None: ax.set_ylim(y_min, y_max)
        # Set y-axis limits appropriately for linear or log scale
        if y_scale == 'linear':
             ax.set_ylim(0, 105) # Y-axis range 0-100% for linear
        else: # Log scale logic
             if transmission is not None and np.any(np.isfinite(transmission[transmission > 0])):
                 min_positive_transmission = transmission[transmission > 0].min()
                 ax.set_ylim(bottom=max(1e-3, min_positive_transmission * 0.5), top=110) # Avoid zero for log
             else:
                 ax.set_ylim(bottom=1e-3, top=110) # Fallback if no positive data

    plt.tight_layout()

    # --- Saving and Showing ---
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            try: os.makedirs(save_dir)
            except OSError as e: warnings.warn(f"Cannot create directory {save_dir}: {e}")
        try:
            fig.savefig(save_path, dpi=300)
            print(f"Chart saved to: {save_path}")
        except Exception as e:
            warnings.warn(f"Error saving chart to {save_path}: {e}")

    if show_plot:
        plt.show()

    if return_fig:
        return fig
    else:
        plt.close(fig)
        return None

def plot_compound_all(elements: Elements,
                     formula: str,
                     e_min: float,
                     e_max: float,
                     density: float,
                     thickness: float,
                     points: int = DEFAULT_INTERPOLATION_POINTS,
                     save_dir: str = None,
                     x_scale: str = 'log',
                     y_scale_atten: str = 'log',
                     y_scale_trans: str = 'linear',
                     y_scale_effect: str = 'log', # New parameter for effect plot y-scale
                     y_min: float = None,
                     y_max: float = None
                     ) -> Optional[str]:
    """Generate and save all charts related to a compound, including effect contributions."""
    # --- Sanitize formula for filename (Define this OUTSIDE the if block) ---
    safe_formula = "".join(c if c.isalnum() else '_' for c in formula)

    # --- save_dir logic --- #
    if not save_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use the already sanitized formula
        save_dir = os.path.join(TMP_PLOTS_DIR, f"{safe_formula}_{timestamp}")

    if not os.path.exists(save_dir):
        try: os.makedirs(save_dir)
        except OSError as e: warnings.warn(f"Cannot create dir {save_dir}: {e}"); return None

    # Plot compound components chart (Uses y_scale_atten)
    components_fig = plot_compound_components(
        elements, formula, e_min, e_max, points, show_plot=False, return_fig=True,
        x_scale=x_scale, y_scale=y_scale_atten, y_min=y_min, y_max=y_max # Pass scales
    )
    if components_fig:
        components_path = os.path.join(save_dir, f"{safe_formula}_components.png") # Use safe_formula
        try: components_fig.savefig(components_path, dpi=300)
        except Exception as e: warnings.warn(f"Error saving compound components chart: {e}")
        finally: plt.close(components_fig)
        print(f"Compound components chart saved to: {components_path}")

    # Plot effect contributions chart (Uses y_scale_effect)
    effects_fig = plot_compound_effect_contributions(
        elements, formula, e_min, e_max, points, show_plot=False, return_fig=True,
        x_scale=x_scale, y_scale=y_scale_effect, y_min=y_min, y_max=y_max # Pass new y_scale parameter
    )
    if effects_fig:
         effects_path = os.path.join(save_dir, f"{safe_formula}_effects.png") # Use safe_formula
         try: effects_fig.savefig(effects_path, dpi=300) # No bbox_inches needed now
         except Exception as e: warnings.warn(f"Error saving effect contributions chart: {e}")
         finally: plt.close(effects_fig)
         print(f"Effect contributions chart saved to: {effects_path}")

    # Plot compound transmission chart (Uses y_scale_trans)
    transmission_fig = plot_compound_transmission(
        elements, formula, e_min, e_max, density, thickness, points,
        show_plot=False, return_fig=True,
        x_scale=x_scale, y_scale=y_scale_trans, y_min=y_min, y_max=y_max # Pass scales
    )
    if transmission_fig:
        transmission_path = os.path.join(save_dir, f"{safe_formula}_transmission.png") # Use safe_formula
        try: transmission_fig.savefig(transmission_path, dpi=300)
        except Exception as e: warnings.warn(f"Error saving compound transmission chart: {e}")
        finally: plt.close(transmission_fig)
        print(f"Compound transmission chart saved to: {transmission_path}")

    return save_dir

def plot_mixture_components(elements: Elements,
                            mixture_definition: List[Dict[str, Union[str, float]]], # Still includes density key, though not used here
                            e_min: float,
                            e_max: float,
                            points: int = DEFAULT_INTERPOLATION_POINTS,
                            save_path: Optional[str] = None,
                            show_plot: bool = False,
                            return_fig: bool = False,
                            x_scale: str = 'log',
                            y_scale: str = 'log',
                            y_min: float = None,
                            y_max: float = None
                            ) -> Optional[matplotlib.figure.Figure]:
    """Plots the contribution of each FORMULA to the total mixture attenuation."""
    # --- Validate Mixture Definition Format (allow missing density for this plot) ---
    if not isinstance(mixture_definition, list) or not all(
            isinstance(item, dict) and 'formula' in item and 'proportion' in item
            for item in mixture_definition):
         warnings.warn("Invalid mixture_definition format. Expected List[Dict] with 'formula', 'proportion'.")
         return None
    # --- Normalization (remains same, using only formula and proportion) ---
    proportions = []
    formulas = []
    valid_items = []
    for item in mixture_definition:
        try:
            prop = float(item['proportion'])
            if prop < 0: warnings.warn(f"Skipping component {item['formula']} due to negative proportion."); continue
            formulas.append(item['formula'])
            proportions.append(prop)
            valid_items.append(item) # Keep track of items used for labels later
        except (ValueError, TypeError, KeyError): warnings.warn(f"Skipping item due to invalid format: {item}"); continue

    if not proportions:
         warnings.warn("No valid components found in mixture definition.")
         return None

    total_proportion = sum(proportions)
    if total_proportion <= 0:
        warnings.warn("Total proportion is zero or negative.")
        # Create empty plot
        fig, ax = plt.subplots(figsize=(10 * 1.1, 6 * 1.1)); base_fontsize=15
        ax.text(0.5, 0.5, 'Cannot plot mixture\n(Total proportion <= 0)', ...)
        if return_fig: return fig
        else: plt.close(fig); return None

    normalized_mixture_data = [{'formula': item['formula'], 'mass_fraction': float(item['proportion']) / total_proportion}
                               for item in valid_items]

    # --- Create energy points ---
    if e_min <= 0 and x_scale == 'log':
        warnings.warn("e_min <= 0 for log scale, using default minimum energy.")
        # Use min energy from valid elements in the mixture
        mixture_elements = set(el_sym for f_dict in normalized_mixture_data for el_sym, _ in elements.parse_chemical_formula(f_dict['formula']))
        valid_element_data = [elements.get(symbol=sym) for sym in mixture_elements if elements.get(symbol=sym) is not None and elements.get(symbol=sym).data is not None and hasattr(elements.get(symbol=sym), 'energy_min') and elements.get(symbol=sym).energy_min > 0]
        if not valid_element_data: e_min = 1e-6
        else: e_min = min(el.energy_min for el in valid_element_data)

    if e_min >= e_max:
        warnings.warn(f"Invalid energy range for mixture: e_min={e_min} >= e_max={e_max}")
        return None

    if x_scale == 'log':
        energies = np.logspace(np.log10(e_min), np.log10(e_max), points)
    else:
        energies = np.linspace(e_min, e_max, points)

    # --- Calculate data using the new component calculation method --- Pass only valid items
    # The calculation function itself expects the original format if density is needed elsewhere
    mixture_results = elements.calculate_mixture_cross_section_components(
        valid_items, energies, with_coherent=True
    )

    # --- Prepare figure ---
    fig, ax = plt.subplots(figsize=(10 * 1.1, 6 * 1.1))
    base_fontsize = 22

    # --- Handle calculation errors ---
    if mixture_results is None:
        warnings.warn(f"Cannot calculate component contributions for the mixture.")
        ax.text(0.5, 0.5, 'Cannot plot mixture components\n(Calc error)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=base_fontsize)
        total_mu_rho_mix, formula_contributions = None, None # Ensure defined
        plot_successful = False
    else:
        total_mu_rho_mix, formula_contributions, _ = mixture_results # Ignore effect dict here

        # --- Plotting ---
        plot_successful = False
        # Plot contributions from each formula
        if formula_contributions:
            for formula, contribution in formula_contributions.items():
                 if contribution is not None and np.any(np.isfinite(contribution)):
                    # Find the corresponding normalized data for the label
                    norm_data = next((item for item in normalized_mixture_data if item['formula'] == formula), None)
                    if norm_data:
                        ax.plot(energies, contribution, label=f"{formula} ({norm_data['mass_fraction']*100:.1f}%)", linewidth=1.5)
                        plot_successful = True
        # Plot total mixture coefficient (black, dashed)
        if total_mu_rho_mix is not None and np.any(np.isfinite(total_mu_rho_mix)):
             ax.plot(energies, total_mu_rho_mix, label='Total Mixture', linestyle='--', color='black', linewidth=2.0)
             plot_successful = True # Ensure this is set if total is plotted

        # --- Set scales ---
        if plot_successful:
            try: ax.set_xscale(x_scale)
            except ValueError: warnings.warn(f"Invalid x_scale: '{x_scale}'. Using 'linear'."); ax.set_xscale('linear')
            try: ax.set_yscale(y_scale)
            except ValueError: warnings.warn(f"Invalid y_scale: '{y_scale}'. Using 'linear'."); ax.set_yscale('linear')

            # --- Aesthetics and Labels ---
            ax.set_xlabel('Photon Energy (MeV)', fontsize=base_fontsize)
            ax.set_ylabel('Mass Attenuation Coefficient (cm²/g)', fontsize=base_fontsize)
            ax.set_title('Formula Contributions to Mixture Attenuation', fontsize=base_fontsize * 1.1)
            ax.legend(loc='best', fontsize=base_fontsize * 0.75)
            _apply_tick_grid_styles(ax) # Apply new grid/tick styles
            ax.tick_params(axis='both', which='major', labelsize=base_fontsize * 0.75)
            if y_min is not None and y_max is not None: ax.set_ylim(y_min, y_max)
        elif not plot_successful: # Handle case where calculation worked but resulted in no plottable data
             ax.text(0.5, 0.5, 'Cannot plot mixture components\n(No valid data points)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=base_fontsize)

    plt.tight_layout()

    # --- Saving and Showing ---
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            try: os.makedirs(save_dir)
            except OSError as e: warnings.warn(f"Cannot create directory {save_dir}: {e}")
        try:
            fig.savefig(save_path, dpi=300)
            print(f"Chart saved to: {save_path}")
        except Exception as e:
            warnings.warn(f"Error saving chart to {save_path}: {e}")

    if show_plot:
        plt.show()

    if return_fig:
        return fig
    else:
        plt.close(fig)
        return None

def plot_mixture_effect_contributions(elements: Elements,
                                      mixture_definition: List[Dict[str, Union[str, float]]],
                                      e_min: float,
                                      e_max: float,
                                      points: int = DEFAULT_INTERPOLATION_POINTS,
                                      save_path: Optional[str] = None,
                                      show_plot: bool = False,
                                      return_fig: bool = False,
                                      y_min: float = None,
                                      y_max: float = None,
                                      x_scale: str = 'log',
                                      y_scale: str = 'log' # Now selectable
                                      ) -> Optional[matplotlib.figure.Figure]:
    """Plots the absolute contribution (cm²/g) of each interaction effect for the mixture."""

    # --- Energy points ---
    if e_min <= 0 and x_scale == 'log':
        warnings.warn("e_min <= 0 for log scale, using default minimum energy.")
        mixture_elements = set(el_sym for item in mixture_definition if 'formula' in item for el_sym, _ in elements.parse_chemical_formula(item['formula']))
        valid_element_data = [elements.get(symbol=sym) for sym in mixture_elements if elements.get(symbol=sym) is not None and elements.get(symbol=sym).data is not None and hasattr(elements.get(symbol=sym), 'energy_min') and elements.get(symbol=sym).energy_min > 0]
        if not valid_element_data: e_min = 1e-6
        else: e_min = min(el.energy_min for el in valid_element_data)

    if e_min >= e_max: warnings.warn(f"Invalid energy range."); return None
    if x_scale == 'log': energies = np.logspace(np.log10(e_min), np.log10(e_max), points)
    else: energies = np.linspace(e_min, e_max, points)

    # --- Calculate data (including effects) ---
    mixture_results = elements.calculate_mixture_cross_section_components(
        mixture_definition, energies, with_coherent=True
    )

    # --- Prepare figure ---
    fig, ax = plt.subplots(figsize=(10 * 1.1, 6 * 1.1))
    base_fontsize = 22

    if mixture_results is None:
        warnings.warn(f"Cannot calculate effect contributions for the mixture.")
        ax.text(0.5, 0.5, 'Cannot plot mixture effect contributions\n(Calc error)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=base_fontsize)
        total_mu_rho_mix, _, mixture_effect_contributions = None, None, None
        plot_successful = False
    else:
        total_mu_rho_mix, _, mixture_effect_contributions = mixture_results

        # --- Plotting Lines ---
        plot_successful = False
        effect_plot_params = { # Use same colors/styles as compound plot
            'Photoelectric':  ('Photoelectric Absorption', None, '-'),
            'Coherent': ('Coherent Scattering', None, '-'), 
            'Incoherent': ('Incoherent Scattering', None, '-'),
            'Nuclear': ('Pair Production (Nuclear)', None, '-'),
            'Electron': ('Pair Production (Electron)', None, '-')
        }

        for effect_key, (label, color, linestyle) in effect_plot_params.items():
            data = mixture_effect_contributions.get(effect_key)
            # 替换 inf、nan、负值为 np.nan
            data[data <= 1e-10] = np.nan
            ax.plot(energies, data, label=label, color=color, linestyle=linestyle, linewidth=1.5)
            plot_successful = True

        # Plot Total mixture line
        if total_mu_rho_mix is not None and np.any(np.isfinite(total_mu_rho_mix)):
             ax.plot(energies, total_mu_rho_mix, label='Total Mixture', color='black', linestyle='--', linewidth=2.0)
             plot_successful = True

    # --- Set scales --- Always done after potential plotting
    if plot_successful:
        try: ax.set_xscale(x_scale)
        except ValueError: warnings.warn(f"Invalid x_scale: '{x_scale}'. Using 'linear'."); ax.set_xscale('linear')
        try: ax.set_yscale(y_scale) # Use selectable y_scale
        except ValueError: warnings.warn(f"Invalid y_scale: '{y_scale}'. Using 'linear'."); ax.set_yscale('linear')

        # --- Aesthetics and Labels ---
        ax.set_xlabel('Photon Energy (MeV)', fontsize=base_fontsize)
        ax.set_ylabel('Contribution to μ/ρ (cm²/g)', fontsize=base_fontsize) # Updated Label
        ax.set_title(f'Interaction Effect Contributions for Mixture', fontsize=base_fontsize * 1.1)
        ax.legend(loc='best', fontsize=base_fontsize * 0.75) # Legend inside
        _apply_tick_grid_styles(ax)
        ax.tick_params(axis='both', which='major', labelsize=base_fontsize * 0.75)
        if y_min is not None and y_max is not None: ax.set_ylim(y_min, y_max)
    elif total_mu_rho_mix is not None: # Calc worked but no valid points
         ax.text(0.5, 0.5, f'Cannot plot mixture effect contributions\n(No valid data points)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=base_fontsize)

    plt.tight_layout() # Legend inside

    # --- Saving and Showing logic remains the same, remove bbox_inches='tight' ---
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            try: os.makedirs(save_dir)
            except OSError as e: warnings.warn(f"Cannot create directory {save_dir}: {e}")
        try:
            fig.savefig(save_path, dpi=300) # Remove bbox_inches
            print(f"Chart saved to: {save_path}")
        except Exception as e: warnings.warn(f"Error saving chart: {e}")

    if show_plot: plt.show()

    if return_fig: return fig
    else: plt.close(fig); return None

def plot_mixture_transmission(elements: Elements,
                              mixture_definition: List[Dict[str, Union[str, float]]],
                              e_min: float,
                              e_max: float,
                              mixture_thickness: float, # Total thickness of the mixture layer
                              points: int = DEFAULT_INTERPOLATION_POINTS,
                              save_path: Optional[str] = None,
                              show_plot: bool = False,
                              return_fig: bool = False,
                              x_scale: str = 'log',
                              y_scale: str = 'linear',
                              mixture_density_override: Optional[float] = None
                             ) -> Optional[matplotlib.figure.Figure]:
    """
    Plots the transmission rate through a mixture.
    Requires mixture_definition where each dict contains 'formula', 'proportion',
    and 'density' of the pure component. Requires overall mixture_thickness.
    Density is calculated from components or can be overridden.
    """
    # --- Validate format (ensure density is present for potential calculation) ---
    if not isinstance(mixture_definition, list) or not all(
            isinstance(item, dict) and 'formula' in item and 'proportion' in item and 'density' in item
            for item in mixture_definition):
         warnings.warn("Invalid mixture_definition format for transmission. Need 'formula', 'proportion', 'density'.")
         # Create empty plot with error
         fig, ax = plt.subplots(figsize=(10 * 1.1, 6 * 1.1)); base_fontsize=15
         ax.text(0.5, 0.5, 'Invalid Mixture Input Format', ...)
         if return_fig: return fig
         else: plt.close(fig); return None

    # --- Determine Mixture Density ---
    density_source = "unknown"
    if mixture_density_override is not None:
        if mixture_density_override <= 0:
            warnings.warn(f"Invalid override density: {mixture_density_override}")
            mixture_density = None
        else:
            mixture_density = mixture_density_override
            density_source = f"provided {mixture_density:.3f}"
    else:
        mixture_density = elements.calculate_mixture_density(mixture_definition)
        if mixture_density is None:
            warnings.warn("Could not calculate mixture density.")
        else:
            density_source = f"calculated {mixture_density:.3f}"

    # Cannot proceed without density
    if mixture_density is None:
        fig, ax = plt.subplots(figsize=(10 * 1.1, 6 * 1.1)); base_fontsize=15
        ax.text(0.5, 0.5, 'Cannot Plot Transmission\n(Unable to determine density)', ...)
        if return_fig: return fig
        else: plt.close(fig); return None

    # --- Create energy points ---
    if e_min <= 0 and x_scale == 'log':
        warnings.warn("e_min <= 0 for log scale, using default minimum energy.")
        mixture_elements = set(el_sym for item in mixture_definition for el_sym, _ in elements.parse_chemical_formula(item['formula']))
        valid_element_data = [elements.get(symbol=sym) for sym in mixture_elements if elements.get(symbol=sym) is not None and elements.get(symbol=sym).data is not None and hasattr(elements.get(symbol=sym), 'energy_min') and elements.get(symbol=sym).energy_min > 0]
        if not valid_element_data: e_min = 1e-6
        else: e_min = min(el.energy_min for el in valid_element_data)

    if e_min >= e_max: warnings.warn(f"Invalid energy range."); return None
    if x_scale == 'log': energies = np.logspace(np.log10(e_min), np.log10(e_max), points)
    else: energies = np.linspace(e_min, e_max, points)

    # --- Calculate Transmission using the specific mixture method ---
    transmission = elements.calculate_mixture_transmission(
        mixture_definition,
        energies,
        total_thickness=mixture_thickness, # Pass the total thickness
        mixture_density=mixture_density, # Pass the determined density
        with_coherent=True # Or make this a parameter? Assume True for now.
    )

    # --- Prepare figure ---
    fig, ax = plt.subplots(figsize=(10 * 1.1, 6 * 1.1)) # Larger figure
    base_fontsize = 22 # Increased base font size

    # --- Handle calculation errors for transmission ---
    if transmission is None or np.all(np.isnan(transmission)):
        warnings.warn(f"Cannot calculate valid transmission rates for the mixture.")
        ax.text(0.5, 0.5, 'Cannot plot mixture transmission\n(Calc error)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=base_fontsize)
    else:
        # --- Plotting ---
        ax.plot(energies, transmission, label='Transmission Rate (%)', linewidth=1.5) # Use plot

        # --- Set scales ---
        try: ax.set_xscale(x_scale)
        except ValueError: warnings.warn(f"Invalid x_scale: '{x_scale}'. Using 'linear'."); ax.set_xscale('linear')
        try: ax.set_yscale(y_scale)
        except ValueError: warnings.warn(f"Invalid y_scale: '{y_scale}'. Using 'linear'."); ax.set_yscale('linear')

        # --- Aesthetics and Labels ---
        ax.set_xlabel('Photon Energy (MeV)', fontsize=base_fontsize)
        ax.set_ylabel('Transmission Rate (%)', fontsize=base_fontsize)
        ax.set_title(f'Mixture Transmission ({density_source} g/cm³, {mixture_thickness:.2f} cm)', fontsize=base_fontsize * 1.1) # Show density source
        ax.legend(loc='best', fontsize=base_fontsize * 0.75)
        _apply_tick_grid_styles(ax) # Apply new grid/tick styles
        ax.tick_params(axis='both', which='major', labelsize=base_fontsize * 0.75)

        # Set y-axis limits
        if y_scale == 'linear':
             ax.set_ylim(0, 105) # Y-axis range 0-100%
        else: # Log scale for transmission
             if transmission is not None and np.any(np.isfinite(transmission[transmission > 0])):
                 min_positive_transmission = transmission[transmission > 0].min()
                 ax.set_ylim(bottom=max(1e-3, min_positive_transmission * 0.5), top=110)
             else:
                 ax.set_ylim(bottom=1e-3, top=110) # Fallback

    plt.tight_layout()

    # --- Saving and Showing ---
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            try: os.makedirs(save_dir)
            except OSError as e: warnings.warn(f"Cannot create directory {save_dir}: {e}")
        try:
            fig.savefig(save_path, dpi=300)
            print(f"Chart saved to: {save_path}")
        except Exception as e:
            warnings.warn(f"Error saving chart to {save_path}: {e}")

    if show_plot:
        plt.show()

    if return_fig:
        return fig
    else:
        plt.close(fig)
        return None

def plot_mixture_all(elements: Elements,
                     mixture_definition: List[Dict[str, Union[str, float]]],
                     e_min: float,
                     e_max: float,
                     mixture_thickness: float, # Total thickness of the mixture layer
                     points: int = DEFAULT_INTERPOLATION_POINTS,
                     save_dir: str = None,
                     x_scale: str = 'log',
                     y_scale_atten: str = 'log',
                     y_scale_trans: str = 'linear',
                     y_scale_effect: str = 'log', # New parameter for effect plot y-scale
                     y_min: float = None,
                     y_max: float = None,
                     mixture_density_override: Optional[float] = None
                     ) -> Optional[str]:
    """
    Generate and save all charts related to a mixture, including effect contributions.
    Accepts mixture_definition where each dict contains 'formula', 'proportion',
    and 'density' of the pure component. Requires overall mixture_thickness.
    Density is calculated or overridden. Allows specifying scales for plots.
    """
    # --- Validate mixture def format early (Ensures density is present for calculation) ---
    if not isinstance(mixture_definition, list) or not all(
            isinstance(item, dict) and 'formula' in item and 'proportion' in item and 'density' in item
            for item in mixture_definition):
         warnings.warn("Invalid mixture_definition format for plot_mixture_all. Need 'formula', 'proportion', 'density'.")
         return None

    # --- mixture_name and save_dir logic --- #
    mixture_name = "_".join([f"{item['formula']}" for item in mixture_definition if isinstance(item, dict) and 'formula' in item])
    # Sanitize name
    mixture_name = "".join(c if c.isalnum() else '_' for c in mixture_name)
    if len(mixture_name) > 50: mixture_name = "mixture"
    elif not mixture_name: mixture_name = "mixture"

    if not save_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(TMP_PLOTS_DIR, f"{mixture_name}_{timestamp}")
    if not os.path.exists(save_dir):
        try: os.makedirs(save_dir)
        except OSError as e: warnings.warn(f"Cannot create dir {save_dir}: {e}"); return None

    # Plot mixture components (attenuation contribution by formula) (Uses y_scale_atten)
    components_fig = plot_mixture_components(
        elements, mixture_definition, e_min, e_max, points,
        show_plot=False, return_fig=True,
        x_scale=x_scale, y_scale=y_scale_atten, y_min=y_min, y_max=y_max
    )
    if components_fig:
        components_path = os.path.join(save_dir, f"{mixture_name}_components.png") # Unique name
        try: components_fig.savefig(components_path, dpi=300)
        except Exception as e: warnings.warn(f"Error saving mixture components chart: {e}")
        finally: plt.close(components_fig)
        print(f"Mixture components chart saved to: {components_path}")

    # Plot mixture effect contributions (Uses y_scale_effect)
    effects_fig = plot_mixture_effect_contributions(
         elements, mixture_definition, e_min, e_max, points,
         show_plot=False, return_fig=True,
         x_scale=x_scale, y_scale=y_scale_effect, y_min=y_min, y_max=y_max # Pass new y_scale parameter
    )
    if effects_fig:
        effects_path = os.path.join(save_dir, f"{mixture_name}_effects.png") # Unique name
        try: effects_fig.savefig(effects_path, dpi=300) # No bbox_inches needed
        except Exception as e: warnings.warn(f"Error saving mixture effects chart: {e}")
        finally: plt.close(effects_fig)
        print(f"Mixture effects chart saved to: {effects_path}")

    # Plot mixture transmission chart (Uses y_scale_trans)
    transmission_fig = plot_mixture_transmission(
        elements, mixture_definition, e_min, e_max,
        mixture_thickness=mixture_thickness, # Pass correct parameter
        points=points,
        show_plot=False, return_fig=True,
        x_scale=x_scale, y_scale=y_scale_trans, y_min=y_min, y_max=y_max,
        mixture_density_override=mixture_density_override # Pass override if provided
    )
    if transmission_fig:
        transmission_path = os.path.join(save_dir, f"{mixture_name}_transmission.png") # Unique name
        try: transmission_fig.savefig(transmission_path, dpi=300)
        except Exception as e: warnings.warn(f"Error saving mixture transmission chart: {e}")
        finally: plt.close(transmission_fig)
        print(f"Mixture transmission chart saved to: {transmission_path}")

    return save_dir 