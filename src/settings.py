"""Default numerical experiment parameters; edit this file for new training runs.

Validation and resume use the parameters embedded in their selected checkpoint.
Physical DGU/line constants and primary equations live in plant.py.
"""
from copy import deepcopy

PARAMETERS = {'schema_version': 1,
 'seed': 202610023,
 'h_s': 5e-05,
 'rho_V': 10.0,
 'primary': {'k1': 0.0, 'k2_ratio': 0.5, 'k3_fraction': 0.35},
 'controller': {'gamma_R': 25.0,
                'a': 0.001,
                'epsilon': 0.02,
                'state_dim': 6,
                'width': 6,
                'hidden_dim': 8,
                'learn_edge_weights': True,
                'edge_weight_bounds': [0.0001, 1.0]},
 'training': {'epochs': 152,
              'duration_s': 0.15,
              'perturbation': 0.2,
              'freeze_edge_weights': True,
              'ren_learning_rate': 0.0025,
              'eta_learning_rate': 0.0008,
              'mad_first_learning_rate': 0.0008,
              'mad_final_learning_rate': 0.004,
              'lr_final_fraction': 0.25,
              'load_step_range_A': [0.8, 2.0],
              'large_load_step_range_A': [2.0, 3.2],
              'large_load_every_epochs': 5,
              'before_range_s': [0.003, 0.007],
              'edge_learning_rate': 0.03},
 'guard': {'seed': 91300,
           'duration_s': 0.5,
           'before_s': 0.005,
           'perturbation': 0.2,
           'load_step_A': 1.1,
           'every_epochs': 4,
           'peak_relative_allowance': 0.05,
           'peak_absolute_allowance_V': 0.02,
           'parent_relative_allowance': 0.1,
           'parent_absolute_allowance_V': 0.02},
 'validation': {'duration_s': 0.5, 'seed': 3100, 'perturbation': 0.4},
 'representative': {'before_s': 0.025, 'after_s': 0.5, 'separate_experiments': True},
 'loss': {'voltage_tolerance': 0.035,
          'smooth_tube_tau': 0.005,
          'tube_radius': 10.0,
          'reference_difference_step': 0.0005,
          'event_window_s': 0.5,
          'settling_delay_s': 0.015,
          'tail_window_s': 0.025},
 'loss_calibration': {'control_scale_V': 2.0, 'smooth_scale_V': 1.0},
 'transient_loss': {'early_window_s': 0.05,
                    'priorities': {'early_voltage_energy': 4.0,
                                   'early_voltage_peak_squared': 4.0,
                                   'early_current_energy': 1.5}},
 'runtime': {'wall_time_hours': 2.5,
             'finalization_reserve_s': 180,
             'minimum_epoch_forecast_s': 240,
             'epoch_time_safety_factor': 1.2}}


def default_parameters():
    """Return an independent parameter dictionary for a new optimizer."""
    return deepcopy(PARAMETERS)
