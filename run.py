import warnings
import tensorflow as tf
import pandas as pd
import numpy as np
from data import data_generation
from calibration import odme_mapping_variables, optimization
import yaml
import logging

warnings.filterwarnings('ignore')  # ignore warning messages
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)  # configuration of logging messages


def run_optimization(od_volume: tf.Tensor,
                     sparse_matrix: dict,
                     path_link_inc: tf.Tensor,
                     path_link_inc_n: tf.Tensor,
                     path_flow: tf.Tensor,
                     bpr_params: dict,
                     optimization_params: dict,
                     target_data: dict,
                     data_imputation: dict,
                     obj_setting: dict,
                     ) -> None:
    """
    Perform ADMM optimization for path flows.

    Parameters:
    - od_volume (tf.Tensor): Origin-Destination (OD) volume matrix.
    - sparse_matrix (dict): Sparse incidence matrix (e.g., mapping origin-destination into paths).
    - path_link_inc (tf.Tensor): Path-link incidence matrix.
    - path_link_inc_n (tf.Tensor): Path-link incidence matrix with one pair.
    - path_flow (tf.Tensor): Initial path flows.
    - bpr_params (dict): Parameters for the BPR (Bureau of Public Roads) function.
    - training_steps (int): Number of ADMM training steps.
    - target_data (dict): observed target data
    - init_path_flows (tf.Tensor): initialized path flows (either DTALite results or randomly generated)
    - data_imputation (dict): proportion of car and truck in a given network
    - obj_setting (dict): dictionary of multi-objective dispersion parameters

    Returns:
    None
    """

    logging.info("Finding Optimal Path Flows...")
    init_odme_mapping_variables = {}
    # get the initialized odme mapping variables
    init_link_volumes, init_od_flows, init_o_flows, _ = odme_mapping_variables(path_flow,
                                                                               od_volume,
                                                                               sparse_matrix,
                                                                               path_link_inc,
                                                                               path_link_inc_n,
                                                                               bpr_params)
    init_odme_mapping_variables["link_counts"] = init_link_volumes
    init_odme_mapping_variables["od_flows"] = init_od_flows
    init_odme_mapping_variables["o_flows"] = init_o_flows

    # start to find optimal path flows
    optimal_path_flow, losses = optimization(path_flow,
                                             bpr_params,
                                             od_volume,
                                             sparse_matrix,
                                             path_link_inc,
                                             path_link_inc_n,
                                             target_data,
                                             init_odme_mapping_variables,
                                             optimization_params,
                                             data_imputation,
                                             obj_setting,
                                             )
    # get the odme mapping variables using the optimal path flows
    estimated_link_volumes, estimated_od_flows, estimated_o_flows, optimal_paths = odme_mapping_variables(
        optimal_path_flow,
        od_volume,
        sparse_matrix,
        path_link_inc,
        path_link_inc_n,
        bpr_params)

    concat_path_flows = get_path_flow_columns(load_data, optimal_paths)
    evaluation(losses,
               concat_path_flows,
               tf.squeeze(estimated_link_volumes),
               tf.squeeze(estimated_od_flows),
               tf.squeeze(estimated_o_flows),
               target_data,
               data_imputation,
               )
    logging.info("Complete!")


def get_path_flow_columns(data_source, optimal_paths):
    """

    Args:
        data_source:
        optimal_paths:

    Returns:

    """
    od_layer = data_source.od_df
    path_layer = data_source.od_to_path_layer()
    restore_path_flows = []
    multi_od_pair_idx = 0
    one_od_pair_idx = 0

    for i in range(len(od_layer)):
        od_id = od_layer.loc[i, 'od_id']
        od_path = path_layer[path_layer['od_id'] == od_id].reset_index(drop=True)

        for j in range(len(od_path)):
            if j < len(od_path) - 1:
                restore_path_flows.append(optimal_paths["multi_od_pairs"][multi_od_pair_idx].numpy().item())
                multi_od_pair_idx += 1
            else:
                restore_path_flows.append(optimal_paths["one_od_pair"][one_od_pair_idx].numpy().item())
                one_od_pair_idx += 1

    return restore_path_flows

def rmse(estimated, observation):
    """

    Args:
        estimated:
        observation:

    Returns:

    """
    squared_diff = (estimated - observation) ** 2
    mean_squared_diff = np.mean(squared_diff)
    rmse_value = np.sqrt(mean_squared_diff)
    return rmse_value


def evaluation(losses,
               optimal_path_flows,
               estimated_link_volumes,
               estimated_od_flows,
               estimated_o_flows,
               target_data,
               data_imputation,
               ):
    """

    Args:
        losses:
        optimal_path_flows:
        estimated_link_volumes:
        estimated_od_flows:
        estimated_o_flows:
        target_data:
        data_imputation:

    Returns:

    """
    logging.info("Saving loss ...")
    get_df_losses = pd.DataFrame(losses, columns=["losses"])
    get_df_losses.index.name = "epoch"
    get_df_losses.to_csv(config["data_path"] + "loss_results.csv")

    logging.info("Saving the optimal path flows ...")
    load_path_df = load_data.route_assignment_data[["path_no",
                                                    "o_zone_id",
                                                    "d_zone_id",
                                                    "node_sequence",
                                                    "link_sequence",
                                                    "geometry",
                                                    ]]
    path_flow_df = pd.DataFrame(optimal_path_flows, columns=["Path_Flows"])
    load_path_df = load_path_df.join(path_flow_df)
    load_path_df.to_csv(config["data_path"] + "calibrated_results.csv", index=False)

    # Goodness of Fit (RMSE)
    car_prop = data_imputation["car_prop"]
    truck_prop = data_imputation["truck_prop"]
    rmse_car_link_volumes = rmse(estimated_link_volumes * car_prop, target_data["car_link_volume"])
    rmse_truck_link_volumes = rmse(estimated_link_volumes * truck_prop, target_data["truck_link_volume"])
    rmse_car_vmt = rmse(estimated_link_volumes * car_prop * target_data["distance_miles"],
                        target_data["car_link_volume"] * target_data["distance_miles"])
    rmse_truck_vmt = rmse(estimated_link_volumes * truck_prop * target_data["distance_miles"],
                          target_data["truck_link_volume"] * target_data["distance_miles"])
    rmse_od_flows = rmse(estimated_od_flows, target_data["observed_od_volume"])
    rmse_o_flows = rmse(estimated_o_flows, target_data["observed_o_volume"])

    # logging messages
    logging.info(f"RMSE: Passenger Car Count: {rmse_car_link_volumes}")
    logging.info(f"RMSE: Truck Count: {rmse_truck_link_volumes}")
    logging.info(f"RMSE: Passenger Car VMT: {rmse_car_vmt}")
    logging.info(f"RMSE: Truck VMT: {rmse_truck_vmt}")
    logging.info(f"RMSE: OD Flow: {rmse_od_flows}")
    logging.info(f"RMSE: O Flow: {rmse_o_flows}")


if __name__ == "__main__":
    # Load YAML configuration file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    load_data = data_generation(config)
    od_volume, spare_od_path_inc, path_link_inc, path_link_inc_n, _, = load_data.reformed_incidence_mat()
    path_flow = load_data.get_init_path_values(init_given=config["avail_initial_path_flow"])
    init_path_flow = path_flow
    bpr_params = load_data.get_bpr_params()
    loaded_link_target = load_data.link_df["volume"]
    total_link_volume = np.array(loaded_link_target, dtype='f')

    sparse_matrix = {"o_od_inc": load_data.get_o_to_od_incidence_mat(), "od_path_inc": spare_od_path_inc}
    target_data = {"observed_o_volume": np.array(load_data.ozone_df["volume"], dtype="f"),
                   "observed_od_volume": np.array(load_data.od_df["volume"], dtype="f"),
                   "total_link_volume": np.array(load_data.link_df["volume"], dtype="f"),
                   "car_link_volume": np.array(load_data.link_df["car_vol"], dtype="f"),
                   "truck_link_volume": np.array(load_data.link_df["truck_vol"], dtype="f"),
                   "distance_miles": np.array(load_data.link_df["distance_mile"], dtype="f")}

    lagrangian_params, lambda_positive = load_data.get_lagrangian_params(path_link_inc_n, path_link_inc)
    run_optimization(od_volume=od_volume,
                     sparse_matrix=sparse_matrix,
                     path_link_inc=path_link_inc,
                     path_link_inc_n=path_link_inc_n,
                     path_flow=path_flow,
                     bpr_params=bpr_params,
                     optimization_params=config["optimization_setting"],
                     target_data=target_data,
                     data_imputation=config["data_imputation"],
                     obj_setting=config["multi_objective_function_setting"]
                     )
