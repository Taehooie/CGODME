import warnings
import tensorflow as tf
import pandas as pd
import numpy as np
from data import data_generation
from calibration import calculateCoreVars, optimizeSupply
import argparse
import yaml
import logging
warnings.filterwarnings('ignore') # ignore warning messages
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG) # configuration of logging messages


def run_optimization(od_volume: tf.Tensor,
                     sparse_matrix: dict,
                     path_link_inc: tf.Tensor,
                     path_link_inc_n: tf.Tensor,
                     path_flow: tf.Tensor,
                     bpr_params: dict,
                     lagrangian_params: dict,
                     lambda_positive: tf.Tensor,
                     training_steps: int,
                     target_data: dict,
                     init_path_flows: tf.Tensor) -> None:

    """
    Perform ADMM optimization for path flows.

    Parameters:
    - od_volume (tf.Tensor): Origin-Destination (OD) volume matrix.
    - sparse_matrix (dict): Sparse incidence matrix (e.g., mapping origin-destination into paths).
    - path_link_inc (tf.Tensor): Path-link incidence matrix.
    - path_link_inc_n (tf.Tensor): Path-link incidence matrix with one pair.
    - path_flow (tf.Tensor): Initial path flows.
    - bpr_params (dict): Parameters for the BPR (Bureau of Public Roads) function.
    - lagrangian_params (dict): Parameters for the Lagrangian multiplier update.
    - lambda_positive (tf.Tensor): Lagrangian multipliers for positive path flows.
    - training_steps (int): Number of ADMM training steps.
    - target_data (dict): observed target data

    Returns:
    None
    """

    logging.info("Finding Optimal Path Flows...")
    path_flow, losses = optimizeSupply(path_flow,
                                       lambda_positive,
                                       bpr_params,
                                       lagrangian_params,
                                       od_volume,
                                       sparse_matrix,
                                       path_link_inc,
                                       path_link_inc_n,
                                       target_data,
                                       init_path_flows,
                                       )

    estimated_link_volumes, estimated_od_flows, estimated_o_flows = calculateCoreVars(path_flow,
                                                                   od_volume,
                                                                   sparse_matrix,
                                                                   path_link_inc,
                                                                   path_link_inc_n,
                                                                   bpr_params)

    evaluation(losses,
               path_flow,
               tf.squeeze(estimated_link_volumes),
               tf.squeeze(estimated_od_flows),
               tf.squeeze(estimated_o_flows),
               target_data,
               )
    logging.info("Complete!")

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

def evaluation(losses, optimal_path_flows, estimated_link_volumes, estimated_od_flows, estimated_o_flows, target_data):
    """

    Args:
        losses:
        optimal_path_flows:
        estimated_link_volumes:
        target_data:

    Returns:

    """
    logging.info("Saving the loss result ...")
    get_df_losses = pd.DataFrame(losses, columns=["losses"])
    get_df_losses.index.name = "epoch"
    get_df_losses.to_csv("loss_results.csv")

    logging.info("Saving the estimated path flows ...")
    pd.DataFrame(optimal_path_flows.numpy(), columns=["Path_Flows"]).to_csv("estimated_path_flows.csv", index=False)

    # RMSE for a goodness of fit
    # NOTE: 0.9 means 90 percent of link volumes is car
    rmse_car_link_volumes = rmse(estimated_link_volumes * 0.9, target_data["car_link_volume"])
    rmse_truck_link_volumes = rmse(estimated_link_volumes * 0.1, target_data["truck_link_volume"])
    rmse_car_vmt = rmse(estimated_link_volumes * 0.9 * target_data["distance_miles"],
                        target_data["car_link_volume"] * 0.9 * target_data["distance_miles"])
    rmse_truck_vmt = rmse(estimated_link_volumes * 0.1 * target_data["distance_miles"],
                        target_data["truck_link_volume"] * 0.1 * target_data["distance_miles"])
    rmse_od_flows = rmse(estimated_od_flows, target_data["observed_od_volume"])
    rmse_o_flows = rmse(estimated_o_flows, target_data["observed_o_volume"])

    # logging messages
    logging.info(f"RMSE: Car Link Volumes: {rmse_car_link_volumes}")
    logging.info(f"RMSE: Truck Link Volumes: {rmse_truck_link_volumes}")
    logging.info(f"RMSE: Car VMT: {rmse_car_vmt}")
    logging.info(f"RMSE: Truck VMT: {rmse_truck_vmt}")
    logging.info(f"RMSE: OD Flow: {rmse_od_flows}")
    logging.info(f"RMSE: O Flow: {rmse_o_flows}")

if __name__ == "__main__":

    # TODO: read the initial setting from a yaml file (e.g., learning rates, training steps, data directory)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default="./data/Sioux_Falls/",
                        help="Directory containing data files")

    parser.add_argument("--demand_rand",
                        type=str,
                        default=False,
                        help="Randomize zonal demand through uniform distribution")

    parser.add_argument('--training_steps',
                        type=int,
                        default=5,
                        help="Number of ADMM training steps")
    args = parser.parse_args()

    load_data = data_generation(args.data_dir, args.demand_rand)
    od_volume, spare_od_path_inc, path_link_inc, path_link_inc_n, _, = load_data.reformed_incidence_mat()
    path_flow = load_data.get_init_path_values(init_given=False)
    init_path_flow = path_flow
    bpr_params = load_data.get_bpr_params()
    loaded_link_target = load_data.link_df["volume"]
    total_link_volume = np.array(loaded_link_target, dtype='f')

    sparse_matrix = {}
    sparse_matrix["o_od_inc"] = load_data.get_o_to_od_incidence_mat()
    sparse_matrix["od_path_inc"] = spare_od_path_inc
    target_data = {}
    target_data["observed_o_volume"] = np.array(load_data.ozone_df["volume"], dtype="f")
    target_data["observed_od_volume"] = np.array(load_data.od_df["volume"], dtype="f")
    target_data["total_link_volume"] = np.array(load_data.link_df["volume"], dtype="f")
    target_data["car_link_volume"] = np.array(load_data.link_df["car_vol"], dtype="f")
    target_data["truck_link_volume"] = np.array(load_data.link_df["truck_vol"], dtype="f")
    target_data["distance_miles"] = np.array(load_data.link_df["distance_mile"], dtype="f")

    lagrangian_params, lambda_positive = load_data.get_lagrangian_params(path_link_inc_n, path_link_inc)
    run_optimization(od_volume=od_volume,
                     sparse_matrix=sparse_matrix,
                     path_link_inc=path_link_inc,
                     path_link_inc_n=path_link_inc_n,
                     path_flow=path_flow,
                     bpr_params=bpr_params,
                     lagrangian_params=lagrangian_params,
                     lambda_positive=lambda_positive,
                     training_steps=args.training_steps,
                     target_data=target_data,
                     init_path_flows=init_path_flow)
