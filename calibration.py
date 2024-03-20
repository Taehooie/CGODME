from functools import partial
import tensorflow as tf


# @tf.function
def odme_mapping_variables(path_flow_: tf.Tensor,
                           od_volume: tf.Tensor,
                           sparse_matrix: dict,
                           path_link_inc: tf.Tensor,
                           path_link_inc_n: tf.Tensor,
                           bpr_params: dict,
                           ):
    """

    Args:
        path_flow_:
        od_volume:
        sparse_matrix:
        path_link_inc:
        path_link_inc_n:
        bpr_params:

    Returns:

    """

    def conv_paths_to_ods(sparse_matrix, flows):
        return tf.sparse.sparse_dense_matmul(sparse_matrix, flows)

    def get_positive_path_flow(od_vols, est_paths):
        return tf.clip_by_value(od_vols - est_paths, 0, tf.float32.max)

    path_flow_ = tf.reshape(path_flow_, (-1, 1))
    path_to_ods = conv_paths_to_ods(sparse_matrix["od_path_inc"], path_flow_)
    path_flow_n = get_positive_path_flow(od_volume, path_to_ods)  # one od path flow

    od_flows = path_flow_n + path_to_ods
    o_flows = conv_paths_to_ods(sparse_matrix["o_od_inc"], od_flows)

    link_flow = tf.matmul(tf.transpose(path_link_inc), path_flow_) + \
                tf.matmul(tf.transpose(path_link_inc_n), path_flow_n)
    link_cost = bpr_params["fftt"] * (1 + bpr_params["alpha"] * (link_flow / bpr_params["cap"]) ** bpr_params["beta"])

    # path_cost = tf.matmul(path_link_inc, link_cost)
    # path_cost_n = tf.matmul(path_link_inc_n, link_cost)
    # path_flow_all = tf.concat([path_flow_, path_flow_n], axis=0)

    optimal_path_flows = {}
    optimal_path_flows["one_od_pair"] = path_flow_n
    optimal_path_flows["multi_od_pairs"] = path_flow_

    return link_flow, od_flows, o_flows, optimal_path_flows

# @tf.function
def multi_objective_loss(path_flow,
                         od_volume,
                         sparse_matrix,
                         path_link_inc,
                         path_link_inc_n,
                         bpr_params,
                         target_data,
                         data_imputation,
                         obj_setting,
                         init_odme_mapping_vars,
                         optimization_params,
                         ):
    """

    Args:
        path_flow:
        od_volume:
        sparse_matrix:
        path_link_inc:
        path_link_inc_n:
        bpr_params:
        target_data:
        data_imputation:
        obj_setting:

    Returns:

    """
    est_link_volumes, est_od_flows, est_o_flows, _ = odme_mapping_variables(path_flow,
                                                                            od_volume,
                                                                            sparse_matrix,
                                                                            path_link_inc,
                                                                            path_link_inc_n,
                                                                            bpr_params)

    def mse(estimation, observation):
        # Reshaping the estimated and observed arrays to avoid OOM
        reshape_estimation = tf.reshape(estimation, (estimation.shape[0], ))
        reshape_observation = tf.reshape(observation, (observation.shape[0], ))

        # calculate mean square error
        mean_square_err = tf.reduce_mean(tf.subtract(reshape_estimation, reshape_observation) ** 2)
        return mean_square_err

    def scaled_mse(initial_estimation, estimation, observation):
        init_mse = mse(initial_estimation, observation)
        if init_mse < 1:
            # If-statement to avoid zero division (e.g., 1.0 / 0.003)
            init_mse = tf.constant(1.0, dtype=tf.float32)
        return mse(estimation, observation) / init_mse

    def positivity_constraints(penalty_coeffi, var):
        return tf.reduce_sum(penalty_coeffi * tf.nn.relu(-var) ** 2)

    def get_vmt(link_volume, link_dist):
        return tf.squeeze(link_volume) * link_dist

    car_prop = data_imputation["car_prop"]
    truck_prop = data_imputation["truck_prop"]
    link_dist = target_data["distance_miles"]
    loss = obj_setting["passenger_car_count"] * scaled_mse(init_odme_mapping_vars["link_counts"] * car_prop,
                                                           est_link_volumes * car_prop,
                                                           target_data["car_link_volume"]) \
           + obj_setting["truck_count"] * scaled_mse(init_odme_mapping_vars["link_counts"] * truck_prop,
                                                     est_link_volumes * truck_prop,
                                                     target_data["truck_link_volume"]) \
           + obj_setting["passenger_car_vmt"] * scaled_mse(
        get_vmt(init_odme_mapping_vars["link_counts"] * car_prop, link_dist),
        get_vmt(est_link_volumes * car_prop, link_dist),
        get_vmt(target_data["car_link_volume"], link_dist)) \
           + obj_setting["truck_vmt"] * scaled_mse(
        get_vmt(init_odme_mapping_vars["link_counts"] * truck_prop, link_dist),
        get_vmt(est_link_volumes * truck_prop, link_dist),
        get_vmt(target_data["truck_link_volume"], link_dist)) \
           + obj_setting["od_split"] * scaled_mse(init_odme_mapping_vars["od_flows"], est_od_flows,
                                                  target_data["observed_od_volume"]) \
           + obj_setting["zonal"] * scaled_mse(init_odme_mapping_vars["o_flows"], est_o_flows,
                                               target_data["observed_o_volume"]) \
           + positivity_constraints(optimization_params["penalty_coefficient"], path_flow)
    return loss


# @tf.function
def optimization(path_flow,
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
                 ):
    """

    Args:
        path_flow:
        bpr_params:
        od_volume:
        sparse_matrix:
        path_link_inc:
        path_link_inc_n:
        target_data:
        init_odme_mapping_variables:
        optimization_params:
        data_imputation:
        obj_setting:

    Returns:

    """

    partial_loss = partial(multi_objective_loss,
                           od_volume=od_volume,
                           sparse_matrix=sparse_matrix,
                           path_link_inc=path_link_inc,
                           path_link_inc_n=path_link_inc_n,
                           bpr_params=bpr_params,
                           target_data=target_data,
                           data_imputation=data_imputation,
                           obj_setting=obj_setting,
                           init_odme_mapping_vars=init_odme_mapping_variables,
                           optimization_params=optimization_params,
                           )

    # FIXME: convergence stopping rule
    # Set optimization parameters
    learning_rate = optimization_params["learning_rates"]
    epochs = optimization_params["training_steps"]

    # Set the optimizer
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)

    trace_loss = []
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = partial_loss(path_flow)

        # Gradients
        gradients = tape.gradient(loss, [path_flow])

        # Update path flows
        optimizer.apply_gradients(zip(gradients, [path_flow]))
        # log losses
        if (epoch + 1) % (epochs / 10) == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')
        trace_loss.append(loss.numpy())

    return path_flow, trace_loss
