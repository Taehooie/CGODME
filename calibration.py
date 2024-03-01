from functools import partial
import tensorflow as tf

# @tf.function
def calculateCoreVars(path_flow_: tf.Tensor,
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

    def conv_paths_to_ods(sparse_matrix, path_flows):
        return tf.sparse.sparse_dense_matmul(sparse_matrix, path_flows)
    def get_positive_path_flow(od_vols, est_paths):
        return tf.clip_by_value(od_vols - est_paths, 0, tf.float32.max)

    path_flow_ = tf.reshape(path_flow_, (-1, 1))
    path_to_ods = conv_paths_to_ods(sparse_matrix["od_path_inc"], path_flow_)
    path_flow_n = get_positive_path_flow(od_volume, path_to_ods) # one od path flow

    # FIXME: convert od_flows to o_flows (incidence matrix?)
    od_flows = path_flow_n + path_to_ods
    o_flows = conv_paths_to_ods(sparse_matrix["o_od_inc"], od_flows)

    link_flow = tf.matmul(tf.transpose(path_link_inc), path_flow_) + \
                tf.matmul(tf.transpose(path_link_inc_n), path_flow_n)
    link_cost = bpr_params["fftt"] * (1 + bpr_params["alpha"] * (link_flow / bpr_params["cap"]) ** bpr_params["beta"])

    path_cost = tf.matmul(path_link_inc, link_cost)
    path_cost_n = tf.matmul(path_link_inc_n, link_cost)
    path_flow_all = tf.concat([path_flow_, path_flow_n], axis=0)

    return link_flow, od_flows, o_flows


# @tf.function
def optimizeSupply(path_flow,
                   lambda_positive_,
                   bpr_params,
                   lagrangian_params,
                   od_volume,
                   sparse_matrix,
                   path_link_inc,
                   path_link_inc_n,
                   target_data,
                   init_path_flows,
                   optimization_params,
                   data_imputation,
                   ):
    """

    Args:
        path_flow:
        lambda_positive_:
        bpr_params:
        lagrangian_params:
        od_volume:
        spare_od_path_inc:
        path_link_inc:
        path_link_inc_n:
        target_data:
        init_path_flows:
        optimization_setting (dict):

    Returns:

    """

    # get initial link volumes and od flows calculated by initial path flows
    est_init_link_volumes, est_init_od_flows, est_init_o_flows = calculateCoreVars(init_path_flows,
                                                                                   od_volume,
                                                                                   sparse_matrix,
                                                                                   path_link_inc,
                                                                                   path_link_inc_n,
                                                                                   bpr_params,
                                                                                   )

    def calculateLoss(path_flow,
                      od_volume,
                      sparse_matrix,
                      path_link_inc,
                      path_link_inc_n,
                      lambda_positive_,
                      bpr_params,
                      lagrangian_params,
                      target_data,
                      data_imputation,
                      ):
        """

        Args:
            path_flow:
            od_volume:
            spare_od_path_inc:
            path_link_inc:
            path_link_inc_n:
            lambda_positive_:
            bpr_params:
            lagrangian_params:
            target_data:

        Returns:

        """
        est_link_volumes, est_od_flows, est_o_flows = calculateCoreVars(path_flow,
                                                           od_volume,
                                                           sparse_matrix,
                                                           path_link_inc,
                                                           path_link_inc_n,
                                                           bpr_params)

        def mse(estimation, observation):
            return tf.reduce_mean((estimation - observation)**2)
        def scaled_mse(initial_estimation, estimation, observation):
            return mse(estimation, observation)/mse(initial_estimation, observation)
        def positivity_constraints(lagrangian_params, var):
            return tf.reduce_sum((lagrangian_params / 2) * tf.nn.relu(-var) ** 2)

        def get_vmt(link_volume, link_dist):
            return tf.squeeze(link_volume) * link_dist
        car_prop = data_imputation["car_prop"]
        truck_prop = data_imputation["truck_prop"]
        link_dist = target_data["distance_miles"]
        loss = scaled_mse(est_init_link_volumes * car_prop,
                          est_link_volumes * car_prop, target_data["car_link_volume"]) + \
               scaled_mse(est_init_link_volumes * truck_prop,
                          est_link_volumes * truck_prop, target_data["truck_link_volume"]) + \
               positivity_constraints(lagrangian_params["rho_factor"], path_flow) + \
               scaled_mse(get_vmt(est_init_link_volumes * car_prop, link_dist),
                          get_vmt(est_link_volumes * car_prop, link_dist),
                          get_vmt(target_data["car_link_volume"], link_dist)) + \
               scaled_mse(get_vmt(est_init_link_volumes * truck_prop, link_dist),
                          get_vmt(est_link_volumes * truck_prop, link_dist),
                          get_vmt(target_data["truck_link_volume"], link_dist)) + \
               scaled_mse(est_init_od_flows, est_od_flows, target_data["observed_od_volume"]) + \
               scaled_mse(est_init_o_flows, est_o_flows, target_data["observed_o_volume"])

        return loss

    partial_loss = partial(calculateLoss,
                           od_volume=od_volume,
                           sparse_matrix=sparse_matrix,
                           path_link_inc=path_link_inc,
                           path_link_inc_n=path_link_inc_n,
                           lambda_positive_=lambda_positive_,
                           bpr_params=bpr_params,
                           lagrangian_params=lagrangian_params,
                           target_data=target_data,
                           data_imputation=data_imputation)

    # FIXME: learning stop rule - eta difference
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
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')
        trace_loss.append(loss.numpy())

    return path_flow, trace_loss
