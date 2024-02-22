from functools import partial
import tensorflow as tf
import tensorflow_probability as tfp

# @tf.function
def calculateCoreVars(path_flow_,
                      od_volume,
                      spare_od_path_inc,
                      path_link_inc,
                      path_link_inc_n,
                      bpr_params,):

    path_flow_ = tf.reshape(path_flow_, (-1, 1))
    path_flow_n = od_volume - tf.sparse.sparse_dense_matmul(spare_od_path_inc, path_flow_)

    link_flow = tf.matmul(tf.transpose(path_link_inc), path_flow_) + tf.matmul(tf.transpose(path_link_inc_n),
                                                                               path_flow_n)
    link_cost = bpr_params["fftt"] * (1 + bpr_params["alpha"] * (link_flow / bpr_params["cap"]) ** bpr_params["beta"])

    path_cost = tf.matmul(path_link_inc, link_cost)
    path_cost_n = tf.matmul(path_link_inc_n, link_cost)
    path_flow_all = tf.concat([path_flow_, path_flow_n], axis=0)

    return link_flow


# @tf.function
def optimizeSupply(path_flow,
                   lambda_positive_,
                   bpr_params,
                   lagrangian_params,
                   od_volume,
                   spare_od_path_inc,
                   path_link_inc,
                   path_link_inc_n,
                   link_data,
                   init_path_flows):

    est_init_link_volumes = calculateCoreVars(init_path_flows, od_volume, spare_od_path_inc, path_link_inc,
                                              path_link_inc_n, bpr_params)

    def calculateLoss(path_flow,
                      od_volume,
                      spare_od_path_inc,
                      path_link_inc,
                      path_link_inc_n,
                      lambda_positive_,
                      bpr_params,
                      lagrangian_params,
                      link_data):
        est_link_volumes = calculateCoreVars(path_flow, od_volume, spare_od_path_inc,
                                             path_link_inc, path_link_inc_n, bpr_params)

        def mse(estimation, observation):
            return tf.reduce_mean((estimation - observation)**2)
        def scaled_mse(initial_estimation, estimation, observation):
            return mse(estimation, observation)/mse(initial_estimation, observation)
        def positivity_constraints(lagrangian_params, var):
            return tf.reduce_sum((lagrangian_params / 2) * tf.nn.relu(-var) ** 2)

        def get_vmt(link_volume, link_dist):
            return tf.squeeze(link_volume) * link_dist
        # FIXME: set a configuration for link volume proportions
        car_prop = 0.9
        truck_prop = 0.1
        link_dist = link_data["distance_miles"]
        loss = scaled_mse(est_init_link_volumes * car_prop,
                          est_link_volumes * car_prop, link_data["car_link_volume"]) + \
               scaled_mse(est_init_link_volumes * truck_prop,
                          est_link_volumes * truck_prop, link_data["truck_link_volume"]) + \
               positivity_constraints(lagrangian_params["rho_factor"], path_flow) + \
               scaled_mse(get_vmt(est_init_link_volumes * car_prop, link_dist),
                          get_vmt(est_link_volumes * car_prop, link_dist),
                          get_vmt(link_data["car_link_volume"], link_dist)) + \
               scaled_mse(get_vmt(est_init_link_volumes * truck_prop, link_dist),
                          get_vmt(est_link_volumes * truck_prop, link_dist),
                          get_vmt(link_data["truck_link_volume"], link_dist))

        # FIXME: add an argument to customize different loss functions
        # loss = tf.reduce_sum(
        #     bpr_params["fftt"] * link_flow + (bpr_params["alpha"] * bpr_params["fftt"]) /
        #     (bpr_params["beta"] + 1) * (link_flow / bpr_params["cap"]) ** (bpr_params["beta"] + 1) * bpr_params["cap"]) \
        #        + tf.reduce_sum(tf.multiply(lambda_positive_, tf.nn.relu(-path_flow_all))) + tf.reduce_sum(
        #     (lagrangian_params["rho_factor"] / 2) * tf.nn.relu(-path_flow_all) ** 2)
        return loss

    partial_loss = partial(calculateLoss,
                           od_volume=od_volume,
                           spare_od_path_inc=spare_od_path_inc,
                           path_link_inc=path_link_inc,
                           path_link_inc_n=path_link_inc_n,
                           lambda_positive_=lambda_positive_,
                           bpr_params=bpr_params,
                           lagrangian_params=lagrangian_params,
                           link_data=link_data)


    # FIXME: create functions for the adam optimizer and bfgs optimizer
    # FIXME: create configuration to set learning parameters
    # FIXME: learning stop rule - eta difference
    # Optimization parameters
    learning_rate = 0.01
    epochs = 100

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

    # def loss_gradient_supply(path_flow_):
    #     return tfp.math.value_and_gradient(calculateLossSupply, path_flow_)
    #
    # supply_opt = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=loss_gradient_supply,
    #                                           initial_position=initial_path,
    #                                           tolerance=1e-08,
    #                                           max_iterations=500)
    return path_flow, trace_loss
