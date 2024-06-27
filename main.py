import Load_Data as ld
import matplotlib.pyplot as plt
import Helper_Functions as hf
import Deep_Neural_Network as dn
import Reshape_Image as ri

#
parameters = dn.two_layer_model(ri.train_x, ri.train_y, layers_dims = (ri.n_x, ri.n_h, ri.n_y), num_iterations = 2500, print_cost=True)
#
# predictions_train = hf.predict(ri.train_x, ri.train_y, parameters)
#
# predictions_test = hf.predict(ri.test_x, ri.test_y, parameters)


# parameters = dn.L_layer_model(ri.train_x, ri.train_y, ri.layers_dims, num_iterations = 2500, print_cost = True)
#
# pred_train = hf.predict(ri.train_x, ri.train_y, parameters)

pred_test = hf.predict(ri.test_x, ri.test_y, parameters)

hf.print_mislabeled_images(ri.classes, ri.test_x, ri.test_y, pred_test)
# parameters, costs = dn.two_layer_model(ri.train_x, ri.train_y, layers_dims = (ri.n_x, ri.n_h, ri.n_y), num_iterations = 2500, print_cost=True)
#
# parameters, costs = dn.L_layer_model(ri.train_x, ri.train_y, ri.layers_dims, num_iterations = 2500, print_cost = True)