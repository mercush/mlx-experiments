import mlx.core as mx
from tensorflow import keras
import random
import networkx as nx
import dwave_networkx as dnx

epochs = 10
learning_rate = 0.003
alpha = 0.6
batch_size = 512
sample_iter = 10
dtype = mx.bfloat16

with open("coloring.txt", "r") as f:
    contents = f.read()
coloring = mx.array(eval(contents), dtype=dtype)
graph = dnx.pegasus_graph(14)
graph_mask = mx.array(nx.to_numpy_array(graph), dtype=dtype)
block_masks = mx.array([coloring == i for i in range(1, 5)], dtype=mx.bool_)

num_edges = graph.number_of_edges()

w, h = 28, 28
num_classes = 10
num_visible_neurons = w * h + num_classes * 5
num_neurons = 4264
# TODO: double check that these random_idx_transforms are being done correctly
randomized_idx_loc = mx.array(random.sample(range(num_neurons), num_visible_neurons))
random_idx_transform = mx.zeros((num_neurons, num_visible_neurons), dtype=dtype) # need to make this one-hot encoding of randomized_idx_loc
random_idx_transform[randomized_idx_loc, mx.arange(num_visible_neurons)] = 1 # Create random index transformation matrix for labels


(x_train, x_labels_train), (x_test, x_labels_test) = keras.datasets.mnist.load_data()
x_train = 2 * mx.array((x_train / 255.0) > 0.5, dtype=dtype) - 1
x_labels_train = mx.array(x_labels_train, dtype=dtype)
train_size = x_train.shape[0]

x_test = 2 * mx.array((x_test / 255.0) > 0.5, dtype=dtype) - 1
x_labels_test = mx.array(x_labels_test, dtype=dtype)

imgs = x_train
labels = x_labels_train
imgs = imgs.reshape(-1, w * h)
one_hot_labels = mx.array([[1 if i == label else 0 for i in range(num_classes)] for label in x_labels_train], dtype=dtype)
one_hot_labels = mx.repeat(one_hot_labels, 5, axis=1).reshape(-1, 50)

random_transform_train_imgs = imgs @ random_idx_transform[:, :w*h].T # (60000, 784) x (784, 4264)
random_transform_train_labels =  one_hot_labels @ random_idx_transform[:, w*h:].T # (60000, 50) x (50, 4264)

# TODO: double check that these transforms are correct
def random_img_to_normal(random_transform_img):
    return (random_transform_img @ random_idx_transform)[:,:w*h].reshape(-1, w, h)

def random_label_to_normal(random_transform_label):
    return mx.mean((random_transform_label @ random_idx_transform)[:, w*h:].reshape(-1, 10, 5), axis=2)

# imgs = x_test
# labels = x_labels_test
# imgs = imgs.reshape(-1, w*h)
# random_transform_test_imgs = random_idx_transform[:,:w*h] @ imgs

with open("visible_bias_init.txt") as f:
    contents = f.read()
visible_bias_init = mx.array(eval(contents), dtype=dtype).reshape((-1,))
