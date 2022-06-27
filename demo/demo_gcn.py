# coding=utf-8

import os

import flax.linen

from jax_gnn.utils.jax_utils import find_params_by_name

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax_gnn as jg

import jax.numpy as jnp
import jax
import optax




graph, (train_index, valid_index, test_index) = jg.datasets.CoraDataset().load_data()

graph = graph.device_put()
train_index = jax.device_put(train_index)
valid_index = jax.device_put(valid_index)
test_index = jax.device_put(test_index)

seed = 100
learning_rate = 1e-2
l2_coef = 5e-4
drop_rate = 0.1
num_classes = jnp.max(graph.y) + 1

rng = jax.random.PRNGKey(seed)
init_rng, dropout_rng = jax.random.split(rng, num=2)


class GCNModel(flax.linen.Module):

    def setup(self):
        self.dropout = flax.linen.Dropout(drop_rate)
        self.gcn0 = jg.nn.GCN(16, activation=jax.nn.relu)
        self.gcn1 = jg.nn.GCN(num_classes)

    @flax.linen.compact
    def __call__(self, x, edge_index, training=False, cache=None):

        h = self.dropout(x, deterministic=not training)
        h = self.gcn0(h, edge_index, cache=cache)
        h = self.dropout(h, deterministic=not training)
        h = self.gcn1(h, edge_index, cache=cache)
        return h


model = GCNModel()
jg.nn.gcn_build_cache_for_graph(graph)
params = model.init(jax.random.PRNGKey(seed), graph.x, graph.edge_index)

optimizer = optax.adam(learning_rate=learning_rate)


def compute_loss(params, mask_index, training=False):
    logits = model.apply(params, graph.x, graph.edge_index, cache=graph.cache, rngs={"dropout": dropout_rng},
                         training=training)
    masked_logits = logits[mask_index]
    masked_y = graph.y[mask_index]
    cls_loss = optax.softmax_cross_entropy(masked_logits, jax.nn.one_hot(masked_y, num_classes)).mean()

    kernel_vars = find_params_by_name(params["params"], lambda name: "kernel" in name)

    # kernel_vars = [param for name, param in params["params"].items() if "kernel" in name]

    l2_losses = [0.5 * (kernel_var ** 2).sum() for kernel_var in kernel_vars]
    l2_loss = sum(l2_losses)

    loss = cls_loss + l2_loss * l2_coef

    return loss


@jax.jit
def evaluate(params, mask_index):
    logits = model.apply(params, graph.x, graph.edge_index, cache=graph.cache)
    masked_y_pred = jnp.argmax(logits[mask_index], axis=-1)
    masked_y_true = graph.y[mask_index]
    accuracy = (masked_y_pred == masked_y_true).astype(jnp.float64).mean()
    return accuracy


def fit(params, optimizer: optax.GradientTransformation):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(compute_loss)(params, train_index, training=True)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for epoch in range(400):
        params, opt_state, loss = step(params, opt_state)
        if epoch % 1 == 0:
            accuracy = evaluate(params, test_index)
            print(epoch, " accuracy: ", accuracy)


fit(params, optimizer)
