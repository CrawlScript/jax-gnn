# coding=utf-8
from typing import Optional, Callable

import tensorflow as tf
# from tf_geometric.sparse.sparse_adj import SparseAdj
import jax.numpy as jnp
import jax_sparse as jsp
from jax_sparse import SparseMatrix
import flax
import numpy as np

# default_kernel_init = flax.linen.initializers.glorot_normal()
default_kernel_init = flax.linen.initializers.glorot_uniform()
default_bias_init = flax.linen.initializers.zeros

# new API
CACHE_KEY_GCN_NORMED_ADJ_TEMPLATE = "gcn_normed_adj_{}_{}"


def compute_cache_key(renorm, improved):
    """
    Compute the cached key based on GCN normalization configurations: renorm and improved

    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :return: The corresponding cached key for the given GCN normalization configuration.
    """
    return CACHE_KEY_GCN_NORMED_ADJ_TEMPLATE.format(renorm, improved)


def gcn_norm_adj(sparse_adj, renorm=True, improved=False, cache: dict = None):
    """
    Compute normed edge (updated edge_index and normalized edge_weight) for GCN normalization.

    :param sparse_adj: SparseMatrix, sparse adjacency matrix.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param cache: A dict for caching the updated edge_index and normalized edge_weight.
    :return: Normed edge (updated edge_index and normalized edge_weight).
    """

    if cache is not None:
        cache_key = compute_cache_key(renorm, improved)
        cached_data = cache.get(cache_key, None)
        if cached_data is not None:
            # return cached_data
            return SparseMatrix(cached_data[0], cached_data[1], cached_data[2])

    fill_weight = 2.0 if improved else 1.0

    if renorm:
        sparse_adj = sparse_adj.add_diag(fill_weight)
        # sparse_adj = sparse_adj.concat_diag(fill_weight)
        # sparse_adj = sparse_adj.add_self_loop(fill_weight=fill_weight)


    deg = sparse_adj.segment_sum(axis=-1)

    deg_inv_sqrt = jnp.power(deg, -0.5)
    deg_inv_sqrt = jnp.where(
        jnp.logical_or(jnp.isinf(deg_inv_sqrt), jnp.isnan(deg_inv_sqrt)),
        jnp.zeros_like(deg_inv_sqrt),
        deg_inv_sqrt
    )
    deg_inv_sqrt = jsp.diags(deg_inv_sqrt)

    # (D^(-1/2)A)D^(-1/2)
    normed_sparse_adj = deg_inv_sqrt @ sparse_adj @ deg_inv_sqrt
    # normed_sparse_adj = tfs.sparse_diag_matmul(tfs.diag_sparse_matmul(deg_inv_sqrt, sparse_adj), deg_inv_sqrt)

    if not renorm:
        normed_sparse_adj = normed_sparse_adj.add_diag(fill_weight)
        # normed_sparse_adj = normed_sparse_adj.concat_diag(fill_weight)
        # normed_sparse_adj = normed_sparse_adj.add_self_loop(fill_weight=fill_weight)

    if cache is not None:
        # cache[cache_key] = normed_sparse_adj
        cache[cache_key] = np.array(normed_sparse_adj.index), np.array(normed_sparse_adj.data), np.array(normed_sparse_adj.shape)
        # cache[cache_key] = normed_sparse_adj.index, normed_sparse_adj.data, normed_sparse_adj.shape

    return normed_sparse_adj


def gcn_build_cache_by_adj(sparse_adj: SparseMatrix, renorm=True, improved=False, override=False, cache=None):
    """
    Manually compute the normed edge based on the given GCN normalization configuration (renorm and improved) and put it in graph.cache.
    If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

    :param sparse_adj: sparse_adj.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param override: Whether to override existing cached normed edge.
    :return: cache
    """

    if cache is None:
        cache = {}
    elif override:
        cache_key = compute_cache_key(renorm, improved)
        cache[cache_key] = None

    gcn_norm_adj(sparse_adj, renorm, improved, cache)
    return cache


def gcn_build_cache_for_graph(graph, renorm=True, improved=False, override=False):
    """
    Manually compute the normed edge based on the given GCN normalization configuration (renorm and improved) and put it in graph.cache.
    If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

    :param graph: tfg.Graph, the input graph.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param override: Whether to override existing cached normed edge.
    :return: None
    """
    graph.cache = gcn_build_cache_by_adj(graph.adj(), renorm=renorm, improved=improved, override=override, cache=graph.cache)
    return graph.cache

    # if override:
    #     cache_key = compute_cache_key(renorm, improved)
    #     graph.cache[cache_key] = None
    #
    # sparse_adj = SparseMatrix(graph.edge_index, graph.edge_weight, [graph.num_nodes, graph.num_nodes])
    # gcn_norm_adj(sparse_adj, renorm, improved, graph.cache)



class GCN(flax.linen.Module):

    units: int
    activation: Optional[Callable] = None
    use_bias: bool = True
    renorm: bool = True
    improved: bool = False

    def build_cache_by_adj(self, sparse_adj, override=False, cache=None):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        """
        return gcn_build_cache_by_adj(sparse_adj, self.renorm, self.improved, override=override, cache=cache)

    def build_cache_for_graph(self, graph, override=False):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        """
        gcn_build_cache_for_graph(graph, self.renorm, self.improved, override=override)

    @flax.linen.compact
    def __call__(self, x, edge_index, edge_weight=None, training=False, cache=None):

        num_nodes = x.shape[0]

        kernel = self.param("kernel", default_kernel_init, [x.shape[-1], self.units])
        h = x @ kernel

        sparse_adj = SparseMatrix(edge_index, edge_weight, [num_nodes, num_nodes])
        normed_sparse_adj = gcn_norm_adj(sparse_adj, self.renorm, self.improved, cache)
            # .dropout(edge_drop_rate, training=training)

        h = normed_sparse_adj @ h

        if self.use_bias:
            bias = self.param("bias", default_bias_init, [self.units])
            h += bias

        if self.activation is not None:
            h = self.activation(h)

        return h
