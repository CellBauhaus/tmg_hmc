from hypothesis import given, assume, strategies as st
from hypothesis.extra.numpy import arrays
import pytest
import os
import numpy as np
import scipy.sparse as sp

from tmg_hmc import TMGSampler, _TORCH_AVAILABLE
from tmg_hmc.constraints import LinearConstraint, SimpleQuadraticConstraint, QuadraticConstraint, ProductConstraint
if _TORCH_AVAILABLE:
    import torch
    gpu_available = torch.cuda.is_available()
else:
    torch = None
    gpu_available = False

@st.composite
def positive_definite_matrices(draw, min_size=2, max_size=10):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # Generate a random matrix
    A = draw(arrays(dtype=np.float64, shape=(size, size), elements=st.floats(-1e-6, 1e-6)))
    # Make it symmetric positive definite
    pd_matrix = A.T @ A + np.eye(size) * 1e-6  # Add small value to diagonal for numerical stability
    return pd_matrix

@st.composite
def square_matrix(draw, min_size=2, max_size=10):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    mat = draw(arrays(dtype=np.float64, shape=(size, size), elements=st.floats(-1e6, 1e6)))
    return mat

@st.composite
def symmetric_matrix(draw, min_size=2, max_size=10):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    mat = draw(arrays(dtype=np.float64, shape=(size, size), elements=st.floats(-1e6, 1e6)))
    sym_mat = (mat + mat.T) / 2
    return sym_mat

@st.composite
def vector(draw, min_size=2, max_size=10):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    vec = draw(arrays(dtype=np.float64, shape=(size, 1), elements=st.floats(-1e6, 1e6)))
    return vec

@st.composite
def sym_mat_vec(draw, min_size=2, max_size=10):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    mat = draw(arrays(dtype=np.float64, shape=(size, size), elements=st.floats(-1e6, 1e6)))
    sym_mat = (mat + mat.T) / 2
    vec = draw(arrays(dtype=np.float64, shape=(size, 1), elements=st.floats(-1e6, 1e6)))
    return sym_mat, vec

@given(positive_definite_matrices())
def test_sampler_initialization_errors(pd_matrix):
    with pytest.raises(ValueError):
        TMGSampler() 
    TMGSampler(Sigma=pd_matrix)  # Should not raise
    TMGSampler(Sigma_half=pd_matrix)  # Should not raise

    with pytest.raises(ValueError):
        TMGSampler(Sigma=pd_matrix[:-1, :])  # Invalid shape
    with pytest.raises(ValueError):
        TMGSampler(Sigma_half=pd_matrix[:-1, :])  # Invalid shape
    
    asym_matrix = pd_matrix.copy()
    asym_matrix[0,1] += 1e-3  # Make it asymmetric
    with pytest.raises(ValueError):
        TMGSampler(Sigma=asym_matrix)  # Not symmetric
    with pytest.raises(ValueError):
        TMGSampler(Sigma_half=asym_matrix)  # Not symmetric

    evals = np.linalg.eigvalsh(pd_matrix)
    npsd_matrix = pd_matrix - 2 * np.min(evals) * np.eye(pd_matrix.shape[0]) - 1e-3 * np.eye(pd_matrix.shape[0])  # Make it not positive semi-definite
    with pytest.raises(ValueError):
        TMGSampler(Sigma=npsd_matrix)  # Not positive semi-definite
    with pytest.raises(ValueError):
        TMGSampler(Sigma_half=npsd_matrix)  # Not positive semi-definite

@given(positive_definite_matrices())
def test_sigma_decomposition(pd_matrix):
    sampler = TMGSampler(Sigma=pd_matrix)
    Sigma_half = sampler.Sigma_half
    reconstructed_Sigma = Sigma_half @ Sigma_half.T
    assert np.allclose(reconstructed_Sigma, pd_matrix)

@given(st.integers(min_value=2, max_value=10))
def test_invalid_constraint_errors(dim):
    sampler = TMGSampler(Sigma=np.eye(dim))
    with pytest.raises(ValueError):
        constraint = sampler._build_constraint() # Neither A nor f provided
    with pytest.raises(ValueError):
        A = np.zeros((dim, dim))
        f = np.zeros((dim, 1))
        constraint = sampler._build_constraint(A=A, f=f)  # Both A and f zero

@given(vector())
def test_build_linear_constraint(f):
    assume(float(np.linalg.norm(f)) > 1e-8)  # Avoid zero vector
    sampler = TMGSampler(Sigma=np.eye(f.shape[0]))
    constraint = sampler._build_constraint(f=f)
    assert isinstance(constraint, LinearConstraint)
    assert np.allclose(constraint.f, f)

@given(symmetric_matrix())
def test_build_simple_quadratic_constraint(A):
    assume(not np.allclose(A, 0))  # Avoid zero matrix
    sampler = TMGSampler(Sigma=np.eye(A.shape[0]))
    constraint = sampler._build_constraint(A=A, sparse=False)
    assert isinstance(constraint, SimpleQuadraticConstraint)
    assert np.allclose(constraint.A, A)

@given(sym_mat_vec())
def test_build_quadratic_constraint(data):
    A, f = data
    assume(not np.allclose(A, 0) and not np.allclose(f, 0))  # Avoid zero matrix and zero vector
    sampler = TMGSampler(Sigma=np.eye(A.shape[0]))
    constraint = sampler._build_constraint(A=A, f=f, sparse=False)
    assert isinstance(constraint, QuadraticConstraint)
    assert np.allclose(constraint.A, A)
    assert np.allclose(constraint.b, f)

def test_add_constraint():
    f = np.ones(2).reshape(-1,1)
    sampler = TMGSampler(Sigma=np.eye(2))
    for i in range(10):
        sampler.add_constraint(f=(i+1)*f)
        assert len(sampler.constraints) == i+1

def test_add_product_constraint_errors():
    sampler = TMGSampler(Sigma=np.eye(2))
    with pytest.raises(ValueError):
        sampler.add_product_constraint(parameters=[np.eye(2), np.ones((2,1))])  # Invalid length
    with pytest.raises(ValueError):
        sampler.add_product_constraint(parameters=[])  # No constraint passed
    with pytest.warns(UserWarning):
        sampler.add_product_constraint(parameters=[{'A': np.eye(2)}])  # Single constraint passed

def test_add_product_constraint_lists():
    sampler = TMGSampler(Sigma=np.eye(2))
    parameters = [[np.eye(2), np.ones((2,1)), 1.0],
                  [2*np.eye(2), 2*np.ones((2,1)), 2.0]]
    sampler.add_product_constraint(parameters=parameters)
    assert len(sampler.constraints) == 1
    constraint = sampler.constraints[0]
    assert isinstance(constraint, ProductConstraint)

def test_add_product_constraint_dicts():
    sampler = TMGSampler(Sigma=np.eye(2))
    parameters = [{'A': np.eye(2), 'f': np.ones((2,1)), 'c': 1.0},
                  {'f': 2*np.ones((2,1)), 'c': 2.0},
                  {'A': 3*np.eye(2)}]
    sampler.add_product_constraint(parameters=parameters)
    assert len(sampler.constraints) == 1
    constraint = sampler.constraints[0]
    assert isinstance(constraint, ProductConstraint)

@given(st.integers(min_value=2, max_value=10))
def test_unconstrained_sampling(dim):
    sampler = TMGSampler(Sigma=np.eye(dim))
    np.random.seed(0)
    sample = sampler.sample(x0=np.zeros((dim,1)), n_samples=5, burn_in=0)
    np.random.seed(0)
    expected_samples = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=5)
    assert np.allclose(sample, expected_samples)

@pytest.mark.gpu
@pytest.mark.skipif(not gpu_available, reason="GPU not available")
@given(st.integers(min_value=2, max_value=10))
def test_unconstrained_sampling_gpu(dim):
    sampler = TMGSampler(Sigma=np.eye(dim), gpu=True)
    torch.manual_seed(0)
    sample = sampler.sample(x0=np.zeros((dim,1)), n_samples=5, burn_in=0)
    torch.manual_seed(0)
    expected_samples = []
    for _ in range(5):
        expected_samples.append(torch.randn(dim, 1, dtype=torch.float64).cpu().numpy())
    expected_samples = np.concatenate(expected_samples, axis=1).T

    assert np.allclose(sample, expected_samples)

def test_save_load():
    sampler = TMGSampler(Sigma=np.eye(3))
    sampler.add_constraint(f=np.array([[1.0], [0.0], [0.0]]), sparse=False)
    sampler.add_constraint(A=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]), sparse=False)
    sampler.add_constraint(A=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 
                           f=np.array([[1.0], [1.0], [1.0]]), sparse=False)

    sampler.save("test_sampler.pkl")
    loaded_sampler = TMGSampler.load("test_sampler.pkl")

    assert np.allclose(loaded_sampler.Sigma_half, sampler.Sigma_half)
    assert len(loaded_sampler.constraints) == len(sampler.constraints)
    for c1, c2 in zip(loaded_sampler.constraints, sampler.constraints):
        assert type(c1) == type(c2)
        if isinstance(c1, LinearConstraint):
            assert np.allclose(c1.f, c2.f)
        elif isinstance(c1, SimpleQuadraticConstraint):
            assert np.allclose(c1.A, c2.A)
        elif isinstance(c1, QuadraticConstraint):
            assert np.allclose(c1.A, c2.A)
            assert np.allclose(c1.b, c2.b)
    os.remove("test_sampler.pkl")

def test_add_linear_constraints_batch():
    dim = 5
    mu = np.random.randn(dim)
    Sigma = np.eye(dim) + 0.1 * np.random.randn(dim, dim)
    Sigma = Sigma @ Sigma.T

    fs = np.random.randn(10, dim)
    cs = np.random.randn(10)

    sampler_batch = TMGSampler(mu=mu, Sigma=Sigma)
    sampler_batch.add_linear_constraints(fs=fs, cs=cs)

    sampler_individual = TMGSampler(mu=mu, Sigma=Sigma)
    for i in range(10):
        sampler_individual.add_constraint(f=fs[i].reshape(-1, 1), c=cs[i])

    assert len(sampler_batch.constraints) == len(sampler_individual.constraints)
    assert len(sampler_batch.constraints) == 10

    for c1, c2 in zip(sampler_batch.constraints, sampler_individual.constraints):
        assert isinstance(c1, LinearConstraint)
        assert isinstance(c2, LinearConstraint)
        assert np.allclose(c1.f, c2.f)
        assert np.isclose(c1.c, c2.c)

    sampler_batch._rebuild_linear_index()
    sampler_individual._rebuild_linear_index()
    assert sampler_batch._linear_F is not None
    assert np.allclose(sampler_batch._linear_F, sampler_individual._linear_F)
    assert np.allclose(sampler_batch._linear_c, sampler_individual._linear_c)


def test_add_linear_constraints_batch_mixed_with_individual():
    dim = 3
    sampler = TMGSampler(Sigma=np.eye(dim))
    sampler.add_constraint(f=np.array([[1.0], [0.0], [0.0]]))

    fs = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cs = np.array([0.5, 0.5])
    sampler.add_linear_constraints(fs=fs, cs=cs)

    assert len(sampler.constraints) == 3
    sampler._rebuild_linear_index()
    assert sampler._linear_F.shape == (3, dim)


def test_add_linear_constraints_batch_sampling():
    sampler = TMGSampler(Sigma=np.eye(2))
    fs = np.array([[1.1, -1.0], [-1.0, 1.0]])
    cs = np.array([0.0, 0.0])
    sampler.add_linear_constraints(fs=fs, cs=cs)

    x0 = np.array([[1.0], [1.05]])
    samples = sampler.sample(x0=x0, n_samples=100, burn_in=10)
    satisfied = samples[:, 1] <= 1.1 * samples[:, 0]
    satisfied &= samples[:, 1] >= samples[:, 0]
    assert np.all(satisfied)


def test_hit_times_with_precomputed_q_values():
    dim = 5
    sampler = TMGSampler(Sigma=np.eye(dim))
    fs = np.random.randn(20, dim)
    cs = np.zeros(20)
    sampler.add_linear_constraints(fs=fs, cs=cs)
    sampler._rebuild_linear_index()

    x = np.random.randn(dim, 1)
    xdot = np.random.randn(dim, 1)

    hs_baseline, cs_baseline = sampler._hit_times(x, xdot)

    F = sampler._linear_F
    q1 = F @ xdot.flatten()
    q2 = F @ x.flatten()
    hs_cached, cs_cached = sampler._hit_times(x, xdot, _q1=q1, _q2=q2)

    assert np.allclose(hs_baseline, hs_cached)
    assert len(cs_baseline) == len(cs_cached)
    for c1, c2 in zip(cs_baseline, cs_cached):
        if c1 is None:
            assert c2 is None
        else:
            assert c1 is c2


def test_q_value_propagation_identity():
    dim = 5
    sampler = TMGSampler(Sigma=np.eye(dim))
    fs = np.random.randn(20, dim)
    cs = np.zeros(20)
    sampler.add_linear_constraints(fs=fs, cs=cs)
    sampler._rebuild_linear_index()

    F = sampler._linear_F
    x = np.random.randn(dim, 1)
    xdot = np.random.randn(dim, 1)

    q1 = F @ xdot.flatten()
    q2 = F @ x.flatten()

    h = 0.7
    x_new, xdot_new = sampler._propagate(x, xdot, h)

    q2_direct = F @ x_new.flatten()
    q1_direct = F @ xdot_new.flatten()

    q2_cached = np.cos(h) * q2 + np.sin(h) * q1
    q1_cached = np.cos(h) * q1 - np.sin(h) * q2

    assert np.allclose(q2_direct, q2_cached)
    assert np.allclose(q1_direct, q1_cached)


def test_q_value_caching_sampling_correctness():
    sampler = TMGSampler(Sigma=np.eye(3))
    fs = np.array([[1.0, -1.0, 0.0],
                   [-1.0, 1.0, 0.0],
                   [0.0, 1.0, -1.0],
                   [0.0, -1.0, 1.0]])
    cs = np.array([0.5, 0.5, 0.5, 0.5])
    sampler.add_linear_constraints(fs=fs, cs=cs)

    x0 = np.array([[0.0], [0.0], [0.0]])
    samples = sampler.sample(x0=x0, n_samples=200, burn_in=20)

    for i in range(len(fs)):
        vals = samples @ fs[i] + cs[i]
        assert np.all(vals >= -1e-10), f"Constraint {i} violated"


@pytest.mark.gpu
@pytest.mark.skipif(not gpu_available, reason="GPU not available")
def test_add_linear_constraints_batch_gpu_types():
    dim = 3
    sampler = TMGSampler(Sigma=np.eye(dim), gpu=True)
    fs = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    cs = np.array([0.0, 0.0])
    sampler.add_linear_constraints(fs=fs, cs=cs)

    for c in sampler._linear_constraints:
        assert isinstance(c.f, torch.Tensor), "Batch-added constraint f should be a CUDA tensor in GPU mode"
        assert c.f.is_cuda, "Batch-added constraint f should be on CUDA"


@pytest.mark.gpu
@pytest.mark.skipif(not gpu_available, reason="GPU not available")
def test_add_linear_constraints_batch_gpu_sampling():
    sampler = TMGSampler(Sigma=np.eye(2), gpu=True)
    fs = np.array([[1.1, -1.0], [-1.0, 1.0]])
    cs = np.array([0.0, 0.0])
    sampler.add_linear_constraints(fs=fs, cs=cs)

    x0 = np.array([[1.0], [1.05]])
    samples = sampler.sample(x0=x0, n_samples=50, burn_in=10)
    satisfied = samples[:, 1] <= 1.1 * samples[:, 0]
    satisfied &= samples[:, 1] >= samples[:, 0]
    assert np.all(satisfied)


def test_tight_constraints_end_to_end():
    sampler = TMGSampler(Sigma=np.eye(2))
    # x2 <= 1.1 x1 => x2 - 1.1 x1 >= 0
    f = np.array([[1.1], [-1.0]])
    sampler.add_constraint(f=f)
    # x2 >= x1 => -x2 + x1 <= 0
    f = np.array([[-1.0], [1.0]])
    sampler.add_constraint(f=f)

    x0 = np.array([[1.0], [1.05]])
    samples = sampler.sample(x0=x0, n_samples=1000, burn_in=100)
    satisfied = samples[:,1] <= 1.1 * samples[:,0]
    satisfied &= samples[:,1] >= samples[:,0]
    assert np.all(satisfied)