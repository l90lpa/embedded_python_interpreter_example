import mpi4py
# Set initialize to False, to stop mpi4py calling MPI_Init when `MPI` is imported
mpi4py.rc.initialize=False 
from mpi4py import MPI

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from py_swe.geometry import RectangularGrid, Vec2, ParGeometry, create_domain_par_geometry, add_halo_geometry, add_ghost_geometry
from py_swe.state import State, create_local_field_zeros, create_local_field_tsunami_height

# ======== Geometry =========

def create_geometry(comm_int, nx, ny, xmax, ymax):
    comm = MPI.Comm.f2py(comm_int)
    rank = comm.Get_rank()
    size = comm.Get_size()

    grid = RectangularGrid(nx, ny)
    geometry = create_domain_par_geometry(rank, size, grid, Vec2(0.0, 0.0), Vec2(xmax, ymax))
    geometry = add_ghost_geometry(geometry, 1)
    geometry = add_halo_geometry(geometry, 1)
    return geometry

# ======== Initial Condition =========

def create_tsunami_pulse_initial_condition(geometry: ParGeometry):
    zero_field = create_local_field_zeros(geometry, jnp.float64)

    u = jnp.copy(zero_field)
    v = jnp.copy(zero_field)
    h = create_local_field_tsunami_height(geometry, jnp.float64)
    u_ = np.array(u)
    v_ = np.array(v)
    h_ = np.array(h)

    return State(u_, v_, h_)
    # return State(u, v, h)

# ======== Model =========
from math import sqrt, ceil
import time

import matplotlib as mpl
import matplotlib.pyplot as plt

from jax.tree_util import tree_map
import numpy as np

from mpi4jax._src.utils import HashableMPIType

from py_swe.geometry import at_local_domain
from py_swe.state import gather_global_field
from py_swe.model import shallow_water_model_w_padding


def gather_global_state_domain(s, geometry, mpi4py_comm, root):

    s_local_domain = tree_map(lambda x: np.array(x[at_local_domain(geometry)]), s)
    s_global = tree_map(lambda x: gather_global_field(x, geometry.global_pg.nxprocs, geometry.global_pg.nyprocs, root, geometry.local_pg.rank, mpi4py_comm), s_local_domain)
    return s_global


def save_state_figure(state, filename):

    def reorientate(x):
        return np.fliplr(np.rot90(x, k=3))
    
    def downsample(x, n):
        nx = np.size(x, axis=0)
        ns = nx // n
        return x[::ns,::ns]

    # make a color map of fixed colors
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['blue', 'white', 'red'], 256)

    # modify data layout so that it displays as expected (x horizontal and y vertical, with origin in bottom left corner)
    u = reorientate(state.u)
    v = reorientate(state.v)
    h = reorientate(state.h)

    x = y = np.linspace(0, np.size(u, axis=0)-1, np.size(u, axis=0))
    xx, yy = np.meshgrid(x, y)

    # downsample velocity vector field to make it easier to read
    xx = downsample(xx, 20)
    yy = downsample(yy, 20)
    u = downsample(u, 20)
    v = downsample(v, 20)

    fig, ax = plt.subplots()
    # tell imshow about color map so that only set colors are used
    img = ax.imshow(h, interpolation='nearest', cmap=cmap, origin='lower')
    ax.quiver(xx,yy,u,v)
    plt.colorbar(img,cmap=cmap)
    plt.grid(True,color='black')
    plt.savefig(filename)


def save_global_state_domain_on_root(s, geometry: ParGeometry, mpi4py_comm, root, filename, msg):
    s_global = gather_global_state_domain(s, geometry, mpi4py_comm, root)
    if geometry.local_pg.rank == root:
        save_state_figure(s_global, filename)
        print(msg)

def step_model(geometry: ParGeometry, s0: State, comm_int, root):
    assert geometry.global_domain.extent.x == geometry.global_domain.extent.y
    assert geometry.global_domain.grid_extent.x == geometry.global_domain.grid_extent.y

    mpi4jax_comm = MPI.Comm.f2py(comm_int)
    mpi4jax_comm_wrapped = HashableMPIType(mpi4jax_comm)
    mpi4py_comm = mpi4jax_comm.Clone()

    rank = mpi4jax_comm.Get_rank()
    
    xmax = geometry.global_domain.extent.x
    nx = geometry.global_domain.grid_extent.x

    dx = dy = xmax / (nx - 1.0)
    g = 9.81
    dt = 0.68 * dx / sqrt(g * 5030)
    tmax = 150
    num_steps = ceil(tmax / dt)

    token = jnp.empty((1,))
    b = create_local_field_zeros(geometry, jnp.float64)

    save_global_state_domain_on_root(s0, geometry, mpi4py_comm, root, "step-0.png", "Saved initial condition.")


    if rank == root:
        print(f"Starting compilation.")
        start = time.perf_counter()


    model_compiled = shallow_water_model_w_padding.lower(s0, geometry, mpi4jax_comm_wrapped, b, num_steps, dt, dx, dy, token).compile()


    if rank == root:
        end = time.perf_counter()
        print(f"Compilation completed in {end - start} seconds.")
        print(f"Starting simulation with {num_steps} steps...")
        start = time.perf_counter()


    sN, _ = model_compiled(s0, b, dt, dx, dy, token)
    sN.u.block_until_ready()


    if rank == root:
        end = time.perf_counter()
        print(f"Simulation completed in {end - start} seconds, with an average time per step of {(end - start) / num_steps} seconds.")


    save_global_state_domain_on_root(sN, geometry, mpi4py_comm, root, f"step-{num_steps}.png", "Saved final condition.")
    

