import firedrake as fd
import matplotlib.pyplot as plt

# Define the mesh
mesh = fd.RectangleMesh(nx=20, ny=20, Lx=2.0, Ly=1.0)

V = fd.VectorFunctionSpace(mesh, "CG", 2)
P = fd.FunctionSpace(mesh, "CG", 1)

nu = 0.01  # Viscosity
rho = 1.0  # Density

u = fd.TrialFunction(V)
v = fd.TestFunction(V)
p = fd.TrialFunction(P)
q = fd.TestFunction(P)

# Define equation
F = (nu * fd.inner(fd.grad(u), fd.grad(v)) - fd.div(v) * p + fd.div(u) * q) * fd.dx

bcu_inflow = fd.DirichletBC(V, fd.Constant((1.0, 0.0)), "on_boundary && x[0] < fd.DOLFIN_EPS")
bcu_walls = fd.DirichletBC(V, fd.Constant((0.0, 0.0)), "on_boundary && (x[1] < fd.DOLFIN_EPS || x[1] > 1.0 - fd.DOLFIN_EPS)")
bcu = [bcu_inflow, bcu_walls]

bcp_outflow = fd.DirichletBC(P, fd.Constant(0.0), "on_boundary && x[0] > 2.0 - fd.DOLFIN_EPS")
bcp = [bcp_outflow]

a = fd.lhs(F)
L = fd.rhs(F)

w = fd.Function(V + P)
fd.solve(a == L, w, bcs=bcu + bcp)

u, p = w.split()

# Plot velocity field
plt.figure(figsize=(8, 4))
fd.plot(u, title="Velocity Field")
plt.colorbar()
plt.show()
