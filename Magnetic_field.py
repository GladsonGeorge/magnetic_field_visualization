import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec
import plotly.graph_objects as go
import sympy as smp
import numba as nb
from sympy.vector import cross


def shape_of_current_vector(angle):
    current_tensor_x = (1 + 0.5*np.cos(10*angle))*np.cos(angle)
    current_tensor_y = (1 + 0.5*np.cos(10*angle))*np.sin(angle)
    current_tensor_z = 0.5*np.sin(10*angle)
    return [current_tensor_x, current_tensor_y, current_tensor_z]


parametric_angle = np.linspace(0, 2*np.pi, 1000)
lx, ly, lz = shape_of_current_vector(parametric_angle)

ax = plt.figure().add_subplot(projection='3d')
ax.plot(lx, ly, lz)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.show()
t, x, y, z = smp.symbols('t, x, y, z')
# t : parametric angle
# x : X-coordinate of magnetic field
# y : Y-coordinate of magnetic field
# z : Z-coordinate of magnetic field
x_v = (1+0.5*smp.cos(10*t))*smp.cos(t)
y_v = (1+0.5*smp.cos(10*t))*smp.sin(t)
z_v = 0.5*smp.sin(10*t)
current_tensor = smp.Matrix([x_v, y_v, z_v])  # current tensor matrix
r = smp.Matrix([x, y, z])  # coordinate vectors
seperation_vector = r - current_tensor  # distance vector b/w wire and point

integrand = smp.diff(current_tensor, t).cross(seperation_vector) / seperation_vector.norm()**3

dBxdt = smp.lambdify([t, x, y, z], integrand[0])
dBydt = smp.lambdify([t, x, y, z], integrand[1])
dBzdt = smp.lambdify([t, x, y, z], integrand[2])

mesh = np.linspace(-2, 2, 10)
xv, yv, zv = np.meshgrid(mesh, mesh, mesh)


def magnetic_field(xc, yc, zc):
    return np.array([quad_vec(dBxdt, 0, 2*np.pi, args=(xc, yc, zc))[0],
                     quad_vec(dBydt, 0, 2*np.pi, args=(xc, yc, zc))[0],
                     quad_vec(dBzdt, 0, 2*np.pi, args=(xc, yc, zc))[0]])


B_field = np.vectorize(magnetic_field, signature='(),(),()->(n)')(xv, yv, zv)
Bx = B_field[:, :, :, 0]
By = B_field[:, :, :, 1]
Bz = B_field[:, :, :, 2]

data = go.Cone(x=xv.ravel(), y=yv.ravel(), z=zv.ravel(),
               u=Bx.ravel(), v=By.ravel(), w=Bz.ravel(),
               colorscale='Inferno', colorbar=dict(title='$x^2$'),
               sizemode="scaled", sizeref=20)

layout = go.Layout(title=r'Plot Title', scene=dict(xaxis_title=r'x', yaxis_title=r'y', zaxis_title=r'z',
                                                   aspectratio=dict(x=1, y=1, z=1),
                                                   camera_eye=dict(x=1.2, y=1.2, z=1.2)))

fig = go.Figure(data=data, layout=layout)
fig.add_scatter3d(x=lx, y=ly, z=lz, mode='lines',
                  line=dict(color='green', width=10))
fig.show()
