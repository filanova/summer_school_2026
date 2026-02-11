import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import torch
import torch.nn as nn
import sys
from scipy.ndimage import gaussian_filter1d


# =======================================================================================
# VISUALIZATION
# =======================================================================================

def plot_eigs(ev, ev2=None, xlim=None):
    
    plt.figure(figsize=(4, 3))#, facecolor='lightskyblue')
    
    plt.plot(ev.real, ev.imag, '*', label='Eigenvalues 1')
    
    if ev2 is not None:
        plt.plot(ev2.real, ev2.imag, 'o', markerfacecolor='none', label='Eigenvalues 2')
        plt.legend() # Only show legend if we have two sets
    
    plt.axvline(x=0, color='red', linestyle='--')
    
    if xlim is not None: 
        plt.xlim(xlim)
        
    plt.grid(True)
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.title('System Eigenvalues')
    plt.show()

def visualize1d( t, dt, snapshots, title=None, red_snapshots = None):

    n = snapshots.shape[0]
    space_grid = np.array([i*1/(n+1) for i in range(1, n+1)])
    nsn = len(t)
    snap = [int(x/dt+1) for x in t]
    leg = []
    plt.figure(figsize=(10.2, 3))#, facecolor='lightskyblue')
    for i in range(nsn):
        plt.plot(space_grid, snapshots[:,snap[i]])
        leg.append(f"t = {t[i]}")

    if red_snapshots is not None:
        step = 10
        
        for i in range(nsn):
            plt.plot(space_grid[::step], red_snapshots[::step,snap[i]],'.')
    plt.legend(leg, ncols = 3)
    if title:
        plt.title(title)
    plt.grid()
    plt.ylabel(r"$x$")
    plt.xlabel(r"$\xi$")
    plt.show()


def visualize2d(time_grid, snapshots, title=None, contour=True,
                logscale=False, plotparams={}):
    
    n = snapshots.shape[0]
    space_grid = np.array([i*1/(n+1) for i in range(1, n+1)])

    X, T = np.meshgrid(space_grid, time_grid)

    # contour plot
    if contour:
        fig = plt.figure(figsize=(4, 3))#,facecolor='white')
        ax = fig.gca()
        if logscale:
            plt.contourf(X, T, snapshots.T, 50,
                         locator=ticker.LogLocator(), **plotparams)
        else:
            plt.contourf(X, T, snapshots.T, 50,  cmap =  'inferno',**plotparams)
        plt.rcParams['axes.grid'] = False
        plt.colorbar()
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$t$')
        if title:
            plt.title(title)
        plt.show()

def visualize3d(time_grid, snapshots, title=None):
    n = snapshots.shape[0]
    space_grid = np.array([i*1/(n+1) for i in range(1, n+1)])

    X, T = np.meshgrid(space_grid, time_grid)
    Z = snapshots.T
    # Plot the surface
    fig, ax = plt.subplots(figsize=(10, 6),subplot_kw={"projection": "3d"})
    ax.plot_surface(X, T, Z, vmin=Z.min() * 2, cmap='inferno')

    ax.set(xlabel=r"$\xi$", ylabel="$t$", zlabel=r"$x(\xi,t)$")

    if title:
        ax.set_title(title)

    
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    
    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel(r"$x(\xi,t)$", rotation=90, labelpad = 7)

    

    # plt.tight_layout()
    ax.dist = 11.8
    # ax.set_box_aspect((4, 4, 0.5))
    # plt.subplots_adjust(right=0.25)

    # Adjust the right or left margin to make space
    # plt.subplots_adjust(left=0.0, right=10.0, bottom=0.0, top=1.0)

    plt.show()


def plot_svd(r, S,E, maxr = 200):
    # also remove it to functions!
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))#,facecolor='lightskyblue')

    # plot the normalized singular values
    if maxr > S.shape[0]:
        maxr = S.shape[0]

    ax[0].semilogy(S[:maxr]/S[0],'o')
    ax[0].grid(color='lightgrey')
    ax[0].set_title('normalized singular values')
    ax[0].set_ylim([1e-16,5])
    ax[0].axvline(x=r,color="r", linestyle="--")
    print(f'Cum. energy E(r) = {E[r]*100} %')

    # plot the cumulative energy
    
    ax[1].plot(E[:maxr],'o')
    ax[1].grid() 
    ax[1].set_title('Cumulative energy')
    ax[1].axvline(x=r,color="r", linestyle="--")

    plt.show()


# =======================================================================================
# AUXILLIARY FUNCTIONS
# =======================================================================================
def rk4th_onestep(model, x, timestep=1e-2):
    """
    It defines 4th-order Runge-Kutta step to predict x at next time-step.

    Args:
        model: it defines vector field
        x: x at time t
        timestep (optional): time-step. Defaults to 1e-2.

    Returns:
        x at time t + timestep
    """
    k1 = model(x)
    k2 = model(x + 0.5 * timestep * k1)
    k3 = model(x + 0.5 * timestep * k2)
    k4 = model(x + 1.0 * timestep * k3)
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timestep

# =======================================================================================
# OPERATOR INFERENCE: MODEL IDENTIFICATION
# =======================================================================================

def training(X_reduced, model, Params, lr_scheduler="cyclic", roll_out = False, smooth_derivative = False):

    """
    This is a training function.
    Inputs:
       models: defines vector field.
       dataloaders: contains training data
       optimizer: optimizer, updates the model at every epoch (iteration)
       Params: contains additional parametes such as epochs
       schedulers: learning rate scheduler


    Returns:
        trained model, loss_track, learning_rate_track
    """

    # Assuming temp_Xr shape is (r, total_width)
    m, k = X_reduced.shape
    chunk_width = k // Params.num_inits

    # Reshape to (r, num_inits, chunk_width) -> Transpose to (num_inits, r, chunk_width)
    X_reduced = X_reduced.reshape(m, Params.num_inits, chunk_width).transpose(1, 0, 2)

    train_dset = list(zip(torch.tensor(X_reduced).permute((0, 2, 1)).double()))
    train_dl = torch.utils.data.DataLoader(train_dset, batch_size=Params.bs, shuffle=True)
    dataloaders = {"train": train_dl}   

    ## OPTIMIZATION DEFINITION

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.parameters(),
                "weight_decay": Params.weight_decay,
            },
        ]
    )    

    if lr_scheduler == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(
                        optimizer,
                        step_size_up=Params.step_size_up,
                        mode=Params.mode,
                        cycle_momentum=Params.cycle_momentum,
                        base_lr=Params.base_lr,
                        max_lr=Params.max_lr,
                    )
    else:
        scheduler = None


    criteria    = nn.MSELoss()   # Loss criteria: Mean squared error
    loss_track  = []             # Vector to save the loss values for every epoch 
    lr_track    = []             # learning rate track

    for epoch in range(Params.num_epochs):

        for data in dataloaders["train"]:

            true_state = data[0]  # Ground truth num_inits x num_time_points x num_DOFs
            
            n = true_state.shape[0] # dimension

            # ===============================================================================================
            # Zero the gradients
            # ===============================================================================================
            optimizer.zero_grad()

            # ===============================================================================================
            # Loss calculation
            # ===============================================================================================
            total_loss = 0.0
            #
            if roll_out:
                # Calculation of the predicted step, performing a time integration step with Runge-Kutta method
                predicted_state = rk4th_onestep(         
                                                model, true_state[:, :-1, :], timestep=Params.time_step
                                                )
                # Loss calculation
                total_loss += (1 / Params.time_step / n) * criteria(predicted_state, true_state[:, 1:, :]) 
            else:
                # Calculation of the true derivative from the state data with finite differences
                true_derivative      = torch.gradient(true_state, spacing=Params.time_step, dim=1)[0]
                if smooth_derivative:
                    # Assuming 'derivative' is your (m, k) tensor
                    # 1. Detach from graph and move to CPU numpy
                    derivative_np = true_derivative.detach().cpu().numpy()
                    

                    # 2. Apply smoothing
                    # sigma = Standard Deviation (higher = smoother)
                    # axis = 1 (Smooth along the time axis)
                    smoothed_np = gaussian_filter1d(derivative_np, sigma=2.0, axis=1)

                    # 3. Convert back to PyTorch
                    true_derivative = torch.from_numpy(smoothed_np).double()
                    # If you are using GPU, add .to(device)
                
                # Calculation of the predicted derivatives
                predicted_derivative = model(true_state)
                
                # Loss calculation
                total_loss += (1 / Params.time_step / n) * criteria(predicted_derivative, true_derivative) 


            
            loss_track.append(total_loss.item())

            # ===============================================================================================
            # Backward pass
            # ===============================================================================================

            total_loss.backward()

            # ===============================================================================================
            # Update parameters
            # ===============================================================================================
            optimizer.step()

            if scheduler:
                scheduler.step()

            lr_track.append(optimizer.param_groups[0]["lr"])

        if epoch and (epoch + 1) % 100 == 0:

            lr = optimizer.param_groups[0]["lr"]
            sys.stdout.write(
                f"\r [{epoch + 1} /{Params.num_epochs}] [Training loss: {loss_track[epoch]:.2e}] [Learning rate: {lr:.2e}]"
            )

    return model, loss_track, lr_track
