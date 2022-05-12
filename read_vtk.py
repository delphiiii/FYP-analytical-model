import pandas as pd

def read_vtk(path):
    """Read and processes VTK files from OpenFOAM particleTracks function

    Parameters:
    ----------
        path : str
            Path to VTK file

    Returns:
    ----------
        x, y : np.array
            Array of x and y values of droplets
    """

    vtk_data = pd.read_csv(path, delimiter=' ').iloc[4:,:3].reset_index(drop=True)
    vtk_data.columns = ['x','y','z']
    vtk_data = vtk_data[~(vtk_data['z'].isnull())].iloc[:-2].astype('float',copy=False)

    return vtk_data['z'].to_numpy(), vtk_data['y'].to_numpy()


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    x, y = read_vtk('vtk_files/7157diam.vtk')
    plt.plot(x,y)
    plt.show()
