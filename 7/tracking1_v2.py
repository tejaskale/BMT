import scipy.io as sio
import scipy.ndimage as ndimage
import numpy as np
import matplotlib.pyplot as plt


def golub(a, b):
    # QR decomposition of A
    [q, r] = np.linalg.qr(a)

    # compute new b
    bnew = np.matmul(q.T, b)

    # get size of A
    [m, n] = a.shape
    # compute solution
    x = np.linalg.solve(r[0:n, :], bnew[0:n])

    # compute residual
    residual = np.sqrt(np.sum((np.matmul(a, x) - b)**2))

    return x, residual


# load video and set up coordinate system
# mat_contents = sio.loadmat(tracking-cars.mat')
mat_contents = sio.loadmat('tracking-leukocyte.mat')

# convert movie to double
movie = np.asarray(mat_contents['movie'], np.double)

# get first image (movie is a 3-d array, where the 3rd dimension is the frame number)
Is = movie[:,:,0]
# generate coordinate system
[r, c] = Is.shape
# x = range(0,r)
# y = range(0,c)
[x,y] = np.meshgrid( range(0,r), range(0,c) )
# display image
plt.imshow(Is, cmap='gray')

# # get center point for patch P
tr = plt.ginput(1)
tr_x = [tr[0][0]]
tr_y = [tr[0][1]]
# # for testing you can use these coordinates
# tr_x = [200-1]
# tr_y = [144-1]

# track the patch
# define 'radius' size of patch
t = 10

# set up patch coordinate system
tx = np.arange(tr_x[0]-t,tr_x[0]+t-1)
ty = np.arange(tr_y[0]-t,tr_y[0]+t-1)
print(tx)
# [tx, ty] = np.meshgrid(range(tr_x[0]-t,tr_x[0]+t-1),range(tr_x[0]-t,tr_x[0]+t-1))


# display image again
plt.imshow(Is, cmap='gray')

# draw trajectory
plt.plot(tr_x, tr_y, 'r.')

# tell pyplot that we want an asynchronous figure
plt.ion()

# loop over all images
for i in range(1, movie.shape[2]):
    # get target image
    It = movie[:,:,i]
    
    # compute function and derivatives
    
    # set up equation system
    A = np.zeros([(2 * t ) ** 2, 2])
    b = np.zeros((2 * t ) ** 2)
    k = 0
    for col in tx:
        for row in ty:
            A[k] = np.array([np.gradient(Is)[0][row][col], np.gradient(Is)[1][row][col]])
            b[k] = It[row][col] - Is[row][col]
            k = k + 1

    # solve equation system using Golub's algorithm
    [v, res] = golub(A, b)
    
    # update trajectory
    # print(v)
    # print(ndimage.map_coordinates(It,[tr_x[i-1] - v[0],tr_y[i-1] - v[1]]))

    tr_x.append(tr_x[i-1] - v[0])
    tr_y.append(tr_y[i-1] - v[1])

    # [tx, ty] = np.meshgrid(range(intx - t, intx + t - 1), range(inty - t, inty + t - 1))
    tx = np.arange(tr_x[i] - t, tr_x[i] + t - 1)
    ty = np.arange(tr_y[i] - t, tr_y[i] + t - 1)
    
    # update the figure
    plt.clf()
    plt.imshow(It, cmap='gray')

    plt.plot(tr_x, tr_y, 'r.-')
    plt.title('frame ' + str(i) + ', residual ' + str(res))

    # give pyplot time to draw
    plt.pause(0.05)

    # update Is
    Is = It

while True:
    plt.pause(0.1)
