from __future__ import division
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse
import scipy.sparse.linalg
from PIL import Image




def _rolling_block(A, block=(3, 3)):
    """Applies sliding window to given matrix."""
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


def compute_laplacian(img, mask=None, eps=10**(-7), win_rad=1):
    
    
    win_size = (win_rad * 2 + 1) ** 2
    h, w, d = img.shape
    c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
    win_diam = win_rad * 2 + 1

    indsM = np.arange(h * w).reshape((h, w))
    ravelImg = img.reshape(h * w, d)
    win_inds = _rolling_block(indsM, block=(win_diam, win_diam))
    win_inds = win_inds.reshape(c_h, c_w, win_size) #Stores the pixels in every window
    if mask is not None:
        mask = cv2.dilate(
            mask.astype(np.uint8),
            np.ones((win_diam, win_diam), np.uint8)
        ).astype(np.bool)
        win_mask = np.sum(mask.ravel()[win_inds], axis=2)
        win_inds = win_inds[win_mask > 0, :]
    else:
        win_inds = win_inds.reshape(-1, win_size)

    
    winI = ravelImg[win_inds]
    win_mu = np.mean(winI, axis=1, keepdims=True)
    win_var = np.einsum('...ji,...jk ->...ik', winI, winI) / win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)
    inv = np.linalg.inv(win_var + (eps/win_size)*np.eye(3))

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = np.eye(win_size) - (1.0/win_size)*(1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))
    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h*w, h*w))
    return L


def matting(img, scribble, levels):
    
    prior = np.ones((scribble.shape[0],scribble.shape[1],levels))*(1/levels) # (h,w,levels)
    colour = np.array([ [0,0,1],[0,1,0], [1,0,0], [1,1,0], [0,1,1]]) # preassigned colour for each level
    #fill the already known pixel levels from the scribble
    for k in range(levels):
        x = np.where((scribble == colour[k]).all(axis = 2))
        temp = np.zeros((1,1,levels))
        temp[0,0,k] = 1
        for (i,j) in zip(x[0],x[1]):
            prior[i,j,:] = temp
            
    #Generate known value map
    known_map = prior != 1/levels #makes True for known values (h, w, levels)
    
    #Generate Laplacian
    laplacian = compute_laplacian(img, ~known_map[:,:,0]) #(h*w, h*w)
    
    confidence = scipy.sparse.diags(100*known_map[:,:,0].ravel()) #create a diagonal matrix with known map elements (h*w, h*w)
    
    prior_reshaped = None #(h*w, levels)
    known_map_reshaped = None  #(h*w, levels)
    
    for i in range(levels):
        if i ==0 :
            prior_reshaped = prior[:,:,0].ravel()
            known_map_reshaped = 100*known_map[:,:,0].ravel()
            prior_reshaped = np.expand_dims(prior_reshaped,axis = 1)
            known_map_reshaped = np.expand_dims(known_map_reshaped,axis = 1)
            #print(prior_reshaped.shape)
            #print(prior_confidence_reshaped.shape)
        else:
            prior_reshaped =np.concatenate([prior_reshaped, np.expand_dims(prior[:,:,i].ravel(),axis= 1)],axis =1)
            known_map_reshaped =np.concatenate([known_map_reshaped, np.expand_dims(100*known_map[:,:,i].ravel(),axis= 1)],axis =1)
        
    solution = scipy.sparse.linalg.spsolve( #AX = B
        laplacian + confidence, #A
        prior_reshaped * known_map_reshaped #B
    )
    alpha = np.minimum(np.maximum(solution.reshape(prior.shape), 0), 1)
    return alpha

def threshold(img):
    img[np.where(img < 127 )] = 0
    img[np.where(img < 127 )] = 0
    return img

def bokah(img_path, scribble_path, level):
    capture = cv2.imread(img_path)[...,::-1]
    scribble = cv2.imread(scribble_path)[...,::-1]
    assert capture.shape == scribble.shape, 'scribbles must have exactly same shape as image.'
    output = matting(capture/255, threshold(scribble)/255, level) 
    blurs = [capture.copy()]
    for i in range(1, level):
        k = 5*i
        blurs.append(cv2.blur(capture.copy(),(k, k)))
    output_img = np.zeros(capture.shape)
    for i in range(capture.shape[0]):
        for j in range(capture.shape[1]):
            amax = np.argmax(output[i,j,:])
            pick = blurs[amax]
            output_img[i,j,:] = pick[i,j,:]
    img_PIL = Image.fromarray(np.array(output_img, dtype = np.uint8))
    return img_PIL