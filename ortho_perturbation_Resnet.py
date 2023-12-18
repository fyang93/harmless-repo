import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import *
from sympy.polys.matrices import DomainMatrix
from scipy.linalg import null_space, svd
from scipy.linalg import qr
import numpy as np
import math
import time
import numpy

EPS = 1e-7

def qr_null(A, tol=None):
    Q, R, P = qr(A.T, mode='full', pivoting=True)
    tol = np.finfo(R.dtype).eps if tol is None else tol
    rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
    return Q[:, rnk:].conj()


def my_null_space(A, rcond=None):
    #print("my_null_space")
    u, s, vh = svd(A, full_matrices=True, lapack_driver='gesvd')  # lapack_driver='gesvd'
    #print("final svd")
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = numpy.finfo(s.dtype).eps * max(M, N)
    tol = numpy.amax(s) * rcond
    num = numpy.sum(s > tol, dtype=int)
    Q = vh[num:, :].T.conj()
    return Q



class HarmlessPerturbation(nn.Module):

    def __init__(
        self,
        kernel: nn.Conv2d,
        stride: int=3,
        padding: int=0,
        perturbation_shape: tuple=(3,32,32),
        device: str = 'cuda:0',
        nSpace_path: str = './'
    )->None:
        super().__init__()
        self.perturbation_shape = perturbation_shape
        self.device = device
        self.stride = stride
        self.padding = padding
        self.out_channels, self.in_channels, self.kernel_size, _ = kernel.weight.shape
        self.original_kernel = kernel.weight
        self.kernel = kernel.weight.reshape(self.out_channels, -1).T
        self.nSpace_path = nSpace_path


    def get_nullspace(self, Matrix_tensor, left_nullspace=True):
        if left_nullspace:
            Matrix_tensor = Matrix_tensor.T
        print("Matrix A's shape", Matrix_tensor.shape)

        t1 = time.time()
        M_nullspace = my_null_space(Matrix_tensor.detach().cpu().numpy())
        t2 = time.time()
        print("time of calculating null space:", t2-t1)
        np.save(self.nSpace_path, np.array(M_nullspace))

        # whether to return nullspace of matrix A
        returnTensor = False
        if returnTensor == True:
            M_nullspace_tensor = torch.from_numpy(np.array(M_nullspace)).squeeze(dim=-1) #torch.from_numpy(np.array(M_nullspace).astype(np.float64)).squeeze(dim=-1)
            #print("tensor.shape:", M_nullspace_tensor.shape)
            return M_nullspace_tensor



    """ Reshape the input before the convolution operation according to kernel_size, stride, padding """
    def extend_layer(self, x):
        # extend x from (batch_size, channel, conv_size, conv_size) to (batch_size, channel, H_out * kernel_size, H_out * kernel_size)
        N, C, H_in  = x.shape[0], x.shape[1], x.shape[2]  # image shape
        H_out = (H_in - self.kernel_size + 2 * self.padding) / self.stride + 1    # output size after applying the equivalent convolution operation
        #assert H_out % 1 == 0, "stride/padding is not suitable"
        H_out = int(H_out)
        #print("H_out", H_out)

        # first, padding x with 0
        pad = (self.padding, self.padding, self.padding, self.padding)  #(1, 1, 1, 1)
        x = F.pad(x, pad)

        # second, gather x twice
        indice = torch.tensor([int(i / self.kernel_size) * self.stride + i % self.kernel_size   \
                               for i in range(0, H_out * self.kernel_size)]).to(self.device)
        row_indice = indice.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N, C, H_in + 2*self.padding, 1)
        col_indice = indice.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(N, C, 1, H_out * self.kernel_size)
        x = torch.gather(x, 3, row_indice)
        x = torch.gather(x, 2, col_indice)

        return x


    def get_index(self):
        C, H, W = self.perturbation_shape  # image shape
        all_players = np.arange(1, C * H * W + 1)
        #print(all_players)
        all_players = torch.from_numpy(all_players).reshape(C, H, W).unsqueeze(dim=0).to(self.device)
        index = self.extend_layer(all_players).squeeze(dim=0)
        return index


    def get_sparse_matrix_each_kernel(self, kernel, index):
        # N, C, H  = x.shape[0], x.shape[1], x.shape[2] # image shape
        #print(kernel.shape)
        C, H, W = self.perturbation_shape  # image shape
        H_out = (H - self.kernel_size + 2 * self.padding) / self.stride + 1
        W_out = (W - self.kernel_size + 2 * self.padding) / self.stride + 1
        #assert H_out % 1 == 0 and W_out % 1 == 0, "stride/padding is not suitable"
        H_out, W_out = int(H_out), int(W_out)
        #print("H_out, W_out", H_out, W_out)

        res = np.zeros((H_out * W_out, C * H * H))
        for dim in range(C):
            for i in range(0, index.shape[1], self.kernel_size):
                for j in range(0, index.shape[2], self.kernel_size):
                    patch_index = index[dim][i:i + self.kernel_size, j:j + self.kernel_size]
                    # print(patch_index)
                    for k in range(len(patch_index)):
                        for l in range(len(patch_index[0])):
                            # print(dim, i , j , k , l)
                            if patch_index[k][l] != 0:
                                res[int((i * H_out + j) / self.kernel_size)][patch_index[k][l] - 1] = kernel[dim][k][l] #kernel[0][dim][k][l]

        return torch.tensor(res)


    def get_sparse_matrix(self, kernel, index):
        res = self.get_sparse_matrix_each_kernel(kernel[0], index)
        for i in range(1, len(kernel)):
            tmp = self.get_sparse_matrix_each_kernel(kernel[i], index)
            res = np.concatenate((res, tmp), axis=0)
        return torch.tensor(res)



    def get_delta(self):
        C, H, W = self.perturbation_shape  # image shape
        print(f"stride: {self.stride}, padding: {self.padding}, kernel.shape:{tuple(self.original_kernel.shape)}")

        # get sparse matrix A
        SparseMatrix = self.get_sparse_matrix(self.original_kernel, self.get_index())

        # get null space of matrix A
        #orthogonal_bases = self.get_nullspace(SparseMatrix.double(), left_nullspace=False).to(self.device)  # load Tensor out of memory
        if not os.path.exists(self.nSpace_path):
            self.get_nullspace(SparseMatrix.double(), left_nullspace=False).to(self.device)
        orthogonal_bases = np.load(self.nSpace_path)
        orthogonal_bases = torch.tensor(orthogonal_bases).float().to(self.device)
        print("orthogonal_bases", orthogonal_bases.shape)

        # get arbitrary perturbation
        if min(orthogonal_bases.shape) == 0:  # if perturbation is zero
            isApprox = True
            if isApprox:  # SVD decomposition to find the approximated perturbation
                print("SparseMatrix", SparseMatrix.shape, SparseMatrix)
                SparseMatrix = np.array(SparseMatrix)
                u, sigma, vt = np.linalg.svd(SparseMatrix)
                delta = torch.tensor(vt[-1]).to(self.device)
                print("delta", delta.shape, delta)
                delta = delta.reshape(C, H, W).unsqueeze(dim=0)
                return delta
            else:  # get zero space
                return torch.zeros(C, H, W).unsqueeze(dim=0).to(self.device)
        else:
            with torch.no_grad():
                rank_OC = torch.linalg.matrix_rank(orthogonal_bases.half())
                print("rank_OC", rank_OC)
                vec_coef = torch.randn(rank_OC).to(self.device)
                vec_coef = torch.diag(vec_coef).to(self.device)
                linear_comb = torch.mm(orthogonal_bases, vec_coef).double()
                delta = torch.sum(linear_comb, dim=1).double()
                delta = delta.reshape(C, H, W).unsqueeze(dim=0)

        return delta, orthogonal_bases






class HarmlessPerturbationLinearLayer(nn.Module):

    def __init__(
        self,
        weight: nn.Linear,
        device: str = 'cuda:0',
        nSpace_path: str = './'
    )->None:
        super().__init__()
        self.device = device
        self.weight = weight
        self.nSpace_path = nSpace_path


    def get_nullspace(self, Matrix_tensor, left_nullspace=True):
        if left_nullspace:
            Matrix_tensor = Matrix_tensor.T
        print("Matrix A's shape", Matrix_tensor.shape)

        M_nullspace = my_null_space(Matrix_tensor.detach().cpu().numpy())
        np.save(self.nSpace_path, np.array(M_nullspace))
        M_nullspace_tensor = torch.from_numpy(np.array(M_nullspace)).squeeze(dim=-1)

        return M_nullspace_tensor


    def get_delta(self):
        # get null space of matrix A
        #orthogonal_bases = self.get_nullspace(self.weight.double(), left_nullspace=False).to(self.device)  # load Tensor out of memory
        if not os.path.exists(self.nSpace_path):
            self.get_nullspace(self.weight.double(), left_nullspace=False).to(self.device)
        orthogonal_bases = np.load(self.nSpace_path)
        orthogonal_bases = torch.tensor(orthogonal_bases).float().to(self.device)
        print("orthogonal_bases", orthogonal_bases.shape)

        # get arbitrary perturbation
        if min(orthogonal_bases.shape) == 0:  # if perturbation is zero
            isApprox = True
            if isApprox:  # SVD decomposition to find the approximated perturbation
                print("Weight", self.weight.shape, self.weight)
                u, sigma, vt = np.linalg.svd(np.array(self.weight))
                delta = torch.tensor(vt[-1]).to(self.device)
                print("delta", delta.shape, delta)
                delta = delta.reshape(self.weight.in_features).unsqueeze(dim=0)
                return delta
            else:  # get zero space
                return torch.zeros(self.weight.in_features).unsqueeze(dim=0).to(self.device)
        else:
            with torch.no_grad():
                rank_OC = torch.linalg.matrix_rank(orthogonal_bases.half())
                print("rank_OC", rank_OC)
                vec_coef = torch.randn(rank_OC).to(self.device)
                vec_coef = torch.diag(vec_coef).to(self.device)
                linear_comb = torch.mm(orthogonal_bases, vec_coef).double()
                delta = torch.sum(linear_comb, dim=1).double()
                delta = delta.reshape(self.weight.in_features).unsqueeze(dim=0)

        return delta, orthogonal_bases







