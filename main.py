import numpy as np
import math
import time
from sklearn.utils.extmath import randomized_svd as rsvd

def multi_left(A, m, n, omega):
  _, r_ = omega.shape
  B = np.zeros((m, r_))
  for i in range(m):
     A_row = np.array([A(i, j) for j in range(n)])
     B[i] = np.dot(A_row, omega[:n, :])
  return B

def multi_right(Q_T, A, m, n):
  r_, _ = Q_T.shape
  B = np.zeros((r_, n))
  for i in range(r_):
    A_col = np.array([[A(j, k) for k in range(n)] for j in range(m)])
    B[i] = np.dot(Q_T[i], A_col)
  return B


def RSVD1(A, r, p):
  _,n = A.shape
  r_=r+p
  omega = np.random.standard_normal((n, r_))
  B = A@omega
  Q, _ = np.linalg.qr(B)
  C_R = Q.T@A
  U_hat, S, Vt = np.linalg.svd(C_R, full_matrices=False)
  U = Q @ U_hat
  return U[:, :r], S[:r], Vt[:r, :]

def RSVD(A, m, n, r, p):
  r_=r+p
  omega = np.random.standard_normal((n, r_))
  B = multi_left(A, m, n, omega)
  Q, _ = np.linalg.qr(B)
  C_R = multi_right(Q.T, A, m, n)
  U_hat, S, Vt = np.linalg.svd(C_R, full_matrices=False)
  U = Q @ U_hat
  return U[:, :r], S[:r], Vt[:r, :]

#m, n = 1500, 1024
A1 = lambda i, j: math.sin(i+j)

m, n = 160, 160
A2 = lambda i, j: math.sqrt(i+j)
start = time.time()
#U, S, Vt = RSVD(A2, m, n, 3, 8)
A = np.fromfunction(np.vectorize(A1), (m, n), dtype=float)
U, S, Vt = RSVD(A1, m, n, 3, 8)#rsvd(A, 3)
finish = time.time()
print(finish - start)
print(S)