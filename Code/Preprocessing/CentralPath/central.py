import numpy as np

def find_central_point(A, b, ss = 10**5, ts = 10**-5, te = 10**5):
    A = A.astype(float)
    b = b.astype(float)
    n = A.shape[1]
    d = A.shape[0]
    y = np.vstack((np.zeros(n)[..., None], [ss]))
    c = np.hstack((np.zeros(n), [1]))[..., None]

    for i in range(d):
        norm = np.linalg.norm(A[i])
        A[i] = A[i]/norm
        b[i] = b[i]/norm
    
    A_tilde = np.hstack((A, np.ones(d)[..., None] * -1))
    t = ts
    
    prev_grad = np.zeros(n + 1)
    
    while(t < te):
        
        slack = 1/(b - np.matmul(A_tilde,y))
        grad = c + np.matmul(A_tilde.T, slack)/t

        slack_mat = np.diag(np.squeeze(np.asarray(slack.T)))
        hessian = np.matmul(slack_mat, slack_mat)
        hessian = np.matmul(np.matmul(A_tilde.T, hessian), A_tilde)/t
        inv_hessian = np.linalg.inv(hessian)
        
        new_y = y - 0.5 * np.matmul(inv_hessian, grad)
        
        if np.linalg.norm(prev_grad - grad)/(np.linalg.norm(prev_grad) + 10**(-8)) < 0.01:
            t *= 2
        
        prev_grad = grad
        y = new_y
        
    return y[:n]