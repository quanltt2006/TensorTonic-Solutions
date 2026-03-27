import numpy as np

def adam_step(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)
    
    m = beta1 * m + (1 - beta1) * grad
    
    v = beta2 * v + (1 - beta2) * (grad**2)
    
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return param.tolist(), m.tolist(), v.tolist()

