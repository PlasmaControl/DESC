import numpy as np

examples = {'Dshaped_tokamak':
                {'Psi_total': 1.0,
                 # profiles in polynomial basis, coeffs in ascending order
                 'pressfun_params': 1.65e3*np.array([1,0,-2,0,1]),
                 'iotafun_params': np.array([1,0,-.67]),
                 # bdry shape        m, n, R       Z 
                 'bdry': np.array([[-1, 0,  0.0,   1.47 ],
                                   [-2, 0,  0.0,   0.16 ],
                                   [ 0, 0,  3.51,  0.0  ],
                                   [ 1, 0, -1.00,  0.0  ],
                                   [ 2, 0,  0.106, 0.0  ]]),
                 'bdry_mode':'spectral',
                 'NFP': 1,
                },
            'heliotron':
                {'Psi_total': 1.0,
                 # profiles in polynomial basis, coeffs in ascending order
                 'pressfun_params': 3.4e3*np.array([1,0,-2,0,1]),
                 'iotafun_params': np.array([.5,0,1.5]),
                 # bdry shape        m, n,  R      Z
                 'bdry': np.array([[-1, 0,  0.0,  1.0  ],
                                   [-1, 1,  0.0, -0.3  ],
                                   [ 0, 0, 10.0,  0.0  ],
                                   [ 1, 0, -1.0,  0.0  ],
                                   [ 1, 1, -0.3,  0.0  ]]),
                 'bdry_mode':'spectral',
                 'NFP': 19,
                },
            'circular_tokamak':
                {'Psi_total': 1.0,
                 # profiles in polynomial basis, coeffs in ascending order
                 'pressfun_params': 1e3*np.array([1,0,-1]),
                 'iotafun_params': np.array([1.618]),
                 # bdry shape        m, n, R   Z
                 'bdry': np.array([[-1, 0, 0.0, 1.0],
                                   [ 0, 0, 2.0, 0.0 ],
                                   [ 1, 0, 1.0, 0.0 ]]),
                 'bdry_mode':'spectral',
                 'NFP': 1,                 
                }
           }
        





