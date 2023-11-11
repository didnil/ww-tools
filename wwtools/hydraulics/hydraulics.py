def calculate_power(Q, H, n, rho=1000, g=9.81):
    """
    Formula to calculate power

    Parameters
    ----------
    Q : float
        Flow in m3/s
    H : float
        Operating head (m)
    """

    P = (rho * g * Q * H) / n
    return P


def calculate_inertia(P, N, pump_type='normal'):
    """
    Empiric formula for pump inertia

    Parameters
    ----------
    P : float
        Power of pump (kW)
    N : float
        Pump speed (r / (min * 1000))

    """
    
    if pump_type == 'normal':
        I_pump = 0.03768 * (P / N**3)**(0.9556)
    elif pump_type == 'light weight':
        I_pump = 0.03407 * (P / N**3)**0.844
    else:
        pass
    I_motor = 0.0043 * (P / N)**1.48
    # I_tot = I_pump + I_motor
    # return I_tot
    return I_pump, I_motor
