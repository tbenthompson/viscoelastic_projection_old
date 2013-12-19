def elastic_stress(x, y, s, D, shear_modulus):
    """
    Use the elastic half-space stress solution from Segall (2010)
    """
    factor = (s * shear_modulus) / (2 * np.pi)
    main_term = -(y - D) / ((y - D) ** 2 + x ** 2)
    image_term = (y + D) / ((y + D) ** 2 + x ** 2)
    Szx = factor * (main_term + image_term)

    main_term = x / (x ** 2 + (y - D) ** 2)
    image_term = -x / (x ** 2 + (y + D) ** 2)
    Szy = factor * (main_term + image_term)
    return Szx, Szy
