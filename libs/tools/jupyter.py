from IPython.display import Math, display


def print_math(expr):
    display(Math(expr))


def print_model_info(model):
    print(model)
    print_math(f"G = {model.G:.3e}" + "\mbox{ (Coupling strength)}")
    print_math(r"\Delta = " + f"{model.delta:.3e}")
    print_math(r"\frac{G^2}{\Delta} = " + f"{model.G**2 / model.delta:.0e}")
    print_math(r"\frac \Delta G = " + f"{model.delta / model.G :.0e}" + " \gg 1")
    print_math(r"\Omega = " + f"{model.Omega}")
