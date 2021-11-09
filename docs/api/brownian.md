# Brownian controls

SDEs are simulated using a Brownian motion as a control. (See the neural SDE example.)

::: diffrax.AbstractBrownianPath
    selection:
        members:
            - evaluate

::: diffrax.UnsafeBrownianPath
    selection:
        members:
            - __init__
            - evaluate