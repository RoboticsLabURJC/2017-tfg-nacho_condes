import numpy as np
from cprint import cprint


class PIDController:
    '''Given the desired gains for each component, create a PID controller to
    compute an appropriate response for the reference person.'''
    def __init__(self, Kp, Ki, Kd, K_loss, limit, stop_range, soften, verbose):
        # Specific parameters
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.K_loss = K_loss
        self.limit = limit
        self.stop_range = stop_range
        self.center = np.mean(stop_range)

        self.lost_limiter = limit / 3.0

        # Internal attributes for the response calculation
        self.prev_error = 0
        self.cumulative_error = 0
        self.last_response = 0

        self.soften = soften
        self.verbose = verbose

    def lostResponse(self):
        """Stop with a controlled inertia."""
        response = self.K_loss * self.last_response
        self.last_response = response
        return response

    def isInRange(self, error):
        """Check if the error is inside the dead range."""
        return self.stop_range[0] <= error <= self.stop_range[1]

    def computeResponse(self, error):
        """Compute the appropriate response."""

        if np.isnan(error):
            return self.last_response

        if self.isInRange(error):
            # Reset the memory and stop moving.
            self.prev_error = 0
            self.cumulative_error = 0
            self.last_response = 0
            if self.verbose:
                cprint.ok('\tNull response (target in range)')
            return 0

        # Compute the response from
        # the relative error otherwise
        rel_error = error - self.center
        P = self.Kp * rel_error
        I = self.Ki * (rel_error + self.cumulative_error)
        if abs(self.prev_error) > 0: # To avoid jumps because of the huge derivative
            D = self.Kd * (rel_error - self.prev_error)
        else:
            D = 0

        response = P + I + D

        if self.verbose: # Print outputs
            cprint.info(f'\tP >> {P:.3f}')
            cprint.info(f'\tI >> {I:.3f}')
            cprint.info(f'\tD >> {D:.3f}')
            cprint.ok(f'\tResponse >> {response:.3f}')


        # Limiting response!
        if abs(response) > self.limit:
            response = np.sign(response) * self.limit

        # Update parameters for the next iteration
        self.prev_error = rel_error
        self.cumulative_error += rel_error

        if self.soften:
            response = np.mean([self.last_response, response])

        self.last_response = response

        # Returning the response value
        return response

    def resetError(self):
        """ Sets the cumulative error again to 0 (called when the target is recovered). """
        self.cumulative_error = 0
        self.prev_error = 0
