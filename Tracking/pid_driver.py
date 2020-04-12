import numpy as np
from cprint import cprint


class PIDController:
    '''Given the desired gains for each component, create a PID controller to
    compute an appropriate response for the reference person.'''
    def __init__(self, Kp, Ki, Kd, K_loss, scaling_factor, limiter):
        # Specific parameters
        # TODO: REWRITE THE DRIVER
        #self.func = func
        self.Kc = Kc
        self.Ki = Ki
        self.Kd = Kd
        self.K_loss = K_loss
        self.scaling_factor = scaling_factor
        self.limiter = limiter

        self.lost_limiter = limiter / 3.0

        # Internal parameters for the response calculation
        self.prev_error = 0
        self.cumulative_error = 0
        self.last_response = 0


    # def _sendCommand(self, response):
    #     ''' This method sends a command to the motors, and updates the last command. '''
    #     # Soften movements:
    #     '''
    #     if abs(response - self.last_response) > 0.2:
    #         speeds = np.linspace(self.last_response, response, num=5)
    #         for speed in speeds:
    #             self.func(speed)
    #             sleep(0.05)
    #     else:
    #         self.func(response)
    #     self.last_response = response
    #     '''
    #     #print "    response: ", response
    #     #print "    last res: ", self.last_response
    #     resp = (response + self.last_response) / 2.0
    #     #print "sending -> ", resp
    #     # self.func(resp)



    # def brake(self):
    #     self._sendCommand(0)


    def lostResponse(self):
        '''Send a softened response from the last sent speed.'''
        response = self.K_loss * self.last_response
        #print "lost called"
        if abs(response) > self.lost_limiter:
            response = np.sign(response) * self.lost_limiter

        self._sendCommand(response)
        # Return for printing
        return response



    def computeResponse(self, error):
        '''Compute the appropriate response'''

        P = self.Kc * error * self.scaling_factor
        I = self.Ki * (error + self.cumulative_error) * self.scaling_factor
        if self.prev_error != 0: # To avoid jumps because of the huge derivative
            D = self.Kd * (error - self.prev_error) * self.scaling_factor
        else:
            D = 0

        response = P + I + D

        if verbose: # Print outputs
            cprint.ok('    P >> %.3f' % (P))
            cprint.ok('    I >> %.3f' % (I))
            cprint.ok('    D >> %.3f' % (D))
            cprint.ok('  Response >> %.3f' % (response))


        # Limiting response!
        if abs(response) > self.limiter:
            response = np.sign(response) * self.limiter

        if not debug:
            self._sendCommand(response)

        # Update parameters for the next iteration
        self.prev_error = error
        self.cumulative_error += error

        # Returning response for printing it
        return response

    def resetError(self):
        ''' Sets the cumulative error again to 0 (called when the target is recovered). '''
        self.cumulative_error = 0
        self.prev_error = 0
