import numpy
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tiltx.support.methods import *
import warnings


class FeatureExtractor(object):
    
    def __init__(self, t, alpha, beta, RT_method='cumsum', correct='left'):
        """
        This class attempts to extract a number of features from the time series
        representing Euler angles of a mobile device. Since we look only
        into the movement along two axes (left-right and up-down), as an
        input we have alpha and beta with t being a time component. The series alpha
        is angle along the left-right axis, and beta along up-down axis.
        
        RT_method is the method used for change point detection. It can be either
        cumsum or stationary.
        The parameter correct spcifies the direction of correct response:
        left, right, up, or down.
        
        :param t: the array representing the time component
        :type t: list or array
        :param alpha: the array representing a time series of left-right axis
        :type alpha: list or array
        :param beta: the array representing a time series of up-down axis
        :type beta: list or array
        :param RT_method: the method used for finding reaction time
        :type RT_method: string
        :param correct: the orientation of the correct/desired response
        :type correct: string
        """
        
        if not len(t) == len(alpha) == len(beta):
            warnings.warn(
                    "t, alpha, and beta must be equal length arrays.'"
                )
            return
        
        if RT_method not in ['cumsum','stationary']:
            warnings.warn(
                    "\n{} is an invalid RT_method parameter. Use either 'cumsum' or 'stationary'.".format(
                        RT_method)
                )
            return
        
        if correct not in ['left','right','up','down']:
            warnings.warn(
                    "\n{} is an invalid target. Use either 'left', 'right', 'up', or 'down'.".format(
                        correct)
                )
            return
        
        """
        Define a triggering axis x and the remaining (non-triggering)
        axis y.
        """
        if abs(alpha[-1]) > abs(beta[-1]):
            # alpha is a trigger
            trigger = 'alpha'
            self.x = numpy.array(alpha)
            self.y = numpy.array(beta)
        else:
            # beta is a trigger
            trigger = 'beta'
            self.x = numpy.array(beta)
            self.y = numpy.array(alpha)
        
        self.t = numpy.array(t)
        self.alpha = numpy.array(alpha)
        self.beta = numpy.array(beta)
        
        """
        Introduce a new parameter target which specifies the correct/desired 
        direction:
            - target = [1,0] --> 'left'
            - target = [-1,0] --> 'right'
            - target = [0,1] --> 'up'
            - target = [0,-1] --> 'down'.
        """
        
        dic = {'left': [1,0],
              'right': [-1,0],
              'up': [0,1],
              'down': [0,-1]}
        
        target = dic[correct]
            
        # Calculate features
        self.find_RT(RT_method)
        self.find_L1(target=target, trigger=trigger)
        self.find_MD(correct=correct)
        self.find_SampEn()
        self.find_flips()
        # Plot x and y
        # self.plot()
             
    def find_RT(self, RT_method):
        """This function finds a point of change in x."""
        if RT_method == 'cumsum':
            # Compute index of change point detection using cumsum.
            self.RT_index = CUMSUM_flip(self.x,self.t)
            # Compute reaction time       
            self.RT = self.t[self.RT_index]-self.t[0]
        elif RT_method == 'stationary':
            # The index of change point is the last stationary point.
            self.RT_index = last_stationary_point(self.x,self.t)
            # Compute reaction time       
            self.RT = self.t[self.RT_index]-self.t[0]
                
    def find_L1(self, target, trigger):
        """Finds L1 norm of x and y with respect to target."""
        # defined L1 norms
        # L1_x : L1 norm of the triggering axis
        # L1_y : L1 norm of the remaining axis
        dt = (self.t[-1]-self.t[0])
        L1_alpha = 1/dt * integrate.simps(numpy.abs(0.5*target[0] - self.alpha), self.t)
        L1_beta = 1/dt * integrate.simps(numpy.abs(0.5*target[1] - self.beta), self.t)
        if trigger == 'alpha':
            self.L1_x, self.L1_y = L1_alpha, L1_beta
        elif trigger == 'beta':
            self.L1_x, self.L1_y = L1_beta, L1_alpha
                
                
    def find_MD(self, correct):
        """Finds maximal deviation of x and y in the direction opposite from
        the target/goal. The goal of the triggering axis is 0.5 for left and up,
        and -0.5 for right and down. The goal of the non-triggering axis is zero.
        If there are any transitive processes at the begining, the parameter
        t_MD can be set accordingly. E.g., if the first m samples are transitive,
        t_MD = m.
        """
        # MD : Maximal Deviation
        # t_MD : time from which we calculate MD (in this way we can 
        # eliminate the transitive process at the begining of the move)
        t_MD = 0 #begining
        t_end = self.RT_index # or t_end = -1
        if correct == 'left':
            self.MD_x = numpy.abs(numpy.min(self.alpha[t_MD:t_end]))
            self.MD_y = numpy.max(numpy.abs(self.beta[t_MD:t_end]))
        elif correct == 'right':
            self.MD_x = numpy.abs(numpy.max(self.alpha[t_MD:t_end]))
            self.MD_y = numpy.max(numpy.abs(self.beta[t_MD:t_end]))
        elif correct == 'up':
            self.MD_x = numpy.abs(numpy.min(self.beta[t_MD:t_end]))
            self.MD_y = numpy.max(numpy.abs(self.alpha[t_MD:t_end]))
        elif correct == 'down':
            self.MD_x = numpy.abs(numpy.max(self.beta[t_MD:t_end]))
            self.MD_y = numpy.max(numpy.abs(self.alpha[t_MD:t_end]))
        
    
    def find_SampEn(self, ):
        """Compute the sample entropy of the interpolated and normalized signals."""
        # Firstly, normalize x and y.
        x_norm = normalize(self.x)
        y_norm = normalize(self.y)
        # Interpolate x and y to 100 samples from t[0] to t[-1]
        # We do this interpolation for computational reasons.
        self.t_int = numpy.linspace(self.t[0], self.t[-1], 101)
        self.x_int = numpy.interp(self.t_int, self.t, x_norm)
        self.y_int = numpy.interp(self.t_int, self.t, y_norm)
        # r: tolerance of the sampling entropy
        r = 0.1
        # Compute sampling entropy of x and y.
        self.SE_x = SampEn(self.x_int, 5, r)
        self.SE_y = SampEn(self.y_int, 5, 0.1)
    
    def find_flips(self, ):
        """Compute the number of turning points of filtered signals x and y.
        We use the Savitzky–Golay filter. For more about this filter see:
        https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
        """
        # Filter the interpolated signals x and y.
        self.x_filt = savgol_filter(self.x_int, 51, 3)
        self.y_filt = savgol_filter(self.y_int, 51, 3)
        # Compute the number of flips.
        self.x_flips = number_of_flips(self.x_filt, self.t_int)
        self.y_flips = number_of_flips(self.y_filt, self.t_int)
        
        
    def plot(self, ):
        """Plot the time series x and y, onset of reaction, and shade the regions
        corresponding to L1 norms.
        """
        plt.plot(self.t,self.x,label = '$x$')
        plt.plot(self.t,self.y,label = '$y$')
        #plt.axvline(x=self.t[0],color='g', linestyle='-') # begining
        #plt.axvline(x=self.t[-1],color='r', linestyle='-') # end
        plt.axvline(x=self.t[self.RT_index],color='k', linestyle='--',label='Onset')
        if self.x[-1] > 0: 
            plt.fill_between(self.t, 0.5, self.x, facecolor='blue', alpha=0.1, label = '$T \cdot L^1_x$')
        else:
            plt.fill_between(self.t, - 0.5, self.x, facecolor='blue', alpha=0.1, label = '$T \cdot L^1_x$')
        plt.fill_between(self.t, 0, self.y, facecolor='green', alpha=0.1, label = '$T \cdot L^1_y$')
        plt.axvline
        plt.xlabel('$t$')
        plt.ylabel('$x,\,y$')
        plt.legend()
        plt.grid()
        plt.show()