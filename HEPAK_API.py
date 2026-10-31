import win32com.client
import win32gui
import win32con
import os
import atexit
import threading
import time
import numpy as np

######################################################################
# Define global variables
######################################################################

ADD_IN_PATH = r"C:\Users\sulta\AppData\Roaming\Microsoft\AddIns\HEPAK.xla"

######################################################################
# Define Excel class
######################################################################

class Excel:

    def __init__(self):

        self.xl = win32com.client.gencache.EnsureDispatch("Excel.Application")
        self.xl.Visible = False
        self.xl.DisplayAlerts = False

        watcher_thread = threading.Thread(target=self.auto_license_popup_checker, daemon=True)
        watcher_thread.start()

        self.hepak_wb = self.xl.Workbooks.Open(ADD_IN_PATH)

        atexit.register(self.close)

    #Automatically hits ok button in license check popup window during initialization  
    def auto_license_popup_checker(self):

        for _ in range(30):
            try:
                hwnd = win32gui.FindWindow(None, "HePak")
                if hwnd != 0:
                    win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)
                    win32gui.PostMessage(hwnd, win32con.WM_KEYUP, win32con.VK_RETURN, 0)
                    break
            except Exception:
                pass
            
            time.sleep(0.2)

    #Kills the backround excel process created during initialization
    def close(self):

        try:
            if hasattr(self, 'hepak_wb') and self.hepak_wb:
                self.hepak_wb.Close(SaveChanges=False)
            if hasattr(self, 'xl') and self.xl:
                self.xl.Quit()
        except:
            pass

######################################################################
# Define HEPAK class
######################################################################

class HEPAK(Excel):

    def __init__(self):

        super().__init__()

    #Returns fluid's density
    def rho(self, T, p):

        return np.vectorize(lambda T, p: self.xl.Evaluate(f'=HeCalc(3;0;1;{p};2;{T};1)'))(T, p)
    
    #Returns fluid's thermal diffusivity
    def kappa(self, T, p):

        return np.vectorize(lambda T, p: self.xl.Evaluate(f'=HeCalc(28;0;1;{p};2;{T};1)'))(T, p)

    #Returns fluid's heat capacity at constant pressure
    def cp(self, T, p):

        return np.vectorize(lambda T, p: self.xl.Evaluate(f'=HeCalc(14;0;1;{p};2;{T};1)'))(T, p)

    #Returns fluid's heat capacity at constant volume
    def cv(self, T, p):

        return np.vectorize(lambda T, p: self.xl.Evaluate(f'=HeCalc(15;0;1;{p};2;{T};1)'))(T, p)

    #Returns fluid's ratio of specific heats
    def gamma(self, T, p):
        
        return self.cp(T,p) / self.cv(T,p)

    #Returns fluid's isothermal compressibility
    def alpha_T(self, T, p):

        return np.vectorize(lambda T, p: self.xl.Evaluate(f'=HeCalc(19;0;1;{p};2;{T};1)'))(T, p)

    #Returns fluid's isobaric expansivity
    def alpha_p(self, T, p):

        return np.vectorize(lambda T, p: self.xl.Evaluate(f'=HeCalc(17;0;1;{p};2;{T};1)') / T)(T, p)

    #Returns fluid's dynamic viscosity
    def eta(self, T, p):

        return np.vectorize(lambda T, p: self.xl.Evaluate(f'=HeCalc(25;0;1;{p};2;{T};1)'))(T, p)

    #Returns fluid's kinematic viscosity
    def nu(self, T, p):
    
        return self.eta(T,p) / self.rho(T,p)

    #Returns fluid's thermal conductivity
    def lam(self, T, p):
    
        return np.vectorize(lambda T, p: self.xl.Evaluate(f'=HeCalc(26;0;1;{p};2;{T};1)'))(T, p)

    def __exit__(self):
        
        self.close()