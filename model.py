import numpy as np

# constant
kb = 1.380649e-23
g = 1e-7
mu0 = (4*np.pi)*g

class Energy:
    def convertToVec(self, angles):
        theta, phi = angles
        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(theta)
        return np.array([x,y,z])

    def computeScalar(self, vec_u, vec_v):
        return np.sum(vec_u*vec_v)/(np.linalg.norm(vec_u)*np.linalg.norm(vec_v))

    def findNearests(self, index, config, volumes, distance):
        lst = [(-1, 0, 0), (1, 0, 0),
                (0, -1, 0), (0, 1, 0),
                (0, 0, -1), (0, 0, 1)]
        radius = lambda vol: 0.5*(6.0*vol/np.pi)**(1/3.)

        nearests = []
        for item in lst:
            try:
                index_nearest = tuple([index[k] + item[k] for k in range(3)])
                angle_j = config[index_nearest]
                vol_i, vol_j  = volumes[index], volumes[index_nearest]
                r = radius(vol_i) + radius(vol_j) + distance
                dic = {
                    'angle': angle_j,
                    'volume': vol_j,
                    'vec_r': np.array(item),
                    'r': r
                }
                nearests.append(dic)
            except IndexError:
                continue

        return nearests

    def computeSingleParticleEnergy(self, vec_si, vec_ni, vec_h, Ku, Ms, vol, h):
        energy_aniso = -Ku*vol*self.computeScalar(vec_si, vec_ni)**2
        energy_zeeman = -mu0*Ms*vol*h*self.computeScalar(vec_si, vec_h)
        return energy_aniso + energy_zeeman

    def computeDipolarEnergy(self, vec_si, vec_sj, vec_r, mui, muj, r):
        firstTerm = self.computeScalar(vec_si, vec_sj)
        secondTerm = -3.0*self.computeScalar(vec_si, vec_r)*self.computeScalar(vec_sj, vec_r)
        eij = ((mui*muj)/r**3)*(firstTerm + secondTerm)
        return eij

    def computeTotalEnergy(self, vec_si, vec_ni, vec_h, Ku, Ms, vol, h, nearests, is_dipolar):
        total_energy = self.computeSingleParticleEnergy(vec_si, vec_ni, vec_h, Ku, Ms, vol, h)
        if is_dipolar:
            mui = mu0*Ms*vol
            energies_dipolar = []
            for item in nearests:
                angle_sj = item['angle']
                vec_sj = self.convertToVec(angle_sj)
                vol_sj = item['volume']
                muj = mu0*Ms*vol_sj
                vec_r = item['vec_r']
                r = item['r']
                eij = self.computeDipolarEnergy(vec_si, vec_sj, vec_r, mui, muj, r)
                energies_dipolar.append(eij)

            total_energy += g*sum(energies_dipolar)

        return total_energy


class Parameters:
    def __init__(self, Ku, Ms, h, box_size, distance, n_iters=int(1e4), is_dipolar=False):
        self.__params = {'Ku': Ku,
                        'Ms': Ms,
                        'h': h,
                        'box_size': box_size,
                        'distance': distance,
                        'n_iters': n_iters,
                        'is_dipolar': is_dipolar
                    }

    def _initAngles(self, low, high):
        return np.random.uniform(low, high, size=self.__params['box_size']+(2,))

    def _createVolumes(self, diameters):
        return (np.pi/6.0)*(diameters)**3

    def setConfigAniso(self, low, high):
        self.__params['config_aniso'] = self._initAngles(low, high)

    def setConfigExternalField(self, low, high):
        self.__params['config_externalfield'] = self._initAngles(low, high)

    def setVolumes(self, diameters):
        self.__params['volumes'] = self._createVolumes(diameters)

    def getParameters(self):
        return self.__params


class MonteCarlo(Energy):
    def __init__(self):
        super().__init__()

    def __call__(self, temp, config, param_obj):
        params = param_obj.getParameters()
        n_iters = params['n_iters']
        box_size = params['box_size']
        Ku = params['Ku']
        Ms = params['Ms']
        h = params['h']
        config_aniso = params['config_aniso']
        config_externalfield = params['config_externalfield']
        volumes = params['volumes']
        dcc = params['distance']
        is_dipolar = params['is_dipolar']
        for i in range(n_iters):
            # get angles
            index = tuple(np.random.randint(box_size))
            angle_si = config[index]
            delta = np.random.uniform(-1.0, 1.0, size=2)
            angle_sj = angle_si + delta
            angle_aniso = config_aniso[index]
            angle_externalfield = config_externalfield[index]

            # get volume of particle
            vol_si = volumes[index]

            # get nearests
            nearests = self.findNearests(index, config, volumes, dcc)

            # convert to vector
            vec_si = self.convertToVec(angle_si)
            vec_sj = self.convertToVec(angle_sj)
            vec_ni = self.convertToVec(angle_aniso)
            vec_h = self.convertToVec(angle_externalfield)

            total_energy_si = self.computeTotalEnergy(vec_si, vec_ni, vec_h, Ku, Ms, vol_si, h, nearests, is_dipolar)
            total_energy_sj = self.computeTotalEnergy(vec_sj, vec_ni, vec_h, Ku, Ms, vol_si, h, nearests, is_dipolar)

            de = -(1.0/(kb*temp))*(total_energy_sj - total_energy_si)
            prob = min(1., np.exp(de, dtype=np.float64))
            u = np.random.uniform(0.0, 1.0)
            if prob > u:
                config[index] = angle_sj.reshape(-1,)

        return config

class Funtions:
    def __init__(self, params):
        self.Ms = params['Ms']
        self.volumes = params['volumes']

    def computeMagnetization(self, config):
        mu_numpy = mu0*self.Ms*self.volumes
        the, phi = config[...,0], config[...,1]
        X = mu_numpy*np.sin(the)*np.cos(phi)
        Y = mu_numpy*np.sin(the)*np.sin(phi)
        Z = mu_numpy*np.cos(the)
        return (np.sqrt(np.sum(X)**2 + np.sum(X)**2 + np.sum(Z)**2))/np.sum(self.volumes)

    def moveAverage(self, data, len_window=5):
        new_data = np.convolve(data, np.ones(len_window), 'valid') / float(len_window)
        return new_data

    def filterNoises(self, data, used_method='Kalman'):
        '''
        here, default is Kalman method to filter noises
        '''
        if used_method=='Kalman':
            from filterpy.kalman import KalmanFilter
            pass
        else:
            return data
