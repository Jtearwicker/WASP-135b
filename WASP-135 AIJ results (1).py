#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#data for WASP-135 b observations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('dark_background')


# In[ ]:


#O-C for AIJ Tc results
#data_date_obs = [Tc_exp, Tc_obs]
d_2020_04_23 = [2458962.8635, 2458962.863485893]
d_2020_05_07 = [2458976.8873, 2458976.869835546]
d_2020_05_27 = [2458997.8980, 2458997.887129746]
d_2020_06_11 = [2459011.9118, 2459011.903662297]
d_2021_03_28 = [2459301.9974, 2459301.989484424]
d_2021_05_12 = [2459346.8415, 2459346.830830686]
d_2021_06_02 = [2459367.8622, 2459367.850342210]
d_2021_07_14 = [2459409.9036, 2459409.904120878]
d_2021_07_26 = [2459422.5160, 2459422.513880192]
d_2021_08_14 = [2459440.7339, 2459440.730359323]
d_2021_08_21 = [2459447.7408, 2459447.741509456]
d_2021_09_04 = [2459461.7546, 2459461.755190148]

data = [d_2020_04_23, d_2020_05_07, d_2020_05_27, d_2020_06_11, d_2021_03_28,
    d_2021_05_12, d_2021_06_02, d_2021_07_14, d_2021_07_26, d_2021_08_14,
    d_2021_08_21, d_2021_09_04]

O_C = np.zeros(len(data))
Tc_obs = np.zeros(len(data))
for i in range(0, len(data)):
    O_C[i] = data[i][1] - data[i][0]
    Tc_obs[i] = data[i][1]


# In[ ]:


plt.plot(Tc_obs, O_C,'o')
plt.ylabel('O-C Tc')
plt.xlabel('Observed Tc (BJD_TDB)')


# In[ ]:




