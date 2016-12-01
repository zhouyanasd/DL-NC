import numpy as np
from src.core import global_constant
from src.core import Base

base = Base()
print(base.get_global_connection())
pre_conn = np.array([[(1,0.5)]],dtype = global_constant.CONNECTION_ARRAY_DTYPE)
post_conn = np.array([[(0,0.0)]],dtype = global_constant.CONNECTION_ARRAY_DTYPE)
self_conn = np.array([[(0,0.0)]],dtype = global_constant.CONNECTION_ARRAY_DTYPE)

base.add_global_connection(pre_conn,post_conn,self_conn)

print(base.get_global_connection())