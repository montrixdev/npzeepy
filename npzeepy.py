"""Main module."""

import os
import inspect
import numpy as np

class Workspace:
    def __init__(self, path) -> None:
        self.path = path

        if not os.path.exists(path):
            raise Exception('{0} does not exist'.format(path))

    def retrieve_name(self, var):
            """
            Gets the name of var. Does it from the out most frame inner-wards.
            :param var: variable to get name from.
            :return: string
            """
            for fi in reversed(inspect.stack()):
                names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
                if len(names) > 0:
                    return names[-1]

    def set_variable(self, arr, name=None):
        nm = name
        if nm is None:
            nm = self.retrieve_name(arr)
        filepath = os.path.join(self.path, nm)
        np.save(filepath, arr)



if __name__ == '__main__':
    ws1 = Workspace('D:/npzee/workspace1')

    var1 = np.random.randn(300,4000)
    var2 = np.ones((3,4))
    var3 = np.random.randn(20, 300, 400)

    ws1.set_variable(var1)
    ws1.set_variable(var2)
    ws1.set_variable(var3)
    ws1.set_variable(np.random.randn(20, 300))

    # set_variable(var2)
    # ws1.all_variables() # ['']
    # ws1.export_npz('filename.npz')



