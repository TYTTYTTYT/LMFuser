from hyperargs import Conf, StrArg
from lmfuser_data.interfaces import SubclassTracer


class RunerConf(Conf, SubclassTracer):

    project_name = StrArg('please set a project name')
    run_name = StrArg('please set the name of this run')
