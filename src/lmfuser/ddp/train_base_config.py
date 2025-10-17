from hyperargs import Conf, StrArg


class TrainConfigBase(Conf):

    project_name = StrArg('please_set_a_project_name')
    run_name = StrArg('please_set_the_name_of_this_run')
