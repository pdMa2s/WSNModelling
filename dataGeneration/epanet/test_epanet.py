from dataGeneration.epanet import ENopen, ENclose, ENopenH, ENinitH, ENrunH, ENnextH, ENcloseH, ENgetcount, EN_TANK, \
    ENgetnodetype, EN_NODECOUNT, EN_TANKLEVEL, ENgetnodevalue, EN_PRESSURE, ENsettimeparam, EN_DURATION, ENsetnodevalue


def tank_indexes():
    n_links = ENgetcount(EN_NODECOUNT)
    tank_idx = []
    for i in range(1, n_links + 1):
        type = ENgetnodetype(i)
        if type == EN_TANK:
            tank_idx.append(i)
    return tank_idx


if __name__ == '__main__':
    err_code = ENopen('Richmond_skeleton.inp', '/dev/null')

    tank_idx = tank_indexes()

    ENopenH()
    ENinitH(10)
    ENsetnodevalue(tank_idx[0], EN_TANKLEVEL, .5)
    t_step = 1

    levels = list()
    ENsettimeparam(EN_DURATION, 82800)

    while t_step > 0:
        t = ENrunH()
        if t % 3600 is 0:
            levels.append(ENgetnodevalue(tank_idx[0], EN_PRESSURE))

        t_step = ENnextH()

    print(levels)
    ENcloseH()
    ENclose()