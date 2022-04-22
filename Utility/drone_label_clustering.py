import pandas as pd


def definisci_label_drone(scenario, app, sensore):
    if sensore == 'ALVIRA':
        if scenario == '1_1':
            app = app[app['LABEL'] == 1]

        if scenario == '1_2':
            app = app[app['LABEL'] == 1]

        if scenario == '1_3':
            app1 = app[app['LABEL'] == 0]
            app2 = app[app['LABEL'] == 2]
            app = pd.concat([app1, app2], axis=0)

        if scenario == '1_4':
            app = app[app['LABEL'] == 1]
        if scenario == '2_1':
            app = app[app['LABEL'] == 1]
        if scenario == '2_2':
            app = app[app['LABEL'] == 1]
        if scenario == '3':
            app = app[app['LABEL'] == 0]
            app1 = app[app['LABEL'] == 2]
            app = pd.concat([app, app1], axis=0)
        if scenario == '1a_test':
            app = app[app['LABEL'] == 1]
        if scenario == '1b_test':
            # app = app[app['LABEL'] == 2]
            app1 = app[app['LABEL'] == 1]
            app2 = app[app['LABEL'] == 2]
            app = pd.concat([app1, app2], axis=0)
        if scenario == '2a_test':
            app1 = app[app['LABEL'] == 2]
            app2 = app[app['LABEL'] == 0]
            app = pd.concat([app1, app2], axis=0)
        if scenario == '2b_test':
            app1 = app[app['LABEL'] == 2]
            app2 = app[app['LABEL'] == 0]
            app = pd.concat([app1, app2], axis=0)
        if scenario == '2c_test':
            app = app[app['LABEL'] == 0]
        if scenario == '2d_test':
            app = app[app['LABEL'] == 0]
        if scenario == '3a_test':
            app1 = app[app['LABEL'] == 2]
            app2 = app[app['LABEL'] == 0]
            app = pd.concat([app1, app2], axis=0)
    else:
        '''if scenario == '1_1':
            app = app[app['LABEL'] == 0]

        if scenario == '1_2':
            app = app[app['LABEL'] == 0]

        if scenario == '1_3':
            app1 = app[app['LABEL'] == 0]
            app2 = app[app['LABEL'] == 2]
            app = pd.concat([app1, app2], axis=0)

        if scenario == '1_4':
            app = app[app['LABEL'] == 2]
        if scenario == '2_1':
            app = app[app['LABEL'] == 1]
        if scenario == '2_2':
            app = app[app['LABEL'] == 1]
        if scenario == '3':
            app = app[app['LABEL'] == 0]
            app1 = app[app['LABEL'] == 2]
            app = pd.concat([app, app1], axis=0)
        if scenario == '1a_test':
            app1 = app[app['LABEL'] == 1]
            app2 = app[app['LABEL'] == 2]
            app = pd.concat([app1, app2], axis=0)
        if scenario == '1b_test':
            # app = app[app['LABEL'] == 2]
            app1 = app[app['LABEL'] == 1]
            app2 = app[app['LABEL'] == 2]
            app = pd.concat([app1, app2], axis=0)
        if scenario == '2a_test':
            app1 = app[app['LABEL'] == 2]
            app2 = app[app['LABEL'] == 0]
            app = pd.concat([app1, app2], axis=0)
        if scenario == '2b_test':
            app1 = app[app['LABEL'] == 2]
            app2 = app[app['LABEL'] == 0]
            app = pd.concat([app1, app2], axis=0)'''
        if scenario == '2c_test':
            app = app[app['LABEL'] == 0]

    return app
