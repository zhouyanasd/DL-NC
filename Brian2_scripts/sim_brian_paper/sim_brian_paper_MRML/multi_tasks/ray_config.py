ray_cluster_one = {
    'head': {
        'manage_address': '192.168.1.179',
        'manage_port': 22,
        'address': '192.168.1.179',
        'name': 'zy',
        'password': '88507840',
        'dashboard': '8265',
        'start_commends': ['source Project/DL-NC/venv/bin/activate',
                          'ray start --head --port=6379 --dashboard-host="0.0.0.0"'],
        'stop_commends': ['source Project/DL-NC/venv/bin/activate',
                           'ray stop'],
    },
    'nodes':{
        'manage_address_list': ['192.168.1.196', '192.168.1.180', '192.168.1.181', '192.168.1.182', '192.168.1.183'],
        'manage_port_list': [22, 22, 22, 22, 22],
        'address_list':['192.168.1.196', '192.168.1.180', '192.168.1.181', '192.168.1.182', '192.168.1.183'],
        'name_list':['zy', 'zy', 'zy', 'zy', 'zy'],
        'password_list':['88507840', '88507840', '88507840', '88507840', '88507840'],
        'start_commends': ['source Project/DL-NC/venv/bin/activate',
                          'ray start --address="192.168.1.179:6379" --redis-password="5241590000000000"'],
        'stop_commends': ['source Project/DL-NC/venv/bin/activate',
                          'ray stop'],
    }
}

ray_cluster_two = {
    'head': {
        'manage_address': '219.216.80.12',
        'manage_port': 22,
        'address': '192.168.111.179',
        'name': 'zy',
        'password': '88507840',
        'dashboard': '8265',
        'start_commends': ['source Project/DL-NC/venv/bin/activate',
                          'ray start --head --port=6379 --dashboard-host="0.0.0.0"'],
        'stop_commends': ['source Project/DL-NC/venv/bin/activate',
                          'ray stop'],
    },
    'nodes':{
        'manage_address_list':['219.216.80.12', '219.216.80.12', '219.216.80.12', '219.216.80.12', '219.216.80.12', '219.216.80.12'],
        'manage_port_list':[23, 24, 25, 26, 27, 28],
        'address_list':['192.168.111.180', '192.168.111.181', '192.168.111.182', '192.168.111.183', '192.168.111.184', '192.168.111.185'],
        'name_list':['zy', 'zy', 'zy', 'zy', 'zy', 'zy'],
        'password_list':['88507840', '88507840', '88507840', '88507840', '88507840', '88507840'],
        'start_commends': ['source Project/DL-NC/venv/bin/activate',
                          'ray start --address="192.168.111.179:6379" --redis-password="5241590000000000"'],
        'stop_commends': ['source Project/DL-NC/venv/bin/activate',
                          'ray stop'],
    }
}


ray_cluster_three = {
    'head': {
        'manage_address': '192.168.101.179',
        'manage_port': 22,
        'address': '192.168.101.179',
        'name': 'zy',
        'password': '88507840',
        'dashboard': '8265',
        'start_commends': ['source Project/DL-NC/venv/bin/activate',
                          'ray start --head --port=6379 --dashboard-host="0.0.0.0"'],
        'stop_commends': ['source Project/DL-NC/venv/bin/activate',
                           'ray stop'],
    },
    'nodes':{
        'manage_address_list': ['192.168.101.180'],
        'manage_port_list': [22],
        'address_list':['192.168.101.180'],
        'name_list':['zy'],
        'password_list':['88507840'],
        'start_commends': ['source Project/DL-NC/venv/bin/activate',
                          'ray start --address="192.168.101.179:6379" --redis-password="5241590000000000"'],
        'stop_commends': ['source Project/DL-NC/venv/bin/activate',
                          'ray stop'],
    }
}