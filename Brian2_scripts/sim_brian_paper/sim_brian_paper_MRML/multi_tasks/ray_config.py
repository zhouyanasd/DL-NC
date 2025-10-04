ray_cluster_one = {
    'head': {
        'manage_address': '192.168.81.179',
        'manage_port': 22,
        'address': '192.168.81.179',
        'name': 'zy',
        'password': '88507840',
        'dashboard': '8265',
        'start_commends': ['source Project/DL-NC/venv/bin/activate',
                          'ray start --head --port=6379 --dashboard-host="0.0.0.0"'],
        'stop_commends': ['source Project/DL-NC/venv/bin/activate',
                           'ray stop'],
    },
    'nodes':{
        'manage_address_list': ['192.168.81.180', '192.168.81.181'],
        'manage_port_list': [22, 22],
        'address_list':['192.168.81.180', '192.168.81.181'],
        'name_list':['zy', 'zy'],
        'password_list':['88507840', '88507840'],
        'start_commends': ['source Project/DL-NC/venv/bin/activate',
                          'ray start --address="192.168.81.179:6379" --redis-password="5241590000000000"'],
        'stop_commends': ['source Project/DL-NC/venv/bin/activate',
                          'ray stop'],
    }
}