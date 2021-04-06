import paramiko

ssh = paramiko.SSHClient()
key = paramiko.AutoAddPolicy()
ssh.set_missing_host_key_policy(key)
ssh.connect('219.216.81.148', 11179, 'zy', '88507840' ,timeout=5)
stdin, stdout, stderr = ssh.exec_command('ls -l')
# stdin, stdout, stderr = ssh.exec_command('source Project/DL-NC/venv/bin/activate')
# stdin, stdout, stderr = ssh.exec_command('ray start --head --port=6379 --dashboard-host="0.0.0.0" ')

for i in stdout.readlines():
	print(i)