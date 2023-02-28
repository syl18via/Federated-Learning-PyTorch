
# Create a container in an interactive mode
nvidia-docker run -it --name fl_hphu -e HOME=/root/user/ -v /home/hphu:/root/user/ juntao/cdpp:0.1 /bin/bash

# start the container
docker start -i fl_hphu && cd && zsh

cd Federated-Learning-PyTorch