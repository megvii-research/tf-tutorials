## CPU version

### 1. Install docker-ce
For OS other than below, please google by yourself.

#### Ubuntu 16.04
We provide a bash file to help install docker-ce. 
```bash
git clone https://github.com/PeiqinSun/tf-tutorials
cd ./tf-tutorials/Docker-materials 
bash install-docker-ce.sh
```

#### Mac OS
https://stackoverflow.com/questions/40112083/can-i-use-docker-for-installing-ubuntu-on-a-mac

### 2. Pull base image
```bash
sudo docker pull ubuntu:16.04
```

### 3. build tensorflow-cpu image
```bash
sudo docker build -t deeplearning:tensorflow .
sudo docker run -it --name=${NAME} deeplearning:test /bin/bash
```

### 4. Common commands
1. If you want to detach docker, please use `ctrl + p + q`
2. If you want to attach docker, please use `sudo docker attach ${container-ID}`
3. If you want to lookup your container ID, please use `sudo docker ps -a`
4. If you want to stop docker, please use `ctrl + d` or `exit`
5. If you want to start docker, please use `sudo docker start ${container-ID}`

---
