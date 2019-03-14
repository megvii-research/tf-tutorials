## CPU version

### 1. Install docker-ce
We provide a bash file to help install docker-ce on ubuntu16.04. For other OS, please google by yourself.
```bash
git clone https://github.com/PeiqinSun/tf-tutorials
cd ./tf-tutorials/Docker-materials 
bash install-docker-ce.sh
```

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
