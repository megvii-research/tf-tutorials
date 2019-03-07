
# Docker
You have to use the Docker to complete all your assignments on servers.

## Instructions
- We will assign an IP address and a port number to everyone, then you can log in via ssh. 
  e.g. `ssh root@40.104.61.196 -p 8100`
- The initial password of docker is **root**. The first thing is to update your password when you log in successfully.
- Run `cd` into your home directory.
- Run `git clone https://github.com/PeiqinSun/tf-tutorials.git` get a repo for course.
- Run `cd tf-tutorials/01-svhn` into your first homework.
- Run `CUDA_VISIBLE_DEVICES=${NUM} python train.py` to start. **NUM can be 0~7**.

##### Warnings
- **You must use your real name and real id. All containers that do not conform to the naming convention will be cleared!!**
- **Don't interrunpt the expriemnt during the data filling stage, otherwise you will generate a large file called core in your directory.**

### GPU Usage

When running your train script,  you should use environment variable **CUDA_VISIBLE_DEVICES** to specify which GPU your program is running on. 

```
CUDA_VISIBLE_DEVICES=0 python train.py
```

To monitor GPU usage, your can use

```
watch nvidia-smi
```

### Tmux
If your program is still running, but you want to temporarily exit the terminal.
You can use tmux, a terminal multiplexer software.
If you want get more information about tmux, please access http://cenalulu.github.io/linux/tmux/.
Run `sudo apt install tmux` to install tmux.


