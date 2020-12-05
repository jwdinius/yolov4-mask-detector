docker run --rm -it \
    --name opencv-course-c \
    --net host \
    --privileged \
    --ipc host \
    --device /dev/video0 \
    --device /dev/video1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/course-materials:/home/opencv/course-materials \
    -e DISPLAY=$DISPLAY \
    jdinius/opencv-4.5.0-nvidia-cuda10.2 \
    /bin/bash
