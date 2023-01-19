#!/bin/bash

name=rcnn
port=8000 # pass-thuru port (for port forwarding)

run()
{
    case "$1" in
    build)
        rm -rf docker/${name}
        mkdir -p docker/${name}
        cp -r data docker/${name}/
        cp -r model docker/${name}/
        cp -r utils docker/${name}/
        cp -r *.py docker/${name}/
        cp -r *.txt docker/${name}/
        cp -r *.sh docker/${name}/
        cp -r *.ipynb docker/${name}/
        sudo docker build -t pushdown99/${name} docker
        ;;
    push)
        sudo docker push pushdown99/${name}
        ;;
    run)
        host="${name}-P${port}"
        sudo docker run -p ${port}:${port}/tcp --name ${host} -h ${host} --ipc=host --mount type=bind,source=/home/hyhwang/repositories/model/${name}/dataset,target=/${name}/dataset --mount type=bind,source=/home/hyhwang/repositories/dataset/NIA/download/origin,target=/${name}/images --mount type=bind,source=/home/hyhwang/repositories/model/${name}/output,target=/${name}/output -it --rm --runtime=nvidia pushdown99/${name} bash
        ;;
    *)
        echo ""
        echo "Usage: docker-build {build|push|torch}"
        echo ""
        echo "       build : build docker image to local repositories"
        echo "       push  : push to remote repositories (docker hub)"
        echo ""
        return 1
        ;;
    esac
}
run "$@"

