#set -x
if [ "$RLAUNCH_REPLICA" == "0" ]; then
    echo "***************************************************************************"
    echo $RLAUNCH_ID
    echo "***************************************************************************"
fi

mkdir -p logs/$RLAUNCH_ID

(
  flock -n 1
  echo $RLAUNCH_REPLICA: `hostname -i` $KUBEBRAIN_NODE_NAME
) >> logs/$RLAUNCH_ID/workers.txt


if [ "$NCCL_DEBUG_SUBSYS" != "" ]; then
    lspci -PP>>logs/$RLAUNCH_ID/$RLAUNCH_REPLICA-lspci.log 2>&1
    nvidia-smi>>logs/$RLAUNCH_ID/$RLAUNCH_REPLICA-smi.log 2>&1
    sudo mst status -v>>logs/$RLAUNCH_ID/$RLAUNCH_REPLICA-mst-status.log 2>&1
fi

(
  export RLAUNCH_REPLICA_0_HOST=$RLAUNCH_ID-0.`hostname -d`
  echo "***************************************************************************"
  echo 'Worker:' `hostname` $RLAUNCH_REPLICA/$RLAUNCH_REPLICA_TOTAL '('$RLAUNCH_COUNTER')' `hostname -i`
  echo 'Host:  ' $KUBEBRAIN_NODE_NAME
  echo "***************************************************************************"
  torchrun --nproc_per_node 8 --master_addr $RLAUNCH_REPLICA_0_HOST --master_port 5678 --nnodes $RLAUNCH_REPLICA_TOTAL --node_rank $RLAUNCH_REPLICA -- ${@:1}

) >> logs/$RLAUNCH_ID/$RLAUNCH_REPLICA.log 2>&1

#echo $KUBEBRAIN_NODE_NAME>>logs/$RLAUNCH_ID/$RLAUNCH_REPLICA.log 2>&1
#torchrun --nproc_per_node 8 --master_addr $RLAUNCH_REPLICA_0_HOST --master_port 5678 --nnodes $RLAUNCH_REPLICA_TOTAL --node_rank $RLAUNCH_REPLICA -- ${@:1}>>logs/$RLAUNCH_ID/$RLAUNCH_REPLICA.log 2>&1

#sleep infinity
