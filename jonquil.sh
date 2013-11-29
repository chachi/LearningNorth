# Sample file for running learning_north.py

TRAIN_SETS="~/data/FollowTheSun/JonquilFarm/2013.11.28_12:57:04/"
 # ~/data/FollowTheSun/JonquilFarm/2013.11.28_13:01:49/ ~/data/FollowTheSun/JonquilFarm/2013.11.28_13:06:58/

TEST_SETS="~/data/FollowTheSun/JonquilFarm/2013.11.28_12:59:40/" #~/data/FollowTheSun/JonquilFarm/2013.11.28_13:03:59/

python learning_north.py -train $TRAIN_SETS  -test $TEST_SETS --lasso
