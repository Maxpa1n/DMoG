python -u -c 'import torch; print(torch.__version__)'
FULL_DATA_PATH=$1
 
echo $FULL_DATA_PATH
 
python -u codes/run_bi.py \
         --cuda \
         --gpu 1 \
         --only_test \
         --do_test \
         --word_embed_path data/$FULL_DATA_PATH/ontology/word_embedding.pytorch \
         --word_graph_path data/$FULL_DATA_PATH/ontology/word_graph.pytorch \
         --data_path data/$FULL_DATA_PATH \
         --ontology_data_path data/$FULL_DATA_PATH/ontology \
         --model TransE \
         --agg MOE  \
         --cpu_num 30  \
         --max_steps 700000 \
         -n 100 -b 2080 -d 100  -g 16.0 -lr 0.0005  \
         --test_batch_size 1 \
         --valid_steps 7000 \
         --test_log_steps 300 \
         --log_steps 500 \
         -save models/ \
         --n-hidden 100 \
         --n_basses 5 \
         --n-layers 2 \
         --dropout 0.1 \
	     --init /home/songran/KGEOntology/models/wiki-zero/TransE/MOE/2022-03-25-16-01 \
         --save_path /home/songran/KGEOntology/models/wiki-zero/TransE/MOE/2022-03-25-16-01
         #--double_entity_embedding
