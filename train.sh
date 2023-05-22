python -u -c 'import torch; print(torch.__version__)'
FULL_DATA_PATH=$1
 
echo $FULL_DATA_PATH
 
python -u codes/run_bi.py \
         --cuda \
         --gpu 1 \
         --do_train \
         --do_valid \
         --do_test \
         --word_embed_path data/$FULL_DATA_PATH/ontology/word_embedding.pytorch \
         --word_graph_path data/$FULL_DATA_PATH/ontology/word_graph.pytorch \
         --data_path data/$FULL_DATA_PATH \
         --ontology_data_path data/$FULL_DATA_PATH/ontology \
         --model TransE \
         --agg TRIPLE  \
         --cpu_num 30  \
         --max_steps 800000 \
         -n 100 -b 2080 -d 100  -g 9.0 -lr 0.0005  \
         --test_batch_size 30 \
         --valid_steps 50000 \
         --test_log_steps 300 \
         --log_steps 500 \
         -save models/ \
         --n-hidden 100 \
         --n_basses 5 \
         --n-layers 2 \
         --dropout 0 \
         #--double_entity_embedding
