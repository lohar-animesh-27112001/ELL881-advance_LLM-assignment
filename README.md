# ELL881-advance_LLM-assignment
1. Implementing a Decoder-Only Transformer: The goal of this assignment is to develop a decoder-only transformer language model from scratch.

2. Training and Inference Enhancements: Beam Search Decoding, KV Caching, Gradient Accumulation, Gradient Checkpointing.

## Hyperparameters:

  ###
  
      vocab_size = 10000,
      
      d_model = 300,
      
      num_layers = 3,
      
      num_heads = 8,
      
      d_ff = 1024,
      
      max_seq_length = 64,  
      
      batch_size = 32,
    
      learning_rate = 3e-4,
      
      num_epochs = 3

## Number of Parameters:

  ### 1. Input Embedding:
  
      Embedding matrix: 10000 × 300 = 3,000,000

      Projection layer (300→296): 300 × 296 + 296 = 89,096

      Total = 3,089,096

  ### 2. Transformer Blocks (3 layers):

      Per block:

      Multi-head Attention:
      
      Q, K, V projections: 3 × (296 × 296 + 296) = 263,736
      
      Output projection: 296 × 296 + 296 = 87,912
      
      Attention total: 351,648
      
      Feed Forward:
      
      First linear: 296 × 1024 + 1024 = 304,128
      
      Second linear: 1024 × 296 + 296 = 303,400
      
      FF total: 607,528
      
      Layer Norms (2 per block): 2 × (296 + 296) = 1,184

      Per block total: 351,648 + 607,528 + 1,184 = 960,360

      3 blocks total: 3 × 960,360 = 2,881,080

  ### 3. Output Layers:
  
      Final LayerNorm: 296 + 296 = 592
      
      Output linear: 296 × 10000 + 10000 = 2,970,000
      
      Total: 2,970,592
  ### Total:

      Input Embedding:   3,089,096
      Transformer Blocks: 2,881,080
      Output Layers:     2,970,592
      ────────────────────────────────
      TOTAL:             8,940,768 parameters



## Run These Commands:

  ###
      $ git clone https://github.com/lohar-animesh-27112001/ELL881-advance_LLM-assignment.git
  ###
      $ cd ELL881-advance_LLM-assignment
  ###
      $ pip install -r requirements.txt
  ###
      $ cd part-i
  ###
      $ cd layers
  ###
      $ python fasttext_model.py
  ###
      $ cd ..
  ###
      $ cp layers/cc.en.300.bin .
  ###
      $ python transformer_model.py
  #### To run this Python file, you need 32GB of RAM. You can run it on Google Colab.
      $ cd ..
  ###
      $ cp part-i/cc.en.300.bin part-ii/transformer_model-with_fasttext_embeddings/
  ###
      $ cd part-ii/transformer_model-with_fasttext_embeddings/
  ###
      $ python transformer_model.py
  #### To run this Python file, you need 40GB of RAM. You can run it on Google Colab.

## decoder-only model architecture:
<img width="856" height="674" alt="architecture_diagram" src="https://github.com/user-attachments/assets/21b9e146-9fd6-4f5c-9f33-98671b157034" />

## output_ii_attention_visualization.png
<img width="1516" height="1030" alt="output_i_training_curves" src="https://github.com/user-attachments/assets/a95296f2-b889-4ab1-88f5-e9e8177e2bf1" />

## output_ii_attention_visualization.png :
<img width="1202" height="980" alt="output_ii_attention_visualization" src="https://github.com/user-attachments/assets/b3053156-d029-4dd1-b080-e63b84a6d730" />

## output_iii
<img width="1402" height="578" alt="output_iii" src="https://github.com/user-attachments/assets/db2f0062-ad8c-4ca8-b58a-0ba252b53ab8" />

## output_iv
<img width="290" height="220" alt="output_iv" src="https://github.com/user-attachments/assets/21878b59-0c3a-488b-af13-b784470e9277" />


