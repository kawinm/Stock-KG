GPU = 1
LR = 0.0001
BS = 128
W = 20
T = 250
LOG = False
D_MODEL = 5
N_HEAD  = 5
DROPOUT = 0.2
D_FF    = 64
ENC_LAYERS = 1
DEC_LAYERS = 1
MAX_EPOCH = 50
USE_POS_ENCODING = False
USE_GRAPH = True
HYPER_GRAPH = True
USE_KG = False
PREDICTION_PROBLEM = 'value'

# run the code with the parameters
python3 model_gnn.py --gpu $GPU --lr $LR --batch_size $BS --window $W --timesteps $T --log $LOG --d_model $D_MODEL --n_head $N_HEAD --dropout $DROPOUT --d_ff $D_FF --enc_layers $ENC_LAYERS --dec_layers $DEC_LAYERS --max_epoch $MAX_EPOCH --use_pos_encoding $USE_POS_ENCODING --use_graph $USE_GRAPH --hyper_graph $HYPER_GRAPH --use_kg $USE_KG --prediction_problem $PREDICTION_PROBLEM



if __name__ == "__main__":
    main()
    GPU = 1
    LR = 0.0001
    BS = 128
    W = 20
    T = 250
    LOG = False
    D_MODEL = 5
    N_HEAD  = 5
    DROPOUT = 0.2
    D_FF    = 64
    ENC_LAYERS = 1
    DEC_LAYERS = 1
    MAX_EPOCH = 50
    USE_POS_ENCODING = False
    USE_GRAPH = True
    HYPER_GRAPH = True
    USE_KG = False
    PREDICTION_PROBLEM = 'value'

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=GPU)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--bs', type=int, default=BS)
    parser.add_argument('--w', type=int, default=W)
    parser.add_argument('--t', type=int, default=T)
    parser.add_argument('--log', type=bool, default=LOG)
    parser.add_argument('--d_model', type=int, default=D_MODEL)
    parser.add_argument('--n_head', type=int, default=N_HEAD)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--d_ff', type=int, default=D_FF)
    parser.add_argument('--enc_layers', type=int, default=ENC_LAYERS)
    parser.add_argument('--dec_layers', type=int, default=DEC_LAYERS)
    parser.add_argument('--max_epoch', type=int, default=MAX_EPOCH)
