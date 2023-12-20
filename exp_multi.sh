
# python run.py --task_name edema_analysis_3rd_trend --adv_weight 0 --num_epoch_discriminator 0 --model_type GNN_simple --adv_D_delay_epochs 200 --adv_E_delay_epochs 200 --gcn_depth 2 --layers 1 --loss_type CE --lr 0.005 --adv False
# python run.py --task_name fatigue_analysis_3rd_trend --adv_weight 0 --num_epoch_discriminator 0 --model_type GNN_simple --adv_D_delay_epochs 200 --adv_E_delay_epochs 200 --gcn_depth 2 --layers 1 --loss_type CE --lr 0.005 --adv False
# python run.py --task_name edema_analysis_3rd_trend --adv_weight 0 --num_epoch_discriminator 0 --model_type LSTM --adv_D_delay_epochs 100 --adv_E_delay_epochs 100 --gcn_depth 2 --layers 1 --loss_type CE --lr 0.005 --symptom_weight 1
# python run.py --task_name fatigue_analysis_3rd_trend --adv_weight 0 --num_epoch_discriminator 0 --model_type LSTM --adv_D_delay_epochs 100 --adv_E_delay_epochs 100 --gcn_depth 2 --layers 1 --loss_type CE --lr 0.005 --symptom_weight 1

# python run.py --task_name edema_analysis_3rd_trend --adv_weight 0 --num_epoch_discriminator 0 --model_type MTGNN --adv_D_delay_epochs 100 --adv_E_delay_epochs 100 --gcn_depth 2 --layers 1 --loss_type CE --lr 0.005 --symptom_weight 1.1
# python run.py --task_name fatigue_analysis_3rd_trend --adv_weight 0 --num_epoch_discriminator 0 --model_type MTGNN --adv_D_delay_epochs 100 --adv_E_delay_epochs 100 --gcn_depth 2 --layers 1 --loss_type CE --lr 0.005 --symptom_weight 1.1

python run.py --task_name edema_analysis_3rd_trend --adv_weight 0.0 --num_epoch_discriminator 20 --model_type MTGNN_lstm --adv_D_delay_epochs 3 --adv_E_delay_epochs 5 --gcn_depth 2 --layers 1 --loss_type CE --lr 0.005 --symptom_weight 1.3

python run.py --task_name fatigue_analysis_3rd_trend --adv_weight 0.0 --num_epoch_discriminator 20 --model_type MTGNN_lstm --adv_D_delay_epochs 3 --adv_E_delay_epochs 5 --gcn_depth 2 --layers 1 --loss_type CE --lr 0.005 --symptom_weight 1.1

