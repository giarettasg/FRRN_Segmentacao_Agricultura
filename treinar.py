from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess


import matplotlib
matplotlib.use('Agg')

from utilitario import utils, helpers
from construcao import construir_modelo

import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epocas', type=int, default=300, help='Numero de epocas')
parser.add_argument('--epoca_inicio', type=int, default=0, help='Inicio da contagem')
parser.add_argument('--checkpoint_passo', type=int, default=5, help='Frequencia dos checkpoints(epocas)')
parser.add_argument('--validacao_step', type=int, default=1, help='Frequencia de validacao (epocas)')
parser.add_argument('--continuar_treinamento', type=str2bool, default=False, help='Continuar treinamento')
parser.add_argument('--dataset', type=str, default="Crop_Weed", help='Dataset')
parser.add_argument('--recorte_altura', type=int, default=512, help='Altura do recorte')
parser.add_argument('--recorte_comprimento', type=int, default=512, help='Comprimento do recorte')
parser.add_argument('--batch_size', type=int, default=1, help='Numero do batchsize')
parser.add_argument('--num_val_images', type=int, default=2, help='Numero de imagens de validacao')
parser.add_argument('--h_flip', type=str2bool, default=True, help='Translacao horizontal aumento de dados')
parser.add_argument('--v_flip', type=str2bool, default=True, help='Translacao vertival aumento de dados')
parser.add_argument('--model', type=str, default="FRRN-A", help='Modelo')
parser.add_argument('--frontend', type=str, default="ResNet101", help='Frontend')
args = parser.parse_args()


def aumento_dados(imagem_entrada, imagem_saida):

    imagem_entrada, imagem_saida = utils.random_crop(imagem_entrada, imagem_saida, args.recorte_altura, args.recorte_comprimento)

    if args.h_flip and random.randint(0,1):
        imagem_entrada = cv2.flip(imagem_entrada, 1)
        imagem_saida = cv2.flip(imagem_saida, 1)
    if args.v_flip and random.randint(0,1):
        imagem_entrada = cv2.flip(imagem_entrada, 0)
        imagem_saida = cv2.flip(imagem_saida, 0)

    return imagem_entrada, imagem_saida


# Atribuicao dos nomes das classes
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)



net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network, init_fn = construir_modelo.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, crop_width=args.recorte_comprimento, crop_height=args.recorte_altura, is_training=True)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))

opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()


if init_fn is not None:
    init_fn(sess)


model_checkpoint_name = "checkpoints/latest_model_" + args.model + "_" + args.dataset + ".ckpt"



train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)



avg_loss_per_epoca = []
avg_scores_per_epoca = []
avg_iou_per_epoca = []


val_indices = []
num_vals = min(args.num_val_images, len(val_input_names))


random.seed(16)
val_indices=random.sample(range(0,len(val_input_names)),num_vals)


 #treinamento
for epoca in range(args.epoca_inicio, args.num_epocas):

    current_losses = []

    cnt=0



    id_list = np.random.permutation(len(train_input_names))

    num_iters = int(np.floor(len(id_list) / args.batch_size))
    st = time.time()
    epoca_st=time.time()
    for i in range(num_iters):


        imagem_entrada_batch = []
        imagem_saida_batch = []


        for j in range(args.batch_size):
            index = i*args.batch_size + j
            id = id_list[index]
            imagem_entrada = utils.load_image(train_input_names[id])
            imagem_saida = utils.load_image(train_output_names[id])

            with tf.device('/cpu:0'):
                imagem_entrada, imagem_saida = aumento_dados(imagem_entrada, imagem_saida)


                # Pre processamento das imagens
                imagem_entrada = np.float32(imagem_entrada) / 255.0
                imagem_saida = np.float32(helpers.one_hot_it(label=imagem_saida, label_values=label_values))

                imagem_entrada_batch.append(np.expand_dims(imagem_entrada, axis=0))
                imagem_saida_batch.append(np.expand_dims(imagem_saida, axis=0))

        if args.batch_size == 1:
            imagem_entrada_batch = imagem_entrada_batch[0]
            imagem_saida_batch = imagem_saida_batch[0]
        else:
            imagem_entrada_batch = np.squeeze(np.stack(imagem_entrada_batch, axis=1))
            imagem_saida_batch = np.squeeze(np.stack(imagem_saida_batch, axis=1))


        _,current=sess.run([opt,loss],feed_dict={net_input:imagem_entrada_batch,net_output:imagem_saida_batch})
        current_losses.append(current)
        cnt = cnt + args.batch_size
        if cnt % 20 == 0:
            string_print = "epoca = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoca,cnt,current,time.time()-st)
            utils.LOG(string_print)
            st = time.time()

    mean_loss = np.mean(current_losses)
    avg_loss_per_epoca.append(mean_loss)

    # Salva checkpoints de validacao
    if not os.path.isdir("%s/%04d"%("checkpoints",epoca)):
        os.makedirs("%s/%04d"%("checkpoints",epoca))

    # salva ultimo arquivo de treinamento

    saver.save(sess,model_checkpoint_name)

    if val_indices != 0 and epoca % args.checkpoint_passo == 0:

        saver.save(sess,"%s/%04d/model.ckpt"%("checkpoints",epoca))


    if epoca % args.validacao_step == 0:

        target=open("%s/%04d/val_scores.csv"%("checkpoints",epoca),'w')
        target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))


        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []


        # Validacao
        for ind in val_indices:

            imagem_entrada = np.expand_dims(np.float32(utils.load_image(val_input_names[ind])[:args.recorte_altura, :args.recorte_comprimento]),axis=0)/255.0
            gt = utils.load_image(val_output_names[ind])[:args.recorte_altura, :args.recorte_comprimento]
            gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))



            imagem_saida = sess.run(network,feed_dict={net_input:imagem_entrada})


            imagem_saida = np.array(imagem_saida[0,:,:,:])
            imagem_saida = helpers.reverse_one_hot(imagem_saida)
            out_vis_image = helpers.colour_code_segmentation(imagem_saida, label_values)

            accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=imagem_saida, label=gt, num_classes=num_classes)

            file_name = utils.filepath_to_name(val_input_names[ind])
            target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
            for item in class_accuracies:
                target.write(", %f"%(item))
            target.write("\n")

            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)

            gt = helpers.colour_code_segmentation(gt, label_values)

            file_name = os.path.basename(val_input_names[ind])
            file_name = os.path.splitext(file_name)[0]
            cv2.imwrite("%s/%04d/%s_pred.png"%("checkpoints",epoca, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
            cv2.imwrite("%s/%04d/%s_gt.png"%("checkpoints",epoca, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))


        target.close()

        avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoca.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        avg_iou_per_epoca.append(avg_iou)

        print("\nValidacao media por epoca # %04d = %f"% (epoca, avg_score))
        print("Acuracia media de validacao por epoca # %04d:"% (epoca))
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names_list[index], item))
        print("V precision = ", avg_precision)
        print("V recall = ", avg_recall)
        print("V F1 score = ", avg_f1)
        print("V IoU score = ", avg_iou)

    epoca_time=time.time()-epoca_st
    remain_time=epoca_time*(args.num_epocas-1-epoca)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Tempo restante de treinamento = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Tempo restante : Treinamento completo.\n"
    utils.LOG(train_time)
    scores_list = []


    fig1, ax1 = plt.subplots(figsize=(11, 8))

    ax1.plot(range(epoca+1), avg_scores_per_epoca)
    ax1.set_title("Acuracia Media de Validacao x Epocas", size=20)
    ax1.set_xlabel("Epoca",fontsize=20)
    ax1.set_ylabel("Acuracia Media de Validacao",fontsize=20)


    plt.savefig('Acuracia_vs_Epocas.png')

    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))

    ax2.plot(range(epoca+1), avg_loss_per_epoca)
    ax2.set_title("Erro Medio x Epocas",size=20)
    ax2.set_xlabel("Epoca",fontsize=20)
    ax2.set_ylabel("Erro Atual",fontsize=20)

    plt.savefig('Erro_vs_Epocas.png')

    plt.clf()

    fig3, ax3 = plt.subplots(figsize=(11, 8))

    ax3.plot(range(epoca+1), avg_iou_per_epoca)
    ax3.set_title("Average IoU vs epocas")
    ax3.set_xlabel("epoca")
    ax3.set_ylabel("Current IoU")

    plt.savefig('iou_vs_epocas.png')



