import os, time, cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np

from utilitario import utils, helpers
from construcao import construir_modelo

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=None, required=True,help='Caminho do modelo salvo')
parser.add_argument('--recorte_altura', type=int, default=512, help='Altura do recorte')
parser.add_argument('--recorte_comprimento', type=int, default=512, help='Comprimento do recorte')
parser.add_argument('--model', type=str, default="FRRN-A", required=False, help='Modelo')
parser.add_argument('--dataset', type=str, default="Crop_Weed", required=False, help='Dataset')
args = parser.parse_args()



class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

# Inicializacao da rede
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

net_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
net_output = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])

network, _ = construir_modelo.build_model(args.model, net_input=net_input, num_classes=num_classes,
                                       crop_width=args.recorte_comprimento, crop_height=args.recorte_altura, is_training=False)

sess.run(tf.global_variables_initializer())


saver = tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)



train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(
    dataset_dir=args.dataset)

# Create directories if needed
if not os.path.isdir("%s" % ("Teste")):
    os.makedirs("%s" % ("Teste"))

target = open("%s/test_scores.csv" % ("Teste"), 'w')
target.write("test_name, test_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))
scores_list = []
class_scores_list = []
precision_list = []
recall_list = []
f1_list = []
iou_list = []
run_times_list = []


for ind in range(len(test_input_names)):
    sys.stdout.write("\rImagem de teste %d / %d" % (ind + 1, len(test_input_names)))
    sys.stdout.flush()

    imagem_entrada = np.expand_dims(
        np.float32(utils.load_image(test_input_names[ind])[:args.recorte_altura, :args.recorte_comprimento]), axis=0) / 255.0
    gt = utils.load_image(test_output_names[ind])[:args.recorte_altura, :args.recorte_comprimento]
    gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

    st = time.time()
    imagem_saida = sess.run(network, feed_dict={net_input: imagem_entrada})

    run_times_list.append(time.time() - st)

    imagem_saida = np.array(imagem_saida[0, :, :, :])
    imagem_saida = helpers.reverse_one_hot(imagem_saida)
    out_vis_image = helpers.colour_code_segmentation(imagem_saida, label_values)

    accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=imagem_saida, label=gt,
                                                                                 num_classes=num_classes)

    file_name = utils.filepath_to_name(test_input_names[ind])
    target.write("%s, %f, %f, %f, %f, %f" % (file_name, accuracy, prec, rec, f1, iou))
    for item in class_accuracies:
        target.write(", %f" % (item))
    target.write("\n")

    scores_list.append(accuracy)
    class_scores_list.append(class_accuracies)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)
    iou_list.append(iou)

    gt = helpers.colour_code_segmentation(gt, label_values)

    cv2.imwrite("%s/%s_pred.png" % ("Teste", file_name), cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
    cv2.imwrite("%s/%s_gt.png" % ("Teste", file_name), cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

target.close()

avg_score = np.mean(scores_list)
class_avg_scores = np.mean(class_scores_list, axis=0)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_iou = np.mean(iou_list)
avg_time = np.mean(run_times_list)
print("M. teste acuracia = ", avg_score)
print("M tete por classe = \n")
for index, item in enumerate(class_avg_scores):
    print("%s = %f" % (class_names_list[index], item))
print("M. precisao = ", avg_precision)
print("M sensibilidade = ", avg_recall)

print("M. IoU  = ", avg_iou)
print("M tempo = ", avg_time)
