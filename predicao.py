import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np

from utilitario import utils, helpers
from construcao import construir_modelo

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None, required=True, help='Nome da imagem para predicao')
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='Caminho do modelo salvo')
parser.add_argument('--recorte_altura', type=int, default=512, help='Altura do recorte')
parser.add_argument('--recorte_comprimento', type=int, default=512, help='Comprimento do recorte')
parser.add_argument('--model', type=str, default="FRRN-A", required=False, help='Modelo')
parser.add_argument('--dataset', type=str, default="Crop_Weed", required=False, help='Dataset')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)


# Inicializacao da rede
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network, _ = construir_modelo.build_model(args.model, net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=args.recorte_comprimento,
                                        crop_height=args.recorte_altura,
                                        is_training=False)

sess.run(tf.global_variables_initializer())


saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)


print("Testando imagem " + args.image)

loaded_image = utils.load_image(args.image)
resized_image =cv2.resize(loaded_image, (args.recorte_comprimento, args.recorte_altura))
imagem_entrada = np.expand_dims(np.float32(resized_image[:args.recorte_altura, :args.recorte_comprimento]),axis=0)/255.0

st = time.time()
imagem_saida = sess.run(network,feed_dict={net_input:imagem_entrada})

run_time = time.time()-st

imagem_saida = np.array(imagem_saida[0,:,:,:])
imagem_saida = helpers.reverse_one_hot(imagem_saida)

out_vis_image = helpers.colour_code_segmentation(imagem_saida, label_values)
file_name = utils.filepath_to_name(args.image)
cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

print("")
print("Terminado")

