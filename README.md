# SEGMENTAÇÃO SEMÂNTICA PARA AGRICULTURA DE PRECISÃO COM REDES NEURAIS

## Descrição
Este reporisório consiste nos códigos utilizados para a implementação de uma rede neural convolucional, utilizado para a identificação de ervas daninhas por meio da segmentação semântica. 

O repositório apresenta a implementação da arquitetura de rede - [Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes](https://arxiv.org/abs/1611.08323) baseado no repositório : https://github.com/GeorgeSeif/Semantic-Segmentation-Suite.

Também foi adicionado o arquivo do Jupyter Notebook "Segmentacao_Semantica.ipynb" com os códigos em alto nível para o treinamento, teste e predição utilizando a interface do Google Colab.

## Dataset
O Conjunto de dados utilzado para avaliar a rede foi -[A Crop/Weed Field Image Dataset for the Evaluation of Computer Vision Based Precision Agriculture Tasks](https://github.com/cwfid/dataset)

## Instalação
O projeto tem as seguintes dependências:

- Numpy `sudo pip install numpy`

- OpenCV Python `sudo apt-get install python-opencv`

- TensorFlow `sudo pip install --upgrade tensorflow-gpu`

## Resultados
-Treinamento

![alt text-1](https://github.com/giarettasg/FRRN_Segmentacao_Agricultura/blob/master/Resultados/ErrovsEpocas.png) |  ![alt text-2](https://github.com/giarettasg/FRRN_Segmentacao_Agricultura/blob/master/Resultados/Evolucao.png)





