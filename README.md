# Desafio

Neste desafio, você deve analisar um conjunto de fotos de um trecho de uma partida de futebol (disponibilizado em anexo) e capturar os jogadores em campo. Para alcançar esse objetivo, siga os passos abaixo:
1. Identifique os jogadores por meio de visão computacional, preferencialmente utilizando OpenCV.
2. Registre as posições dos jogadores em um mapeamento bidimensional (com coordenadas x e y).
3. Gere uma tabela contendo o ID do(s) jogador(es) e a posição de cada um dos atletas.

## Requisitos
* A implementação deve ser feita em Python.
* Utilize bibliotecas atuais, com as dependências ajustadas.
* Inclua a biblioteca OpenCV em sua implementação.
* Você pode escolher entre utilizar ou não bibliotecas como TensorFlow/Keras ou YOLO.
* É permitido usar um modelo já treinado.
* Por favor, descreva sua abordagem para resolver este desafio e forneça o código-fonte, juntamente com instruções detalhadas sobre como executá-lo.

# Solução
A solução tomou como base a sequencia indicada no desafio, sendo que a primeira etapa foi a identificação dos jogadores, a segunda a captura das posições e a terceira a geração da tabela com os dados. 

## Ambiente
Para a execução do projeto, foi utilizado o MacOS, com o Python 3.10.9, e as biliotecas presentes no arquivo `requirements.txt`.

Para facilitar a execução do projeto, foi criado um ambiente virtual, utilizando o [virtualenv], e instalado as dependências utilizando o [pip].

### Execução em ambiente virtual
Para executar o projeto em um ambiente virtual, é necessário instalar o [virtualenv], e então executar os seguintes comandos:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```


## Identificação dos jogadores
Para a identificação dos jogadores foi utilizado o modelo YOLOv3, que é um modelo de detecção de objetos, que foi treinado com o dataset COCO. O modelo foi treinado para identificar 80 classes de objetos, dentre elas, pessoas.

Foi possível perceber que de maneira geral, o modelo conseguiu identificar corretamente os jogadores, porém, em alguns casos, ele identificou objetos que não eram jogadores, como por exemplo, o juiz de linha, ou mesmo outras pessoas que estavam foram do campo.
Este problema pode ser resolvido facilmente após termos a tabela com os dados, pois podemos filtrar os dados e remover os jogadores que não estiverem dentro do campo.


![](docs/yol.png)

Além disso, é possível utilizar a cor da camiseta para identificar jogadores de cada time, ou o juiz.

> Etapa disponível em: `python3 yol.py Teste_1.jpg`

## Mapeamento dos jogadores
Primeiramente, com base nas bounding boxes geradas pelo modelo YOLOv3, foi possível identificar a posição dos jogadores em relação a imagem. Para isso, foi utilizado o centro inferior da bounding box, que é o ponto mais baixo do jogador (seus pés). Visto que este é provavelmente o ponto menos distorcido da imagem, e que a posição do jogador em relação a imagem é a mesma que a posição do jogador em relação ao campo.

### Estimativa de Homografia
Para identificar a posição dos jogadores em relação ao campo, seria necessário estimar a homografia entre a imagem e o campo.

#### Linhas principais do campo
Inicialmente realizei processamentos simples na imagem para destacar as linhas principais do campo.

![](docs/linhas.png)

> Etapa disponível em: `python3 lines.py Teste_1.jpg`

Para a estimativa de homografia, o desafio indicava uma rede neural já treinada, apresentada no artigo [narya], porém este projeto necessita de uma biblioteca chamada [mxnet], que não é compativel com MacOS.

Tentei utilizar outras abordagens, como a biblioteca [scikit-image], o método [RANSAC] e também o método de classificação [KNN], porém, nenhum deles foi capaz de identificar os pontos de interesse com precisão.
Imagino que com mais tempo, e um bom banco de dados de treinamento, seria possível obter bons resultados com uma [CNN] específica para este problema.

Como alternativa, resolvi utilizar por enquanto uma classificação manual, onde o usuário iria clicar nos pontos de interesse, e o programa iria calcular a homografia. Isto como uma solução temporária, até que eu consiga implementar uma solução mais robusta e automatizada.

![](docs/pontos.png)

O vídeo ![docs/manual.mov](docs/manual.mov) mostra o processo manual de estimativa de homografia.





