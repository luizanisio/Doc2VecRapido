# Doc2VecRapido

==EM PREPARAÇÃO==

Componente python que simplifica o processo de criação de um modelo `Doc2Vec` [`Gensim 4.0.1`](https://radimrehurek.com/gensim/models/doc2vec.html) sem tatnos parâmetros de configuração como o [Doc2VecFacil](/Doc2VecFacil). Dicas de agrupamento de documentos similares, uso de `ElasticSearch` e `SingleStore`.
- se você não sabe o que é um modelo de similaridade, em resumo é um algoritmo não supervisionado para criar um modelo que transforma frases ou documentos em vetores matemáticos que podem ser comparados retornando um valor equivalente à similaridade semântica de documentos do mesmo contexto/domínio dos documentos usados no treinamento do modelo (doc2vec). Nesse cenário a máquina 'aprende' o vocabulário treinado e o contexto em que as palavras aparecem (word2vec), permitindo identificar a similaridade entre os termos, as frases e os documentos. O doc2vec amplia o treinamento do word2vec para frases ou documentos.
- alguns links para saber mais:
  - [`Paragraph Vector 2014`](https://cs.stanford.edu/~quocle/paragraph_vector.pdf) - a origem do Doc2Vec
  - [`Gensim 4.0.1 Doc2Vec`](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html) - documentação do framework
  - [`me Amilar 2018`](https://repositorio.idp.edu.br/handle/123456789/2635) - Dissertação de Mestrado IDP - Doc2Vec em documentos jurídicos
  - [`Representação Semântica Vetorial`](https://sol.sbc.org.br/index.php/erigo/article/view/7125) - Artigo UFG
  - [`Word2Vec Explained`](https://www.youtube.com/watch?v=yFFp9RYpOb0) - vídeo no Youtube
  - [`Word Embedding Explained and Visualized`](https://www.youtube.com/watch?v=D-ekE-Wlcds) - vídeo no Youtube
  - [`ti-exame`](https://www.ti-enxame.com/pt/python/como-calcular-similaridade-de-sentenca-usando-o-modelo-word2vec-de-gensim-com-python/1045257495/) - Word2Vec
  - [`Tomas Mikolov paper`](https://arxiv.org/pdf/1301.3781.pdf) - mais um artigo sobre o Doc2Vec

- Com essa comparação vetorial, é possível encontrar documentos semelhantes a um indicado ou [`agrupar documentos semelhantes`](https://github.com/luizanisio/Doc2VecFacil/blob/main/docs/agrupamento.md) entre si de uma lista de documentos (será disponibilizado um código específico para o Doc2VecRapido). Pode-se armazenar os vetores no `SingleStore` ou `ElasticSearch` para permitir uma pesquisa vetorial rápida e combinada com metadados dos documentos, como nas dicas [aqui](#dicas).

- Em um recorte do espaço vetorial criado pelo treinamento do modelo, pode-se perceber que documentos semelhantes ficam próximos enquanto documentos diferentes ficam distantes entre si. Então agrupar ou buscar documentos semelhantes é uma questão de identificar a distância vetorial dos documentos após o treinamento. Armazenando os vetores dos documentos no `ElasticSearch` ou `SingleStore`, essa tarefa é simplificada, permitindo construir sistemas de busca semântica com um esforço pequeno. Uma técnica parecida pode ser usada para treinar e armazenar vetores de imagens para encontrar imagens semelhantes, mas isso fica para outro projeto. Segue aqui uma [`view`](docs/readme_dicas.md) e uma [`procedure`](docs/readme_dicas.md) para busca de similares e agrupamentos no SingleStore.

![exemplo recorte espaço vetorial](./exemplos/img_recorte_espaco_vetorial.png?raw=true "Exemplo recorte de espaço vetorial")

- O uso da similaridade permite também um sistema sugerir rótulos para documentos novos muito similares a documentos rotulados previamente – como uma classificação rápida, desde que o rótulo esteja relacionado ao conteúdo geral do documento e não a informações externas a ele. Rotulação por informações muito específicas do documento pode não funcionar muito bem, pois detalhes do documento podem não ser ressaltados na similaridade semântica. 
- Outra possibilidade seria o sistema sugerir revisão de rotulação/classificação quando dois documentos possuem similaridades muito altas, mas rótulos distintos (como no exemplo do assunto A e B na figura abaixo), ou rótulos iguais para similaridades muito baixas (não é necessariamente um erro, mas sugere-se conferência nesses casos). Ou o sistema pode auxiliar o usuário a identificar rótulos que precisam ser revistos, quando rótulos diferentes são encontrados para documentos muito semelhantes e os rótulos poderiam ser unidos em um único rótulo, por exemplo. Essas são apenas algumas das possibilidades de uso da similaridade. 

![exemplo recorte espaço vetorial e assuntos](./exemplos/img_agrupamento_assuntos.png?raw=true "Exemplo recorte de espaço vetorial e assuntos")

> :bulb: Uma dica para conjuntos de documentos com pouca atualização, é fazer o cálculo da similaridade dos documentos e armazenar em um banco transacional qualquer para busca simples pelos metadados da similaridade. Por exemplo uma tabela com as colunas `seq_doc_1`, `seq_doc_2` e `sim` para todos os documentos que possuam similaridade acima de nn% a ser avaliado de acordo com o projeto. Depois basta fazer uma consulta simples para indicar documentos similares ao que o usuário está acessando, por exemplo.

- O core desse componente é o uso de um Tokenizador Inteligente que usa as configurações dos arquivos contidos na pasta do modelo para tokenizar os arquivos de treinamento e os arquivos novos para comparação no futuro (toda a configuração do tokenizador é opcional).
- Esse é um repositório de estudos. Analise, ajuste, corrija e use os códigos como desejar.
> :thumbsup: <sub> Agradecimentos especiais ao Miguel Angelo Neto do Paraná e ao me Amilar Martins por vários feedbacks contribuindo para a correção de bugs e a melhoria da documentação.</sub><br>

> :warning: <sub>A quantidade de documentos treinados e de épocas de treinamento são valores que dependem do objetivo e do tipo de texto de cada projeto.</sub><br>
> :warning: <sub>É importante lembrar que ao atualizar o modelo com mais épocas de treinamento ou mais documentos, todos os vetores gerados anteriormente e guardados para comparação no seu sistema devem ser atualizados. Uma dica é criar uma tabela nova no SingleStore ou uma coluna nova no ElasticSearch e, após a geração dos novos vetores, fazer a atualização em bloco substituindo os vetores antigos pelos novos.</sub>

### As etapas de um treinamento são simples:
1) reservar um volume de documentos que represente a semântica que será treinada. Então o primeiro passo é extrair e separar em uma pasta os documentos que serão usados no treinamento. É interessante que sejam documentos “texto puro” (não ocerizados), mas não impede que sejam usados documentos ocerizados na falta de documentos “texto puro”. Com textos com muito ruído, como em textos ocerizados, o vocabulário "aprendido" pode não ser tão eficiente.
2) preparar o ambiente python caso ainda não tenha feito isso: [`anaconda`](https://www.anaconda.com/) + [`requirements`](./src/requirements.txt)
3) baixar os arquivos do [`projeto`](./src/) 
4) baixar um [`modelo`](./exemplos/) ou criar a sua estrutura de pastas
5) rodar o treinamento e explorar os recursos que o uso do espaço vetorial permite
> :bulb: <sub> Nota: Esse é o [tutorial oficial do gensim](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#introducing-paragraph-vector), a ideia do componente é simplificar a geração e uso do modelo treinado, mas não há nada muito especial se comparado aos códigos da documentação. </sub>

<hr>

## Você pode configurar alguns parâmetros antes do treinamento para o `Doc2VecRapido`:
 - criar a pasta do modelo (por exemplo "meu_modelo") e criar o arquivo `config.json` com os seguintes parâmetros.
  - `config.json = {"vector_size": 300, "strip_numeric":true, "stemmer":false, "min_count": 5 , "token_br": true}`
  - strip_numeric = remove números (padrão true)
  - stemmer = utiliza o stemmer dos tokens (padrão false)
  - min_count = ocorrência mínima do token no corpus para ser treinado (padrão 5)
  - token_br = cria o token #br para treinar simulando a quebra de parágrafos (padrão true)
  - vector_size = número de dimensões do vetor que será treinado (padrão 300)

 - treinamento do modelo usando a estrutura de tokenização criada 
   - `python util_doc2vec_rapido.py -pasta ./meu_modelo -treinar` -textos ./textos
   - o modelo será gravado a cada 50 iterações para continuação do treino se ocorrer alguma interrupção
> 💡 <sub>Nota: para interromper o treino sem correr o risco corromper o modelo durante a gravação, basta criar um arquivo `meu_modelo/parar.txt` na pasta do modelo que o treinamento será interrompido ao final da iteração em andamento.</sub>

 - carregando o modelo para comparação
 ```python 
   from util_doc2vec_rapido import Doc2VecRapido
   dv = Doc2VecRapido(pasta_modelo = 'minha_pasta')
   texto_1 = 'esse é um texto de teste para comparação'
   texto_2 = 'esse outro texto de teste para uma nova comparação'
   sim = 100*dv.similaridade(texto_1, texto_2)
   print(f'Similaridade texto 1 e 2: {sim:.2f}')       
```    
Resultado (a similaridade vai depender dos documentos usados no treinamento):
```
  Similaridade texto 1 e 2: 83.25%
```

## Dicas de uso: <a name="dicas">
- gravar os vetores, textos e metadados dos documentos no [`ElasticSearch`](https://www.elastic.co/pt/), e usar os recursos de pesquisas: More Like This, vetoriais e por proximidade de termos como disponibilizado no componente [`PesquisaElasticFacil`](https://github.com/luizanisio/PesquisaElasticFacil) ou criar sua própria estrutura de dados com [`essas dicas`](https://github.com/luizanisio/PesquisaElasticFacil/blob/main/docs/ElasticQueries.md).
- gravar os vetores, textos e metadados no [`SingleStore`](https://www.singlestore.com/) e criar views de similaridade para consulta em tempo real dos documentos inseridos na base, incluindo filtros de metadados e textuais como nos exemplos disponíveis aqui: [`dicas SingleStore`](./docs/readme_dicas.md).
