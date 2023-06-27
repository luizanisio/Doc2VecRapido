## Doc2VecRapido, Doc2BertRapido e AgrupamentoRapido

### Vetorização para agrupamento e similaridade semântica com Doc2Vec, BERTimbau e LLms

Classe em python que simplifica o processo de criação de um modelo `Doc2Vec` [`Gensim 4.0.1`](https://radimrehurek.com/gensim/models/doc2vec.html) sem tantos parâmetros de configuração como o [Doc2VecFacil](/Doc2VecFacil), mas já traz um resultado excelente em vários contextos. Também tem algumas dicas de agrupamento de documentos similares, uso de `ElasticSearch` e `SingleStore`.<br>

Agora com a alternativa [Doc2BertRapido](./src/util_doc2bert_rapido.py) que permite usar modelos do Bert, como o [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) para vetorizar e agrupar os documentos sem a necessidade de treinar um modelo específico. <i>(estou atualizando a documentação e incluindo os códigos de exemplo de fine tune para domínios específicos como psicologia, jurídico, tecnologia etc)</i><br>
:warning: Estou trabalhando no código para treinar a similaridade do `BERTimbau` ou outras LLMs como o `ult5-pt-small` (com 1024 tokens de entrada e bem leve) e o `
gtr-t5-xxl` (muito bom para similaridade em PTBR, mas precisa de muita máquina). Em breve disponibilizo para uso no agrupamento com modelos prontos ou personalizados.

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

- Com essa comparação vetorial, é possível encontrar documentos semelhantes a um indicado ou [`agrupar documentos semelhantes`](./README_agrupamento.md) entre si de uma lista de documentos (disponível para uso com o Doc2VecRapido ou o Doc2BertRapido). Pode-se armazenar os vetores no `SingleStore` ou `ElasticSearch` para permitir uma pesquisa vetorial rápida e combinada com metadados dos documentos, como nas dicas [aqui](#dicas).

- Em um recorte do espaço vetorial criado pelo treinamento do modelo, pode-se perceber que documentos semelhantes ficam próximos enquanto documentos diferentes ficam distantes entre si. Então agrupar ou buscar documentos semelhantes é uma questão de identificar a distância vetorial dos documentos após o treinamento. Armazenando os vetores dos documentos no `ElasticSearch` ou `SingleStore`, essa tarefa é simplificada, permitindo construir sistemas de busca semântica com um esforço pequeno. Uma técnica parecida pode ser usada para treinar e armazenar vetores de imagens para encontrar imagens semelhantes, mas isso fica para outro projeto. Segue aqui uma [`view`](docs/README_siglestore.md) e uma [`procedure`](docs/README_siglestore.md) para busca de similares e agrupamentos no SingleStore.

![exemplo recorte espaço vetorial](./exemplos/img_recorte_espaco_vetorial.png?raw=true "Exemplo recorte de espaço vetorial")

- O uso da similaridade permite também um sistema sugerir rótulos para documentos novos muito similares a documentos rotulados previamente – como uma classificação rápida, desde que o rótulo esteja relacionado ao conteúdo geral do documento e não a informações externas a ele. Rotulação por informações muito específicas do documento pode não funcionar muito bem, pois detalhes do documento podem não ser ressaltados na similaridade semântica. 
- Outra possibilidade seria o sistema sugerir revisão de rotulação/classificação quando dois documentos possuem similaridades muito altas, mas rótulos distintos (como no exemplo do assunto A e B na figura abaixo), ou rótulos iguais para similaridades muito baixas (não é necessariamente um erro, mas sugere-se conferência nesses casos). Ou o sistema pode auxiliar o usuário a identificar rótulos que precisam ser revistos, quando rótulos diferentes são encontrados para documentos muito semelhantes e os rótulos poderiam ser unidos em um único rótulo, por exemplo. Essas são apenas algumas das possibilidades de uso da similaridade. 

![exemplo recorte espaço vetorial e assuntos](./exemplos/img_agrupamento_assuntos.png?raw=true "Exemplo recorte de espaço vetorial e assuntos")

> :bulb: <sub>Uma dica para conjuntos de documentos com pouca atualização, é fazer o cálculo da similaridade dos documentos e armazenar em um banco transacional qualquer para busca simples pelos metadados da similaridade. Por exemplo uma tabela com as colunas `seq_doc_1`, `seq_doc_2` e `sim` para todos os documentos que possuam similaridade acima de nn% a ser avaliado de acordo com o projeto. Depois basta fazer uma consulta simples para indicar documentos similares ao que o usuário está acessando, por exemplo.</sub>

- Esse é um repositório de estudos. Analise, ajuste, corrija e use os códigos como desejar.
> :warning: <sub>A quantidade de documentos treinados e de épocas de treinamento são valores que dependem do objetivo e do tipo de texto de cada projeto.</sub><br>
> :warning: <sub>É importante lembrar que ao atualizar o modelo com mais épocas de treinamento ou mais documentos, todos os vetores gerados anteriormente e guardados para comparação no seu sistema devem ser atualizados. Uma dica é criar uma tabela nova no SingleStore ou uma coluna nova no ElasticSearch e, após a geração dos novos vetores, fazer a atualização em bloco substituindo os vetores antigos pelos novos.</sub>

### As etapas de um treinamento são simples:
1) reservar um volume de documentos que represente a semântica que será treinada. Então o primeiro passo é extrair e separar em uma pasta os documentos que serão usados no treinamento. É interessante que sejam documentos “texto puro” (não ocerizados), mas não impede que sejam usados documentos ocerizados na falta de documentos “texto puro”. Com textos com muito ruído, como em textos ocerizados, o vocabulário "aprendido" pode não ser tão eficiente.
2) preparar o ambiente python caso ainda não tenha feito isso: [`anaconda`](https://www.anaconda.com/) + [`requirements`](./src/requirements.txt)
3) baixar os arquivos do [`projeto`](./src/) 
4) preparar um conjunto de textos como no exemplo [`textos_legislacoes.zip`](./exemplos/) 
5) rodar o treinamento e explorar os recursos que o uso do espaço vetorial permite
> :bulb: <sub> Nota: Esse é o link do [tutorial oficial do gensim](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#introducing-paragraph-vector), a ideia do componente é simplificar a geração e uso do modelo treinado, mas não há nada muito especial se comparado aos códigos da documentação. </sub>

<hr>

## Você pode configurar alguns parâmetros antes do treinamento para o `Doc2VecRapido`:
 - criar a pasta do modelo (por exemplo "meu_modelo") e criar o arquivo `config.json` com os seguintes parâmetros.
   - `config.json = {"vector_size": 300, "strip_numeric":true, "min_count": 5 , "token_br": true}`
   - **strip_numeric** = remove números (padrão true)
   - **stemmer** = utiliza o stemmer dos tokens para treinamento (padrão false)
   - **min_count** = ocorrência mínima do token no corpus para ser treinado (padrão 5)
   - **token_br** = cria o token #br para treinar simulando a quebra de parágrafos (padrão true)
   - **vector_size** = número de dimensões do vetor que será treinado (padrão 300)
   - **window** = a distância máxima entre a palavra atual e a prevista em uma frase (padrão 5)
   - **skip_gram** = True/False se o treinamento será feito com skip-gram
   - **skip_gram_window** = a distância máxima entre a palavra atual e a prevista usando skip-gram (padrão 10)
   - **max_total_epocas** = número máximo de épocas para treinar (facilita para o caso de desejar completar até um valor treinando parcialmente - padrão 0 = sem limite)
> :bulb: <sub> Nota: Você pode criar o arquivo `stopwords.txt` e colocar uma lista de palavras que serão excluídas dos textos durante o treinamento. Essas palavras não serão "vistas" pelo modelo na leitura dos textos. Se quiser criar o arquivo de configuração padrão para ajustar antes do treinamento, use o parâmetro `-config` como no exemplo: `python util_doc2vec_rapido.py -pasta ./meu_modelo -config`.</sub>

## Arquivo de dados e exemplos de código
- alés dos códigos abaixo, foi disponibilizado o arquivo [dados_exemplos.zip](https://github.com/luizanisio/Doc2VecRapido/blob/main/exemplos/dados_exemplos.zip) com textos públicos de exemplo para treinamento e dataframes de exemplo de treinamento. E o arquivo [exemplos.py](https://github.com/luizanisio/Doc2VecRapido/blob/main/src/exemplos.py) com um exemplo para cada cenário de treinamento e preparação de dados.

### Treinamento do modelo usando a estrutura de tokenização criada 
   - `python util_doc2vec_rapido.py -pasta ./meu_modelo -textos ./textos -epocas 1000`
   - o modelo será gravado a cada 50 iterações para continuação do treino se ocorrer alguma interrupção
   - durante o treinamento o arquivo de configuração será atualizado com a chave `log_treino_epocas` (total de épocas treinadas até o momento) e `log_treino_vocab` (número de termos usados no vocabulário do modelo).
   - ao final do treinamento serão criados dois arquivos para consulta: 
     - `vocab_treinado.txt` com os termos treinados 
     - `vocab_similares.txt` com alguns termos e os termos mais similares a eles.

### Finetunning de uma LLM (Bert, LongBert, T5)
- python util_treinallm_rapido.py -pasta ./meut5br -base t5br -textos ./textos_sim -epocas 5
> :bulb: <sub> Nota: Finalizando os testes e disponibilizo em breve.</sub>

### Testando o modelo (vai carregar o modelo e comparar alguns textos internos)
  - `python util_doc2vec_rapido.py -pasta ./meu_modelo`
  - `python util_doc2llm_rapido.py -pasta T5BR`

Resultado: 
```
 >>>> TESTE DO MODELO <<<<
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
- Texto 01:   esse é um texto de teste para comparação - o teste depende de existirem os termos no vocab treinado
  > tokens:  esse|um|texto|teste|comparação|teste|depende|existirem|os|termos|vocab|treinado
- Texto 02:   esse outro texto de teste para uma nova comparação - lembrando que o teste depende de existirem os termos no vocab treinado
  > tokens:  esse|outro|texto|teste|uma|nova|comparação|lembrando|que|teste|depende|existirem|os|termos|vocab|treinado
- Texto 03:   esse é um texto de teste para comparação \n o teste depende de existirem os termos no vocab treinado
  > tokens:  esse|um|texto|teste|comparação|#br|teste|depende|existirem|os|termos|vocab|treinado
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
Similaridade entre o texto 01 e ele mesmo: 98.73%
Similaridade entre o texto 01 e ele com oov: 98.81%
Similaridade entre os textos 01 e 02: 86.25%
Similaridade entre os textos 01 e 03: 98.88%
```

### Carregando o modelo Doc2Vec para comparação
 ```python 
   from util_doc2vec_rapido import Doc2VecRapido
   dv = Doc2VecRapido(pasta_modelo = 'minha_pasta')
   texto_1 = 'esse é um texto de teste para comparação'
   texto_2 = 'esse outro texto de teste para uma nova comparação'
   sim = 100 * dv.similaridade(texto_1, texto_2)
   print(f'Similaridade texto 1 e 2: {sim:.2f}')       
```    

- Resultado: `Similaridade texto 1 e 2: 83.25%`

### Carregando um modelo Transformers para comparação
- alguns mapeados com constantes:
  - `T5BR` = 'tgsc/sentence-transformer-ult5-pt-small'
    - <sub>200Mb 1024 tokens entrada - 512 dimensões de saída - 51Mi param</sub>
  - `GTRT5XXL` = 'sentence-transformers/gtr-t5-xxl'
    - <sub>*treinado similaridade semântica* - 9.1 Gb - 512 tokens entrada - 768 dimensões de saída - 11Bi param</sub>
  - `BERT` ou `BERT_LARGE` = 'neuralmind/bert-large-portuguese-cased'
    - <sub>2.5 Gb 512 tokens entrada - 1024 dimensões de saída - 334Mi param</sub>
  - `BERT_BASE` = 'neuralmind/bert-base-portuguese-cased'
    - <sub>1.3 Gb 512 tokens entrada - 768 dimensões de saída - 109Mi param</sub>
  - `BERT_4K` = 'allenai/longformer-base-4096'
    - <sub>600Mb 4096 tokens e entrada - 768 dimensões de saída - 148Mi param</sub>
  - outros no código (classe Doc2LLMRapido)
  - Opções para carregar:
    - dv = Doc2LLMRapido(modelo = 'meut5br') # busca a pasta 'meut5br'
    - dv = Doc2LLMRapido(modelo = 'T5BR')  # busca pela constante T5BR
    - dv = Doc2LLMRapido(modelo = Doc2LLMRapido.T5BR)  # busca pela constante T5BR
    - dv = Doc2LLMRapido(modelo = Doc2LLMRapido.BERT)  # busca pela constante BERT = Bert Large

 ```python 
   from util_doc2llm_rapido import Doc2LLMRapido
   dv = Doc2LLMRapido(modelo = 'meut5br')
   texto_1 = 'esse é um texto de teste para comparação'
   texto_2 = 'esse outro texto de teste para uma nova comparação'
   sim = 100 * dv.similaridade(texto_1, texto_2)
   print(f'Similaridade texto 1 e 2: {sim:.2f}')       
```    

- Resultado: `Similaridade texto 1 e 2: 85.32%`

### Mostrando o vetor do texto
 ```python 
   from util_doc2vec_rapido import Doc2VecRapido
   dv = Doc2VecRapido(pasta_modelo = 'minha_pasta')
   texto_1 = 'esse é um texto de teste para comparação'
   vetor = dv.vetor(texto_1)
   print(f'Vetor do texto 1: {vetor}')       
```    
- Resultado (300 números do vetor): 
```
[0.012920759618282318, -0.04087100550532341, .. 0.00844051968306303, -0.029573174193501472]
```

### Treinando o modelo pelo código
- os parâmetros passados pelas constantes são os obrigatórios
 ```python 
   from util_doc2vec_rapido import Doc2VecRapido
   PASTA_MODELO = './meu_modelo'
   PASTA_TEXTOS = './meus_textos'
   EPOCAS = 200
   dv = Doc2VecRapido(pasta_modelo=PASTA_MODELO, 
                      documentos=PASTA_TEXTOS, 
                      epochs=EPOCAS, 
                      skip_gram = True,
                      stemmer = False,
                      strip_numeric = True,
                      arq_tagged_docs = None,
                      workers = 30 )
   dv.treinar()
   print('Treino concluído')
   texto_1 = 'esse é um texto de teste para comparação'
   texto_2 = 'esse é outro texto para comparação'
   sim = dv.similaridade(texto_1, texto_2)
   print(f'Similaridade entre os textos: {sim}')       
```    
- o parâmetro **arq_tagged_docs** pode ser usado para um conjunto muito grande de documentos que não cabem na memória. 
  - Você pode criar um arquivo `documentos.json`, por exemplo, e colocar em cada linha uma chave tokens e, opcionalmente, uma chave tags. 
  - Cada linha deve conter um json válido: {"tokens": ["token1","token2" ....], "tags" : ["tag1","tag2"]}
  - Os documentos serão iterados em blocos de **bloco_tagged_docs** textos de cada vez para que o treinamento ocorra de forma rápida e com pouca memória.
  - É necessário que os documentos já estejam processados para tornar o treinamento mais eficiente. 
  - Abaixo um exemplo de como preparar o arquivo com tokens processados.

### Criando um arquivo de treino gigante com tokens processados
- você pode pré-processar um volume grande de textos para um treinamento longo e com milhares/milhões de arquivos
- você pode ter um ou vários dataframes com uma coluna texto e outra de tags (tags é opcional)
```python
    from util_doc2vec_rapido import Doc2VecRapido, DocumentosDoArquivo
    import pandas as pd

    arq1 = './dataframes/arquivo.feather'
    arq2 = './dataframes/outro_arquivo.json'
    df1 = pd.read_feather(arq1)
    df2 = pd.read_json(arq2, lines=True)
    # nome do arquivo de treino que será gerado
    arq_saida = './dataframes/textos_para_treino.json'
    # apaga o arquivo de saída se existir
    if os.path.isfile(arq_saida):
        os.remove(arq_saida)
    # aqui o arquivo config já é criado para o modelo 
    # na pasta indicada com os parâmetros do tokenizador
    dv = Doc2VecRapido(pasta_modelo='./meu_modelo', 
                       strip_numeric=True, 
                       stemmer=False, 
                       skip_gram=True)        
    docs = DocumentosDoArquivo(arq_saida, dv)
    # aqui os dataframes são processados e inseridos no arquivo de treino    
    docs.incluir_textos([df1,df2], coluna_texto='texto', coluna_tags='rotulos')    
    print(f'Arquivo de treino criado: ', arq_saida)```

## Criando um arquivo de treino gigante de uma pasta de documentos
- a estratégia de rótulos de cada arquivo pode ser definida pelo nome do arquivo, pode ter outro arquivo associado ou deixar sem rótulos

```python
    from util_doc2vec_rapido import Doc2VecRapido,DocumentosDoArquivo
    from util_doc2util import UtilDocs
    tags_aleatorias = ['Rótulo A', 'Rótulo B', 'Rótulo C']
    textos = UtilDocs.carregar_documentos('./textos_grupos')
    # aqui o arquivo config já é criado para o modelo 
    # na pasta indicada com os parâmetros do tokenizador
    dv = Doc2VecRapido(pasta_modelo='./meu_modelo', 
                       strip_numeric=True, 
                       stemmer=False, 
                       skip_gram=True)        
    # aqui os textos são processados e inseridos no arquivo de treino
    arq_saida = './textos/textos_para_treino.json'
    # apaga o arquivo de saída se existir
    if os.path.isfile(arq_saida):
        os.remove(arq_saida)
    docs = DocumentosDoArquivo(arq_saida, dv)
    for texto in textos:
        docs.incluir_texto(texto, tags = [random.choice(tags_aleatorias)])
    print(f'Arquivo de treino criado: ', arq_saida)
```

### Treinando com um arquivo de treino pronto e com todos os textos tokenizados
- o arquivo de treinamento estará pronto para ser usado no treinamento.
- é necessário passar o objeto dv carregado com a tokenização desejada. Então é interessante ter o arquivo `config.json` na pasta do modelo ou passar os parâmetros de tokenização para que a chamada crie o arquivo config.json, permitindo a mesma configuração de tokenização no treino e nas futuras predições.
 ```python 
   from util_doc2vec_rapido import Doc2VecRapido
   PASTA_MODELO = './meu_modelo'
   EPOCAS = 50
   # esse arquivo pode ter milhões de linhas
   arquivo_treino = './dataframes/textos_para_treino.json'
   # entendendo que o arquivo config já existe com as informações
   # da tokenização e bloco_tagged_docs pode ter o valor que caiba na memória
   dv = Doc2VecRapido(pasta_modelo=PASTA_MODELO, 
                      arq_tagged_docs= arquivo_treino,
                      bloco_tagged_docs= 50000,
                      epochs=EPOCAS,
                      workers = 30 )
   dv.treinar()
   print('Treino concluído')
   texto_1 = 'esse é um texto de teste para comparação'
   texto_2 = 'esse é outro texto para comparação'
   sim = dv.similaridade(texto_1, texto_2)
   print(f'Similaridade entre os textos: {sim}')       
```    

## Dicas de uso: <a name="dicas">
- gravar os vetores, textos e metadados dos documentos no [`ElasticSearch`](https://www.elastic.co/pt/), e usar os recursos de pesquisas: More Like This, vetoriais e por proximidade de termos como disponibilizado no componente [`PesquisaElasticFacil`](https://github.com/luizanisio/PesquisaElasticFacil) ou criar sua própria estrutura de dados com [`essas dicas`](https://github.com/luizanisio/PesquisaElasticFacil/blob/main/docs/ElasticQueries.md).
- gravar os vetores, textos e metadados no [`SingleStore`](https://www.singlestore.com/) e criar views de similaridade para consulta em tempo real dos documentos inseridos na base, incluindo filtros de metadados e textuais como nos exemplos disponíveis aqui: [`dicas SingleStore`](./README_siglestore.md).
