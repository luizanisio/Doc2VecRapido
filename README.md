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

> :bulb: <sub>Uma dica para conjuntos de documentos com pouca atualização, é fazer o cálculo da similaridade dos documentos e armazenar em um banco transacional qualquer para busca simples pelos metadados da similaridade. Por exemplo uma tabela com as colunas `seq_doc_1`, `seq_doc_2` e `sim` para todos os documentos que possuam similaridade acima de nn% a ser avaliado de acordo com o projeto. Depois basta fazer uma consulta simples para indicar documentos similares ao que o usuário está acessando, por exemplo.</sub>

- O core desse componente é o uso de um Tokenizador Inteligente que usa as configurações dos arquivos contidos na pasta do modelo para tokenizar os arquivos de treinamento e os arquivos novos para comparação no futuro (toda a configuração do tokenizador é opcional).
- Esse é um repositório de estudos. Analise, ajuste, corrija e use os códigos como desejar.
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
   - **strip_numeric** = remove números (padrão true)
   - **stemmer** = utiliza o stemmer dos tokens (padrão false)
   - **min_count** = ocorrência mínima do token no corpus para ser treinado (padrão 5)
   - **token_br** = cria o token #br para treinar simulando a quebra de parágrafos (padrão true)
   - **vector_size** = número de dimensões do vetor que será treinado (padrão 300)
   - **window** = a distância máxima entre a palavra atual e a prevista em uma frase (padrão 10)

 - treinamento do modelo usando a estrutura de tokenização criada 
   - `python util_doc2vec_rapido.py -pasta ./meu_modelo -treinar -textos ./textos`
   - o modelo será gravado a cada 50 iterações para continuação do treino se ocorrer alguma interrupção
   - durante o treinamento o arquivo de configuração será atualizado com a chave `log_treino_epocas` (total de épocas treinadas até o momento) e `log_treino_vocab` (número de termos usados no vocabulário do modelo).
   - ao final do treinamento serão criados dois arquivos para consulta: 
     - `vocab_treinado.txt` com os termos treinados 
     - `vocab_similares.txt` com alguns termos e os termos mais similares a eles.
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

- mostrando o vetor do texto
 ```python 
   from util_doc2vec_rapido import Doc2VecRapido
   dv = Doc2VecRapido(pasta_modelo = 'minha_pasta')
   texto_1 = 'esse é um texto de teste para comparação'
   vetor = dv.vetor(texto)
   print(f'Vetor do texto 1: {vetor}')       
```    
- resultado (os 300 números do vetor criado para o documento):
```
[0.012920759618282318, -0.04087100550532341, -0.004641648381948471, -0.012162852101027966, -0.02776987850666046, -0.0918320044875145, -0.06542444974184036, 0.04195054993033409, 0.06541111320257187, 0.03410173952579498, 0.053852327167987823, 0.08504167199134827, 0.015855900943279266, -0.10877902060747147, -0.027269462123513222, -0.03065289556980133, 0.019806699827313423, 0.15604981780052185, -0.07018765807151794, 0.021327996626496315, -0.06504599750041962, 0.0001805076317396015, 0.0027813573833554983, -0.04120273515582085, -0.013223088346421719, -0.022825639694929123, 0.01981930062174797, -0.07640938460826874, 0.04173220321536064, 0.01945299655199051, -0.05070667713880539, -0.0034932922571897507, 0.025652656331658363, -0.03199231997132301, -0.02671872265636921, 0.090485580265522, 0.07531122863292694, 0.048449743539094925, 0.06450975686311722, -0.08393879979848862, 0.010060732252895832, -0.030927082523703575, -0.03262617066502571, -0.06292419135570526, 0.05825110152363777, 0.07299591600894928, -0.013516977429389954, -0.027202310040593147, -0.0314081609249115, -0.08265126496553421, 0.01811525784432888, -0.043320558965206146, -0.06904895603656769, -0.07839549332857132, -0.04671350494027138, -0.013496444560587406, -0.020125245675444603, -0.05444709211587906, -0.05520816519856453, -0.05243867263197899, 0.030940717086195946, 0.07326696813106537, 0.03600494563579559, -0.034752897918224335, -0.0824853852391243, -0.027615351602435112, 0.0545431450009346, -0.07841676473617554, -0.04878977686166763, 0.04111477732658386, -0.06795225292444229, 0.10315079987049103, -0.1361357569694519, 0.031071996316313744, -0.11413531005382538, 0.034696534276008606, -0.12201303988695145, -0.11269417405128479, -0.010762154124677181, 0.010574258863925934, 0.0007393507985398173, 0.017735207453370094, -0.0050237737596035, 0.002368019428104162, 0.04512730613350868, -0.03597823530435562, 0.08752517402172089, 0.03505466878414154, 0.10579521954059601, -0.0019503976218402386, -0.10433682799339294, 0.015327941626310349, -0.06886322051286697, -0.0007313766400329769, -0.07066147029399872, -0.03380739316344261, -0.09928388893604279, 0.04090247303247452, -0.05565746873617172, -0.01526807714253664, 0.0368751659989357, 0.03262645751237869, 0.01462192740291357, -0.05435190349817276, -0.028663190081715584, 0.02889184094965458, 0.05761609598994255, 0.1505211442708969, 0.04083414003252983, 0.03921075165271759, -0.017866753041744232, -0.04700610041618347, 0.017440928146243095, 0.07673367857933044, 0.005186847876757383, -0.009692703373730183, 0.04838212579488754, -0.06802864372730255, 0.06181338056921959, 0.07300572097301483, -0.003749487455934286, -0.09380146116018295, 0.02069896087050438, 0.03695311397314072, 0.07580381631851196, -0.06309019774198532, -0.07735433429479599, 0.05104733631014824, -0.0037158785853534937, 0.09973426908254623, -0.062467530369758606, -0.027031419798731804, -0.0005807244451716542, 0.0770687535405159, -0.016339099034667015, 0.056829120963811874, -0.0043184878304600716, 0.04647742211818695, -0.04316911846399307, -0.09943560510873795, -0.013671353459358215, -0.06822573393583298, 0.0038762527983635664, 0.032989270985126495, 0.03303777426481247, 0.00712445005774498, -0.05544329807162285, -0.0748615711927414, -0.029634563252329826, 0.036060381680727005, 0.09232290089130402, 0.012167858891189098, -0.0744403749704361, 0.0019100881181657314, 0.01921442337334156, 0.03672565519809723, 0.12814582884311676, 0.10923463851213455, -0.010623936541378498, -0.04302908480167389, -0.0005301024066284299, 0.036657579243183136, 0.0077194287441670895, 0.025975247845053673, 0.07468866556882858, -0.0689973533153534, -0.08426027745008469, -0.08752048015594482, -0.046413782984018326, 0.01733304001390934, -0.10253778100013733, 0.06422658264636993, -0.0414365790784359, -0.014761026948690414, 0.07573938369750977, 0.021467110142111778, 0.055104583501815796, -0.0703710988163948, -0.018213676288723946, -0.0007552398601546884, 0.014600371941924095, -0.004022100009024143, -0.07852470874786377, 0.1213875487446785, 0.06272368878126144, -0.005116534885019064, -0.033831678330898285, 0.059692684561014175, 0.06671664118766785, -0.07839513570070267, 0.012300166301429272, -0.15148226916790009, 0.051535625010728836, 0.05420878157019615, -0.02225036360323429, -0.08883121609687805, -0.05721248686313629, 0.05417277291417122, 0.025657080113887787, -0.007609791122376919, 0.05027920752763748, 0.005914983339607716, 0.017599258571863174, -0.029538771137595177, -0.08776085823774338, -0.015400253236293793, 0.02319120243191719, 0.014885499142110348, -0.001613891334272921, 0.016327839344739914, 0.05984141677618027, 0.036976683884859085, -0.02593727968633175, 0.021512366831302643, 0.0954013466835022, 0.07817462831735611, -0.006996919866651297, -0.06498533487319946, 0.05237214267253876, -0.04124525934457779, -0.12574072182178497, 0.006290863733738661, -0.09249759465456009, 0.002293388359248638, -0.09310562908649445, -0.058742012828588486, 0.020117659121751785, -0.03490052372217178, -0.07356352359056473, -0.035756915807724, 0.09404479712247849, -0.023259861394762993, -0.03035222738981247, 0.007756620645523071, -0.08849190175533295, -0.014709297567605972, -0.0028694546781480312, 0.02071950025856495, -0.04880058020353317, 0.03152783587574959, 0.07081560045480728, 0.08089449256658554, 0.05952193960547447, -0.08201968669891357, -0.058699093759059906, -0.04740528762340546, -0.061847537755966187, 0.0572635792195797, 0.0556931272149086, -0.03509797900915146, 0.08692574501037598, 0.009880123659968376, -0.025141024962067604, -0.06957271695137024, -0.0022108787670731544, -0.021313413977622986, -0.009271993301808834, 0.06245402991771698, 0.15824054181575775, -0.04907536506652832, 0.001510693458840251, -0.03677589073777199, -0.0004158232477493584, 0.09581030905246735, -0.07913174480199814, -0.04817294329404831, -0.026326734572649002, 0.04445841535925865, 0.06241689994931221, 0.009005681611597538, 0.036332953721284866, 0.02881227247416973, -0.07866812497377396, -0.005385654978454113, 0.01598750241100788, -0.02502330020070076, 0.005288539454340935, 0.09915148466825485, 0.02919670194387436, -0.054862625896930695, 0.07328227907419205, -0.03035156987607479, 0.09086304157972336, -0.020074155181646347, -0.026729542762041092, 0.04237302765250206, 0.12820285558700562, 0.0386740118265152, -0.11401347070932388, -0.030602941289544106, -0.024829067289829254, -0.007230975199490786, 0.0025534487795084715, 0.042160242795944214, 0.0380859449505806, 0.15053176879882812, -0.028883542865514755, 0.054057639092206955, 0.00844051968306303, -0.029573174193501472]
```


## Dicas de uso: <a name="dicas">
- gravar os vetores, textos e metadados dos documentos no [`ElasticSearch`](https://www.elastic.co/pt/), e usar os recursos de pesquisas: More Like This, vetoriais e por proximidade de termos como disponibilizado no componente [`PesquisaElasticFacil`](https://github.com/luizanisio/PesquisaElasticFacil) ou criar sua própria estrutura de dados com [`essas dicas`](https://github.com/luizanisio/PesquisaElasticFacil/blob/main/docs/ElasticQueries.md).
- gravar os vetores, textos e metadados no [`SingleStore`](https://www.singlestore.com/) e criar views de similaridade para consulta em tempo real dos documentos inseridos na base, incluindo filtros de metadados e textuais como nos exemplos disponíveis aqui: [`dicas SingleStore`](./docs/readme_dicas.md).
